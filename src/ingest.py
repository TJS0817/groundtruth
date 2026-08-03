"""Fetch FastAPI docs, chunk them with metadata, and populate the vector + BM25 indexes."""
import logging
import pickle
import re
from datetime import date
from pathlib import Path

import chromadb
import requests
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

import config

logger = logging.getLogger(__name__)

# FastAPI's docs reference code samples via {* ../../docs_src/foo/bar.py *}
# (optionally with a trailing "hl[...]" line-highlight annotation) instead of
# inlining the code. Left unresolved, facts that only exist in the referenced
# .py file (e.g. a numeric default) are never indexed and the system correctly
# but needlessly refuses questions about them.
INCLUDE_RE = re.compile(r"\{\*\s*([^\s*]+\.py)[^*]*\*\}")


def _inline_code_snippets(text: str, cache: dict[str, str | None]) -> str:
    """Replaces each {* path *} include directive with the actual code, fetched once per path."""

    def replace(m: re.Match) -> str:
        path = m.group(1)
        if path not in cache:
            repo_path = re.sub(r"^(\.\./)+", "", path)
            url = f"{config.FASTAPI_REPO_BASE}/{repo_path}"
            try:
                resp = requests.get(url, timeout=15)
                resp.raise_for_status()
                cache[path] = resp.text
            except requests.RequestException as e:
                logger.warning("skipping code include %s: %s", path, e)
                cache[path] = None
        code = cache[path]
        return f"\n```python\n{code}\n```\n" if code else ""

    return INCLUDE_RE.sub(replace, text)


def fetch_docs() -> list[dict]:
    """Download the configured FastAPI doc pages, inlining referenced code
    snippets. Skips pages that fail to fetch."""
    docs = []
    snippet_cache: dict[str, str | None] = {}
    for page in config.FASTAPI_DOC_PAGES:
        url = f"{config.FASTAPI_DOCS_BASE}/{page}"
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.warning("skipping %s: %s", page, e)
            continue
        docs.append({"source": page, "text": _inline_code_snippets(resp.text, snippet_cache)})
    return docs


def _split_by_headers(text: str) -> list[tuple[str, str]]:
    """Split markdown into (section_title, section_body) preserving header context."""
    lines = text.splitlines()
    sections: list[tuple[str, str]] = []
    current_title = "Introduction"
    current_lines: list[str] = []
    for line in lines:
        m = re.match(r"^(#{2,3})\s+(.*)", line)
        if m:
            if current_lines:
                sections.append((current_title, "\n".join(current_lines).strip()))
            current_title = m.group(2).strip()
            current_lines = []
        else:
            current_lines.append(line)
    if current_lines:
        sections.append((current_title, "\n".join(current_lines).strip()))
    return [(t, b) for t, b in sections if b]


def _split_oversized(body: str) -> list[str]:
    """Recursively split a section's body on paragraph/sentence boundaries with overlap."""
    if len(body) <= config.CHUNK_SIZE:
        return [body]

    paragraphs = [p for p in body.split("\n\n") if p.strip()]
    if len(paragraphs) <= 1:
        paragraphs = re.split(r"(?<=[.!?])\s+", body)

    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        candidate = f"{current}\n\n{para}".strip() if current else para
        if len(candidate) > config.CHUNK_SIZE and current:
            chunks.append(current)
            overlap = current[-config.CHUNK_OVERLAP:]
            current = f"{overlap}\n\n{para}".strip()
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def chunk_document(source: str, text: str) -> list[dict]:
    """Recursive chunker: split on headers first, then on paragraph/sentence within oversized sections."""
    today = date.today().isoformat()
    chunks = []
    for section, body in _split_by_headers(text):
        for i, piece in enumerate(_split_oversized(body)):
            chunks.append(
                {
                    "id": f"{source}#{section}#{i}",
                    "text": f"{section}\n\n{piece}".strip(),
                    "source": source,
                    "section": section,
                    "ingested_date": today,
                }
            )
    return chunks


def run() -> None:
    config.STORAGE_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Fetching FastAPI docs...")
    docs = fetch_docs()
    logger.info("fetched %d/%d pages", len(docs), len(config.FASTAPI_DOC_PAGES))

    logger.info("Chunking...")
    chunks = [c for d in docs for c in chunk_document(d["source"], d["text"])]
    logger.info("produced %d chunks", len(chunks))

    logger.info("Embedding with %s...", config.EMBEDDING_MODEL)
    embedder = SentenceTransformer(config.EMBEDDING_MODEL)
    embeddings = embedder.encode([c["text"] for c in chunks], show_progress_bar=True).tolist()

    logger.info("Writing Chroma collection...")
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    try:
        client.delete_collection(config.COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(config.COLLECTION_NAME)
    collection.add(
        ids=[c["id"] for c in chunks],
        embeddings=embeddings,
        documents=[c["text"] for c in chunks],
        metadatas=[
            {"source": c["source"], "section": c["section"], "ingested_date": c["ingested_date"]}
            for c in chunks
        ],
    )

    logger.info("Building BM25 index...")
    tokenized = [c["text"].lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)
    with open(config.BM25_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "ids": [c["id"] for c in chunks], "texts": [c["text"] for c in chunks]}, f)

    logger.info("Done. Indexed %d chunks from %d documents.", len(chunks), len(docs))


if __name__ == "__main__":
    run()
