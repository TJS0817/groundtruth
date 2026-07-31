"""Fetch FastAPI docs, chunk them with metadata, and populate the vector + BM25 indexes."""
import pickle
import re
from datetime import date
from pathlib import Path

import chromadb
import requests
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

import config


def fetch_docs() -> list[dict]:
    """Download the configured FastAPI doc pages. Skips pages that fail to fetch."""
    docs = []
    for page in config.FASTAPI_DOC_PAGES:
        url = f"{config.FASTAPI_DOCS_BASE}/{page}"
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"  skip {page}: {e}")
            continue
        docs.append({"source": page, "text": resp.text})
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

    print("Fetching FastAPI docs...")
    docs = fetch_docs()
    print(f"  fetched {len(docs)}/{len(config.FASTAPI_DOC_PAGES)} pages")

    print("Chunking...")
    chunks = [c for d in docs for c in chunk_document(d["source"], d["text"])]
    print(f"  produced {len(chunks)} chunks")

    print(f"Embedding with {config.EMBEDDING_MODEL}...")
    embedder = SentenceTransformer(config.EMBEDDING_MODEL)
    embeddings = embedder.encode([c["text"] for c in chunks], show_progress_bar=True).tolist()

    print("Writing Chroma collection...")
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

    print("Building BM25 index...")
    tokenized = [c["text"].lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)
    with open(config.BM25_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "ids": [c["id"] for c in chunks], "texts": [c["text"] for c in chunks]}, f)

    print(f"Done. Indexed {len(chunks)} chunks from {len(docs)} documents.")


if __name__ == "__main__":
    run()
