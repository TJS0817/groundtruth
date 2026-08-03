"""Guardrailed answer generation: strict grounding, mandatory citations, fail-closed refusal."""
import re

import ollama

import config

_client = ollama.Client(host=config.OLLAMA_HOST, timeout=config.OLLAMA_TIMEOUT_SECONDS)

SYSTEM_PROMPT = """You are a documentation assistant. Answer ONLY using the provided context chunks.

Rules:
- Every factual claim must be followed by a citation tag in the exact form [source: <source>#<section>], copied verbatim from the chunk's tag below.
- Do not use any knowledge beyond the provided context.
- If the context does not contain enough information to answer, respond with EXACTLY this sentence and nothing else:
"{refusal}"

Example of correctly formatted output, given a chunk tagged [source: tutorial/cors.md#Use `CORSMiddleware` {{ #use-corsmiddleware }}]:
Question: What is the default value of expose_headers in CORSMiddleware?
Answer: The default value of expose_headers is an empty list. [source: tutorial/cors.md#Use `CORSMiddleware` {{ #use-corsmiddleware }}]
""".format(refusal=config.REFUSAL_TEXT)


def _format_context(chunks: list[dict]) -> str:
    blocks = []
    for c in chunks:
        tag = f"{c['source']}#{c['section']}"
        blocks.append(f"[source: {tag}]\n{c['text']}")
    return "\n\n---\n\n".join(blocks)


def _valid_tags(chunks: list[dict]) -> set[str]:
    return {f"{c['source']}#{c['section']}" for c in chunks}


def _split_claims(answer: str) -> tuple[list[tuple[str, str]], str]:
    """Splits the answer at each citation tag into (claim_text, tag) pairs,
    plus any trailing text after the last tag (which has no citation)."""
    parts = re.split(r"(\[source:\s*[^\]]+\])", answer)
    claims: list[tuple[str, str]] = []
    buffer = ""
    for part in parts:
        m = re.match(r"^\[source:\s*([^\]]+)\]$", part)
        if m:
            claims.append((buffer.strip(), m.group(1).strip()))
            buffer = ""
        else:
            buffer += part
    return claims, buffer.strip()


def _verify_claims(answer: str, chunks: list[dict]) -> tuple[str, list[str]]:
    """Keeps only claims whose citation tag matches a retrieved chunk, dropping
    hallucinated-citation or uncited claims individually rather than refusing
    the whole answer over one bad tag."""
    valid = _valid_tags(chunks)
    claims, _trailing_uncited = _split_claims(answer)
    kept_text, kept_tags = [], []
    for text, tag in claims:
        if text and tag in valid:
            kept_text.append(f"{text} [source: {tag}]")
            kept_tags.append(tag)
    return " ".join(kept_text), kept_tags


def generate(query: str, chunks: list[dict]) -> dict:
    """Returns {"answer": str, "citations": list[str], "refused": bool}."""
    if not chunks:
        return {"answer": config.REFUSAL_TEXT, "citations": [], "refused": True}

    context = _format_context(chunks)
    user_prompt = f"Context:\n{context}\n\nQuestion: {query}"

    response = _client.chat(
        model=config.OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        # Greedy decoding: grounded answers must be reproducible. At Ollama's
        # default temperature the model non-deterministically refuses on context
        # that does support an answer.
        options={"temperature": 0},
    )
    answer = response["message"]["content"].strip()

    if config.REFUSAL_TEXT in answer:
        return {"answer": config.REFUSAL_TEXT, "citations": [], "refused": True}

    filtered, kept_tags = _verify_claims(answer, chunks)
    if not filtered:
        # Fail closed: no claim in the answer survived citation verification.
        return {"answer": config.REFUSAL_TEXT, "citations": [], "refused": True}

    return {"answer": filtered, "citations": kept_tags, "refused": False}
