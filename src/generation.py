"""Guardrailed answer generation: strict grounding, mandatory citations, fail-closed refusal."""
import re

import ollama

import config

SYSTEM_PROMPT = """You are a documentation assistant. Answer ONLY using the provided context chunks.

Rules:
- Every factual claim must be followed by a citation tag in the exact form [source: <source>#<section>], copied verbatim from the chunk's tag below.
- Do not use any knowledge beyond the provided context.
- If the context does not contain enough information to answer, respond with EXACTLY this sentence and nothing else:
"{refusal}"
""".format(refusal=config.REFUSAL_TEXT)


def _format_context(chunks: list[dict]) -> str:
    blocks = []
    for c in chunks:
        tag = f"{c['source']}#{c['section']}"
        blocks.append(f"[source: {tag}]\n{c['text']}")
    return "\n\n---\n\n".join(blocks)


def _valid_tags(chunks: list[dict]) -> set[str]:
    return {f"{c['source']}#{c['section']}" for c in chunks}


def _citations_in(answer: str) -> list[str]:
    return re.findall(r"\[source:\s*([^\]]+)\]", answer)


def generate(query: str, chunks: list[dict]) -> dict:
    """Returns {"answer": str, "citations": list[str], "refused": bool}."""
    if not chunks:
        return {"answer": config.REFUSAL_TEXT, "citations": [], "refused": True}

    context = _format_context(chunks)
    user_prompt = f"Context:\n{context}\n\nQuestion: {query}"

    response = ollama.chat(
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

    valid = _valid_tags(chunks)
    cited = [tag.strip() for tag in _citations_in(answer)]
    hallucinated = [tag for tag in cited if tag not in valid]

    if not cited or hallucinated:
        # Fail closed: no citations, or a citation that doesn't match retrieved context.
        return {"answer": config.REFUSAL_TEXT, "citations": [], "refused": True}

    return {"answer": answer, "citations": cited, "refused": False}
