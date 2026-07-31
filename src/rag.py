"""Ties retrieval + generation into a single query() call."""
from src import generation, retrieval


def query(question: str) -> dict:
    chunks = retrieval.search(question)
    result = generation.generate(question, chunks)
    result["retrieved"] = chunks
    return result
