"""Evaluation harness: context recall (keyword overlap), faithfulness + answer relevance (LLM-judge)."""
import json
import re

import ollama

import config
from src import rag

JUDGE_PROMPT = """You are grading a RAG system's answer. Respond with ONLY a single integer from 0 to 5, nothing else.

Question: {question}

Context provided to the system:
{context}

Answer given by the system:
{answer}

Score how well the answer is: {criterion}
Score (0-5 only):"""

FAITHFULNESS_CRITERION = "supported by the context — every claim must be traceable to the context, with no invented information."
RELEVANCE_CRITERION = "directly and completely addressing the question asked."


def _judge(question: str, context: str, answer: str, criterion: str) -> float:
    response = ollama.chat(
        model=config.OLLAMA_MODEL,
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(
            question=question, context=context, answer=answer, criterion=criterion,
        )}],
    )
    match = re.search(r"\d", response["message"]["content"])
    return int(match.group()) / 5.0 if match else 0.0


def _context_recall(expected_facts: list[str], retrieved: list[dict]) -> float:
    if not expected_facts:
        return 1.0
    combined = " ".join(c["text"].lower() for c in retrieved)
    hits = sum(1 for fact in expected_facts if fact.lower() in combined)
    return hits / len(expected_facts)


def run() -> list[dict]:
    with open(config.ROOT / "tests" / "test_queries.json") as f:
        test_cases = json.load(f)

    report = []
    for case in test_cases:
        result = rag.query(case["query"])
        context = "\n\n".join(c["text"] for c in result["retrieved"])

        row = {
            "id": case["id"],
            "type": case["type"],
            "query": case["query"],
            "answer": result["answer"],
            "refused": result["refused"],
            "context_recall": _context_recall(case["expected_facts"], result["retrieved"]),
        }

        if case["type"] == "negative":
            row["refusal_correct"] = result["refused"]
        else:
            row["faithfulness"] = _judge(case["query"], context, result["answer"], FAITHFULNESS_CRITERION)
            row["answer_relevance"] = _judge(case["query"], context, result["answer"], RELEVANCE_CRITERION)

        report.append(row)
    return report


def print_report(report: list[dict]) -> None:
    for row in report:
        print(f"\n=== {row['id']} ({row['type']}) ===")
        print(f"Q: {row['query']}")
        print(f"A: {row['answer'][:300]}")
        print(f"context_recall: {row['context_recall']:.2f}")
        if row["type"] == "negative":
            status = "PASS" if row["refusal_correct"] else "FAIL"
            print(f"refusal_correct: {row['refusal_correct']} [{status}]")
        else:
            print(f"faithfulness: {row['faithfulness']:.2f}")
            print(f"answer_relevance: {row['answer_relevance']:.2f}")


if __name__ == "__main__":
    print_report(run())
