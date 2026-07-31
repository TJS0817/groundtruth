"""CLI: ask a single question against the built index."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import rag

if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or input("Question: ")
    result = rag.query(question)
    print("\n--- Answer ---")
    print(result["answer"])
    if result["retrieved"]:
        print("\n--- Retrieved chunks ---")
        for c in result["retrieved"]:
            print(f"  {c['source']}#{c['section']}")
