"""CLI: fetch docs, chunk, embed, and build the dense + BM25 indexes."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import ingest

if __name__ == "__main__":
    ingest.run()
