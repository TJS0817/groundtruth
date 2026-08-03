"""CLI: run the evaluation suite against the built index."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from . import eval as eval_

if __name__ == "__main__":
    eval_.print_report(eval_.run())
