"""CLI: run the web UI server."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import uvicorn

import config

if __name__ == "__main__":
    uvicorn.run("src.web:app", host=config.SERVER_HOST, port=config.SERVER_PORT, reload=False)
