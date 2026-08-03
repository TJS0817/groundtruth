"""FastAPI backend: serves the web UI and exposes rag.query() as a JSON API."""
import logging
import time
from pathlib import Path
from typing import Optional

import ollama
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import config
from src import rag

logger = logging.getLogger(__name__)

WEB_DIR = Path(__file__).resolve().parents[1] / "web"

app = FastAPI(title="GroundTruth")
app.mount("/static", StaticFiles(directory=WEB_DIR / "static"), name="static")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.monotonic()
    response = await call_next(request)
    elapsed_ms = (time.monotonic() - start) * 1000
    logger.info("%s %s -> %d (%.0fms)", request.method, request.url.path, response.status_code, elapsed_ms)
    return response


def require_api_key(x_api_key: Optional[str] = Header(default=None)):
    """No-op when API_KEY is unset (local/dev use). Set API_KEY to require a matching header."""
    if config.API_KEY and x_api_key != config.API_KEY:
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/health")
def health():
    try:
        ollama.Client(host=config.OLLAMA_HOST, timeout=5).list()
        ollama_ok = True
    except Exception:
        ollama_ok = False
    return JSONResponse(
        {"status": "ok" if ollama_ok else "degraded", "ollama_reachable": ollama_ok},
        status_code=200 if ollama_ok else 503,
    )


@app.post("/api/ask", dependencies=[Depends(require_api_key)])
def ask(req: AskRequest):
    try:
        result = rag.query(req.question)
    except Exception:
        logger.exception("ask failed for question=%r", req.question[:200])
        raise HTTPException(status_code=502, detail="Upstream generation failed or timed out")
    return {
        "answer": result["answer"],
        "citations": result["citations"],
        "refused": result["refused"],
        "retrieved": [
            {"id": c["id"], "source": c["source"], "section": c["section"], "text": c["text"]}
            for c in result["retrieved"]
        ],
    }
