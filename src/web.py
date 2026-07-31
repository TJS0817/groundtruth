"""FastAPI backend: serves the web UI and exposes rag.query() as a JSON API."""
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src import rag

WEB_DIR = Path(__file__).resolve().parents[1] / "web"

app = FastAPI(title="RAG System")
app.mount("/static", StaticFiles(directory=WEB_DIR / "static"), name="static")


class AskRequest(BaseModel):
    question: str


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


@app.post("/api/ask")
def ask(req: AskRequest):
    result = rag.query(req.question)
    return {
        "answer": result["answer"],
        "citations": result["citations"],
        "refused": result["refused"],
        "retrieved": [
            {"id": c["id"], "source": c["source"], "section": c["section"], "text": c["text"]}
            for c in result["retrieved"]
        ],
    }
