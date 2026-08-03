"""Central configuration for the RAG pipeline.

Deployment-relevant values (Ollama location/model, auth, timeouts, logging)
read from the environment so the same image can run in different
environments without a code change. Everything else (chunking, retrieval
tuning) is a build-time knob edited directly here, not an env var.
"""
import logging
import os
from pathlib import Path

ROOT = Path(__file__).parent
STORAGE_DIR = Path(os.environ.get("STORAGE_DIR") or ROOT / "storage")
CHROMA_DIR = STORAGE_DIR / "chroma"
BM25_PATH = STORAGE_DIR / "bm25_index.pkl"

COLLECTION_NAME = "fastapi_docs"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Ollama connection. OLLAMA_HOST lets the app reach a sidecar/remote Ollama
# instance (e.g. the "ollama" service in docker-compose) instead of localhost.
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:4b")
OLLAMA_TIMEOUT_SECONDS = float(os.environ.get("OLLAMA_TIMEOUT_SECONDS", "300"))

# API auth: if unset, the /api/ask endpoint is open (local/dev use). Set this
# to require an X-API-Key header matching it.
API_KEY = os.environ.get("API_KEY")

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
SERVER_HOST = os.environ.get("SERVER_HOST", "127.0.0.1")
SERVER_PORT = int(os.environ.get("SERVER_PORT", "8000"))

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)

# Chunking
CHUNK_SIZE = 800          # max chars per chunk
CHUNK_OVERLAP = 120       # char overlap between adjacent chunks within a section

# Retrieval
DENSE_TOP_K = 15
BM25_TOP_K = 15
RRF_K = 60                # standard RRF smoothing constant
RERANK_TOP_K = 5          # final number of chunks passed to the LLM

# Generation
REFUSAL_TEXT = "I don't have enough information in the provided documents to answer that."

FASTAPI_DOCS_BASE = "https://raw.githubusercontent.com/fastapi/fastapi/master/docs/en/docs"
FASTAPI_DOC_PAGES = [
    "tutorial/first-steps.md",
    "tutorial/path-params.md",
    "tutorial/query-params.md",
    "tutorial/body.md",
    "tutorial/query-params-str-validations.md",
    "tutorial/dependencies/index.md",
    "tutorial/security/oauth2-jwt.md",
    "tutorial/security/first-steps.md",
    "tutorial/background-tasks.md",
    "tutorial/middleware.md",
    "tutorial/cors.md",
    "advanced/settings.md",
    "tutorial/testing.md",
    "tutorial/sql-databases.md",
    "tutorial/bigger-applications.md",
]
