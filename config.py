"""Central configuration for the RAG pipeline."""
from pathlib import Path

ROOT = Path(__file__).parent
STORAGE_DIR = ROOT / "storage"
CHROMA_DIR = STORAGE_DIR / "chroma"
BM25_PATH = STORAGE_DIR / "bm25_index.pkl"

COLLECTION_NAME = "fastapi_docs"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
OLLAMA_MODEL = "qwen3:4b"

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
