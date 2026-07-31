# Local RAG System

A fully local Retrieval-Augmented Generation pipeline over the FastAPI documentation.
No API keys, no hosted services — embeddings, reranking, generation, and evaluation
all run on your machine.

## Pipeline

```
ingest:  fetch docs -> header-aware recursive chunking (+metadata) -> embed -> Chroma + BM25
query:   dense search + BM25 -> reciprocal rank fusion -> cross-encoder rerank
         -> guardrailed prompt -> Ollama -> citation verification (fail closed)
eval:    context recall (keyword overlap) + faithfulness & answer relevance (LLM-judge)
```

## Stack

| Layer | Choice |
|---|---|
| Vector DB | ChromaDB (embedded, persisted to `storage/`) |
| Embeddings | `BAAI/bge-small-en-v1.5` |
| Keyword search | `rank_bm25` (BM25Okapi) |
| Fusion | Reciprocal Rank Fusion |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Generation | Ollama, `qwen3:4b` |

## Setup

Requires [Ollama](https://ollama.com) running locally with the model pulled:

```bash
ollama pull qwen3:4b

python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt   # Windows
# source .venv/bin/activate && pip install -r requirements.txt  # macOS/Linux
```

## Usage

```bash
# Build the index (fetches 15 FastAPI doc pages, ~300 chunks)
.venv/Scripts/python.exe scripts/build_index.py

# Ask a question
.venv/Scripts/python.exe scripts/ask.py "What is the default max_age for CORSMiddleware?"

# Run the evaluation suite
.venv/Scripts/python.exe scripts/run_eval.py
```

## Web UI

A dark, terminal-inspired web UI for asking questions interactively — same
guardrailed pipeline, no CLI required.

```bash
.venv/Scripts/python.exe scripts/serve.py
# open http://127.0.0.1:8000
```

Ask a question and watch it move through Retrieving → Reranking → Generating →
Verifying. The result shows a **Grounded** or **Refused** badge, the answer text,
clickable citation chips that jump to and highlight the matching retrieved chunk,
and a collapsible list of everything retrieved for that query. Local generation on
CPU is slow (tens of seconds to a few minutes per question on a 4B model) — the
staged status track is there so the wait isn't a blank screen.

Built with FastAPI + vanilla HTML/CSS/JS (no frontend framework or build step —
`web/index.html`, `web/static/style.css`, `web/static/app.js`). The answer is not
token-streamed: the citation-verification guardrail runs after the full response is
generated, so streaming partial text would risk showing content the verifier later
rejects. The typewriter-style reveal you see is a client-side animation of the
already-verified final answer, not a live stream.

## Guardrails

The generation prompt permits answers only from retrieved context and requires a
`[source: file#section]` citation per claim. After generation, every citation tag is
verified against the actually-retrieved chunks; an answer with no citations or a
fabricated one is replaced with the refusal string rather than returned. Failure mode
is a refusal, never an ungrounded answer.

## Evaluation

`tests/test_queries.json` holds three scenarios: a specific-fact lookup, a multi-hop
synthesis question, and an out-of-scope question that must trigger the refusal.
Tuning knobs (chunk size, top-k, model names) live in `config.py`.
