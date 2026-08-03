# GroundTruth

A fully local Retrieval-Augmented Generation pipeline over the FastAPI documentation.
No API keys, no hosted services — embeddings, reranking, generation, and evaluation
all run on your machine. Named for what it refuses to compromise on: every answer is
either grounded in a verified citation, or it says it doesn't know.

## Use case

Built as an **internal engineering-docs assistant a company self-hosts for its own
team**: point it at your internal docs, runbooks, or API references instead of the
FastAPI corpus, and it answers employee questions with mandatory citations and a
hard refusal — never an invented answer — when the docs don't cover the question.
Fully local by design, so nothing in the corpus or the questions asked against it
leaves the company's own infrastructure. It is not built for high-throughput
multi-tenant serving (see [Limitations](#limitations)).

### What to ask it

With the FastAPI docs indexed (the demo corpus), it's tuned for the three question
shapes it's evaluated on:

- **Precise fact lookup** — e.g. "What's the default `max_age` for CORSMiddleware?",
  "What's the default access token expiration in the JWT tutorial?"
- **Multi-hop synthesis** — questions needing two doc sections combined, e.g. "How do
  dependency injection and OAuth2PasswordBearer work together?"
- **Anything outside FastAPI's docs** — it refuses rather than guesses. That's the
  point of the system, not a gap to work around.

Try it via `scripts/ask.py "your question"` (CLI) or `scripts/serve.py` + the web UI
at `localhost:8000` (citations you can click through to the source chunk).

**Not good for:** general knowledge (refuses by design), high query volume (single
local model, one request at a time — see [Limitations](#limitations)), or anything
where a citation being *structurally* real isn't enough on its own — see the known
semantic-verification gap in [Guardrails](#guardrails) before using this for anything
like a compliance or legal use case.

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
# Build the index (fetches 15 FastAPI doc pages + referenced code snippets, ~360 chunks)
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
`[source: file#section]` citation per claim. After generation, the answer is split at
each citation tag and **every claim is checked individually**: a claim citing a chunk
that wasn't actually retrieved is dropped, a claim with no citation at all is dropped,
and only claims that survive are returned. If nothing survives, the response falls
back to a fixed refusal string. This is deliberately per-claim rather than
all-or-nothing — one hallucinated citation drops that one sentence, not the whole
answer — while still failing closed: nothing ships without a verified citation behind
it. See `tests/test_generation.py` for the behavior under valid/hallucinated/uncited
citations.

**Known gap: citation checking is structural, not semantic.** It verifies a claim
cites a chunk that was actually retrieved — it does not verify the claim accurately
reflects what that chunk says. Confirmed case: asked how dependency injection works
with `OAuth2PasswordBearer`, the system answered "FastAPI will verify the token...
if valid" citing `tutorial/security/first-steps.md#Use it` — but that chunk only
says the dependency "will provide a `str` that is assigned to the parameter `token`,"
with no claim about verification (actual token validation is covered later, in the
JWT tutorial). The citation is real and the chunk is real; the claim attached to it
overstates what the chunk says. This is reproducible (greedy decoding) and is what
`tests/test_queries.json`'s `multi_hop_dependencies_oauth2` faithfulness score of
0.40 is catching — not eval noise. Closing this needs a semantic per-claim verifier
(an extra judge call per claim at generation time, with real added latency), which
isn't implemented.

## Evaluation

`tests/test_queries.json` holds three scenarios: a specific-fact lookup, a multi-hop
synthesis question, and an out-of-scope question that must trigger the refusal.
Tuning knobs (chunk size, top-k, model names) live in `config.py`.

## Configuration

Deployment-relevant settings are environment variables (see `.env.example`); copy it
to `.env` and adjust, or set them directly in your shell/orchestrator:

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Where the app reaches Ollama |
| `OLLAMA_MODEL` | `qwen3:4b` | Generation model |
| `OLLAMA_TIMEOUT_SECONDS` | `480` | Per-request timeout to Ollama |
| `API_KEY` | unset (auth disabled) | Required `X-API-Key` header value for `POST /api/ask` |
| `LOG_LEVEL` | `INFO` | Python logging level |
| `SERVER_HOST` / `SERVER_PORT` | `127.0.0.1` / `8000` | Where `scripts/serve.py` binds |
| `STORAGE_DIR` | `./storage` | Where the Chroma DB and BM25 index live |

Everything else (chunk size, retrieval `top_k`, model choices for embedding/rerank)
is a build-time tuning knob in `config.py`, not an env var — those change what gets
indexed, not how the service is deployed.

## Deployment

```bash
cp .env.example .env   # set API_KEY before exposing this beyond localhost
docker compose up -d --build
docker compose exec ollama ollama pull qwen3:4b   # one-time, downloads the model into the ollama volume
docker compose exec app python scripts/build_index.py   # one-time, builds the index into the storage volume
```

The app then listens on `http://localhost:8000`, talking to the `ollama` service over
the compose network. Both the index (`storage_data` volume) and the pulled model
(`ollama_data` volume) persist across restarts.

**Not build-tested.** This Dockerfile/compose setup was traced by hand against
`config.py` and never run through an actual `docker build` (no Docker daemon was
available while writing it) — verify it works before relying on it. One specific,
known risk: `chromadb` has historically required a newer `libsqlite3` than some slim
Debian base images ship. If the `app` container fails on Chroma import with an sqlite
version error, install `pysqlite3-binary` and alias it over the stdlib `sqlite3`
module (see chromadb's own troubleshooting docs) rather than downgrading chromadb.

Operational surface for this deployment:
- **Health check:** `GET /health` — returns `503` if Ollama isn't reachable, so a
  container orchestrator can detect and restart a broken instance.
- **Auth:** if `API_KEY` is set, `POST /api/ask` requires a matching `X-API-Key`
  header; unset means the endpoint is open (fine for local dev, not for anything else).
- **Logging:** structured request logs (method, path, status, latency) to stdout at
  `LOG_LEVEL`, ready to be picked up by any log collector.
- **Input validation:** questions outside 3–2000 characters are rejected with `422`
  before reaching the model.
- **Timeouts:** a stuck/unreachable Ollama call surfaces as a clean `502` rather than
  hanging the request indefinitely.

## Limitations

Known constraints worth knowing before adopting this for real use, not fixed here:
- **Single-request throughput.** One local model instance handles one generation at a
  time; concurrent users queue behind each other. Fine for a small team, not for
  wide rollout without a request queue and/or a hosted/GPU-backed model.
- **No multi-tenancy.** One corpus, one index, one shared API key. Separate customers
  or business units need separate deployments, not a shared one.
- **No persistent conversation history.** Each question is independent; the web UI's
  session view is client-side only and resets on page reload.
- **CPU generation is slow.** Tens of seconds to a few minutes per question on a 4B
  model on CPU. A GPU or a larger/hosted model would be the next lever, not something
  this deployment config changes on its own.
