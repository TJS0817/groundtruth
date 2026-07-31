# RAG System — Design Spec

Date: 2026-07-31

## Goal

A production-grade, fully local Retrieval-Augmented Generation system: ingest a
technical doc set, retrieve with hybrid dense+keyword search, rerank, generate
grounded/cited answers, and evaluate the whole pipeline against a fixed query
set (fact lookup, multi-hop, out-of-scope).

## Stack

| Layer | Choice | Rationale |
|---|---|---|
| Language | Python 3.11 | Standard for this ecosystem |
| Vector DB | ChromaDB (embedded, persisted under `storage/`) | Zero-infra, embedded, metadata filtering built in |
| Embeddings | `BAAI/bge-small-en-v1.5` via `sentence-transformers` | Small, CPU-friendly, retrieval-tuned |
| Keyword search | `rank_bm25` | Tiny pure-Python BM25, no server |
| Fusion | Reciprocal Rank Fusion, hand-rolled | Combines dense + BM25 rankings without a dependency |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` via `sentence-transformers.CrossEncoder` | Standard, small, local cross-encoder |
| Generation LLM | Ollama, `qwen3:4b` (local, already pulled) | No API key/cost, decent instruction following |
| Orchestration | Plain Python modules/functions, no LangChain/LlamaIndex | Pipeline is linear; a framework adds indirection with no benefit here |
| Eval | Hand-rolled harness: LLM-judge (same local Ollama model) for faithfulness + answer relevance, deterministic keyword-overlap for context recall | RAGAS/TruLens assume OpenAI-shaped LLMs and pull in `langchain` transitively; wiring to Ollama is more fragile than ~80 lines of direct judge prompts. Same 3 metrics, no extra framework. |

## Data flow

**Ingestion:** `scripts/build_index.py` → `src/ingest.py`
1. Fetch ~15 FastAPI tutorial doc pages from `raw.githubusercontent.com` via `requests`.
2. Markdown-aware recursive chunking: split on `##`/`###` headers first (preserves
   section boundaries + captures `{source, section, ingested_date}` metadata),
   then recursively split oversized sections by paragraph/sentence with overlap.
3. Embed each chunk with `bge-small-en-v1.5`, upsert into Chroma collection.
4. Build a `rank_bm25` index over the same chunk texts, persist alongside (pickle).

**Query:** `scripts/ask.py` → `src/rag.py`
1. Dense search (Chroma, top ~15) + BM25 search (top ~15) over the query.
2. Reciprocal Rank Fusion merges both rankings into one candidate list.
3. Cross-encoder reranks fused candidates, keep top ~4-5.
4. `src/generation.py` builds a strict grounding prompt with the retrieved
   chunks (each tagged `[source: file#section]`), calls Ollama.
5. System prompt requires: answer only from provided context, cite every claim
   with `[source: file#section]`, and if the context is insufficient, respond
   with the fixed refusal string: *"I don't have enough information in the
   provided documents to answer that."*
6. Post-check: regex-verify every citation tag in the answer matches an actually
   retrieved chunk id; if a hallucinated citation is found, drop to the refusal.

**Eval:** `scripts/run_eval.py` → `src/eval.py`
- Reads `tests/test_queries.json` — 3 queries:
  1. **Fact retrieval** — a precise FastAPI config/parameter question.
  2. **Multi-hop** — a question requiring dependency-injection + security concepts together.
  3. **Negative/out-of-scope** — an unrelated question (must trigger the refusal).
- For each query: run the full pipeline, then score:
  - **Context recall** — deterministic keyword-overlap between retrieved chunks and the query's `expected_facts` list.
  - **Faithfulness** — LLM-judge: are all claims in the answer supported by the retrieved context? (0-1 score)
  - **Answer relevance** — LLM-judge: does the answer address the question? (0-1 score)
  - **Refusal check** (negative query only) — answer must contain the fixed refusal string.
- Prints a pass/fail report per query per metric.

## File layout

```
RAG System/
  requirements.txt
  config.py                 # model names, chunk size/overlap, top_k, paths
  src/
    ingest.py
    retrieval.py
    generation.py
    rag.py
    eval.py
  scripts/
    build_index.py
    ask.py
    run_eval.py
  tests/
    test_queries.json
  storage/                  # chroma db + bm25 pickle, gitignored
```

## Error handling

- Ingestion: skip and log any doc page that fails to fetch (don't abort the whole build).
- Generation: if Ollama is unreachable, raise a clear error rather than silently returning an empty answer.
- Citation check failure → forced refusal rather than an ungrounded answer (fail closed).

## Explicitly out of scope

Hosted vector DB, continuous/scheduled eval runs, RAGAS/TruLens as literal
dependencies, a dedicated MCP fetch server (plain `requests` covers the one
public GitHub source).
