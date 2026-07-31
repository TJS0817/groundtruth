"""Hybrid dense + BM25 retrieval with reciprocal rank fusion and cross-encoder reranking."""
import pickle

import chromadb
from sentence_transformers import CrossEncoder, SentenceTransformer

import config

_embedder = None
_reranker = None
_collection = None
_bm25_data = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(config.EMBEDDING_MODEL)
    return _embedder


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(config.RERANKER_MODEL)
    return _reranker


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        _collection = client.get_collection(config.COLLECTION_NAME)
    return _collection


def _get_bm25():
    global _bm25_data
    if _bm25_data is None:
        with open(config.BM25_PATH, "rb") as f:
            _bm25_data = pickle.load(f)
    return _bm25_data


def _dense_search(query: str, top_k: int) -> list[str]:
    """Returns chunk ids ranked by dense similarity."""
    embedding = _get_embedder().encode([query]).tolist()
    result = _get_collection().query(query_embeddings=embedding, n_results=top_k)
    return result["ids"][0]


def _bm25_search(query: str, top_k: int) -> list[str]:
    """Returns chunk ids ranked by BM25 score."""
    data = _get_bm25()
    scores = data["bm25"].get_scores(query.lower().split())
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [data["ids"][i] for i in ranked[:top_k]]


def _reciprocal_rank_fusion(rankings: list[list[str]], k: int = config.RRF_K) -> list[str]:
    """Merges multiple ranked id lists into one, by sum of 1/(k + rank)."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)


def _fetch_chunks(ids: list[str]) -> list[dict]:
    """Look up full chunk text + metadata for a list of ids from the BM25 store
    (which holds every chunk, unlike a Chroma query result)."""
    data = _get_bm25()
    id_to_text = dict(zip(data["ids"], data["texts"]))
    collection = _get_collection()
    got = collection.get(ids=ids)
    id_to_meta = dict(zip(got["ids"], got["metadatas"]))
    return [
        {"id": cid, "text": id_to_text[cid], **id_to_meta.get(cid, {})}
        for cid in ids
        if cid in id_to_text
    ]


def search(query: str, top_k: int = config.RERANK_TOP_K) -> list[dict]:
    """Hybrid search: dense + BM25 -> RRF fusion -> cross-encoder rerank -> top_k chunks."""
    dense_ids = _dense_search(query, config.DENSE_TOP_K)
    bm25_ids = _bm25_search(query, config.BM25_TOP_K)
    fused_ids = _reciprocal_rank_fusion([dense_ids, bm25_ids])

    candidates = _fetch_chunks(fused_ids)
    if not candidates:
        return []

    pairs = [(query, c["text"]) for c in candidates]
    rerank_scores = _get_reranker().predict(pairs)
    ranked = sorted(zip(candidates, rerank_scores), key=lambda x: x[1], reverse=True)
    return [c for c, _ in ranked[:top_k]]
