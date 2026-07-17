"""
Retrieval: embed the query, search ChromaDB, and (optionally) re-rank.

Heavy imports (sentence-transformers, chromadb) are done lazily inside the
loader functions so that unit tests covering pure logic don't drag in the whole
ML stack. State is cached at module level so Streamlit reruns stay fast.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import settings

_embedder = None
_collection = None
_reranker = None


@dataclass
class Retrieved:
    text: str
    page: int
    article: str | None
    score: float  # cosine similarity in [0, 1]; higher is better


def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        _embedder = SentenceTransformer(settings.embed_model)
    return _embedder


def get_collection():
    global _collection
    if _collection is None:
        import chromadb

        client = chromadb.PersistentClient(path=settings.chroma_dir)
        _collection = client.get_collection(settings.collection)
    return _collection


def get_reranker():
    """Load the cross-encoder lazily. Returns None if it can't be loaded (offline)."""
    global _reranker
    if _reranker is None:
        try:
            from sentence_transformers import CrossEncoder

            _reranker = CrossEncoder(settings.rerank_model)
        except Exception:  # noqa: BLE001 - reranking is best-effort
            _reranker = False  # sentinel: tried and failed, don't retry
    return _reranker or None


def _to_similarity(distance: float) -> float:
    """ChromaDB returns cosine distance; convert to a bounded similarity."""
    return max(0.0, min(1.0, 1.0 - float(distance)))


def rerank(query: str, hits: list[Retrieved], top_k: int) -> list[Retrieved]:
    """Re-order hits with a cross-encoder, falling back to the original order."""
    model = get_reranker()
    if model is None or not hits:
        return hits[:top_k]
    scores = model.predict([(query, h.text) for h in hits])
    order = sorted(range(len(hits)), key=lambda i: scores[i], reverse=True)
    return [hits[i] for i in order[:top_k]]


def retrieve(query: str, top_k: int | None = None) -> list[Retrieved]:
    """Return the most relevant chunks for a query.

    When re-ranking is enabled we over-fetch ``fetch_k`` candidates by embedding
    similarity, then let the cross-encoder pick the final ``top_k``.
    """
    top_k = top_k or settings.top_k
    n = settings.fetch_k if settings.use_reranker else top_k

    emb = get_embedder().encode([query]).tolist()
    result = get_collection().query(
        query_embeddings=emb,
        n_results=n,
        include=["documents", "metadatas", "distances"],
    )

    hits = [
        Retrieved(
            text=doc,
            page=meta.get("page", "?"),
            article=meta.get("article"),
            score=_to_similarity(dist),
        )
        for doc, meta, dist in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0], strict=True
        )
    ]

    if settings.use_reranker:
        return rerank(query, hits, top_k)
    return hits[:top_k]
