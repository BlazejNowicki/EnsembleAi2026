"""
T9: Reranker – CrossEncoder reranking of retrieval candidates.
"""
from __future__ import annotations

from sentence_transformers import CrossEncoder

from src.config import get_config
from src.retriever import RetrievalResult

_reranker_cache: CrossEncoder | None = None


def _get_reranker() -> CrossEncoder:
    global _reranker_cache
    if _reranker_cache is None:
        cfg = get_config()["reranker"]
        _reranker_cache = CrossEncoder(cfg["model"])
    return _reranker_cache


def rerank(
    query_text: str,
    candidates: list[RetrievalResult],
    top_k: int | None = None,
) -> list[RetrievalResult]:
    if not candidates:
        return []

    cfg = get_config()["reranker"]
    if top_k is None:
        top_k = cfg["top_k"]

    model = _get_reranker()
    pairs = [(query_text, c.chunk.source) for c in candidates]
    scores = model.predict(pairs, batch_size=16)

    for candidate, score in zip(candidates, scores):
        candidate.score = float(score)

    candidates.sort(key=lambda r: r.score, reverse=True)
    return candidates[:top_k]
