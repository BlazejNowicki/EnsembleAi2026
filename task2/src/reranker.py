"""
T9: Reranker – CrossEncoder reranking of retrieval candidates.

Runs on CPU to avoid GPU OOM (only 25 pairs per query - fast enough on CPU).
"""
from __future__ import annotations

import torch
from sentence_transformers import CrossEncoder

from src.config import get_config
from src.retriever import RetrievalResult

_reranker_cache: CrossEncoder | None = None


def _get_reranker() -> CrossEncoder:
    global _reranker_cache
    if _reranker_cache is None:
        cfg = get_config()["reranker"]
        _reranker_cache = CrossEncoder(cfg["model"], device="cpu")
        _reranker_cache.max_length = cfg.get("max_length", 512)
        # FP16 on CPU for faster inference (needs PyTorch >= 2.0)
        _reranker_cache.model.half()
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
    with torch.no_grad():
        scores = model.predict(pairs, batch_size=cfg.get("batch_size", 4))

    for candidate, score in zip(candidates, scores):
        candidate.score = float(score)

    candidates.sort(key=lambda r: r.score, reverse=True)
    return candidates[:top_k]
