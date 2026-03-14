"""
T6: Hybrid Query – Qdrant RRF (Reciprocal Rank Fusion) over dense + sparse vectors.
"""
from __future__ import annotations

from dataclasses import dataclass

from qdrant_client.models import (
    Fusion,
    FusionQuery,
    Prefetch,
)

from src.enrichment import EnrichedChunk
from src.qdrant_indexer import (
    _embed_dense,
    _embed_sparse,
    get_qdrant_client,
)


@dataclass
class RetrievalResult:
    chunk: EnrichedChunk
    score: float


def hybrid_retrieve(
    collection_name: str,
    query_text: str,
    query_bm25_text: str,
    top_n: int = 25,
) -> list[RetrievalResult]:
    """Hybrid search: encode query dense + sparse, then Qdrant RRF fusion."""
    # encode query
    dense_vec = _embed_dense([query_text], batch_size=1)[0]
    sparse_vecs = _embed_sparse([query_bm25_text])
    sparse_vec = sparse_vecs[0]

    client = get_qdrant_client()

    hits = client.query_points(
        collection_name=collection_name,
        prefetch=[
            Prefetch(query=dense_vec, using="dense", limit=top_n * 2),
            Prefetch(query=sparse_vec, using="sparse_bm25", limit=top_n * 2),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=top_n,
        with_payload=True,
    )

    results: list[RetrievalResult] = []
    for point in hits.points:
        p = point.payload
        chunk = EnrichedChunk(
            file_path=p["file_path"],
            chunk_type=p["chunk_type"],
            name=p["name"],
            parent_class=p["parent_class"],
            start_line=p["start_line"],
            end_line=p["end_line"],
            source=p["source"],
            called_symbols=p.get("called_symbols", []),
            used_imports=p.get("used_imports", []),
        )
        results.append(RetrievalResult(chunk=chunk, score=point.score))

    return results
