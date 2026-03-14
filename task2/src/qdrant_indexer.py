"""
T5: Hybrid Indexer – Dense (SentenceTransformer) + Sparse (fastembed BM25) in Qdrant.
"""
from __future__ import annotations

import re

from fastembed import SparseTextEmbedding
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from src.config import get_config
from src.enrichment import EnrichedChunk

_dense_model_cache: SentenceTransformer | None = None
_sparse_model_cache: SparseTextEmbedding | None = None
_qdrant_client_cache: QdrantClient | None = None


def _get_dense_model() -> SentenceTransformer:
    global _dense_model_cache
    if _dense_model_cache is None:
        cfg = get_config()["embedder"]
        _dense_model_cache = SentenceTransformer(cfg["model"])
    return _dense_model_cache


def _get_sparse_model() -> SparseTextEmbedding:
    global _sparse_model_cache
    if _sparse_model_cache is None:
        cfg = get_config()["fastembed"]
        _sparse_model_cache = SparseTextEmbedding(model_name=cfg["model"])
    return _sparse_model_cache


def get_qdrant_client() -> QdrantClient:
    global _qdrant_client_cache
    if _qdrant_client_cache is None:
        cfg = get_config()["qdrant"]
        _qdrant_client_cache = QdrantClient(path=cfg["path"])
    return _qdrant_client_cache


_SAFE_NAME_RE = re.compile(r"[^a-zA-Z0-9_-]")


def _safe_collection_name(repo_id: str) -> str:
    return _SAFE_NAME_RE.sub("_", repo_id)


def init_qdrant_collection(client: QdrantClient, collection_name: str) -> None:
    cfg = get_config()["embedder"]
    dim = cfg["embedding_dim"]

    # stateless: always rebuild from scratch
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": VectorParams(size=dim, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            "sparse_bm25": SparseVectorParams(),
        },
    )


def _embed_dense(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    model = _get_dense_model()
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=False)
    return embeddings.tolist()


def _embed_sparse(texts: list[str]) -> list[SparseVector]:
    model = _get_sparse_model()
    results: list[SparseVector] = []
    for emb in model.embed(texts):
        results.append(
            SparseVector(
                indices=emb.indices.tolist(),
                values=emb.values.tolist(),
            )
        )
    return results


def build_hybrid_index(chunks: list[EnrichedChunk], repo_id: str) -> str:
    cfg = get_config()["embedder"]
    batch_size = cfg["batch_size"]

    client = get_qdrant_client()
    col_name = _safe_collection_name(repo_id)
    init_qdrant_collection(client, col_name)

    dense_texts = [c.text_for_embedding for c in chunks]
    print(f"  Embedding {len(dense_texts)} chunks (dense)...")
    dense_vecs = _embed_dense(dense_texts, batch_size)

    sparse_texts = [c.text_for_bm25 for c in chunks]
    print(f"  Embedding {len(sparse_texts)} chunks (sparse BM25)...")
    sparse_vecs = _embed_sparse(sparse_texts)

    upsert_batch = 128
    for i in range(0, len(chunks), upsert_batch):
        points = []
        for j, chunk in enumerate(chunks[i : i + upsert_batch]):
            idx = i + j
            points.append(
                PointStruct(
                    id=idx,
                    vector={
                        "dense": dense_vecs[idx],
                        "sparse_bm25": sparse_vecs[idx],
                    },
                    payload={
                        "file_path": chunk.file_path,
                        "chunk_type": chunk.chunk_type,
                        "name": chunk.name,
                        "parent_class": chunk.parent_class,
                        "start_line": chunk.start_line,
                        "end_line": chunk.end_line,
                        "source": chunk.source,
                        "called_symbols": chunk.called_symbols,
                        "used_imports": chunk.used_imports,
                    },
                )
            )
        client.upsert(collection_name=col_name, points=points)

    print(f"  Qdrant: upserted {len(chunks)} points into '{col_name}'.")
    return col_name
