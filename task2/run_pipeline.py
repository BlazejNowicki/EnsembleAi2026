"""
Combined pipeline: index each repo, predict its datapoints.

Models (embedder, reranker) are loaded once and reused across all repos.
Qdrant collections are built per repo and reused if they already exist.
Per-repo predictions are cached to disk; on re-run only missing repos are predicted.

Usage:
    poetry run python run_pipeline.py --stage start --lang python
"""

import json
import os
import argparse
from collections import defaultdict

import torch
import jsonlines
from tqdm import tqdm

from src.chunker import chunk_repository, print_stats
from src.enrichment import enrich_chunks
from src.qdrant_indexer import build_hybrid_index
from src.query_analyzer import analyze_query
from src.retriever import hybrid_retrieve
from src.reranker import rerank
from src.context_composer import compose_context


def get_repo_root(language: str, stage: str, datapoint: dict) -> str:
    repo_path = datapoint["repo"].replace("/", "__")
    revision = datapoint["revision"]
    return os.path.join("data", f"repositories-{language}-{stage}", f"{repo_path}-{revision}")


def predict_one(datapoint: dict, collection_name: str) -> str:
    prefix = datapoint.get("prefix", "")
    suffix = datapoint.get("suffix", "")

    query = analyze_query(prefix, suffix)

    candidates = hybrid_retrieve(
        collection_name=collection_name,
        query_text=query.dense_query,
        query_bm25_text=" ".join(query.sparse_query_tokens),
    )

    ranked = rerank(query.dense_query, candidates)

    context = compose_context(ranked)
    return context


def _prediction_cache_dir(language: str, stage: str) -> str:
    return os.path.join("predictions", "cache", f"{language}-{stage}")


def _repo_cache_path(cache_dir: str, repo_name: str) -> str:
    return os.path.join(cache_dir, f"{repo_name}.json")


def _load_cached_predictions(cache_path: str) -> dict[int, str] | None:
    """Load cached per-repo predictions. Returns {original_idx: context} or None."""
    if not os.path.exists(cache_path):
        return None
    with open(cache_path) as f:
        data = json.load(f)
    return {int(k): v for k, v in data.items()}


def _save_cached_predictions(cache_path: str, preds: dict[int, str]) -> None:
    """Save per-repo predictions to cache."""
    with open(cache_path, "w") as f:
        json.dump({str(k): v for k, v in preds.items()}, f)


def main():
    parser = argparse.ArgumentParser(description="Index + predict pipeline")
    parser.add_argument("--stage", type=str, default="start")
    parser.add_argument("--lang", type=str, default="python")
    args = parser.parse_args()

    language = args.lang
    stage = args.stage

    completion_points_file = os.path.join("data", f"{language}-{stage}.jsonl")
    if not os.path.exists(completion_points_file):
        print(f"Data file not found: {completion_points_file}")
        print(f"Run: ./prepare_data.sh {stage} {language}")
        return

    # read all datapoints, preserving original order (index = output order)
    datapoints: list[dict] = []
    with jsonlines.open(completion_points_file, "r") as reader:
        for dp in reader:
            datapoints.append(dp)

    # group datapoints by repo_root, preserving their original indices
    repo_groups: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for idx, dp in enumerate(datapoints):
        repo_root = get_repo_root(language, stage, dp)
        repo_groups[repo_root].append((idx, dp))

    print(f"Stage: {stage}  |  Language: {language}")
    print(f"Datapoints: {len(datapoints)}  |  Unique repos: {len(repo_groups)}")
    print()

    # prepare output array (predictions must match input order)
    predictions: list[str | None] = [None] * len(datapoints)

    cache_dir = _prediction_cache_dir(language, stage)
    os.makedirs(cache_dir, exist_ok=True)

    cached_count = 0
    for repo_root, items in tqdm(repo_groups.items(), desc="Repos"):
        repo_name = os.path.basename(repo_root)
        cache_path = _repo_cache_path(cache_dir, repo_name)

        # --- CHECK CACHE ---
        cached = _load_cached_predictions(cache_path)
        if cached is not None:
            for idx, _ in items:
                predictions[idx] = cached.get(idx, "")
            cached_count += 1
            print(f"\n[{repo_name}] cached ✓")
            continue

        print(f"\n[{repo_name}]")

        if not os.path.isdir(repo_root):
            print(f"  SKIP - directory not found: {repo_root}")
            repo_preds: dict[int, str] = {}
            for idx, _ in items:
                predictions[idx] = ""
                repo_preds[idx] = ""
            _save_cached_predictions(cache_path, repo_preds)
            continue

        # --- INDEX ---
        chunks = chunk_repository(repo_root, exclude_path=None)
        print_stats(chunks)

        enriched = enrich_chunks(chunks)
        print(f"  Enriched: {len(enriched)}")

        col_name = build_hybrid_index(enriched, repo_name)

        # free intermediate data after indexing
        del enriched
        del chunks
        torch.cuda.empty_cache()

        # --- PREDICT all datapoints for this repo ---
        repo_preds = {}
        for idx, dp in tqdm(items, desc=f"  Predicting ({repo_name})", leave=False):
            ctx = predict_one(dp, col_name)
            predictions[idx] = ctx
            repo_preds[idx] = ctx

        # --- SAVE per-repo cache ---
        _save_cached_predictions(cache_path, repo_preds)
        print(f"  Cached predictions to {cache_path}")

    if cached_count:
        print(f"\n{cached_count}/{len(repo_groups)} repos loaded from cache")

    # --- AGGREGATE: write predictions in original order ---
    os.makedirs("predictions", exist_ok=True)
    output_path = os.path.join("predictions", f"{language}-{stage}-rag.jsonl")

    with jsonlines.open(output_path, "w") as writer:
        for ctx in predictions:
            writer.write({"context": ctx or ""})

    print(f"\nPredictions saved to {output_path}")


if __name__ == "__main__":
    main()
