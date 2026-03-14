"""
Indexer pipeline for the RAG-based context collection strategy.

Currently implements:
  - T1: DFS file walking
  - T2: AST parsing into structural chunks
  - T3: Chunker (split oversized chunks)
  - T4: Enrichment (called_symbols, used_imports, text_for_embedding, text_for_bm25)

Usage:
    poetry run python indexer.py --stage start --lang python
"""

import os
import argparse

import jsonlines
from tqdm import tqdm

from src.chunker import chunk_repository, print_stats
from src.enrichment import enrich_chunks


def get_repo_root(language: str, stage: str, datapoint: dict) -> str:
    repo_path = datapoint["repo"].replace("/", "__")
    revision = datapoint["revision"]
    return os.path.join("data", f"repositories-{language}-{stage}", f"{repo_path}-{revision}")


def index_repo(repo_root: str, completion_file_path: str):
    """Chunk repository, enrich chunks, return enriched list."""
    chunks = chunk_repository(repo_root, completion_file_path)
    print_stats(chunks)

    enriched = enrich_chunks(chunks)
    print(f"  Enriched: {len(enriched)}")

    return enriched


def main():
    parser = argparse.ArgumentParser(description="Index repositories for RAG context collection")
    parser.add_argument("--stage", type=str, default="start", help="Competition stage (start, practice, public)")
    parser.add_argument("--lang", type=str, default="python", help="Language (python, kotlin)")
    args = parser.parse_args()

    language = args.lang
    stage = args.stage

    completion_points_file = os.path.join("data", f"{language}-{stage}.jsonl")
    if not os.path.exists(completion_points_file):
        print(f"Data file not found: {completion_points_file}")
        print(f"Run: ./prepare_data.sh {stage} {language}")
        return

    # Collect unique repos from datapoints
    repos: dict[str, str] = {}  # repo_root -> completion_file_path (last seen)
    with jsonlines.open(completion_points_file, "r") as reader:
        for dp in reader:
            root = get_repo_root(language, stage, dp)
            repos[root] = dp["path"]

    print(f"Stage: {stage}  |  Language: {language}  |  Unique repos: {len(repos)}")
    print()

    for repo_root, completion_path in tqdm(repos.items(), desc="Indexing repos"):
        repo_name = os.path.basename(repo_root)
        print(f"\n[{repo_name}]")

        if not os.path.isdir(repo_root):
            print(f"  SKIP - directory not found: {repo_root}")
            continue

        index_repo(repo_root, completion_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
