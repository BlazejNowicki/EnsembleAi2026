"""
Indexer pipeline for the RAG-based context collection strategy.

Currently implements:
  - T1: DFS file walking
  - T2: AST parsing into structural chunks

Usage:
    poetry run python indexer.py --stage start --lang python
"""

import os
import argparse
from collections import Counter

import jsonlines
from tqdm import tqdm

from src.file_walker import walk_py_files
from src.ast_parser import parse_file


def get_repo_root(language: str, stage: str, datapoint: dict) -> str:
    repo_path = datapoint["repo"].replace("/", "__")
    revision = datapoint["revision"]
    return os.path.join("data", f"repositories-{language}-{stage}", f"{repo_path}-{revision}")


def index_repo(repo_root: str, completion_file_path: str) -> None:
    """Walk files, parse AST, print summary. Will grow with T3-T6."""
    files = walk_py_files(repo_root, exclude_relative=completion_file_path)
    print(f"  Files found: {len(files)}  |  Total lines: {sum(f.num_lines for f in files)}")

    all_chunks = []
    type_counts: Counter = Counter()

    for fi in files:
        chunks = parse_file(fi.content, fi.relative_path)
        all_chunks.extend(chunks)
        for c in chunks:
            type_counts[c.chunk_type] += 1

    print(f"  Chunks: {len(all_chunks)}")
    for t, count in type_counts.most_common():
        print(f"    {t:15s}: {count}")

    return all_chunks


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
