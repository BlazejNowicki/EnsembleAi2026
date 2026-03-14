"""
Test script for file_walker.py

Usage:
    poetry run python test_file_walker.py
    poetry run python test_file_walker.py --stage start --lang python
    poetry run python test_file_walker.py --repo data/repositories-python-start/MaterialsDiscovery__PyChemia-dee8d4f6a9db07a52cc4a47e063ab28f5a9b9967
"""

import os
import argparse

import jsonlines

from src.file_walker import walk_py_files


def print_separator(char="-", width=60):
    print(char * width)


def test_from_datapoint(language: str, stage: str) -> None:
    data_file = os.path.join("data", f"{language}-{stage}.jsonl")
    if not os.path.exists(data_file):
        print(f"Data file not found: {data_file}")
        return

    with jsonlines.open(data_file) as reader:
        datapoints = list(reader)

    print(f"Datapoints: {len(datapoints)}")
    print_separator()

    for i, dp in enumerate(datapoints):
        repo_dir = dp["repo"].replace("/", "__")
        revision = dp["revision"]
        repo_root = os.path.join("data", f"repositories-{language}-{stage}", f"{repo_dir}-{revision}")
        completion_file = dp["path"]

        print(f"Datapoint #{i+1}")
        print(f"  repo            : {dp['repo']}")
        print(f"  revision        : {revision[:12]}...")
        print(f"  completion file : {completion_file}")
        print(f"  repo root       : {repo_root}")
        print()

        if not os.path.isdir(repo_root):
            print("  SKIP: repo directory not found (run prepare_data.sh?)")
            print_separator()
            continue

        # Without exclusion
        all_files = walk_py_files(repo_root)
        # With exclusion
        filtered_files = walk_py_files(repo_root, exclude_relative=completion_file)

        print(f"  Files (all)              : {len(all_files)}")
        print(f"  Files (excl. completion) : {len(filtered_files)}")
        print(f"  Total lines              : {sum(f.num_lines for f in filtered_files)}")

        # Size distribution
        buckets = {"<10": 0, "10-50": 0, "50-200": 0, "200+": 0}
        for f in filtered_files:
            if f.num_lines < 10:
                buckets["<10"] += 1
            elif f.num_lines < 50:
                buckets["10-50"] += 1
            elif f.num_lines < 200:
                buckets["50-200"] += 1
            else:
                buckets["200+"] += 1

        print(f"  Size distribution:")
        for bucket, count in buckets.items():
            bar = "#" * count
            print(f"    {bucket:>8} lines : {count:3d}  {bar}")

        print()
        print(f"  First 10 files:")
        for f in filtered_files[:10]:
            print(f"    {f.relative_path:<60}  ({f.num_lines:4d} lines)")

        print()
        print(f"  Largest 5 files:")
        for f in sorted(filtered_files, key=lambda x: x.num_lines, reverse=True)[:5]:
            print(f"    {f.relative_path:<60}  ({f.num_lines:4d} lines)")

        print()
        excluded = [f for f in all_files if f.relative_path not in {x.relative_path for x in filtered_files}]
        if excluded:
            print(f"  Excluded file:")
            for f in excluded:
                print(f"    {f.relative_path}  ({f.num_lines} lines)")

        print_separator()


def test_from_dir(repo_root: str) -> None:
    if not os.path.isdir(repo_root):
        print(f"Directory not found: {repo_root}")
        return

    files = walk_py_files(repo_root)

    print(f"Repo root : {repo_root}")
    print(f"Files     : {len(files)}")
    print(f"Lines     : {sum(f.num_lines for f in files)}")
    print_separator()

    for f in files:
        print(f"  {f.relative_path:<60}  ({f.num_lines:4d} lines)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=str, default="start")
    parser.add_argument("--lang", type=str, default="python")
    parser.add_argument("--repo", type=str, default=None, help="Path to a repo dir (skips .jsonl lookup)")
    args = parser.parse_args()

    if args.repo:
        test_from_dir(args.repo)
    else:
        test_from_datapoint(args.lang, args.stage)


if __name__ == "__main__":
    main()
