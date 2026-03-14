import argparse
import os
from src.chunker import chunk_repository, print_stats
from src.ast_parser import parse_file

parser = argparse.ArgumentParser()
parser.add_argument("--repo", type=str,
                    default="data/repositories-python-start/MaterialsDiscovery__PyChemia-dee8d4f6a9db07a52cc4a47e063ab28f5a9b9967")
parser.add_argument("--exclude", type=str, default="pychemia/code/fireball/fireball.py")
parser.add_argument("--file", type=str, default=None,
                    help="Absolute path to a single .py file to chunk instead of whole repo")
args = parser.parse_args()


def print_chunks(chunks):
    print_stats(chunks)
    print("=" * 60)
    for i, c in enumerate(chunks):
        name = f"{c.parent_class}.{c.name}" if c.parent_class else c.name
        print(f"[{i}] {c.chunk_type}  {name}  lines {c.start_line}-{c.end_line}  ({c.file_path})")
        print("---")
        print(c.source)
        print("=" * 60)


if args.file:
    abs_path = os.path.abspath(args.file)
    with open(abs_path, "r", encoding="utf-8", errors="replace") as fh:
        content = fh.read()
    chunks = parse_file(content, abs_path)
    print_chunks(chunks)
else:
    chunks = chunk_repository(args.repo, args.exclude)
    print_chunks(chunks)
