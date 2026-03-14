import argparse
import os
from src.chunker import chunk_repository
from src.ast_parser import parse_file
from src.enrichment import enrich_chunks

parser = argparse.ArgumentParser()
parser.add_argument("--repo", type=str,
                    default="data/repositories-python-start/MaterialsDiscovery__PyChemia-dee8d4f6a9db07a52cc4a47e063ab28f5a9b9967")
parser.add_argument("--exclude", type=str, default="pychemia/code/fireball/fireball.py")
parser.add_argument("--file", type=str, default=None,
                    help="Absolute path to a single .py file to enrich instead of whole repo")
args = parser.parse_args()


def print_enriched(enriched):
    print(f"Total chunks: {len(enriched)}")
    print()
    by_calls = sorted(enriched, key=lambda c: len(c.called_symbols), reverse=True)
    print("=== Top 10 by called_symbols count ===")
    for c in by_calls[:10]:
        name = f"{c.parent_class}.{c.name}" if c.parent_class else c.name
        print(f"  {c.chunk_type:15s}  {name:40s}  calls={len(c.called_symbols):3d}  ({c.file_path})")
        print(f"    symbols: {c.called_symbols[:15]}")
    print()
    print("=== All chunks ===")
    for i, c in enumerate(enriched):
        name = f"{c.parent_class}.{c.name}" if c.parent_class else c.name
        print(f"[{i}] {c.chunk_type}  {name}  lines {c.start_line}-{c.end_line}  ({c.file_path})")
        print(f"  called_symbols : {c.called_symbols}")
        print(f"  used_imports   : {c.used_imports}")
        print(f"  --- text_for_embedding ---")
        print(c.text_for_embedding)
        print(f"  --- text_for_bm25 ---")
        print(c.text_for_bm25)
        print("=" * 60)


if args.file:
    # single file mode - provide absolute path
    abs_path = os.path.abspath(args.file)
    with open(abs_path, "r", encoding="utf-8", errors="replace") as fh:
        content = fh.read()
    chunks = parse_file(content, abs_path)
    enriched = enrich_chunks(chunks)
    print_enriched(enriched)
else:
    # whole repo mode
    chunks = chunk_repository(args.repo, args.exclude)
    enriched = enrich_chunks(chunks)
    print_enriched(enriched)
