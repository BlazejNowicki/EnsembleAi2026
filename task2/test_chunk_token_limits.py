"""
Test whether chunks produced by the chunker exceed the embedding model's
max sequence length (token limit).

Usage:
    poetry run python test_chunk_token_limits.py [--repo-dir PATH]

If --repo-dir is not given, scans data/repositories-* for available repos.
"""

import argparse
import os
import glob
import sys

from transformers import AutoTokenizer

from src.config import get_config
from src.chunker import chunk_repository
from src.enrichment import enrich_chunks, _build_embedding_text


def main():
    parser = argparse.ArgumentParser(description="Check chunk token counts vs model limit")
    parser.add_argument("--repo-dir", type=str, default=None,
                        help="Path to a single repo to test. If omitted, scans data/repositories-*.")
    parser.add_argument("--max-repos", type=int, default=5,
                        help="Max repos to scan when auto-discovering (default: 5)")
    args = parser.parse_args()

    cfg = get_config()
    model_name = cfg["embedder"]["model"]
    max_chunk_lines = cfg["chunker"]["max_chunk_lines"]
    chunk_overlap = cfg["chunker"]["chunk_overlap_lines"]

    print(f"Config:")
    print(f"  model:              {model_name}")
    print(f"  max_chunk_lines:    {max_chunk_lines}")
    print(f"  chunk_overlap_lines:{chunk_overlap}")
    print()

    # Load tokenizer to get actual token counts and max length
    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    max_seq_len = getattr(tokenizer, "model_max_length", None)
    if max_seq_len is None or max_seq_len > 1_000_000:
        # fallback: many Qwen models use 8192
        max_seq_len = 8192
    print(f"  model max_seq_length: {max_seq_len} tokens")
    print()

    # Find repos to test
    repo_dirs: list[str] = []
    if args.repo_dir:
        repo_dirs = [args.repo_dir]
    else:
        for pattern in glob.glob("data/repositories-*"):
            if os.path.isdir(pattern):
                for entry in sorted(os.listdir(pattern)):
                    full = os.path.join(pattern, entry)
                    if os.path.isdir(full):
                        repo_dirs.append(full)
                        if len(repo_dirs) >= args.max_repos:
                            break
            if len(repo_dirs) >= args.max_repos:
                break

    if not repo_dirs:
        print("No repos found. Use --repo-dir to point to a repository directory.")
        sys.exit(1)

    # Stats
    total_chunks = 0
    exceeded_chunks = 0
    all_token_counts: list[int] = []
    exceeded_details: list[tuple[str, str, int, int, int]] = []  # (repo, file, lines, tokens, max)

    for repo_dir in repo_dirs:
        repo_name = os.path.basename(repo_dir)
        print(f"[{repo_name}]")

        chunks = chunk_repository(repo_dir, exclude_path=None)
        print(f"  chunks: {len(chunks)}")

        for chunk in chunks:
            # Build the same text that gets embedded
            emb_text = _build_embedding_text(chunk)
            tokens = tokenizer.encode(emb_text, add_special_tokens=True)
            n_tokens = len(tokens)
            n_lines = chunk.end_line - chunk.start_line + 1

            total_chunks += 1
            all_token_counts.append(n_tokens)

            if n_tokens > max_seq_len:
                exceeded_chunks += 1
                exceeded_details.append((
                    repo_name, chunk.file_path, n_lines, n_tokens, max_seq_len
                ))

    # Report
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Total chunks tested:   {total_chunks}")
    print(f"Model max seq length:  {max_seq_len} tokens")
    print(f"max_chunk_lines:       {max_chunk_lines}")
    print()

    if all_token_counts:
        all_token_counts.sort()
        print(f"Token count distribution:")
        print(f"  min:    {all_token_counts[0]}")
        print(f"  p50:    {all_token_counts[len(all_token_counts) // 2]}")
        print(f"  p90:    {all_token_counts[int(len(all_token_counts) * 0.9)]}")
        print(f"  p95:    {all_token_counts[int(len(all_token_counts) * 0.95)]}")
        print(f"  p99:    {all_token_counts[int(len(all_token_counts) * 0.99)]}")
        print(f"  max:    {all_token_counts[-1]}")
        print()

    if exceeded_chunks:
        print(f"EXCEEDED: {exceeded_chunks} / {total_chunks} chunks exceed the limit!")
        print()
        for repo, fpath, lines, tokens, limit in exceeded_details[:20]:
            print(f"  {repo} | {fpath}:{lines} lines | {tokens} tokens (limit {limit})")
        if len(exceeded_details) > 20:
            print(f"  ... and {len(exceeded_details) - 20} more")
    else:
        print("ALL CLEAR: no chunks exceed the token limit.")


if __name__ == "__main__":
    main()
