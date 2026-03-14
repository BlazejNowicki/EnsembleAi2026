from __future__ import annotations

from collections import Counter

from src.config import get_config
from src.file_walker import walk_py_files
from src.ast_parser import parse_file, CodeChunk


def chunk_repository(repo_root: str, exclude_path: str | None = None) -> list[CodeChunk]:
    """Walk all .py files in *repo_root*, parse each with AST, return flat list of chunks.

    *exclude_path* is an optional relative path to skip (e.g. the completion file).
    """
    cfg = get_config()["chunker"]
    max_lines = cfg["max_chunk_lines"]
    overlap = cfg["chunk_overlap_lines"]

    files = walk_py_files(repo_root, exclude_relative=exclude_path)

    all_chunks: list[CodeChunk] = []
    for fi in files:
        chunks = parse_file(fi.content, fi.relative_path)
        for chunk in chunks:
            lines_count = chunk.end_line - chunk.start_line + 1
            if lines_count > max_lines:
                all_chunks.extend(_split_chunk(chunk, max_lines, overlap))
            else:
                all_chunks.append(chunk)

    return all_chunks


def _split_chunk(chunk: CodeChunk, max_lines: int, overlap: int) -> list[CodeChunk]:
    """Split an oversized chunk into smaller pieces with overlap."""
    source_lines = chunk.source.splitlines(keepends=True)
    parts: list[CodeChunk] = []
    start = 0
    total = len(source_lines)

    while start < total:
        end = min(start + max_lines, total)
        part_source = "".join(source_lines[start:end])
        parts.append(CodeChunk(
            file_path=chunk.file_path,
            chunk_type=chunk.chunk_type,
            name=chunk.name,
            parent_class=chunk.parent_class,
            start_line=chunk.start_line + start,
            end_line=chunk.start_line + end - 1,
            source=part_source,
        ))
        if end >= total:
            break
        start = end - overlap

    return parts


def print_stats(chunks: list[CodeChunk]) -> None:
    """Print a summary of chunk counts by type."""
    counts = Counter(c.chunk_type for c in chunks)
    print(f"  Total chunks: {len(chunks)}")
    for t, n in counts.most_common():
        print(f"    {t:15s}: {n}")
