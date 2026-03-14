"""
T10: Context Composer – assemble final context string from ranked chunks.
"""
from __future__ import annotations

from src.config import get_config
from src.retriever import RetrievalResult


def compose_context(
    ranked_chunks: list[RetrievalResult],
    max_chars: int | None = None,
) -> str:
    if max_chars is None:
        cfg = get_config().get("context_composer", {})
        max_chars = cfg.get("max_chars", 8000)

    # group consecutive chunks from the same file under one <|file_sep|>
    blocks: list[str] = []
    current_len = 0

    seen_files: dict[str, list[str]] = {}  # file_path -> list of sources
    order: list[str] = []  # preserve first-seen order

    for r in ranked_chunks:
        fp = r.chunk.file_path
        src = r.chunk.source

        if fp not in seen_files:
            seen_files[fp] = []
            order.append(fp)
        seen_files[fp].append(src)

    for fp in order:
        header = f"<|file_sep|>{fp}\n"
        file_block = header + "\n\n".join(seen_files[fp]) + "\n"

        if current_len + len(file_block) > max_chars:
            remaining = max_chars - current_len
            if remaining > len(header) + 50:
                blocks.append(file_block[:remaining])
            break

        blocks.append(file_block)
        current_len += len(file_block)

    return "".join(blocks)
