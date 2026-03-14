from __future__ import annotations

import ast
import re
import textwrap
from dataclasses import dataclass, field

from src.ast_parser import CodeChunk
from src.config import get_config


@dataclass
class EnrichedChunk:
    """CodeChunk with extra fields useful for retrieval."""

    # --- original CodeChunk fields ---
    file_path: str
    chunk_type: str
    name: str
    parent_class: str
    start_line: int
    end_line: int
    source: str

    # --- enrichment fields ---
    called_symbols: list[str] = field(default_factory=list)
    used_imports: list[str] = field(default_factory=list)
    text_for_embedding: str = ""
    text_for_bm25: str = ""


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def enrich_chunks(
    chunks: list[CodeChunk],
    file_imports: dict[str, list[str]] | None = None,
) -> list[EnrichedChunk]:
    """Enrich a list of CodeChunks with retrieval-useful metadata.

    *file_imports* maps file_path -> list of imported names for that file.
    If not provided, imports are extracted per-chunk from import chunks.
    """
    # build import lookup: file_path -> set of imported names
    if file_imports is None:
        file_imports = _build_import_map(chunks)

    stopwords = set(get_config()["enrichment"]["bm25_stopwords"])

    results: list[EnrichedChunk] = []
    for chunk in chunks:
        called = _extract_called_symbols(chunk.source)
        imports_for_file = file_imports.get(chunk.file_path, [])
        used = [name for name in imports_for_file if name in chunk.source]
        emb_text = _build_embedding_text(chunk)
        bm25_text = _tokenize_for_bm25(chunk.source, stopwords)

        results.append(EnrichedChunk(
            file_path=chunk.file_path,
            chunk_type=chunk.chunk_type,
            name=chunk.name,
            parent_class=chunk.parent_class,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            source=chunk.source,
            called_symbols=called,
            used_imports=used,
            text_for_embedding=emb_text,
            text_for_bm25=bm25_text,
        ))

    return results


# ---------------------------------------------------------------------------
# internal helpers
# ---------------------------------------------------------------------------

def _build_import_map(chunks: list[CodeChunk]) -> dict[str, list[str]]:
    """From import chunks, extract the imported names per file."""
    mapping: dict[str, list[str]] = {}
    for chunk in chunks:
        if chunk.chunk_type != "import":
            continue
        names: list[str] = []
        try:
            tree = ast.parse(chunk.source)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    names.append(alias.asname or alias.name.split(".")[-1])
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    names.append(alias.asname or alias.name)
        mapping[chunk.file_path] = names
    return mapping


def _extract_called_symbols(source: str) -> list[str]:
    """Extract names of called functions/methods from source via AST."""
    try:
        tree = ast.parse(textwrap.dedent(source))
    except SyntaxError:
        return []

    symbols: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            symbols.append(func.id)
        elif isinstance(func, ast.Attribute):
            symbols.append(func.attr)
    # deduplicate but preserve order
    seen: set[str] = set()
    unique: list[str] = []
    for s in symbols:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    return unique


def _build_embedding_text(chunk: CodeChunk) -> str:
    """Format chunk source with file/class context for embedding models."""
    parts: list[str] = [f"# File: {chunk.file_path}"]
    if chunk.parent_class:
        parts.append(f"# Class: {chunk.parent_class}")
    parts.append(chunk.source)
    return "\n".join(parts)


_TOKEN_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")


def _tokenize_for_bm25(source: str, stopwords: set[str]) -> str:
    """Tokenize source code for BM25: split on non-alnum, lowercase, drop stopwords."""
    tokens = _TOKEN_RE.findall(source)
    filtered = [t.lower() for t in tokens if t not in stopwords]
    return " ".join(filtered)
