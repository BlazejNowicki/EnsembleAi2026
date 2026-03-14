from __future__ import annotations

import ast
from dataclasses import dataclass


@dataclass
class CodeChunk:
    file_path: str
    chunk_type: str   # "import" | "class" | "function" | "method" | "module_level"
    name: str
    parent_class: str
    start_line: int
    end_line: int
    source: str

def _lines_source(lines: list[str], start: int, end: int) -> str:
    """Grab source from 1-indexed start..end (inclusive)."""
    return "".join(lines[start - 1 : end])


def _node_start(node: ast.AST) -> int:
    """Start line including decorators."""
    if hasattr(node, "decorator_list") and node.decorator_list:
        return min(d.lineno for d in node.decorator_list)
    return node.lineno


def _node_end(node: ast.AST) -> int:
    return node.end_lineno or node.lineno


def parse_file(content: str, file_path: str) -> list[CodeChunk]:
    """Parse a Python file into structural code chunks.

    Returns a single module_level chunk as fallback on SyntaxError.
    """
    lines = content.splitlines(keepends=True)
    n_lines = len(lines)

    try:
        tree = ast.parse(content, filename=file_path)
    except SyntaxError:
        return [CodeChunk(file_path, "module_level", "", "", 1, n_lines, content)]

    chunks: list[CodeChunk] = []
    claimed: set[int] = set()  # line numbers already assigned to a chunk

    def claim(start: int, end: int) -> None:
        for ln in range(start, end + 1):
            claimed.add(ln)

    # --- pass 1: imports (grouped into one chunk) ---
    import_lines: list[int] = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            s, e = node.lineno, _node_end(node)
            for ln in range(s, e + 1):
                import_lines.append(ln)
            claim(s, e)

    if import_lines:
        import_lines.sort()
        src = "".join(lines[ln - 1] for ln in import_lines)
        chunks.append(CodeChunk(file_path, "import", "", "", import_lines[0], import_lines[-1], src))

    # --- pass 2: classes and top-level functions ---
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            _process_class(node, file_path, lines, chunks, claimed)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _emit_func(node, file_path, lines, chunks, claimed, parent_class="")

    # --- pass 3: everything else = module_level ---
    # collect contiguous runs of unclaimed non-blank lines
    run_start = None
    for i in range(1, n_lines + 1):
        is_content = i not in claimed and lines[i - 1].strip() != ""
        if is_content and run_start is None:
            run_start = i
        elif not is_content and run_start is not None:
            src = _lines_source(lines, run_start, i - 1)
            chunks.append(CodeChunk(file_path, "module_level", "", "", run_start, i - 1, src))
            run_start = None
    if run_start is not None:
        src = _lines_source(lines, run_start, n_lines)
        chunks.append(CodeChunk(file_path, "module_level", "", "", run_start, n_lines, src))

    chunks.sort(key=lambda c: c.start_line)
    return chunks


def _process_class(
    node: ast.ClassDef,
    file_path: str,
    lines: list[str],
    chunks: list[CodeChunk],
    claimed: set[int],
) -> None:
    cls_start = _node_start(node)
    cls_end = _node_end(node)

    # claim all class lines
    for ln in range(cls_start, cls_end + 1):
        claimed.add(ln)

    methods = [n for n in ast.iter_child_nodes(node)
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]

    if not methods:
        src = _lines_source(lines, cls_start, cls_end)
        chunks.append(CodeChunk(file_path, "class", node.name, "", cls_start, cls_end, src))
        return

    # header = everything before first method
    first_method_start = min(_node_start(m) for m in methods)
    header_end = first_method_start - 1
    if header_end >= cls_start:
        src = _lines_source(lines, cls_start, header_end)
        chunks.append(CodeChunk(file_path, "class", node.name, "", cls_start, header_end, src))

    for m in methods:
        _emit_func(m, file_path, lines, chunks, claimed, parent_class=node.name)


def _emit_func(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    file_path: str,
    lines: list[str],
    chunks: list[CodeChunk],
    claimed: set[int],
    parent_class: str,
) -> None:
    start = _node_start(node)
    end = _node_end(node)

    for ln in range(start, end + 1):
        claimed.add(ln)

    src = _lines_source(lines, start, end)
    chunk_type = "method" if parent_class else "function"
    chunks.append(CodeChunk(file_path, chunk_type, node.name, parent_class, start, end, src))
