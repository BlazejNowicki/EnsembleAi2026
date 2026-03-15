"""
T7: Query Analyzer – extract search signals from prefix/suffix.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field

from src.config import get_config


@dataclass
class QueryInfo:
    imports: list[str] = field(default_factory=list)
    recent_symbols: list[str] = field(default_factory=list)
    current_class: str | None = None
    current_function: str | None = None
    dense_query: str = ""
    sparse_query_tokens: list[str] = field(default_factory=list)


_CALL_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_UPPER_RE = re.compile(r"\b([A-Z][a-zA-Z0-9_]+)\b")
_ATTR_CALL_RE = re.compile(r"\.\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(")


def analyze_query(prefix: str, suffix: str) -> QueryInfo:
    cfg = get_config()["query_analyzer"]
    n = cfg["recent_lines"]

    imports = _extract_imports(prefix)
    recent_symbols = _extract_recent_symbols(prefix, n)
    current_class, current_function = _detect_context(prefix)
    dense_query = _build_dense_query(prefix, n)
    sparse_tokens = list(dict.fromkeys(imports + recent_symbols))

    return QueryInfo(
        imports=imports,
        recent_symbols=recent_symbols,
        current_class=current_class,
        current_function=current_function,
        dense_query=dense_query,
        sparse_query_tokens=sparse_tokens,
    )


def _extract_imports(prefix: str) -> list[str]:
    names: list[str] = []
    try:
        tree = ast.parse(prefix)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    names.append(alias.asname or alias.name.split(".")[-1])
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    names.append(alias.asname or alias.name)
    except SyntaxError:
        for m in re.finditer(
            r"^(?:from\s+\S+\s+import\s+(.+)|import\s+(.+))$",
            prefix,
            re.MULTILINE,
        ):
            raw = m.group(1) or m.group(2)
            for part in raw.split(","):
                part = part.strip()
                if " as " in part:
                    part = part.split(" as ")[-1].strip()
                if part:
                    names.append(part)

    seen: set[str] = set()
    unique: list[str] = []
    for n in names:
        if n not in seen:
            seen.add(n)
            unique.append(n)
    return unique


def _extract_recent_symbols(prefix: str, n_lines: int) -> list[str]:
    lines = prefix.splitlines()
    recent = "\n".join(lines[-n_lines:])

    symbols: list[str] = (
        _CALL_RE.findall(recent)
        + _UPPER_RE.findall(recent)
        + _ATTR_CALL_RE.findall(recent)
    )

    seen: set[str] = set()
    unique: list[str] = []
    for s in symbols:
        if s and s not in seen:
            seen.add(s)
            unique.append(s)
    return unique


def _detect_context(prefix: str) -> tuple[str | None, str | None]:
    """Detect current class and function from prefix (best-effort)."""
    current_class = None
    current_function = None

    try:
        tree = ast.parse(prefix)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                current_class = node.name
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                current_function = node.name
        return current_class, current_function
    except SyntaxError:
        pass

    for m in re.finditer(r"^class\s+([A-Za-z_][A-Za-z0-9_]*)", prefix, re.MULTILINE):
        current_class = m.group(1)
    for m in re.finditer(
        r"^\s*(?:async\s+)?def\s+([A-Za-z_][A-Za-z0-9_]*)", prefix, re.MULTILINE
    ):
        current_function = m.group(1)

    return current_class, current_function


def _build_dense_query(prefix: str, n_lines: int) -> str:
    lines = prefix.splitlines()
    base_query = "\n".join(lines[-n_lines:])
    
    # HyDE with Ollama qwen:0.5b / qwen2.5:0.5b for ultra-fast generation
    import requests
    import json
    try:
        prompt = f"Summarize the intent of this code in 5 keywords:\n{base_query}"
        resp = requests.post("http://localhost:11434/api/generate", json={
            "model": "qwen2.5:0.5b", # you can change it to qwen:0.5b if needed
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": 15,
                "num_ctx": 300,
                "temperature": 0.1
            }
        }, timeout=1.5)
        if resp.status_code == 200:
            hyde_keywords = resp.json().get("response", "").strip()
            if hyde_keywords:
                base_query = f"{base_query}\n\n# Context Keywords:\n# {hyde_keywords}"
    except Exception:
        pass # fail silently if Ollama is not running to not break the pipeline
        
    return base_query
