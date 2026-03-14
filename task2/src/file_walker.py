from __future__ import annotations

import os
from dataclasses import dataclass


SKIP_DIRS = {"__pycache__", ".git", ".tox", "venv", ".venv", "node_modules", ".eggs", ".mypy_cache"}


@dataclass
class FileInfo:
    absolute_path: str
    relative_path: str
    content: str
    num_lines: int


def walk_py_files(root_dir: str, exclude_relative: str | None = None) -> list[FileInfo]:
    """Recursively walk *root_dir* and return ``FileInfo`` for every readable ``.py`` file.

    Parameters
    ----------
    root_dir:
        Absolute path to the repository root on disk.
    exclude_relative:
        Optional relative path (inside *root_dir*) of the completion-point file
        that should be excluded from the results. This will be used to exclude the completion
        points file itself when indexing the repo for context collection.
    """
    root_dir = os.path.normpath(root_dir)
    results: list[FileInfo] = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        # prune directories we never want to enter (in-place)
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]

        for filename in filenames:
            if not filename.endswith(".py"):
                continue

            abs_path = os.path.join(dirpath, filename)
            rel_path = os.path.relpath(abs_path, root_dir)

            if exclude_relative and os.path.normpath(rel_path) == os.path.normpath(exclude_relative):
                continue

            try:
                with open(abs_path, "r", encoding="utf-8", errors="replace") as fh:
                    content = fh.read()
            except OSError:
                continue

            num_lines = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
            if num_lines < 1:
                continue

            results.append(FileInfo(
                absolute_path=abs_path,
                relative_path=rel_path,
                content=content,
                num_lines=num_lines,
            ))

    results.sort(key=lambda fi: fi.relative_path)
    return results
