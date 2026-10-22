"""Higiene del repositorio: nombres de archivo inválidos y presencia de .gitattributes."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_INVALID_NAME = re.compile(r"[\s()]")
_IGNORED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "data",
    ".ruff_cache",
    ".pytest_cache",
    "__pycache__",
}


def _iter_repo_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        rel_parts = path.relative_to(ROOT).parts
        if any(part in _IGNORED_DIRS for part in rel_parts):
            continue
        if path.is_file():
            files.append(path)
    return files


def test_no_files_with_spaces_or_parentheses() -> None:
    offenders = [
        str(p.relative_to(ROOT)) for p in _iter_repo_files() if _INVALID_NAME.search(p.name)
    ]
    assert not offenders, f"Archivos con nombres inválidos: {offenders}"


def test_gitattributes_enforces_lf() -> None:
    gitattributes = ROOT / ".gitattributes"
    assert gitattributes.exists(), "Falta .gitattributes en la raíz"
    content = gitattributes.read_text(encoding="utf-8")
    assert "eol=lf" in content
    assert "*.wav" in content and "binary" in content
