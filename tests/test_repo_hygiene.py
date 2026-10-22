"""Higiene del repositorio. Absorbe test_root_filenames.py y test_no_absolute_paths.py."""

import re
from pathlib import Path

import pytest

from tests.conftest import ROOT, requires_git, tracked_files

TEXT_DIRS = ("src", "scripts", "configs", "docs", "tests")
TEXT_SUFFIXES = {".py", ".yaml", ".yml", ".md", ".toml", ".cfg", ".ini"}
ABS_PATH = re.compile(r"(?<![\w.])(?:[A-Za-z]:\\\\?[\w]|/home/\w|/Users/\w)")


def _text_files() -> list[Path]:
    files = [
        p
        for d in TEXT_DIRS
        for p in (ROOT / d).rglob("*")
        if p.is_file() and p.suffix in TEXT_SUFFIXES
    ]
    return files + [ROOT / n for n in ("README.md", "pyproject.toml", "tasks.py")]


def test_root_names_are_portable() -> None:
    bad = [p.name for p in ROOT.iterdir() if re.search(r"[\s()]", p.name)]
    assert not bad, f"Nombres con espacios o paréntesis en la raíz: {bad}"


def test_no_absolute_paths() -> None:
    this_file = Path(__file__).resolve()
    hits = []
    for f in _text_files():
        if not f.exists() or f.resolve() == this_file:
            continue
        for n, line in enumerate(f.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            if ABS_PATH.search(line):
                hits.append(f"{f.relative_to(ROOT)}:{n}")
    assert not hits, f"Rutas absolutas encontradas: {hits}"


@pytest.mark.parametrize(
    "required",
    [
        "README.md",
        "pyproject.toml",
        ".env.example",
        "configs/config.yaml",
        "configs/dsp.yaml",
        "configs/optuna.yaml",
        "configs/logging.yaml",
        "configs/smoke.yaml",
    ],
)
def test_required_files_exist(required: str) -> None:
    assert (ROOT / required).is_file()


@requires_git
def test_no_tracked_data_or_caches() -> None:
    bad = [
        f
        for f in tracked_files()
        if f.lower().endswith(".wav")
        or f.startswith(".ruff_cache/")
        or (f.startswith("data/") and not f.endswith(".gitkeep"))
    ]
    assert not bad, f"Archivos indexados que no deberían estarlo ({len(bad)}): {bad[:5]}"
