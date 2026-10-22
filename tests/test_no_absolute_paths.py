"""Evita rutas absolutas de Windows y referencias a la ubicación antigua del proyecto."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

ABS_WIN_PATH = re.compile(r"(?<![\w/])[A-Za-z]:[\\/]")
OLD_PROJECT = "Proyecto_Data_Science"

TARGETS = ["configs", "src", "scripts", "tasks.py", ".env.example", "pyproject.toml"]
SUFFIXES = {".py", ".yaml", ".yml", ".toml", ".example", ".cfg", ".ini"}
# check_env.py menciona la ruta antigua a propósito para detectarla.
EXCLUDED = {Path("scripts/check_env.py")}


def _iter_files():
    for target in TARGETS:
        path = ROOT / target
        if path.is_file():
            yield path
        elif path.is_dir():
            for f in path.rglob("*"):
                if f.is_file() and f.suffix in SUFFIXES and "__pycache__" not in f.parts:
                    yield f


def _offending_lines(text: str) -> list[tuple[int, str]]:
    return [
        (n, line.strip())
        for n, line in enumerate(text.splitlines(), start=1)
        if ABS_WIN_PATH.search(line) or OLD_PROJECT in line
    ]


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("path = 'Z:/dummy'", True),
        ("path = 'D:/data'", True),
        ("url = 'https://example.com'", False),
        ("ratio = a:b", False),
        ("root = Path(__file__).parents[1]", False),
    ],
)
def test_detector_regex(line, expected):
    assert bool(_offending_lines(line)) is expected


def test_no_hardcoded_absolute_paths():
    offenders = []
    for f in _iter_files():
        rel = f.relative_to(ROOT)
        if rel in EXCLUDED:
            continue
        text = f.read_text(encoding="utf-8", errors="ignore")
        offenders += [f"{rel}:{n}: {line}" for n, line in _offending_lines(text)]
    assert not offenders, "Rutas absolutas u obsoletas:\n" + "\n".join(offenders)
