"""Higiene de la suite. Absorbe test_no_duplicate_tests.py y test_duplicate_detection.py."""

import ast
import sys
from collections import defaultdict
from pathlib import Path

from tests.conftest import ROOT

TESTS = ROOT / "tests"


def _test_functions() -> dict[str, list[tuple[str, str]]]:
    """Devuelve {nombre: [(archivo, cuerpo_normalizado)]} de cada test_*."""
    found: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for f in TESTS.glob("test_*.py"):
        tree = ast.parse(f.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                body = ast.dump(ast.Module(body=node.body, type_ignores=[]))
                found[node.name].append((f.name, body))
    return found


def test_no_duplicate_test_names_within_file() -> None:
    dups = [
        (name, files)
        for name, entries in _test_functions().items()
        if len(files := [e[0] for e in entries]) != len(set(files))
    ]
    assert not dups, f"Tests redefinidos en el mismo archivo (se sobrescriben): {dups}"


def test_no_identical_test_bodies() -> None:
    by_body: dict[str, list[str]] = defaultdict(list)
    for name, entries in _test_functions().items():
        for file, body in entries:
            by_body[body].append(f"{file}::{name}")
    dups = [locs for locs in by_body.values() if len(locs) > 1]
    assert not dups, f"Tests con cuerpo idéntico: {dups}"


def test_test_files_follow_naming() -> None:
    allowed = {"__init__.py", "conftest.py"}
    bad = [
        p.name
        for p in TESTS.glob("*.py")
        if p.name not in allowed and not p.name.startswith("test_")
    ]
    assert not bad, f"Archivos en tests/ que pytest no recoge: {bad}"


def test_tests_package_exists() -> None:
    assert (TESTS / "__init__.py").is_file()


def test_detector_finds_duplicates(tmp_path: Path, monkeypatch) -> None:
    """Comprueba que el detector funciona usando un directorio de tests simulado."""
    (tmp_path / "test_a.py").write_text("def test_x():\n    assert 1 == 1\n")
    (tmp_path / "test_b.py").write_text("def test_y():\n    assert 1 == 1\n")

    monkeypatch.setattr(
        sys.modules[__name__],
        "TESTS",
        tmp_path,
    )

    bodies = [b for entries in _test_functions().values() for _, b in entries]
    assert len(bodies) == 2 and len(set(bodies)) == 1
