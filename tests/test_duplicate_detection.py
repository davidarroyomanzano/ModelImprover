"""Detección de tests redundantes o duplicados en la suite."""

from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).parent


def _test_files() -> list[Path]:
    return sorted(TESTS_DIR.glob("test_*.py"))


def _test_functions(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith(
            "test_"
        ):
            yield node


def _body_fingerprint(func: ast.FunctionDef) -> str:
    body = func.body
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(getattr(body[0], "value", None), ast.Constant)
    ):
        body = body[1:]  # ignora el docstring
    return "".join(ast.dump(stmt, annotate_fields=False) for stmt in body)


@pytest.fixture(scope="module")
def parsed() -> dict[Path, ast.AST]:
    return {p: ast.parse(p.read_text(encoding="utf-8")) for p in _test_files()}


def test_no_shadowed_test_names_within_module(parsed):
    problems = []
    for path, tree in parsed.items():
        seen: dict[str, int] = {}
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith(
                "test_"
            ):
                if node.name in seen:
                    problems.append(f"{path.name}:{node.lineno} redefine {node.name}")
                seen[node.name] = node.lineno
    assert not problems, "\n".join(problems)


def test_no_identical_test_bodies(parsed):
    groups: dict[str, list[str]] = defaultdict(list)
    for path, tree in parsed.items():
        for func in _test_functions(tree):
            fp = _body_fingerprint(func)
            if len(fp) > 80:  # ignora tests triviales (p. ej. `assert True`)
                groups[fp].append(f"{path.name}::{func.name}")
    dups = [names for names in groups.values() if len(names) > 1]
    assert not dups, "Tests con cuerpo idéntico:\n" + "\n".join(map(str, dups))


def test_every_test_file_has_tests(parsed):
    empty = [p.name for p, tree in parsed.items() if not any(_test_functions(tree))]
    assert not empty, f"Archivos de test sin tests: {empty}"
