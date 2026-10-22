"""Higiene de la suite: detecta tests duplicados, ocultos o sin aserciones."""

from __future__ import annotations

import ast
import hashlib
from collections import defaultdict
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).parent
TEST_FILES = sorted(TESTS_DIR.glob("test_*.py"))


class _Normalizer(ast.NodeTransformer):
    """Renombra variables y argumentos locales para comparar solo la estructura."""

    def __init__(self) -> None:
        self.mapping: dict[str, str] = {}

    def _alias(self, name: str) -> str:
        return self.mapping.setdefault(name, f"v{len(self.mapping)}")

    def visit_arg(self, node: ast.arg) -> ast.arg:
        node.arg = self._alias(node.arg)
        node.annotation = None
        return node

    def visit_Name(self, node: ast.Name) -> ast.Name:
        if node.id in self.mapping or isinstance(node.ctx, ast.Store):
            node.id = self._alias(node.id)
        return node


def _strip_docstring(body: list[ast.stmt]) -> list[ast.stmt]:
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        return body[1:]
    return body


def _iter_tests(tree: ast.Module):
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith(
            "test_"
        ):
            yield node.name, node
        elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) and sub.name.startswith(
                    "test_"
                ):
                    yield f"{node.name}.{sub.name}", sub


def _fingerprint(func: ast.FunctionDef | ast.AsyncFunctionDef) -> str | None:
    clone = ast.parse(ast.unparse(func)).body[0]
    clone.body = _strip_docstring(clone.body)
    if not clone.body or all(isinstance(s, ast.Pass) for s in clone.body):
        return None
    clone.name = "_"
    clone.decorator_list = []
    clone.returns = None
    normalized = _Normalizer().visit(clone)
    dump = ast.dump(normalized, annotate_fields=False, include_attributes=False)
    return hashlib.sha1(dump.encode()).hexdigest()


def _has_assertion(func: ast.AST) -> bool:
    for node in ast.walk(func):
        if isinstance(node, ast.Assert):
            return True
        if isinstance(node, ast.Call):
            target = node.func
            name = target.attr if isinstance(target, ast.Attribute) else getattr(target, "id", "")
            if name.startswith("assert") or name in {
                "raises",
                "warns",
                "fail",
                "approx",
            }:
                return True
    return False


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_hay_archivos_de_test() -> None:
    assert TEST_FILES, "No se encontraron tests/test_*.py"


def test_sin_tests_con_cuerpo_identico() -> None:
    groups: dict[str, list[str]] = defaultdict(list)
    for path in TEST_FILES:
        for name, func in _iter_tests(_parse(path)):
            fp = _fingerprint(func)
            if fp is not None:
                groups[fp].append(f"{path.name}::{name}")

    duplicates = [locs for locs in groups.values() if len(locs) > 1]
    report = "\n".join(" == ".join(locs) for locs in duplicates)
    assert not duplicates, f"Tests con cuerpo idéntico (consolidar o parametrizar):\n{report}"


@pytest.mark.parametrize("path", TEST_FILES, ids=lambda p: p.name)
def test_sin_nombres_repetidos_en_modulo(path: Path) -> None:
    seen: set[str] = set()
    repeated: list[str] = []
    for name, _ in _iter_tests(_parse(path)):
        if name in seen:
            repeated.append(name)
        seen.add(name)
    assert not repeated, f"{path.name}: nombres repetidos (el primero nunca se ejecuta): {repeated}"


@pytest.mark.parametrize("path", TEST_FILES, ids=lambda p: p.name)
def test_todos_los_tests_verifican_algo(path: Path) -> None:
    empty = [name for name, func in _iter_tests(_parse(path)) if not _has_assertion(func)]
    assert not empty, f"{path.name}: tests sin assert ni pytest.raises: {empty}"
