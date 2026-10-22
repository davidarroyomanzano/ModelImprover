"""Tests de la guía docs/full_run.md y su coherencia con tasks.py."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "full_run.md"
TASKS = ROOT / "tasks.py"

REQUIRED_SECTIONS = (
    "Preparación",
    "Calidad del código",
    "Ejecución rápida",
    "Pipeline completa",
    "Verificación de artefactos",
)
REQUIRED_CONFIGS = (
    "configs/smoke.yaml",
    "configs/config.yaml",
    "configs/dsp.yaml",
    "configs/optuna.yaml",
)


@pytest.fixture(scope="module")
def doc_text() -> str:
    if not DOC.exists():
        pytest.skip("docs/full_run.md no existe todavía")
    return DOC.read_text(encoding="utf-8")


def _load_tasks_module():
    spec = importlib.util.spec_from_file_location("project_tasks", TASKS)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("section", REQUIRED_SECTIONS)
def test_doc_has_section(doc_text: str, section: str) -> None:
    assert section in doc_text


@pytest.mark.parametrize("config", REQUIRED_CONFIGS)
def test_doc_references_existing_configs(doc_text: str, config: str) -> None:
    assert config in doc_text
    assert (ROOT / config).is_file(), f"{config} citado pero inexistente"


def test_doc_referenced_scripts_exist(doc_text: str) -> None:
    for script in re.findall(r"scripts/[\w./-]+\.py", doc_text):
        assert (ROOT / script).is_file(), f"Script inexistente: {script}"


def test_doc_referenced_source_files_exist(doc_text: str) -> None:
    for src in re.findall(r"src/ds_project/[\w./-]+\.py", doc_text):
        assert (ROOT / src).is_file(), f"Módulo inexistente: {src}"


def test_doc_uses_poetry_invoke(doc_text: str) -> None:
    assert "poetry run invoke" in doc_text


@pytest.mark.parametrize("task_name", ["lint", "test"])
def test_tasks_exposes_documented_task(task_name: str) -> None:
    if not TASKS.exists():
        pytest.skip("tasks.py no existe")
    pytest.importorskip("invoke")
    module = _load_tasks_module()
    assert hasattr(module, task_name), f"tasks.py no define '{task_name}'"


def test_lint_task_runs_without_real_shell() -> None:
    """La tarea lint debe delegar en ctx.run (sin ejecutar procesos reales)."""
    if not TASKS.exists():
        pytest.skip("tasks.py no existe")
    invoke = pytest.importorskip("invoke")
    module = _load_tasks_module()
    lint = getattr(module, "lint", None)
    if lint is None:
        pytest.skip("tasks.py no define 'lint'")

    ctx = mock.MagicMock(spec=invoke.Context)
    body = getattr(lint, "body", lint)
    try:
        body(ctx)
    except TypeError:
        pytest.skip("lint requiere argumentos obligatorios")
    assert ctx.run.called, "lint no invoca ctx.run"
