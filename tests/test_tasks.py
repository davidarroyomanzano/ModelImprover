"""Tests de tasks.py: intérprete local, verificación del paquete y comandos (con mocks)."""

import importlib.util
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def tasks():
    spec = importlib.util.spec_from_file_location("repo_tasks", ROOT / "tasks.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def ctx():
    return MagicMock()


def _cmds(ctx):
    return [call.args[0] for call in ctx.run.call_args_list]


def test_venv_interpreter_is_preferred(tasks):
    venv = ROOT / ".venv"
    if (venv / "Scripts" / "python.exe").exists() or (venv / "bin" / "python").exists():
        assert Path(tasks.VENV_PY).resolve().is_relative_to(venv.resolve())


@pytest.mark.parametrize("task_name", ["lint", "format", "test", "smoke", "train"])
def test_commands_use_venv_interpreter(tasks, ctx, task_name):
    with patch.object(tasks, "_assert_local_package"):
        getattr(tasks, task_name).body(ctx)
    python_cmds = [c for c in _cmds(ctx) if "-m " in c or ".py" in c or "-c " in c]
    assert python_cmds
    assert all(c.startswith(tasks.PY) for c in python_cmds)


def test_lint_does_not_modify_files(tasks, ctx):
    tasks.lint.body(ctx)
    joined = " ".join(_cmds(ctx))
    assert "--no-fix" in joined
    assert "format --check" in joined


def test_format_runs_manual_stage(tasks, ctx):
    tasks.format.body(ctx)
    assert any("--hook-stage manual" in c for c in _cmds(ctx))


def test_assert_local_package_ok(tasks):
    pkg = tasks.SRC / "ds_project" / "__init__.py"
    fake = subprocess.CompletedProcess([], 0, stdout=f"{pkg}\n", stderr="")

    with patch.object(tasks.subprocess, "run", return_value=fake):
        assert tasks._assert_local_package() is None


def test_assert_local_package_rejects_foreign_project(tasks, tmp_path):
    foreign = tmp_path / "otro" / "ds_project" / "__init__.py"
    fake = subprocess.CompletedProcess([], 0, stdout=str(foreign), stderr="")
    with patch.object(tasks.subprocess, "run", return_value=fake):
        with pytest.raises(SystemExit, match="otro proyecto"):
            tasks._assert_local_package()


def test_assert_local_package_import_error(tasks):
    fake = subprocess.CompletedProcess([], 1, stdout="", stderr="ImportError")
    with patch.object(tasks.subprocess, "run", return_value=fake):
        with pytest.raises(SystemExit, match="poetry install"):
            tasks._assert_local_package()


def test_train_aborts_before_running_if_package_is_foreign(tasks, ctx):
    with patch.object(tasks, "_assert_local_package", side_effect=SystemExit(1)):
        with pytest.raises(SystemExit):
            tasks.train.body(ctx)
    ctx.run.assert_not_called()


def test_all_runs_lint_test_then_train_evaluate_and_export_with_scope(tasks, ctx):
    assert {t.name for t in tasks.all_.pre} == {"lint", "test"}
    with (
        patch.object(tasks, "train") as train,
        patch.object(tasks, "evaluate") as evaluate,
        patch.object(tasks, "export") as export,
    ):
        tasks.all_.body(ctx, product="p", turbine="t", data_version="v2", version="v7")
    train.assert_called_once_with(ctx, product="p", turbine="t", data_version="v2")
    evaluate.assert_called_once_with(ctx, product="p", turbine="t", data_version="v2")
    export.assert_called_once_with(ctx, product="p", turbine="t", data_version="v2", version="v7")


def test_evaluate_passes_scope_env_and_thresholds(tasks, ctx):
    with patch.object(tasks, "_assert_local_package"):
        tasks.evaluate.body(ctx, product="p", turbine="t", data_version="v2", min_f1=0.9)
    command = ctx.run.call_args.args[0]
    assert "evaluate_for_project" in command and '"0.9"' in command
    assert ctx.run.call_args.kwargs["env"] == {
        "DS_PRODUCT": "p",
        "DS_TURBINE": "t",
        "DS_DATA_VERSION": "v2",
    }


def test_train_passes_project_env(tasks, ctx):
    with patch.object(tasks, "_assert_local_package"):
        tasks.train.body(ctx, product="pinbolt", turbine="g5x", data_version="v3")
    assert ctx.run.call_args.kwargs["env"] == {
        "DS_PRODUCT": "pinbolt",
        "DS_TURBINE": "g5x",
        "DS_DATA_VERSION": "v3",
    }


def test_train_without_scope_has_empty_env(tasks, ctx):
    with patch.object(tasks, "_assert_local_package"):
        tasks.train.body(ctx)
    assert ctx.run.call_args.kwargs["env"] == {}


def test_export_uses_project_resolver(tasks, ctx):
    with patch.object(tasks, "_assert_local_package"):
        tasks.export.body(ctx, product="pinbolt", turbine="g114", data_version="v2", version="v7")
    cmd = _cmds(ctx)[0]
    assert "export_for_project" in cmd
    assert '"pinbolt" "g114" "v7"' in cmd and '"v2"' in cmd
