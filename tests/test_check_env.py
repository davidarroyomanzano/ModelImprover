"""Tests unitarios de scripts/check_env.py con subprocess e importlib mockeados."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_env.py"


@pytest.fixture()
def check_env(monkeypatch):
    # Copia sys.path: imported_package_path() lo modifica.
    monkeypatch.setattr(sys, "path", list(sys.path))
    spec = importlib.util.spec_from_file_location("check_env", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_main(check_env, monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["check_env.py", *argv])
    return check_env.main()


# --- helpers -----------------------------------------------------------


def test_is_relative_to(check_env, tmp_path):
    assert check_env._is_relative_to(tmp_path / "a" / "b.py", tmp_path)
    assert not check_env._is_relative_to(tmp_path, tmp_path / "otro")


@pytest.mark.parametrize(
    ("exe", "stale"),
    [
        ("Proyecto_Data_Science", True),
        ("ModelosPinbolt", False),
    ],
)
def test_interpreter_is_stale(check_env, monkeypatch, exe, stale):
    monkeypatch.setattr(sys, "executable", exe)
    assert check_env.interpreter_is_stale() is stale


def test_imported_package_path_returns_none_when_missing(check_env):
    with mock.patch.object(check_env.importlib, "import_module", side_effect=ImportError):
        assert check_env.imported_package_path() is None


def test_imported_package_path_returns_module_file(check_env, tmp_path):
    fake = SimpleNamespace(__file__=str(tmp_path / "ds_project" / "__init__.py"))
    with mock.patch.object(check_env.importlib, "import_module", return_value=fake):
        assert check_env.imported_package_path() == Path(fake.__file__).resolve()


def test_reinstall_calls_pip_uninstall_then_editable_install(check_env):
    with mock.patch.object(check_env.subprocess, "run") as run:
        run.return_value = SimpleNamespace(returncode=0)
        assert check_env.reinstall() == 0

    first, second = (c.args[0] for c in run.call_args_list)
    assert first[1:4] == ["-m", "pip", "uninstall"]
    assert second[1:5] == ["-m", "pip", "install", "-e"]
    assert second[5] == str(check_env.ROOT)


# --- main --------------------------------------------------------------


def test_main_ok_when_package_in_repo(check_env, monkeypatch):
    monkeypatch.setattr(check_env, "interpreter_is_stale", lambda: False)
    monkeypatch.setattr(
        check_env,
        "imported_package_path",
        lambda: check_env.EXPECTED_SRC / "ds_project" / "__init__.py",
    )
    assert _run_main(check_env, monkeypatch) == 0


def test_main_stale_interpreter_returns_2(check_env, monkeypatch):
    monkeypatch.setattr(check_env, "interpreter_is_stale", lambda: True)
    assert _run_main(check_env, monkeypatch) == 2


def test_main_misaligned_without_fix_returns_1(check_env, monkeypatch, tmp_path):
    monkeypatch.setattr(check_env, "interpreter_is_stale", lambda: False)
    monkeypatch.setattr(check_env, "imported_package_path", lambda: tmp_path / "x.py")
    reinstall = mock.Mock()
    monkeypatch.setattr(check_env, "reinstall", reinstall)
    assert _run_main(check_env, monkeypatch) == 1
    reinstall.assert_not_called()


@pytest.mark.parametrize("code", [0, 3])
def test_main_fix_propagates_reinstall_code(check_env, monkeypatch, code):
    monkeypatch.setattr(check_env, "interpreter_is_stale", lambda: False)
    monkeypatch.setattr(check_env, "imported_package_path", lambda: None)
    monkeypatch.setattr(check_env, "reinstall", lambda: code)
    assert _run_main(check_env, monkeypatch, "--fix") == code
