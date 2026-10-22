"""Tests de scripts/lint_all.py con subprocess mockeado."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "lint_all.py"


@pytest.fixture()
def lint_all() -> ModuleType:
    spec = importlib.util.spec_from_file_location("lint_all", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _result(code: int) -> MagicMock:
    return MagicMock(returncode=code)


def test_commands_cover_black_and_ruff(lint_all: ModuleType) -> None:
    assert ["black", "."] in lint_all.COMMANDS
    assert ["ruff", "check", ".", "--fix"] in lint_all.COMMANDS
    # La validación final sin --fix debe ir la última.
    assert lint_all.COMMANDS[-1] == ["ruff", "check", "."]


def test_main_returns_zero_when_all_pass(lint_all: ModuleType) -> None:
    with patch.object(lint_all.subprocess, "run", return_value=_result(0)) as run:
        assert lint_all.main() == 0
    assert run.call_count == len(lint_all.COMMANDS)


def test_main_invokes_tools_via_current_interpreter(lint_all: ModuleType) -> None:
    with patch.object(lint_all.subprocess, "run", return_value=_result(0)) as run:
        lint_all.main()
    for call, cmd in zip(run.call_args_list, lint_all.COMMANDS):
        assert call.args[0] == [sys.executable, "-m", *cmd]
        assert call.kwargs["check"] is False


@pytest.mark.parametrize("failing_index", [0, 1, 2])
def test_main_returns_one_if_any_command_fails(lint_all: ModuleType, failing_index: int) -> None:
    codes = [0] * len(lint_all.COMMANDS)
    codes[failing_index] = 1
    with patch.object(lint_all.subprocess, "run", side_effect=[_result(c) for c in codes]) as run:
        assert lint_all.main() == 1
    # Aunque un comando falle, se ejecutan todos.
    assert run.call_count == len(lint_all.COMMANDS)
