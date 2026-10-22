"""Las tareas de invoke deben usar el intérprete del .venv local."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_tasks():
    spec = importlib.util.spec_from_file_location("repo_tasks", ROOT / "tasks.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_venv_interpreter_is_preferred():
    tasks = _load_tasks()
    local = ROOT / ".venv"
    if (local / "Scripts" / "python.exe").exists() or (local / "bin" / "python").exists():
        assert Path(tasks.VENV_PY).resolve().is_relative_to(local.resolve())


def test_commands_do_not_use_bare_python():
    source = (ROOT / "tasks.py").read_text(encoding="utf-8")
    assert 'c.run("python ' not in source
    assert 'c.run(f"python ' not in source


def test_precommit_hooks_do_not_modify_files():
    cfg = (ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    assert '"--no-fix"' in cfg
    assert '"--check"' in cfg
