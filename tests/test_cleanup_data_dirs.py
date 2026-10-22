"""Tests de scripts/cleanup_data_dirs.py."""

import importlib.util
from pathlib import Path
from unittest import mock

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "cleanup_data_dirs.py"
if not _SCRIPT.exists():
    pytest.skip("scripts/cleanup_data_dirs.py no existe", allow_module_level=True)

_SPEC = importlib.util.spec_from_file_location("cleanup_data_dirs", _SCRIPT)
cleanup = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cleanup)


@pytest.fixture
def project(tmp_path: Path) -> Path:
    for name in ("raw", "test", "interim", "processed", "external"):
        (tmp_path / "data" / name).mkdir(parents=True)
    (tmp_path / "data" / "README.md").write_text("x")  # fichero, no carpeta
    (tmp_path / "src").mkdir()
    return tmp_path


def _unused(root: Path) -> set[str]:
    return {d.name for d in cleanup.find_unused_dirs(root)}


# --- find_unused_dirs ------------------------------------------------------


def test_no_references_returns_all_but_protected(project: Path) -> None:
    assert _unused(project) == {"interim", "processed", "external"}


@pytest.mark.parametrize(
    ("relpath", "content"),
    [
        ("src/m.py", 'OUT = "data/processed"\n'),
        ("src/m.py", "name = 'processed'\n"),
        ("src/m.py", 'name = "processed"\n'),
        ("configs/config.yaml", "dir: processed\n"),
        ("tasks.py", "p = 'data/processed/x'\n"),
        ("pyproject.toml", 'x = "processed"\n'),
        ("tests/t.json", '{"d": "processed"}'),
    ],
)
def test_reference_patterns_mark_dir_as_used(project, relpath, content) -> None:
    target = project / relpath
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    assert "processed" not in _unused(project)


def test_unscanned_suffix_is_ignored(project: Path) -> None:
    (project / "src" / "notes.md").write_text("data/processed")
    assert "processed" in _unused(project)


def test_unscanned_location_is_ignored(project: Path) -> None:
    (project / "docs").mkdir()
    (project / "docs" / "x.py").write_text("'data/processed'")
    assert "processed" in _unused(project)


def test_bare_word_is_not_a_reference(project: Path) -> None:
    (project / "src" / "m.py").write_text("# processed data goes here\n")
    assert "processed" in _unused(project)


def test_protected_never_returned(project: Path) -> None:
    assert not {"raw", "test"} & _unused(project)


def test_non_utf8_files_do_not_crash(project: Path) -> None:
    (project / "src" / "bin.py").write_bytes(b"\xff\xfe'data/interim'")
    assert "interim" not in _unused(project)


# --- main ------------------------------------------------------------------


@pytest.fixture
def patched(project: Path, monkeypatch):
    dirs = [project / "data" / "interim"]
    monkeypatch.setattr(cleanup, "PROJECT_ROOT", project)
    monkeypatch.setattr(cleanup, "find_unused_dirs", lambda: dirs)
    with mock.patch.object(cleanup.shutil, "rmtree") as rmtree:
        yield rmtree, dirs


def test_main_dry_run_does_not_delete(patched, capsys) -> None:
    rmtree, _ = patched
    assert cleanup.main([]) == 0
    rmtree.assert_not_called()
    assert "se eliminaría con --apply" in capsys.readouterr().out


def test_main_apply_deletes(patched, capsys) -> None:
    rmtree, dirs = patched
    assert cleanup.main(["--apply"]) == 0
    rmtree.assert_called_once_with(dirs[0])
    assert "Eliminada: data" in capsys.readouterr().out


def test_main_nothing_to_do(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cleanup, "find_unused_dirs", lambda: [])
    with mock.patch.object(cleanup.shutil, "rmtree") as rmtree:
        assert cleanup.main(["--apply"]) == 0
    rmtree.assert_not_called()
    assert "No hay carpetas sin usar" in capsys.readouterr().out
