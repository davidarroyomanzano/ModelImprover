"""Reglas de .gitignore: entorno, cachés, datos y audio. Sustituye a test_gitignore_data.py."""

import pytest

from tests.conftest import ROOT, git, requires_git

pytestmark = requires_git


def _is_ignored(path: str) -> bool:
    # --no-index: evalúa las reglas aunque el archivo siga indexado
    return git("check-ignore", "-q", "--no-index", path).returncode == 0


@pytest.mark.parametrize(
    "path",
    [
        ".env",
        ".venv/lib/x.py",
        "src/ds_project/__pycache__/a.pyc",
        ".pytest_cache/v",
        ".ruff_cache/0.4.10/123",
        "data/raw/Good/x.wav",
        "data/test/Stopped/y.wav",
        "data/processed/features.npy",
        "data/interim/tmp.csv",
        "otro/z.wav",
        "MAYUS.WAV",
        "projects/test/all/v1/data/raw/Good/x.wav",
        "projects/test/all/v1/data/test/Stopped/y.wav",
        "projects/test/all/v1/data/processed/features.npy",
        "projects/test/all/v1/models/20260101_000000_abc/model.tflite",
        "projects/test/all/v1/models/20260101_000000_abc/evaluation/report.pdf",
        "projects/pinbolt/g5x/export/v150/nn_pinbolt_g5x-v150.tflite",
        "projects/pinbolt/g5x/export/v150/dsp_pinbolt_g5x-v150.py",
    ],
)
def test_ignored(path: str) -> None:
    assert _is_ignored(path), f"{path} debería estar ignorado"


@pytest.mark.parametrize(
    "path",
    [
        ".env.example",
        "data/external/.gitkeep",
        "projects/pinbolt/g5x/project.yaml",
        "projects/pinbolt/g5x/templates/dsp_template.py",
        "projects/pinbolt/g5x/templates/v3/dsp_template.py",
        "src/ds_project/__init__.py",
    ],
)
def test_not_ignored(path: str) -> None:
    assert not _is_ignored(path), f"{path} debería versionarse"


def test_no_duplicate_gitignore_file() -> None:
    extras = [p.name for p in ROOT.glob(".gitignore*") if p.name != ".gitignore"]
    assert not extras, f"Archivos .gitignore inválidos: {extras}"
