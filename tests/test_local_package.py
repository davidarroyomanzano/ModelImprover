"""Garantiza que ds_project se importa desde este repositorio y no de otro proyecto."""

from pathlib import Path

import ds_project
import ds_project.pipeline as pipeline

REPO_SRC = Path(__file__).resolve().parents[1] / "src"


def test_ds_project_is_imported_from_this_repo():
    pkg = Path(ds_project.__file__).resolve()
    assert pkg.is_relative_to(
        REPO_SRC.resolve()
    ), f"ds_project importado desde {pkg}, se esperaba dentro de {REPO_SRC}"


def test_pipeline_exposes_run_pipeline():
    assert callable(getattr(pipeline, "run_pipeline", None))
