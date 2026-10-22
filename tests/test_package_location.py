"""El paquete se instala desde src/ (layout src). Absorbe test_local_package.py."""

import importlib
from pathlib import Path

import pytest

from tests.conftest import ROOT

SUBPACKAGES = ["data", "dsp", "features", "models", "utils", "visualization"]


def test_package_imports_from_src() -> None:
    import ds_project

    pkg = Path(ds_project.__file__).resolve()
    assert pkg.is_relative_to(ROOT / "src"), f"ds_project importado desde {pkg}"


@pytest.mark.parametrize("sub", SUBPACKAGES)
def test_subpackages_have_init(sub: str) -> None:
    assert (ROOT / "src" / "ds_project" / sub / "__init__.py").is_file()


@pytest.mark.parametrize("module", ["config", "logger", "pipeline"])
def test_top_level_modules_import(module: str) -> None:
    imported = importlib.import_module(f"ds_project.{module}")
    assert imported is not None


def test_virtualenv_is_local() -> None:
    poetry_toml = (ROOT / "poetry.toml").read_text(encoding="utf-8")
    assert "in-project = true" in poetry_toml.replace(" ", " ")
