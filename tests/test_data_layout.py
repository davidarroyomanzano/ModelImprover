"""
Estructura de projects/<producto>/<turbina>/<datos>/data/<split>/<clase>/*.wav.

Los tests con datos reales se saltan si no hay audio local.
"""

import wave
from pathlib import Path

import pytest

from ds_project.config import load_settings
from tests.conftest import CLASSES, ROOT

SCOPE = ROOT / load_settings().scope_dir()
DATA = SCOPE / "data"
HAS_AUDIO = any(DATA.rglob("*.wav")) if DATA.exists() else False
requires_audio = pytest.mark.skipif(
    not HAS_AUDIO,
    reason="Datos no versionados; no hay audio local",
)


@pytest.mark.skipif(not DATA.exists(), reason="Ámbito por defecto sin crear")
@pytest.mark.parametrize("sub", ["raw", "test", "interim", "processed"])
def test_data_dirs_exist(sub: str) -> None:
    assert (DATA / sub).is_dir()


def test_synthetic_layout(synthetic_dataset: Path) -> None:
    for split in ("raw", "test"):
        labels = {p.parent.name for p in (synthetic_dataset / split).rglob("*.wav")}
        assert labels == set(CLASSES)
    with wave.open(str(next(synthetic_dataset.rglob("*.wav")))) as wf:
        assert wf.getnchannels() == 1 and wf.getsampwidth() == 2


def _split_dir(split: str) -> Path:
    return DATA / split


@requires_audio
@pytest.mark.parametrize("split", ["raw", "test"])
def test_only_known_classes(split: str) -> None:
    labels = {p.name for p in _split_dir(split).iterdir() if p.is_dir()}
    unknown = labels - set(CLASSES)
    assert not unknown, f"Clases desconocidas en {split}: {unknown}"


@requires_audio
@pytest.mark.parametrize("split", ["raw", "test"])
def test_filename_prefix_matches_folder(split: str) -> None:
    bad = [
        str(p.relative_to(DATA))
        for p in (DATA / split).rglob("*.wav")
        if not p.name.startswith(p.parent.name + ".")
    ]
    assert not bad, f"Prefijo distinto a la carpeta: {bad[:5]}"


@requires_audio
def test_no_leakage_between_raw_and_test() -> None:
    raw = {p.name for p in (DATA / "raw").rglob("*.wav")}
    test = {p.name for p in (DATA / "test").rglob("*.wav")}
    assert not raw & test, f"Archivos en raw y test a la vez: {sorted(raw & test)[:5]}"


@requires_audio
def test_filenames_have_no_spaces() -> None:
    bad = [
        str(p.relative_to(DATA))
        for split in ["raw", "test"]
        for p in (DATA / split).rglob("*.wav")
        if " " in p.name or "(" in p.name
    ]
    assert not bad, f"Renombra estos archivos: {bad}"
