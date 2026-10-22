"""Fixtures compartidas: parámetros DSP pequeños, señales y módulos falsos (tf, optuna, msal...)."""

from __future__ import annotations

import math
import os
import shutil
import struct
import subprocess
import sys
import wave
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from ds_project.config import OptunaCfg, PathsCfg, Settings
from ds_project.dsp.params import DSPParams

ROOT = Path(__file__).resolve().parents[1]

CLASSES = (
    "Good",
    "Stopped",
    "Hydraulic_charge_start",
    "Hydraulic_charge_stop",
    "Impacts_hub",
    "Impacts_pitch",
    "Impacts_yaw",
    "Tendon_rupture",
)

HAS_GIT = shutil.which("git") is not None and (ROOT / ".git").exists()

requires_git = pytest.mark.skipif(
    not HAS_GIT,
    reason="Requiere git y repositorio",
)


def git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def tracked_files() -> list[str]:
    result = git("ls-files")
    return result.stdout.splitlines() if result.returncode == 0 else []


@pytest.fixture(autouse=True)
def _isolate_ds_env(monkeypatch):
    """Los tests no heredan ``DS_*`` del terminal ni del ``.env`` (p. ej. ``DS_DSP_CONFIG``)."""
    for name in [k for k in os.environ if k.startswith("DS_")]:
        monkeypatch.delenv(name)
    try:
        import dotenv

        monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    except ImportError:  # pragma: no cover
        pass


@pytest.fixture(scope="session")
def root() -> Path:
    return ROOT


def write_sine_wav(
    path: Path,
    freq: float = 440.0,
    seconds: float = 1.0,
    sr: int = 16000,
) -> Path:
    """Escribe un WAV mono de 16 bits usando solo la stdlib."""
    path.parent.mkdir(parents=True, exist_ok=True)

    n = int(seconds * sr)

    frames = b"".join(
        struct.pack(
            "<h",
            int(32767 * 0.5 * math.sin(2 * math.pi * freq * i / sr)),
        )
        for i in range(n)
    )

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(frames)

    return path


@pytest.fixture
def synthetic_dataset(tmp_path: Path) -> Path:
    """Crea data/<split>/<clase>/*.wav sin depender de datos reales."""
    base = tmp_path / "data"

    for split in ("raw", "test"):
        for i, label in enumerate(CLASSES):
            write_sine_wav(
                base / split / label / f"{label}.s{i}.{split}.wav",
                200 + 50 * i,
            )

    return base


@pytest.fixture
def small_params():
    """DSP reducido: 800 muestras, 5 frames, 10 MFCC."""
    return DSPParams(
        sample_rate=8000,
        clip_duration=0.1,
        frame_length=256,
        hop_length=128,
        n_fft=256,
        n_mels=20,
        n_mfcc=10,
        fmin=20.0,
        fmax=4000.0,
    )


@pytest.fixture
def sine():
    def _sine(freq=1000.0, sr=16000, dur=1.0):
        t = np.arange(int(sr * dur)) / sr
        return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)

    return _sine


@pytest.fixture
def fake_module(monkeypatch):
    """Inyecta un MagicMock en sys.modules."""

    def _make(name):
        mod = MagicMock(name=name)
        monkeypatch.setitem(sys.modules, name, mod)
        return mod

    return _make


@pytest.fixture
def fake_tf(fake_module):
    return fake_module("tensorflow")


@pytest.fixture
def fake_optuna(fake_module):
    opt = fake_module("optuna")
    fake_module("optuna.study")
    fake_module("optuna.trial")
    opt.load_study.return_value = MagicMock(
        best_value=0.9,
        best_params={"units": 8},
    )
    return opt


@pytest.fixture
def make_settings(tmp_path, small_params):
    def _make(**overrides):
        paths = PathsCfg(
            raw_dir=str(tmp_path / "raw"),
            interim_dir=str(tmp_path / "interim"),
            processed_dir=str(tmp_path / "processed"),
            models_dir=str(tmp_path / "models"),
        )

        base = {
            "paths": paths,
            "dsp": small_params,
            "optuna": OptunaCfg(
                enabled=False,
                storage=f"sqlite:///{(tmp_path / 'o.db').as_posix()}",
            ),
        }

        base.update(overrides)
        return Settings(**base)

    return _make
