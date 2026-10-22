"""Validación de los manifests de origen frente a lo que hay en disco."""

from __future__ import annotations

import csv
import wave
from pathlib import Path

from ds_project.data.source_manifest import validate_source_manifests


def _wav(path: Path, sr: int = 16000, seconds: float = 0.5) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(sr)
        fh.writeframes(b"\x00\x00" * int(sr * seconds))


def _manifest(path: Path, rows: list) -> None:
    fields = [
        "output_file",
        "source_file",
        "source_stem",
        "class_role",
        "sample_rate_hz",
        "extract_seconds",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _row(name: str, stem: str = "rec1") -> dict:
    return {
        "output_file": name,
        "source_file": "/x/Good/" + stem + ".wav",
        "source_stem": stem,
        "class_role": "background",
        "sample_rate_hz": 16000,
        "extract_seconds": 0.5,
    }


def test_validate_detects_missing_unlisted_and_leak(tmp_path) -> None:
    data = tmp_path / "data"
    _wav(data / "raw" / "Good" / "a.wav")
    _wav(data / "raw" / "Good" / "extra.wav")
    _wav(data / "test" / "Good" / "b.wav")
    _manifest(
        data / "m_manifest1.csv",
        [_row("a.wav"), _row("b.wav"), _row("gone.wav", "rec2")],
    )
    res = validate_source_manifests(data, data / "raw", data / "test", sample_rate=16000)
    assert res["n_rows"] == 3 and res["n_wavs"] == 3
    assert res["missing_on_disk"]["count"] == 1
    assert res["unlisted_on_disk"]["count"] == 1
    assert res["shared_sources"]["count"] >= 1
    assert res["sample_rate_mismatch"] == 0


def test_validate_flags_sample_rate_mismatch(tmp_path) -> None:
    data = tmp_path / "data"
    _wav(data / "raw" / "Good" / "a.wav", sr=8000)
    (data / "test").mkdir(parents=True)
    _manifest(data / "m_manifest1.csv", [_row("a.wav")])
    res = validate_source_manifests(data, data / "raw", data / "test", sample_rate=16000)
    assert res["sample_rate_mismatch"] == 1 and not res["ok"]


def test_validate_without_manifests_is_available_false(tmp_path) -> None:
    res = validate_source_manifests(tmp_path, tmp_path / "raw", tmp_path / "test")
    assert res["available"] is False and res["ok"] is True
