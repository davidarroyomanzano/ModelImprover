import logging
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from ds_project.data import process
from ds_project.data.ingest import list_audio_files, sync_raw_data
from ds_project.data.process import (
    load_audio,
    load_split,
    resample,
    segment_signal,
    split_files,
)


def _touch(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"")
    return p


# ---------- ingest ----------
def test_list_audio_files(tmp_path):
    _touch(tmp_path / "dog/a.wav")
    _touch(tmp_path / "dog/sub/b.FLAC")
    _touch(tmp_path / "dog/notes.txt")
    _touch(tmp_path / "cat/c.ogg")
    _touch(tmp_path / "root.wav")
    (tmp_path / "empty").mkdir()
    res = list_audio_files(tmp_path)
    assert list(res) == ["cat", "dog"]
    assert [p.name for p in res["dog"]] == ["a.wav", "b.FLAC"]
    assert list(list_audio_files(tmp_path, [".ogg"])) == ["cat"]


def test_list_audio_files_missing_dir(tmp_path):
    assert list_audio_files(tmp_path / "nope") == {}


def test_sync_raw_data(tmp_path):
    storage = MagicMock()
    storage.list_files.return_value = ["raw/a/1.wav", "raw/a/readme.txt", "raw/b/2.wav"]
    _touch(tmp_path / "a/1.wav")
    assert sync_raw_data(storage, "raw", tmp_path) == 1
    storage.list_files.assert_called_once_with("raw")
    storage.download_file.assert_called_once_with("raw/b/2.wav", tmp_path / "b/2.wav")


def test_sync_without_prefix(tmp_path):
    storage = MagicMock()
    storage.list_files.return_value = ["a/1.wav"]
    sync_raw_data(storage, "", tmp_path)
    storage.download_file.assert_called_once_with("a/1.wav", tmp_path / "a/1.wav")


# ---------- process ----------
def test_resample():
    x = np.random.default_rng(0).normal(size=800)
    same = resample(x, 8000, 8000)
    assert same.dtype == np.float32 and np.allclose(same, x, atol=1e-6)
    assert resample(x, 8000, 16000).shape == (1600,)


def test_load_audio_stereo_mocked(fake_module):
    sf = fake_module("soundfile")
    stereo = np.stack([np.ones(800), np.zeros(800)], axis=1).astype(np.float32)
    sf.read.return_value = (stereo, 8000)
    out = load_audio("x.wav", 16000)
    sf.read.assert_called_once_with("x.wav", dtype="float32", always_2d=False)
    assert out.shape == (1600,)
    assert np.allclose(out[200:-200], 0.5, atol=1e-2)


def test_load_audio_real_file(tmp_path):
    sf = pytest.importorskip("soundfile")
    sig = np.random.default_rng(0).uniform(-0.5, 0.5, 800).astype(np.float32)
    sf.write(tmp_path / "x.wav", sig, 8000, subtype="FLOAT")
    assert np.allclose(load_audio(tmp_path / "x.wav", 8000), sig, atol=1e-6)


@pytest.mark.parametrize(
    "length,overlap,min_ratio,expected",
    [
        (50, 0.0, 0.5, 1),
        (100, 0.0, 0.5, 1),
        (250, 0.0, 0.5, 3),
        (230, 0.0, 0.5, 2),
        (200, 0.5, 0.6, 3),
    ],
)
def test_segment_signal(length, overlap, min_ratio, expected):
    segs = segment_signal(np.ones(length, np.float32), 100, overlap, min_ratio)
    assert len(segs) == expected
    assert all(s.shape == (100,) for s in segs)


def _files(n_a, n_b):
    return {
        "b": [Path(f"b{i}.wav") for i in range(n_b)],
        "a": [Path(f"a{i}.wav") for i in range(n_a)],
    }


def test_split_files_no_leak():
    splits, labels = split_files(_files(10, 10), 0.2, 0.2, seed=0)
    assert labels == ["a", "b"]
    sets = {k: {p for p, _ in v} for k, v in splits.items()}
    assert not (
        sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"]
    )
    assert sum(len(s) for s in sets.values()) == 20
    assert (len(sets["train"]), len(sets["val"]), len(sets["test"])) == (12, 4, 4)
    assert all(p.name[0] == labels[y] for v in splits.values() for p, y in v)
    assert split_files(_files(10, 10), 0.2, 0.2, seed=0)[0] == splits


def test_split_small_class_warns(caplog):
    with caplog.at_level(logging.WARNING, logger="ds_project.data.process"):
        splits, _ = split_files(_files(10, 1), 0.2, 0.2, seed=0)
    assert sum(len(v) for v in splits.values()) == 11
    assert "sin estratificar" in caplog.text


def test_load_split(monkeypatch, small_params):
    audios = {"a.wav": np.ones(1600, np.float32), "b.wav": np.ones(100, np.float32)}
    mock = MagicMock(side_effect=lambda path, sr: audios[str(path)])
    monkeypatch.setattr(process, "load_audio", mock)
    X, y = load_split([(Path("a.wav"), 0), (Path("b.wav"), 1)], small_params)
    assert X.shape == (3, 800) and X.dtype == np.float32
    assert y.tolist() == [0, 0, 1] and y.dtype == np.int64
    mock.assert_any_call(Path("a.wav"), 8000)


def test_load_split_empty(small_params):
    X, y = load_split([], small_params)
    assert X.shape == (0, 800) and y.shape == (0,)
