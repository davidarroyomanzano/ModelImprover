import json
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import numpy as np

from ds_project.dsp.params import DSPParams
from ds_project.utils.io import (
    config_hash,
    ensure_dir,
    load_json,
    load_yaml,
    make_run_id,
    save_json,
    save_yaml,
    to_builtin,
)


@dataclass
class _D:
    a: int
    p: Path


def test_to_builtin():
    data = {
        1: np.int64(2),
        "f": np.float32(1.5),
        "b": np.bool_(True),
        "arr": np.arange(3),
        "t": (1, 2),
        "s": {3},
        "d": _D(1, Path("x")),
    }
    out = to_builtin(data)
    assert out == {
        "1": 2,
        "f": 1.5,
        "b": True,
        "arr": [0, 1, 2],
        "t": [1, 2],
        "s": [3],
        "d": {"a": 1, "p": "x"},
    }
    assert isinstance(out["1"], int) and isinstance(out["f"], float) and isinstance(out["b"], bool)
    json.dumps(out)


def test_ensure_dir(tmp_path):
    p = ensure_dir(tmp_path / "a" / "b")
    assert p.is_dir() and ensure_dir(p) == p


def test_json_roundtrip(tmp_path):
    data = {"x": np.float64(1.0), "ñ": [1, 2]}
    path = save_json(data, tmp_path / "n" / "d.json")
    assert load_json(path) == {"x": 1.0, "ñ": [1, 2]}
    assert "ñ" in path.read_text(encoding="utf-8")


def test_yaml_roundtrip_dataclass(tmp_path):
    p = DSPParams(features=("mfcc", "rms"))
    loaded = load_yaml(save_yaml(p, tmp_path / "d.yaml"))
    assert DSPParams.from_dict(loaded) == p


def test_config_hash():
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})
    assert config_hash({"a": 1}) != config_hash({"a": 2})
    assert len(config_hash({"a": 1}, length=12)) == 12


def test_make_run_id():
    with patch("ds_project.utils.io.datetime") as dt:
        dt.now.return_value.strftime.return_value = "20240101_120000"
        rid = make_run_id({"a": 1})
    assert rid == f"20240101_120000_{config_hash({'a': 1})}"
