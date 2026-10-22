from pathlib import Path

import pytest
import yaml

from ds_project.config import (
    DataCfg,
    ExportCfg,
    OptunaCfg,
    PathsCfg,
    Settings,
    StorageCfg,
    TrainCfg,
    _from_dict,
    load_settings,
)

ROOT = Path(__file__).resolve().parents[1]


def _w(path, data):
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_missing_files_use_defaults(tmp_path):
    assert (
        load_settings(tmp_path / "a.yaml", tmp_path / "b.yaml", tmp_path / "c.yaml") == Settings()
    )


def test_empty_file_uses_defaults(tmp_path):
    empty = tmp_path / "e.yaml"
    empty.write_text("", encoding="utf-8")
    assert load_settings(empty, empty, empty) == Settings()


def test_project_configs_are_valid():
    s = load_settings(
        ROOT / "configs/config.yaml",
        ROOT / "configs/optuna.yaml",
        ROOT / "configs/dsp.yaml",
    )
    assert s.dsp.feature_shape == (s.dsp.n_frames, s.dsp.frame_feature_dim)
    assert s.dsp.aux_feature_shape[0] > 0
    assert s.export.quantization in ("float32", "float16", "dynamic", "int8")
    assert s.optuna.search_space


def test_overrides(tmp_path):
    cfg = _w(
        tmp_path / "c.yaml",
        {"data": {"source": "s3", "seed": 1}, "train": {"scaler": "minmax"}},
    )
    opt = _w(tmp_path / "o.yaml", {"n_workers": 4, "direction": "minimize"})
    dsp = _w(tmp_path / "d.yaml", {"features": ["rms", "zcr"], "mode": "aggregate"})
    s = load_settings(cfg, opt, dsp)
    assert (s.data.source, s.data.seed, s.train.scaler) == ("s3", 1, "minmax")
    assert (s.optuna.n_workers, s.optuna.direction) == (4, "minimize")
    assert s.dsp.features == ("rms", "zcr")


def test_env_paths(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "DS_CONFIG", str(_w(tmp_path / "c.yaml", {"export": {"quantization": "int8"}}))
    )
    monkeypatch.setenv("DS_OPTUNA_CONFIG", str(tmp_path / "none1.yaml"))
    monkeypatch.setenv("DS_DSP_CONFIG", str(tmp_path / "none2.yaml"))
    assert load_settings().export.quantization == "int8"


def test_unknown_section(tmp_path):
    with pytest.raises(ValueError, match="Secciones"):
        load_settings(_w(tmp_path / "c.yaml", {"foo": {}}), tmp_path / "x", tmp_path / "y")


def test_unknown_key(tmp_path):
    with pytest.raises(ValueError, match="Claves desconocidas"):
        load_settings(
            _w(tmp_path / "c.yaml", {"data": {"bogus": 1}}),
            tmp_path / "x",
            tmp_path / "y",
        )


def test_from_dict_none():
    assert _from_dict(PathsCfg, None) == PathsCfg()


@pytest.mark.parametrize(
    "factory",
    [
        lambda: DataCfg(source="ftp"),
        lambda: DataCfg(segment_overlap=1.0),
        lambda: DataCfg(test_size=0.5, val_size=0.5),
        lambda: TrainCfg(scaler="robust"),
        lambda: ExportCfg(quantization="int4"),
        lambda: StorageCfg(backend="gcs"),
        lambda: OptunaCfg(direction="up"),
        lambda: OptunaCfg(n_workers=0),
    ],
)
def test_invalid_values(factory):
    with pytest.raises(ValueError):
        factory()
