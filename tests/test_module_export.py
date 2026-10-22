"""Tests de la exportación del paquete DSP + TFLite de PinBolt."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from ds_project.dsp.module_export import export_pinbolt_package, find_latest_run

TEMPLATE = Path(__file__).resolve().parents[1] / "templates" / "dsp_template.py"
LABELS = ["a", "b", "c"]


def _artifact(aux_dim: int = 25) -> dict:
    block = {
        "frame_length": 0.025,
        "frame_stride": 0.4,
        "n_filters": 8,
        "fft_length": 2048,
        "low_freq": 550,
        "high_freq": 3350,
        "noise_floor_db": -30,
    }
    return {
        "dsp": {
            "sample_rate": 16000,
            "clip_duration": 0.9,
            "pre_emphasis": 0.98,
            "profile": "pinbolt",
            "zero_padding": True,
            "n_filters_final": 0,
            "mfe_blocks": [block, {**block, "n_filters": 16, "low_freq": 50.5}],
        },
        "labels": LABELS,
        "aux_scaler": {
            "kind": "standard",
            "offset": [0.5 + i for i in range(aux_dim)],
            "divisor": [1.0 + i for i in range(aux_dim)],
        },
    }


def _make_run(tmp_path: Path, aux_dim: int = 25) -> Path:
    run = tmp_path / "run"
    run.mkdir()
    (run / "dsp_params.json").write_text(json.dumps(_artifact(aux_dim)), encoding="utf-8")
    (run / "model.tflite").write_bytes(b"tflite")
    return run


def _load(path: Path):
    spec = importlib.util.spec_from_file_location("generated_dsp", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_export_fills_constants(tmp_path):
    out = export_pinbolt_package(
        _make_run(tmp_path), tmp_path / "out", TEMPLATE, "demo", "v1", 0.45
    )
    assert out["tflite"].read_bytes() == b"tflite"
    assert out["dsp"].name == "dsp_demo-v1.py"
    mod = _load(out["dsp"])
    assert mod.get_labels() == LABELS
    assert mod.WINDOW_SIZE_SEC == 0.9
    assert mod.WINDOW_STEP_SEC == 0.45
    assert mod.ZERO_PADDING is True
    assert mod.N_FILTERS_FINAL == 8
    assert mod.PREEMPHASIS_COEFF == 0.98
    assert [b["N_FILTERS"] for b in mod.MFE_BLOCKS] == [8, 16]
    assert mod.MFE_BLOCKS[1]["LOW_FREQ"] == 50.5
    assert mod.AUX_MEAN[1] == pytest.approx(1.5)
    assert mod.AUX_SCALE[24] == pytest.approx(25.0)


def test_export_rejects_aux_mismatch(tmp_path):
    run = _make_run(tmp_path, aux_dim=22)
    with pytest.raises(ValueError, match="22"):
        export_pinbolt_package(run, tmp_path / "out", TEMPLATE)


def test_export_aux_mismatch_override(tmp_path):
    run = _make_run(tmp_path, aux_dim=22)
    out = export_pinbolt_package(run, tmp_path / "out", TEMPLATE, allow_aux_mismatch=True)
    assert _load(out["dsp"]).AUX_FEATURE_DIM == 22


def test_export_rejects_non_pinbolt_profile(tmp_path):
    run = _make_run(tmp_path)
    art = json.loads((run / "dsp_params.json").read_text(encoding="utf-8"))
    art["dsp"]["profile"] = "default"
    (run / "dsp_params.json").write_text(json.dumps(art), encoding="utf-8")
    with pytest.raises(ValueError, match="pinbolt"):
        export_pinbolt_package(run, tmp_path / "out", TEMPLATE)


def test_project_matches_generated_dsp(tmp_path):
    import numpy as np

    from ds_project.dsp.export import build_dsp_artifact
    from ds_project.dsp.params import DSPParams, MFEBlock
    from ds_project.features.audio import extract_auxiliary_features, extract_features
    from ds_project.features.scaling import FeatureScaler

    def block(**kw):
        base = dict(frame_length=0.025, frame_stride=0.4, n_filters=8, fft_length=2048)
        base.update(low_freq=550, high_freq=3350, noise_floor_db=-30)
        return MFEBlock(**{**base, **kw})

    params = DSPParams(
        profile="pinbolt",
        sample_rate=16000,
        clip_duration=0.9,
        pre_emphasis=0.98,
        features=("mfe",),
        aux_features=(),
        mfe_blocks=(
            block(),
            block(
                frame_length=0.14,
                frame_stride=0.1,
                n_filters=128,
                low_freq=50,
                high_freq=6400,
                noise_floor_db=-44,
            ),
        ),
    )
    assert params.mfe_flat_length == 1048
    assert params.mfe_reshape == (131, 8)

    rng = np.random.default_rng(0)
    clips = [rng.normal(0, 0.1, params.n_samples).astype(np.float32) for _ in range(6)]
    aux = np.stack([extract_auxiliary_features(c, params) for c in clips])
    assert aux.shape == (6, 25)
    scaler = FeatureScaler("standard").fit(aux)

    artifact = build_dsp_artifact(params, LABELS, aux_scaler_dict=scaler.to_dict())
    run = tmp_path / "run"
    run.mkdir()
    (run / "dsp_params.json").write_text(json.dumps(artifact), encoding="utf-8")
    (run / "model.tflite").write_bytes(b"x")
    mod = _load(export_pinbolt_package(run, tmp_path / "out", TEMPLATE)["dsp"])

    for clip in clips[:3]:
        mfe, aux_scaled = mod.extract_model_inputs(clip, 16000)
        np.testing.assert_allclose(mfe, extract_features(clip, params), atol=1e-6)
        expected = scaler.transform(extract_auxiliary_features(clip, params)[None, :])[0]
        np.testing.assert_allclose(aux_scaled, expected, rtol=1e-3, atol=1e-3)


def test_find_latest_run(tmp_path):
    _make_run(tmp_path)
    assert find_latest_run(tmp_path) == tmp_path / "run"
    with pytest.raises(FileNotFoundError):
        find_latest_run(tmp_path / "vacio")
