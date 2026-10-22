import dataclasses

import pytest

from ds_project.dsp.export import build_dsp_artifact, export_dsp_header, export_dsp_json
from ds_project.dsp.params import DSPParams
from ds_project.utils.io import load_json, save_yaml


def test_defaults_derived():
    p = DSPParams()
    assert p.n_samples == 16000
    assert p.n_frames == 1 + (16000 - 512) // 256
    assert p.n_bins == 257
    assert p.feature_shape == (p.n_frames, 13)


def test_short_clip_has_one_frame():
    p = DSPParams(clip_duration=0.01)
    assert (p.n_samples, p.n_frames) == (160, 1)


def test_aggregate_shape():
    p = DSPParams(mode="aggregate", features=("mfcc", "rms"), stats=("mean", "std"))
    assert p.feature_shape == ((13 + 1) * 2,)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sample_rate": 0},
        {"n_fft": 256},
        {"hop_length": 0},
        {"hop_length": 1024},
        {"window": "kaiser"},
        {"mode": "x"},
        {"features": ()},
        {"features": ("foo",)},
        {"stats": ("p90",)},
        {"mode": "aggregate", "stats": ()},
        {"fmax": 9000.0},
        {"fmin": 8000.0},
        {"n_mfcc": 50},
        {"pre_emphasis": 1.0},
        {"rolloff_percent": 1.0},
    ],
)
def test_invalid(kwargs):
    with pytest.raises(ValueError):
        DSPParams(**kwargs)


def test_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        DSPParams().n_mels = 10


def test_roundtrip_and_coercion(tmp_path):
    p = DSPParams(features=("mfcc", "zcr"))
    assert DSPParams.from_dict(p.to_dict()) == p
    assert DSPParams.from_dict(None) == DSPParams()
    q = DSPParams.from_dict({"sample_rate": "8000", "fmax": 4000, "features": ["rms"]})
    assert q.sample_rate == 8000 and isinstance(q.sample_rate, int)
    assert isinstance(q.fmax, float) and q.features == ("rms",)
    assert DSPParams.from_yaml(save_yaml(p.to_dict(), tmp_path / "d.yaml")) == p
    with pytest.raises(ValueError):
        DSPParams.from_dict({"unknown": 1})


def test_artifact(small_params):
    art = build_dsp_artifact(small_params, ["a", "b"])
    assert len(art["derived"]["feature_names"]) == small_params.frame_feature_dim
    assert art["scaler"] is None and "extra" not in art
    assert build_dsp_artifact(small_params, ["a"], extra={"k": 1})["extra"] == {"k": 1}


def test_exports_aggregate(tmp_path):
    p = DSPParams(mode="aggregate", features=("mfcc",), stats=("mean",))
    scaler = {"kind": "standard", "offset": [0.0] * 13, "divisor": [1.0] * 13}
    js = load_json(export_dsp_json(p, ["a", "b"], tmp_path / "d.json", scaler))
    assert js["dsp"] == p.to_dict()
    assert js["derived"]["feature_shape"] == [13]
    assert js["labels"] == ["a", "b"]
    header = export_dsp_header(p, ["a", "b"], tmp_path / "x" / "d.h", scaler).read_text()
    for token in [
        "DSP_SAMPLE_RATE 16000",
        "DSP_N_MFCC 13",
        "DSP_USE_MFCC 1",
        "DSP_USE_RMS 0",
        "DSP_STAT_MEAN 1",
        "DSP_STAT_STD 0",
        "DSP_MODE_AGGREGATE 1",
        "DSP_PRE_EMPHASIS 0.97f",
        "DSP_SCALER_OFFSET[13]",
        "DSP_N_CLASSES 2",
        'DSP_LABELS[2] = {"a", "b"}',
        "#endif /* DSP_PARAMS_H */",
    ]:
        assert token in header


def test_header_sequence_without_scaler(tmp_path):
    header = export_dsp_header(DSPParams(), ["a"], tmp_path / "d.h", guard="MY_H").read_text()
    assert "#ifndef MY_H" in header and "DSP_MODE_SEQUENCE 1" in header
    assert "DSP_STAT_MEAN 0" in header and "DSP_SCALER" not in header
