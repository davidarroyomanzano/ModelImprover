"""Dashboard HTML: estimaciones, payload y fichero autocontenido."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from ds_project.dsp.params import DSPParams, MFEBlock
from ds_project.reporting import dashboard as dash
from ds_project.utils.io import save_json, save_yaml


def _dsp(**kw) -> DSPParams:
    blocks = (
        MFEBlock(frame_length=0.03, frame_stride=0.02, n_filters=16, fft_length=256),
        MFEBlock(frame_length=0.04, frame_stride=0.04, n_filters=16, fft_length=256),
    )
    return DSPParams(
        profile="pinbolt", features=("mfe",), mfe_blocks=blocks, clip_duration=0.5, **kw
    )


@pytest.mark.slow
def test_estimate_matches_keras_param_count():
    from ds_project.models.architectures import build_model

    dsp = _dsp()
    params = {"n_conv_layers": 3, "filters": 16, "kernel_size": 5, "n_dense_layers": 2, "units": 32}
    model = build_model((dsp.mfe_flat_length,), 5, params, (25,), dsp.mfe_reshape)
    est = dash.estimate_network(dsp, params, 5)
    assert est is not None
    assert int(est["n_params"]) == model.count_params()


def test_estimate_none_without_pinbolt_profile():
    assert dash.estimate_network(DSPParams(), {}, 3) is None


def test_derive_trial_scales_with_window_and_step():
    base = _dsp()
    small = dash.derive_trial(
        {"params": {"window_size": 0.5, "window_step": 0.25}}, base, {}, 4, 2.0
    )
    big = dash.derive_trial({"params": {"window_size": 1.0, "window_step": 0.25}}, base, {}, 4, 2.0)
    assert big["window_s"] == 1.0 and small["window_s"] == 0.5
    assert big["size_kb"] > small["size_kb"]
    assert big["dsp_mops"] > small["dsp_mops"]
    assert big["mops"] == pytest.approx(big["dsp_mops"] + big["net_mops"])


def test_derive_trial_invalid_dsp_is_empty():
    bad = {"params": {"low_freq_1": 5000, "high_freq_1": 100}}
    assert dash.derive_trial(bad, _dsp(), {}, 4, 2.0) == {}


def test_derive_trial_prefers_recorded_n_params():
    t = {"params": {}, "user_attrs": {"n_params": 1000}}
    out = dash.derive_trial(t, _dsp(), {}, 4, 2.0)
    assert out["n_params"] == 1000


def _make_run(tmp_path: Path, two_phase: bool = False, evaluation: bool = True) -> Path:
    run = tmp_path / "run1"
    run.mkdir()
    base = _dsp()
    save_json({"dsp": _dsp_dict(base)}, run / "dsp_params.json")
    save_yaml(
        {
            "data": {"product": "p", "turbine": "t", "version": "v0"},
            "export": {"quantization": "float16"},
            "train": {"model_params": {"units": 16}},
            "optuna": {"metric": "val_accuracy"},
        },
        run / "config_snapshot.yaml",
    )
    trial = lambda n, v, w: {  # noqa: E731
        "number": n,
        "value": v,
        "state": "COMPLETE" if v is not None else "PRUNED",
        "duration_s": 3.0,
        "params": {"window_size": w, "window_step": w / 2, "units": 32},
    }
    study = {
        "study_name": "s",
        "direction": "MAXIMIZE",
        "best_trial": 1,
        "trials": [trial(0, 0.7, 0.5), trial(1, 0.8, 1.0), trial(2, None, 0.6)],
        "importances": {"units": 1.0},
    }
    if two_phase:
        study["dsp_phase"] = {**study, "study_name": "s__dsp"}
    save_json(study, run / "optuna_trials.json")
    save_json(
        {
            "labels": ["Good", "Bad"],
            "keras": {"accuracy": 0.9, "confusion_matrix": [[9, 1], [1, 9]]},
        },
        run / "metrics.json",
    )
    if evaluation:
        ev = run / "evaluation"
        ev.mkdir()
        save_json(
            {
                "labels": ["Good", "Bad"],
                "segment": {"accuracy": 0.8, "f1_macro": 0.8, "confusion_matrix": [[1, 0], [0, 1]]},
                "files_by_class": {"Good": 1, "Bad": 1},
                "verdict": {},
            },
            ev / "metrics_test.json",
        )
        (ev / "predictions.csv").write_text(
            "file,segment,true,pred,confidence,p_Good,p_Bad\n"
            "a.wav,0,Good,Good,0.9,0.9,0.1\nb.wav,0,Bad,Bad,0.8,0.2,0.8\n",
            encoding="utf-8",
        )
    return run


def _dsp_dict(dsp: DSPParams) -> dict:
    from dataclasses import asdict

    return asdict(dsp)


def test_build_payload_joint_and_evaluation(tmp_path):
    payload = dash.build_payload(_make_run(tmp_path))
    assert {t["phase"] for t in payload["trials"]} == {"joint"}
    assert len(payload["trials"]) == 3
    assert payload["trials"][1]["d"]["window_s"] == 1.0
    assert payload["evaluation"]["predictions"]["labels"] == ["Good", "Bad"]
    assert payload["limits"]["max_window_s"] == dash.DEFAULT_LIMITS["max_window_s"]


def test_build_payload_two_phase_tags_phases(tmp_path):
    payload = dash.build_payload(_make_run(tmp_path, two_phase=True))
    assert {t["phase"] for t in payload["trials"]} == {"dsp", "net"}
    assert payload["best_phase"] == "net"


def test_build_payload_without_evaluation(tmp_path):
    payload = dash.build_payload(_make_run(tmp_path, evaluation=False))
    assert payload["evaluation"] is None
    assert payload["internal"]["keras"]["accuracy"] == 0.9


def test_write_dashboard_is_self_contained(tmp_path):
    pytest.importorskip("plotly")
    run = _make_run(tmp_path)
    path = dash.write_dashboard(run, limits={"max_size_kb": 500})
    html = path.read_text(encoding="utf-8")
    assert path.name == "dashboard_run1.html"
    assert "__DATA_JSON__" not in html and "__PLOTLY_JS__" not in html
    assert "<script src=" not in html and "<link " not in html
    data = re.search(r'<script id="data" type="application/json">(.*?)</script>', html, re.S)
    assert json.loads(data.group(1))["limits"]["max_size_kb"] == 500


def test_clean_replaces_non_finite_numbers():
    assert dash._clean({"a": [float("nan"), 1.0, float("inf")]}) == {"a": [None, 1.0, None]}
