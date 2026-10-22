"""Búsqueda de hiperparámetros en dos fases: primero el DSP, después la red."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from ds_project import pipeline, wizard
from ds_project.config import OptunaCfg, phase_trial_counts
from ds_project.models import tuning
from ds_project.utils.io import load_json

DSP_SPACE = {"window_size": {"type": "float", "low": 0.1, "high": 0.4, "step": 0.1}}


def _data():
    rng = np.random.default_rng(0)

    def X(n):
        return rng.normal(size=(n, 5, 3)).astype(np.float32)

    return {
        "labels": ["a", "b"],
        "n_classes": 2,
        "X_train": X(8),
        "y_train": np.arange(8) % 2,
        "X_val": X(4),
        "y_val": np.arange(4) % 2,
        "X_test": X(4),
        "y_test": np.arange(4) % 2,
    }


def test_phase_trial_counts():
    assert phase_trial_counts(10) == (10, 10)
    assert phase_trial_counts(10, 3) == (3, 10)
    assert phase_trial_counts(10, None, 4) == (10, 4)
    assert OptunaCfg(n_trials=8, phase1_trials=20).phase_trials() == (20, 8)


def test_two_phase_needs_a_dsp_space_to_search():
    assert OptunaCfg(strategy="two_phase", dsp_search_space=DSP_SPACE).two_phase
    assert not OptunaCfg(strategy="joint", dsp_search_space=DSP_SPACE).two_phase
    assert not OptunaCfg(strategy="two_phase").two_phase  # sin espacio de DSP
    assert not OptunaCfg(strategy="two_phase", dsp_search_space=DSP_SPACE, tune_dsp=False).two_phase


@pytest.mark.parametrize(
    "kwargs",
    [
        {"strategy": "tres_fases"},
        {"phase1_epochs": 0},
        {"phase1_trials": 0},
        {"phase2_trials": 0},
    ],
)
def test_invalid_two_phase_options(kwargs):
    with pytest.raises(ValueError):
        OptunaCfg(**kwargs)


def test_phase_settings(make_settings):
    s = make_settings(
        optuna=OptunaCfg(
            n_trials=10,
            trial_epochs=40,
            phase1_epochs=3,
            phase1_model_params={"units": 8},
            search_space={"units": {"type": "categorical", "choices": [8, 16]}},
            dsp_search_space=DSP_SPACE,
            study_name="est",
        )
    )
    first = tuning.dsp_phase_settings(s, 4)
    assert first.optuna.study_name == "est__dsp"
    assert (first.optuna.n_trials, first.optuna.trial_epochs) == (4, 3)
    assert first.optuna.search_space == {} and first.optuna.active_dsp_space == DSP_SPACE
    assert first.train.model_params["units"] == 8
    assert first.train.model_params["n_conv_layers"] == tuning.PHASE1_MODEL_PARAMS["n_conv_layers"]

    second = tuning.net_phase_settings(s, 6)
    assert second.optuna.study_name == "est__net"
    assert (second.optuna.n_trials, second.optuna.trial_epochs) == (6, 40)
    assert second.optuna.active_dsp_space == {} and second.optuna.search_space
    assert second.train.model_params == s.train.model_params


def test_train_mode_two_phase():
    assert "two_phase" in wizard.TRAIN_MODES and set(wizard.TRAIN_MODES) == set(
        wizard.TRAIN_MODE_LABELS
    )
    on = {"enabled": True, "tune_dsp": True}
    assert wizard.train_mode({"optuna": {**on, "strategy": "two_phase"}}) == "two_phase"
    assert wizard.train_mode({"optuna": {**on, "strategy": "joint"}}) == "search_all"
    assert (
        wizard.train_mode({"optuna": {**on, "tune_dsp": False, "strategy": "two_phase"}})
        == "search_net"
    )
    assert wizard.train_mode({"optuna": {"enabled": False, "strategy": "two_phase"}}) == "fixed"


@pytest.fixture
def mocked_run(monkeypatch):
    train = MagicMock(return_value=(MagicMock(), {"loss": [1.0], "val_loss": [0.9]}))
    metrics = {"accuracy": 1.0, "confusion_matrix": [[2, 0], [0, 2]]}
    datasets = MagicMock(side_effect=lambda s: _data())
    monkeypatch.setattr(pipeline, "prepare_dataset", datasets)
    monkeypatch.setattr("ds_project.models.train.train_model", train)
    monkeypatch.setattr("ds_project.models.train.set_global_seed", MagicMock())
    monkeypatch.setattr("ds_project.models.export.export_tflite", lambda m, path, *a, **k: path)
    monkeypatch.setattr("ds_project.models.evaluate.evaluate_keras", lambda *a, **k: metrics)
    monkeypatch.setattr("ds_project.models.evaluate.evaluate_tflite", lambda *a, **k: metrics)
    for name in ("plot_training_history", "plot_confusion_matrix", "plot_optuna_history"):
        monkeypatch.setattr(f"ds_project.visualization.plots.{name}", MagicMock())
    return SimpleNamespace(train=train, datasets=datasets)


def _study(name, best_params, value, n):
    return MagicMock(
        study_name=name, best_params=best_params, best_value=value, trials=list(range(n))
    )


def test_run_pipeline_two_phases(make_settings, mocked_run, monkeypatch, tmp_path):
    dsp_study = _study("e__dsp", {"window_size": 0.2}, 0.7, 3)
    net_study = _study("e__net", {"units": 16}, 0.9, 5)
    run_study = MagicMock(side_effect=[dsp_study, net_study])
    monkeypatch.setattr("ds_project.models.tuning.run_study", run_study)
    saved = MagicMock()
    monkeypatch.setattr("ds_project.reporting.optuna_info.save_study_info", saved)
    s = make_settings(
        optuna=OptunaCfg(
            enabled=True,
            n_trials=8,
            phase2_trials=5,
            strategy="two_phase",
            dsp_search_space=DSP_SPACE,
            storage=f"sqlite:///{(tmp_path / 'o.db').as_posix()}",
        )
    )

    run_dir = pipeline.run_pipeline(s)

    first, second = run_study.call_args_list
    assert first.args[1].optuna.study_name.endswith("__dsp")
    assert first.args[1].optuna.n_trials == 8 and first.args[1].optuna.search_space == {}
    assert second.args[1].optuna.study_name.endswith("__net")
    assert second.args[1].optuna.n_trials == 5 and second.args[1].optuna.active_dsp_space == {}
    assert second.args[1].dsp.clip_duration == 0.2  # la fase 2 usa el mejor DSP
    assert first.args[2]["phase"] == "dsp" and second.args[2]["phase"] == "net"
    assert first.args[0] != second.args[0] and second.args[0].name.endswith("_net.npz")
    assert mocked_run.datasets.call_args.args[0].dsp.clip_duration == 0.2

    saved.assert_called_once_with(net_study, run_dir, dsp_study)
    params = mocked_run.train.call_args.kwargs["params"]
    assert params["units"] == 16 and "window_size" not in params
    summary = load_json(run_dir / "optuna_summary.json")
    assert summary["search_mode"] == "two_phase" and summary["n_trials"] == 3 + 5
    search = load_json(run_dir / "search_config.json")
    assert search["mode"] == "two_phase"
    assert [(p["phase"], p["n_trials"]) for p in search["phases"]] == [("dsp", 8), ("net", 5)]
    assert search["phases"][0]["searches"] == ["window_size"]
    assert search["phases"][0]["model_params"]["n_conv_layers"] == 1
    assert summary["best_params"] == {"window_size": 0.2, "units": 16}
    assert summary["dsp_phase"] == {
        "best_value": 0.7,
        "best_params": {"window_size": 0.2},
        "n_trials": 3,
    }
    assert load_json(run_dir / "dsp_params.json")["dsp"]["clip_duration"] == 0.2


def test_joint_strategy_keeps_a_single_study(make_settings, mocked_run, monkeypatch, tmp_path):
    study = _study("e", {"units": 16}, 0.95, 3)
    run_study = MagicMock(return_value=study)
    monkeypatch.setattr("ds_project.models.tuning.run_study", run_study)
    s = make_settings(
        optuna=OptunaCfg(
            enabled=True,
            dsp_search_space=DSP_SPACE,
            storage=f"sqlite:///{(tmp_path / 'o.db').as_posix()}",
        )
    )
    run_dir = pipeline.run_pipeline(s)
    assert run_study.call_count == 1
    summary = load_json(run_dir / "optuna_summary.json")
    assert "dsp_phase" not in summary and summary["search_mode"] == "joint"
    search = load_json(run_dir / "search_config.json")
    assert search["mode"] == "joint" and len(search["phases"]) == 1
    assert "window_size" in search["phases"][0]["searches"]
