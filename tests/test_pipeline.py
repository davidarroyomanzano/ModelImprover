from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from ds_project import pipeline
from ds_project.config import DataCfg, ExportCfg, OptunaCfg, StorageCfg
from ds_project.utils.io import load_json


# ---------- prepare_dataset ----------
@pytest.fixture
def patched_prepare(monkeypatch):
    files = {"a": [Path("a1.wav")], "b": [Path("b1.wav")]}
    splits = {"train": [(Path("a1.wav"), 0)], "val": [], "test": [(Path("b1.wav"), 1)]}
    m = SimpleNamespace(
        files=files,
        get_storage=MagicMock(return_value=None),
        sync=MagicMock(),
        list_files=MagicMock(return_value=files),
        split=MagicMock(return_value=(splits, ["a", "b"])),
        load=MagicMock(
            side_effect=lambda items, *a: (
                np.zeros((len(items), 800), np.float32),
                np.zeros(len(items), np.int64),
            )
        ),
        build=MagicMock(side_effect=lambda s, p, d, name: np.ones((len(s), 5, 10), np.float32)),
        build_aux=MagicMock(side_effect=lambda s, p, d, name: np.ones((len(s), 4), np.float32)),
    )
    monkeypatch.setattr(pipeline, "get_storage", m.get_storage)
    monkeypatch.setattr(pipeline, "sync_raw_data", m.sync)
    monkeypatch.setattr(pipeline, "list_audio_files", m.list_files)
    monkeypatch.setattr(pipeline, "split_files", m.split)
    monkeypatch.setattr(pipeline, "load_split", m.load)
    monkeypatch.setattr(pipeline, "build_features_cached", m.build)
    monkeypatch.setattr(pipeline, "build_auxiliary_features_cached", m.build_aux)
    return m


def test_prepare_dataset_local(make_settings, patched_prepare):
    s = make_settings()
    data = pipeline.prepare_dataset(s)
    patched_prepare.sync.assert_not_called()
    patched_prepare.split.assert_called_once_with(
        patched_prepare.files, s.data.test_size, s.data.val_size, s.data.seed
    )
    assert data["labels"] == ["a", "b"] and data["n_classes"] == 2
    assert data["X_train"].shape == (1, 5, 10) and data["X_val"].shape[0] == 0
    calls = patched_prepare.build.call_args_list
    assert [c.args[3] for c in calls] == ["train", "val", "test"]
    assert all(c.args[2] == s.paths.interim_dir for c in calls)


def test_prepare_dataset_with_auxiliary_features(make_settings, patched_prepare):
    s = make_settings(dsp=replace(make_settings().dsp, aux_features=("rms",)))
    data = pipeline.prepare_dataset(s)
    assert data["X_train_feats"].shape == (1, 4)
    assert [call.args[3] for call in patched_prepare.build_aux.call_args_list] == [
        "train_aux",
        "val_aux",
        "test_aux",
    ]


def test_prepare_dataset_remote_requires_storage(make_settings, patched_prepare):
    with pytest.raises(ValueError, match="remoto"):
        pipeline.prepare_dataset(make_settings(data=DataCfg(source="s3")))


def test_prepare_dataset_remote_syncs(make_settings, patched_prepare):
    storage = MagicMock()
    patched_prepare.get_storage.return_value = storage
    s = make_settings(data=DataCfg(source="s3", remote_prefix="audio"))
    pipeline.prepare_dataset(s)
    patched_prepare.sync.assert_called_once_with(
        storage, "audio", s.paths.raw_dir, s.data.extensions
    )


def test_prepare_dataset_needs_two_classes(make_settings, patched_prepare):
    patched_prepare.list_files.return_value = {"a": [Path("a.wav")]}
    with pytest.raises(ValueError, match=">=2 clases"):
        pipeline.prepare_dataset(make_settings())


# ---------- run_pipeline (todo lo pesado simulado) ----------
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


@pytest.fixture
def mocked_run(monkeypatch):
    model = MagicMock()
    train = MagicMock(return_value=(model, {"loss": [1.0], "val_loss": [0.9]}))
    export = MagicMock(side_effect=lambda m, path, *a, **k: Path(path))
    metrics = {"accuracy": 1.0, "confusion_matrix": [[2, 0], [0, 2]]}
    monkeypatch.setattr(pipeline, "prepare_dataset", lambda s: _data())
    monkeypatch.setattr("ds_project.models.train.train_model", train)
    monkeypatch.setattr("ds_project.models.train.set_global_seed", MagicMock())
    monkeypatch.setattr("ds_project.models.export.export_tflite", export)
    monkeypatch.setattr(
        "ds_project.models.evaluate.evaluate_keras", MagicMock(return_value=metrics)
    )
    monkeypatch.setattr(
        "ds_project.models.evaluate.evaluate_tflite", MagicMock(return_value=metrics)
    )
    for name in (
        "plot_training_history",
        "plot_confusion_matrix",
        "plot_optuna_history",
    ):
        monkeypatch.setattr(f"ds_project.visualization.plots.{name}", MagicMock())
    return SimpleNamespace(model=model, train=train, export=export)


ARTIFACTS = [
    "best_params.json",
    "scaler.joblib",
    "dsp_params.json",
    "dsp_params.h",
    "metrics.json",
    "history.json",
    "config_snapshot.yaml",
]


def test_run_pipeline_without_tuning(make_settings, mocked_run):
    s = make_settings(export=ExportCfg(quantization="int8"))
    run_dir = pipeline.run_pipeline(s, tune=False)
    for name in ARTIFACTS:
        assert (run_dir / name).exists(), name
    assert not (run_dir / "optuna_summary.json").exists()
    assert len(list(Path(s.paths.processed_dir).glob("dataset_*.npz"))) == 1

    kw = mocked_run.train.call_args.kwargs
    assert (kw["n_classes"], kw["epochs"]) == (2, s.train.epochs)
    X_scaled = mocked_run.train.call_args.args[0]
    assert np.allclose(X_scaled.reshape(-1, 3).mean(axis=0), 0.0, atol=1e-5)
    assert mocked_run.export.call_args.args[2] == "int8"
    mocked_run.model.save.assert_called_once_with(run_dir / "model.keras")

    js = load_json(run_dir / "dsp_params.json")
    assert js["labels"] == ["a", "b"] and js["scaler"]["kind"] == "standard"
    assert js["extra"]["quantization"] == "int8"
    assert load_json(run_dir / "metrics.json")["tflite"]["accuracy"] == 1.0


def test_run_pipeline_without_header(make_settings, mocked_run):
    run_dir = pipeline.run_pipeline(
        make_settings(export=ExportCfg(export_header=False)), tune=False
    )
    assert not (run_dir / "dsp_params.h").exists()


def test_run_pipeline_with_tuning(make_settings, mocked_run, monkeypatch, tmp_path):
    study = MagicMock(best_params={"units": 16}, best_value=0.95, trials=[1, 2, 3])
    run_study = MagicMock(return_value=study)
    monkeypatch.setattr("ds_project.models.tuning.run_study", run_study)
    s = make_settings(
        optuna=OptunaCfg(enabled=True, storage=f"sqlite:///{(tmp_path / 'o.db').as_posix()}")
    )
    run_dir = pipeline.run_pipeline(s)
    assert run_study.call_args.args[0].suffix == ".npz"
    assert load_json(run_dir / "best_params.json")["units"] == 16
    assert load_json(run_dir / "optuna_summary.json") == {
        "best_value": 0.95,
        "best_params": {"units": 16},
        "n_trials": 3,
        "search_mode": "net_only",
    }
    search = load_json(run_dir / "search_config.json")
    assert search["mode"] == "net_only" and search["phases"][0]["n_trials"] == s.optuna.n_trials
    assert mocked_run.train.call_args.kwargs["params"]["units"] == 16


def test_run_pipeline_uploads(make_settings, mocked_run, tmp_path):
    remote = tmp_path / "remote"
    s = make_settings(
        storage=StorageCfg(backend="local", local_root=str(remote), upload_artifacts=True)
    )
    run_dir = pipeline.run_pipeline(s, tune=False)
    assert (remote / "models" / run_dir.name / "metrics.json").exists()
