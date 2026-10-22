import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ds_project.config import OptunaCfg, Settings, TrainCfg
from ds_project.models import tuning


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    return {
        "X_train": rng.normal(size=(8, 5, 3)).astype(np.float32),
        "y_train": np.array([0, 1] * 4),
        "X_val": rng.normal(size=(4, 5, 3)).astype(np.float32),
        "y_val": np.array([0, 1] * 2),
        "n_classes": np.asarray(2),
    }


@pytest.fixture
def db_url(tmp_path):
    return f"sqlite:///{(tmp_path / 'o.db').as_posix()}"


# ---------- suggest_params ----------
def test_suggest_params():
    trial = MagicMock()
    trial.suggest_float.return_value = 0.1
    trial.suggest_int.return_value = 2
    trial.suggest_categorical.return_value = 16
    space = {
        "lr": {"type": "float", "low": 0.001, "high": 0.01, "log": True},
        "n": {"type": "int", "low": 1, "high": 3},
        "c": {"type": "categorical", "choices": [8, 16]},
    }
    assert tuning.suggest_params(trial, space) == {"lr": 0.1, "n": 2, "c": 16}
    trial.suggest_float.assert_called_once_with("lr", 0.001, 0.01, log=True)
    trial.suggest_int.assert_called_once_with("n", 1, 3, step=1)
    trial.suggest_categorical.assert_called_once_with("c", [8, 16])


def test_suggest_params_invalid():
    with pytest.raises(ValueError, match="'x'"):
        tuning.suggest_params(MagicMock(), {"x": {"type": "bad"}})


# ---------- objective ----------
@pytest.mark.parametrize("direction,expected", [("maximize", 0.8), ("minimize", 0.5)])
def test_objective(fake_tf, data, direction, expected):
    settings = Settings(
        train=TrainCfg(model_params={"units": 8}),
        optuna=OptunaCfg(
            direction=direction,
            seed=10,
            trial_epochs=2,
            trial_patience=1,
            search_space={"dropout": {"type": "float", "low": 0.0, "high": 0.5}},
        ),
    )
    trial = MagicMock(number=3)
    trial.suggest_float.return_value = 0.25
    history = {"val_accuracy": [0.5, 0.8, 0.6]}
    model = MagicMock()
    model.count_params.return_value = 1234
    with patch("ds_project.models.train.train_model", return_value=(model, history)) as tm, patch(
        "ds_project.models.train.set_global_seed"
    ) as seed, patch.object(tuning, "make_pruning_callback", return_value="cb"):
        value = tuning.make_objective(data, settings)(trial)
    assert value == expected
    trial.set_user_attr.assert_any_call("n_params", 1234)
    trial.set_user_attr.assert_any_call("epochs_run", 3)
    seed.assert_called_once_with(13, True)
    fake_tf.keras.backend.clear_session.assert_called_once()
    kw = tm.call_args.kwargs
    assert kw["params"]["units"] == 8 and kw["params"]["dropout"] == 0.25
    assert (kw["epochs"], kw["patience"], kw["n_classes"]) == (2, 1, 2)
    assert kw["extra_callbacks"] == ["cb"]
    X_scaled = tm.call_args.args[0]
    assert np.allclose(X_scaled.reshape(-1, 3).mean(axis=0), 0.0, atol=1e-5)


def test_objective_missing_metric(fake_tf, data):
    with patch("ds_project.models.train.train_model", return_value=(None, {"loss": [1.0]})), patch(
        "ds_project.models.train.set_global_seed"
    ), patch.object(tuning, "make_pruning_callback"):
        with pytest.raises(ValueError, match="val_accuracy"):
            tuning.make_objective(data, Settings())(MagicMock(number=0))


# ---------- storage / study ----------
def test_make_storage_sqlite(fake_optuna, tmp_path):
    url = f"sqlite:///{(tmp_path / 'a' / 'b' / 'o.db').as_posix()}"
    tuning._make_storage(url)
    assert (tmp_path / "a" / "b").is_dir()
    fake_optuna.storages.RDBStorage.assert_called_once_with(
        url=url, engine_kwargs={"connect_args": {"timeout": 60}}
    )


def test_make_storage_other(fake_optuna):
    tuning._make_storage("postgresql://u@h/db")
    fake_optuna.storages.RDBStorage.assert_called_once_with(
        url="postgresql://u@h/db", engine_kwargs={}
    )


def test_create_study(fake_optuna, db_url):
    cfg = OptunaCfg(storage=db_url, study_name="s", direction="minimize", seed=5)
    tuning.create_study(cfg)
    kw = fake_optuna.create_study.call_args.kwargs
    assert (kw["study_name"], kw["direction"], kw["load_if_exists"]) == (
        "s",
        "minimize",
        True,
    )
    fake_optuna.samplers.TPESampler.assert_called_once_with(seed=5)


def test_optimize(fake_optuna, tmp_path, data, db_url):
    path = tmp_path / "d.npz"
    np.savez(path, **data)
    settings = Settings(optuna=OptunaCfg(n_trials=7, timeout=5.0, storage=db_url))
    with patch.object(tuning, "make_objective", return_value="obj") as mo:
        tuning._optimize(str(path), settings)
    loaded = mo.call_args.args[0]
    assert set(loaded) == set(data) and np.allclose(loaded["X_train"], data["X_train"])
    study = fake_optuna.load_study.return_value
    assert study.optimize.call_args.args[0] == "obj"
    kw = study.optimize.call_args.kwargs
    assert (kw["n_trials"], kw["timeout"], kw["catch"]) == (
        7,
        5.0,
        (RuntimeError, ValueError),
    )
    sys.modules["optuna.study"].MaxTrialsCallback.assert_called_once()
    fake_optuna.samplers.TPESampler.assert_called_once_with(seed=settings.optuna.seed)


def test_optimize_worker_seed_differs(fake_optuna, tmp_path, data, db_url):
    path = tmp_path / "d.npz"
    np.savez(path, **data)
    settings = Settings(optuna=OptunaCfg(storage=db_url, seed=10))
    with patch.object(tuning, "make_objective", return_value="obj"):
        tuning._optimize(str(path), settings, worker_idx=2)
    fake_optuna.samplers.TPESampler.assert_called_once_with(seed=12)


# ---------- run_study / workers ----------
def test_run_study_sequential(fake_optuna, tmp_path, db_url):
    settings = Settings(optuna=OptunaCfg(n_workers=1, storage=db_url))
    with patch.object(tuning, "create_study") as cs, patch.object(
        tuning, "detect_gpu_count", return_value=0
    ), patch.object(tuning, "configure_memory_growth") as mg, patch.object(
        tuning, "_optimize"
    ) as opt:
        study = tuning.run_study(tmp_path / "d.npz", settings)
    cs.assert_called_once_with(settings.optuna, None)
    mg.assert_called_once()
    opt.assert_called_once_with(str(tmp_path / "d.npz"), settings)
    assert study is fake_optuna.load_study.return_value


@pytest.mark.parametrize("exitcodes,ok", [([0, 0], True), ([0, 1], False)])
def test_run_study_parallel(fake_optuna, tmp_path, db_url, exitcodes, ok):
    settings = Settings(optuna=OptunaCfg(n_workers=2, storage=db_url))
    procs = [MagicMock(exitcode=c) for c in exitcodes]
    ctx = MagicMock()
    ctx.Process.side_effect = procs
    with patch.object(tuning, "create_study"), patch.object(
        tuning, "detect_gpu_count", return_value=2
    ), patch.object(tuning.mp, "get_context", return_value=ctx) as gc:
        if ok:
            tuning.run_study(tmp_path / "d.npz", settings)
        else:
            with pytest.raises(RuntimeError, match=r"\[1\]"):
                tuning.run_study(tmp_path / "d.npz", settings)
    gc.assert_called_once_with("spawn")
    calls = [c.kwargs for c in ctx.Process.call_args_list]
    assert all(k["target"] is tuning._worker for k in calls)
    assert [k["args"][:2] for k in calls] == [(0, 2), (1, 2)]
    for p in procs:
        p.start.assert_called_once()
        p.join.assert_called_once()


def test_worker_order():
    parent = MagicMock()
    with patch.object(tuning, "assign_gpu", parent.assign), patch.object(
        tuning, "configure_memory_growth", parent.grow
    ), patch.object(tuning, "_optimize", parent.opt):
        tuning._worker(1, 2, "d.npz", "S")
    assert [c[0] for c in parent.mock_calls] == ["assign", "grow", "opt"]
    parent.assign.assert_called_once_with(1, 2)
    parent.opt.assert_called_once_with("d.npz", "S", 1)


# ---------- Integración (TF + Optuna reales) ----------
@pytest.mark.slow
def test_pruning_callback_real():
    pytest.importorskip("tensorflow")
    optuna = pytest.importorskip("optuna")
    trial = MagicMock()
    trial.should_prune.return_value = True
    cb = tuning.make_pruning_callback(trial, "val_accuracy")
    cb.on_epoch_end(0, {"loss": 1.0})
    trial.report.assert_not_called()
    with pytest.raises(optuna.TrialPruned):
        cb.on_epoch_end(1, {"val_accuracy": 0.5})
    trial.report.assert_called_once_with(0.5, step=1)


@pytest.mark.slow
def test_run_study_smoke(tmp_path, monkeypatch, data, db_url):
    pytest.importorskip("tensorflow")
    pytest.importorskip("optuna")
    monkeypatch.setenv("DS_NUM_GPUS", "0")
    ds = tmp_path / "ds.npz"
    np.savez(ds, **data)
    settings = Settings(
        train=TrainCfg(model_params={"units": 8, "filters": 4, "n_conv_layers": 1}),
        optuna=OptunaCfg(
            storage=db_url,
            n_trials=2,
            trial_epochs=1,
            trial_patience=1,
            search_space={"dropout": {"type": "float", "low": 0.0, "high": 0.5}},
        ),
    )
    study = tuning.run_study(ds, settings)
    assert len(study.trials) >= 2
    assert "dropout" in study.best_params
