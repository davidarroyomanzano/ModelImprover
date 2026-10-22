"""Búsqueda de hiperparámetros con Optuna: RDBStorage compartido + N procesos (1 GPU c/u)."""

from __future__ import annotations

import logging
import multiprocessing as mp
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Optional, Set, Union

import numpy as np

from ds_project.config import OptunaCfg, Settings
from ds_project.dsp.params import MFEBlock
from ds_project.models.architectures import DEFAULT_MODEL_PARAMS
from ds_project.utils.gpu import assign_gpu, configure_memory_growth, detect_gpu_count

logger = logging.getLogger(__name__)
PathLike = Union[str, Path]

# Red de la fase del DSP en la búsqueda en dos fases: pequeña y rápida, y la misma en todos los
# trials para que las diferencias entre ellos se deban solo al DSP.
PHASE1_MODEL_PARAMS: Dict[str, Any] = {
    "learning_rate": 1e-3,
    "batch_size": 32,
    "n_conv_layers": 1,
    "filters": 16,
    "kernel_size": 3,
    "n_dense_layers": 1,
    "units": 32,
    "dropout": 0.2,
}
PHASE_DSP = "dsp"
PHASE_NET = "net"


def suggest_params(trial, search_space: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    for name, spec in search_space.items():
        kind = spec.get("type")
        if kind == "float":
            kwargs: Dict[str, Any] = {}
            if spec.get("step") is not None:
                kwargs["step"] = float(spec["step"])
            else:
                kwargs["log"] = bool(spec.get("log", False))
            params[name] = trial.suggest_float(
                name, float(spec["low"]), float(spec["high"]), **kwargs
            )
        elif kind == "int":
            params[name] = trial.suggest_int(
                name, int(spec["low"]), int(spec["high"]), step=int(spec.get("step", 1))
            )
        elif kind == "categorical":
            params[name] = trial.suggest_categorical(name, list(spec["choices"]))
        else:
            raise ValueError(f"Tipo de parámetro no soportado en '{name}': {kind}")
    return params


def _flat_dsp_specs(space: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Aplana dsp_search_space: los parámetros del bloque i se llaman '<campo>_<i>' (i desde 1)."""
    specs = {k: v for k, v in space.items() if k != "mfe_blocks"}
    for i, block in enumerate(space.get("mfe_blocks") or [], start=1):
        specs.update({f"{k}_{i}": v for k, v in block.items()})
    return specs


def dsp_param_names(space: Dict[str, Any]) -> Set[str]:
    return set(_flat_dsp_specs(space or {}))


def suggest_dsp_params(trial, space: Dict[str, Any]) -> Dict[str, Any]:
    specs = _flat_dsp_specs(space)
    params: Dict[str, Any] = {}
    if "window_size" in specs:
        params.update(suggest_params(trial, {"window_size": specs.pop("window_size")}))
    if "window_step" in specs:
        spec = dict(specs.pop("window_step"))
        if "window_size" in params:  # el paso no puede superar la ventana
            spec["high"] = min(float(spec["high"]), params["window_size"])
        params.update(suggest_params(trial, {"window_step": spec}))
    params.update(suggest_params(trial, specs))
    return params


def resolve_dsp_settings(
    settings: Settings, params: Dict[str, Any], local: bool = False
) -> Settings:
    """Devuelve Settings con DSP y solape de segmentos aplicados desde `params`."""
    base = settings.dsp
    changes: Dict[str, Any] = {}
    if "window_size" in params:
        changes["clip_duration"] = float(params["window_size"])
    n_blocks = len((settings.optuna.active_dsp_space or {}).get("mfe_blocks") or [])
    if n_blocks:
        blocks = []
        for i in range(1, n_blocks + 1):
            prev = base.mfe_blocks[i - 1] if i <= len(base.mfe_blocks) else MFEBlock()
            over = {
                f: params[f"{f}_{i}"]
                for f in MFEBlock.__dataclass_fields__
                if f"{f}_{i}" in params  # pylint: disable=E1101
            }
            blocks.append(replace(prev, **over))
        changes["features"] = ("mfe",)
        changes["mfe_blocks"] = tuple(blocks)
    dsp = replace(base, **changes)

    data = settings.data
    overlap = data.segment_overlap
    if "window_step" in params:
        overlap = max(0.0, 1.0 - float(params["window_step"]) / dsp.clip_duration)
    data = replace(data, segment_overlap=overlap, source="local" if local else data.source)
    return replace(settings, dsp=dsp, data=data)


def _phase_study_name(settings: Settings, phase: str) -> str:
    return f"{settings.optuna.study_name}__{phase}"


def dsp_phase_settings(settings: Settings, n_trials: int) -> Settings:
    """Fase 1 (solo DSP): red fija pequeña, pocas épocas y sin espacio de búsqueda de red."""
    cfg = settings.optuna
    model_params = {**PHASE1_MODEL_PARAMS, **cfg.phase1_model_params}
    return replace(
        settings,
        train=replace(settings.train, model_params=model_params),
        optuna=replace(
            cfg,
            search_space={},
            n_trials=n_trials,
            trial_epochs=cfg.phase1_epochs,
            study_name=_phase_study_name(settings, PHASE_DSP),
        ),
    )


def net_phase_settings(settings: Settings, n_trials: int) -> Settings:
    """Fase 2 (solo red): ``settings`` ya lleva el mejor DSP, que se mantiene fijo."""
    return replace(
        settings,
        optuna=replace(
            settings.optuna,
            tune_dsp=False,
            n_trials=n_trials,
            study_name=_phase_study_name(settings, PHASE_NET),
        ),
    )


def make_pruning_callback(trial, monitor: str):
    import optuna
    import tensorflow as tf

    class _PruningCallback(tf.keras.callbacks.Callback):  # pylint: disable=E1101
        def on_epoch_end(self, epoch, logs=None):
            value = (logs or {}).get(monitor)
            if value is None:
                return
            trial.report(float(value), step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned(f"Poda en época {epoch}: {monitor}={value:.4f}")

    return _PruningCallback()


def make_objective(data: Dict[str, np.ndarray], settings: Settings):
    from ds_project.features.scaling import FeatureScaler
    from ds_project.models.train import set_global_seed, train_model

    cfg = settings.optuna

    def objective(trial) -> float:
        import tensorflow as tf

        tf.keras.backend.clear_session()  # pylint: disable=E1101
        set_global_seed(cfg.seed + trial.number, settings.train.deterministic)
        params = {**DEFAULT_MODEL_PARAMS, **settings.train.model_params}
        trial_data = data
        if cfg.active_dsp_space:
            import optuna

            from ds_project.pipeline import prepare_dataset

            dsp_params = suggest_dsp_params(trial, cfg.active_dsp_space)
            try:
                trial_settings = resolve_dsp_settings(settings, dsp_params, local=True)
            except ValueError as exc:
                raise optuna.TrialPruned(f"DSP inválido: {exc}") from exc
            trial_data = prepare_dataset(trial_settings, cache=False)
        params.update(suggest_params(trial, cfg.search_space))

        trial_dsp = trial_settings.dsp if cfg.active_dsp_space else settings.dsp
        mfe_kind, aux_kind = trial_dsp.scaler_kinds(settings.train.scaler)
        scaler = FeatureScaler(mfe_kind)
        X_train = scaler.fit_transform(trial_data["X_train"])
        X_val = scaler.transform(trial_data["X_val"])
        X_train_feats = X_val_feats = None
        if trial_dsp.has_aux:
            aux_scaler = FeatureScaler(aux_kind)
            X_train_feats = aux_scaler.fit_transform(trial_data["X_train_feats"])
            X_val_feats = aux_scaler.transform(trial_data["X_val_feats"])

        model, history = train_model(
            X_train,
            trial_data["y_train"],
            X_val,
            trial_data["y_val"],
            n_classes=int(trial_data["n_classes"]),
            params=params,
            epochs=cfg.trial_epochs,
            patience=cfg.trial_patience,
            use_class_weights=settings.train.use_class_weights,
            extra_callbacks=[make_pruning_callback(trial, cfg.metric)],
            X_train_feats=X_train_feats,
            X_val_feats=X_val_feats,
            mfe_reshape=trial_dsp.mfe_reshape,
        )
        if cfg.metric not in history:
            raise ValueError(f"Métrica '{cfg.metric}' no disponible: {list(history)}")
        values = history[cfg.metric]
        trial.set_user_attr("n_params", int(model.count_params()))
        trial.set_user_attr("epochs_run", len(values))
        for name in ("val_accuracy", "val_loss"):
            if name in history and name != cfg.metric:
                best = max(history[name]) if "acc" in name else min(history[name])
                trial.set_user_attr(f"best_{name}", float(best))
        return float(max(values) if cfg.direction == "maximize" else min(values))

    return objective


def _make_storage(url: str):
    import optuna

    engine_kwargs: Dict[str, Any] = {}
    if url.startswith("sqlite:///"):
        Path(url[len("sqlite:///") :]).parent.mkdir(parents=True, exist_ok=True)
        engine_kwargs = {"connect_args": {"timeout": 60}}
    return optuna.storages.RDBStorage(url=url, engine_kwargs=engine_kwargs)


def create_study(cfg: OptunaCfg, user_attrs: Optional[Dict[str, Any]] = None):
    import optuna

    study = optuna.create_study(
        study_name=cfg.study_name,
        storage=_make_storage(cfg.storage),
        direction=cfg.direction,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=cfg.seed),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=cfg.n_startup_trials, n_warmup_steps=cfg.n_warmup_steps
        ),
    )
    for key, value in (user_attrs or {}).items():
        study.set_user_attr(key, value)
    return study


def _optimize(dataset_path: str, settings: Settings, worker_idx: int = 0) -> None:
    import optuna
    from optuna.study import MaxTrialsCallback
    from optuna.trial import TrialState

    cfg = settings.optuna
    with np.load(dataset_path) as npz:
        data = {k: npz[k] for k in npz.files}
    study = optuna.load_study(
        study_name=cfg.study_name,
        storage=_make_storage(cfg.storage),
        sampler=optuna.samplers.TPESampler(seed=cfg.seed + worker_idx),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=cfg.n_startup_trials, n_warmup_steps=cfg.n_warmup_steps
        ),
    )
    study.optimize(
        make_objective(data, settings),
        n_trials=cfg.n_trials,
        timeout=cfg.timeout,
        gc_after_trial=True,
        catch=(RuntimeError, ValueError),
        callbacks=[
            MaxTrialsCallback(
                cfg.n_trials,
                states=(TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL),
            )
        ],
    )


def _worker(worker_idx: int, n_gpus: int, dataset_path: str, settings: Settings) -> None:
    # Debe ejecutarse ANTES de importar TensorFlow en el proceso hijo.
    assign_gpu(worker_idx, n_gpus)
    configure_memory_growth()
    _optimize(dataset_path, settings, worker_idx)


def run_study(
    dataset_path: PathLike,
    settings: Settings,
    user_attrs: Optional[Dict[str, Any]] = None,
):
    """Lanza la búsqueda. dataset_path: .npz con X_train, y_train, X_val, y_val, n_classes.

    ``user_attrs`` (proyecto, versión de datos, hash del dataset...) se guarda en el estudio.
    """
    import optuna

    cfg = settings.optuna
    create_study(cfg, user_attrs)
    n_gpus = detect_gpu_count()
    logger.info("Optuna: %d trials, %d workers, %d GPUs", cfg.n_trials, cfg.n_workers, n_gpus)

    if cfg.n_workers <= 1:
        configure_memory_growth()
        _optimize(str(dataset_path), settings)
    else:
        ctx = mp.get_context("spawn")
        procs = [
            ctx.Process(target=_worker, args=(i, n_gpus, str(dataset_path), settings))
            for i in range(cfg.n_workers)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join()
        failed = [p.exitcode for p in procs if p.exitcode != 0]
        if failed:
            raise RuntimeError(f"Workers de Optuna fallaron con códigos {failed}")

    study = optuna.load_study(study_name=cfg.study_name, storage=_make_storage(cfg.storage))
    logger.info("Mejor valor %.4f con %s", study.best_value, study.best_params)
    return study
