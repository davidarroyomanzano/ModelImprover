"""Pipeline completo: ingest -> process -> features -> (tune) -> train -> evaluate -> export."""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from ds_project.config import Settings, load_settings
from ds_project.data.ingest import list_audio_files, sync_raw_data
from ds_project.data.manifest import build_manifest
from ds_project.data.process import load_split, split_files
from ds_project.dsp.export import export_dsp_header, export_dsp_json
from ds_project.features.build import build_auxiliary_features_cached, build_features_cached
from ds_project.features.scaling import FeatureScaler
from ds_project.models.architectures import DEFAULT_MODEL_PARAMS
from ds_project.utils.io import (
    config_hash,
    ensure_dir,
    make_run_id,
    save_json,
    save_yaml,
    to_builtin,
)
from ds_project.utils.storage import get_storage

logger = logging.getLogger(__name__)

SEARCH_MODE_LABELS = {
    "joint": "DSP y red a la vez (un estudio)",
    "two_phase": "dos fases: 1) DSP con red mínima, 2) red con ese DSP",
    "net_only": "solo la red, DSP fijo",
}


def search_description(settings: Settings) -> Dict[str, Any]:
    """Qué búsqueda va a hacer Optuna (se guarda en el run antes de empezar)."""
    cfg = settings.optuna
    if cfg.two_phase:
        from ds_project.models.tuning import PHASE1_MODEL_PARAMS

        n_dsp, n_net = cfg.phase_trials()
        mode = "two_phase"
        phases = [
            {
                "phase": "dsp",
                "study_name": f"{cfg.study_name}__dsp",
                "n_trials": n_dsp,
                "trial_epochs": cfg.phase1_epochs,
                "model_params": {**PHASE1_MODEL_PARAMS, **cfg.phase1_model_params},
                "searches": sorted(cfg.active_dsp_space),
            },
            {
                "phase": "net",
                "study_name": f"{cfg.study_name}__net",
                "n_trials": n_net,
                "trial_epochs": cfg.trial_epochs,
                "searches": sorted(cfg.search_space),
            },
        ]
    else:
        mode = "joint" if cfg.active_dsp_space else "net_only"
        phases = [
            {
                "phase": mode,
                "study_name": cfg.study_name,
                "n_trials": cfg.n_trials,
                "trial_epochs": cfg.trial_epochs,
                "searches": sorted(cfg.active_dsp_space) + sorted(cfg.search_space),
            }
        ]
    return {
        "mode": mode,
        "mode_label": SEARCH_MODE_LABELS[mode],
        "strategy": cfg.strategy,
        "tune_dsp": cfg.tune_dsp,
        "metric": cfg.metric,
        "direction": cfg.direction,
        "timeout_s": cfg.timeout,
        "n_workers": cfg.n_workers,
        "phases": phases,
    }


def prepare_dataset(settings: Settings, cache: bool = True) -> Dict[str, Any]:
    paths = settings.resolved_paths()
    storage = get_storage(settings.storage)
    if settings.data.source != "local":
        if storage is None:
            raise ValueError("data.source remoto requiere storage.backend configurado")
        sync_raw_data(
            storage,
            settings.data.remote_prefix,
            paths.raw_dir,
            settings.data.extensions,
        )

    files = list_audio_files(paths.raw_dir, settings.data.extensions)
    if len(files) < 2:
        raise ValueError(f"Se necesitan >=2 clases en {paths.raw_dir}/<etiqueta>/")

    splits, labels = split_files(
        files, settings.data.test_size, settings.data.val_size, settings.data.seed
    )
    manifest = build_manifest(files, paths.raw_dir, Path(paths.interim_dir) / "manifest_cache.json")
    data: Dict[str, Any] = {"labels": labels, "n_classes": len(labels), "manifest": manifest}
    for name, items in splits.items():
        signals, y = load_split(
            items,
            settings.dsp,
            settings.data.segment_overlap,
            settings.data.min_segment_ratio,
        )
        X = build_features_cached(signals, settings.dsp, paths.interim_dir if cache else None, name)
        data[f"X_{name}"], data[f"y_{name}"] = X, y
        if settings.dsp.has_aux:
            X_feats = build_auxiliary_features_cached(
                signals,
                settings.dsp,
                paths.interim_dir if cache else None,
                f"{name}_aux",
            )
            data[f"X_{name}_feats"] = X_feats
            logger.info("Split %s: X_feats=%s", name, X_feats.shape)
        logger.info("Split %s: X=%s", name, X.shape)
    return data


def make_study_name(settings: Settings, run_hash: str) -> str:
    """Nombre del estudio Optuna: base + proyecto + versión de datos + hash del run."""
    parts = [
        settings.optuna.study_name,
        settings.project_name or "default",
        settings.data.version or "nover",
        run_hash,
    ]
    return "__".join(parts)


def scoped_storage(settings: Settings) -> str:
    """URL de Optuna; una SQLite local cuelga del ámbito ``<proyecto>/<turbina>/<datos>``."""
    url = settings.optuna.storage
    prefix = "sqlite:///"
    scope = settings.scope_dir()
    if scope is None or not url.startswith(prefix):
        return url
    return prefix + (scope / url[len(prefix) :]).as_posix()


def _save_search_dataset(path: Path, data: Dict[str, Any], has_aux: bool) -> Path:
    """Guarda el .npz que leen los workers de Optuna (train/val y, si hay, features auxiliares)."""
    np.savez(
        path,
        X_train=data["X_train"],
        y_train=data["y_train"],
        X_val=data["X_val"],
        y_val=data["y_val"],
        n_classes=np.asarray(data["n_classes"]),
        **(
            {"X_train_feats": data["X_train_feats"], "X_val_feats": data["X_val_feats"]}
            if has_aux
            else {}
        ),
    )
    return path


def run_pipeline(settings: Optional[Settings] = None, tune: Optional[bool] = None) -> Path:
    from ds_project.models.evaluate import evaluate_keras, evaluate_tflite
    from ds_project.models.export import export_tflite
    from ds_project.models.train import set_global_seed, train_model
    from ds_project.visualization.plots import (
        plot_confusion_matrix,
        plot_optuna_history,
        plot_training_history,
    )

    settings = settings or load_settings()
    tune = settings.optuna.enabled if tune is None else tune
    set_global_seed(settings.data.seed, settings.train.deterministic)

    # Misma configuración y datos => misma carpeta: se renombra con la fecha y se sobrescribe.
    data = prepare_dataset(settings)
    labels = data["labels"]
    manifest = data.get("manifest") or {}
    paths = settings.resolved_paths()
    identity = {"settings": to_builtin(settings), "dataset_hash": manifest.get("dataset_hash", "")}
    cfg_hash = config_hash(identity)
    run_id = make_run_id(identity)
    models_dir = ensure_dir(paths.models_dir)
    for old in models_dir.glob(f"*_{cfg_hash}"):
        if old.is_dir() and old.name != run_id:
            old.rename(models_dir / run_id)
            break
    run_dir = ensure_dir(models_dir / run_id)
    logger.info("Run: %s", run_dir)
    if manifest:
        save_json(manifest, run_dir / "dataset_manifest.json")

    dataset_path = _save_search_dataset(
        ensure_dir(paths.processed_dir) / f"dataset_{cfg_hash}.npz", data, settings.dsp.has_aux
    )

    params = {**DEFAULT_MODEL_PARAMS, **settings.train.model_params}
    if tune:
        from ds_project.models.tuning import (
            dsp_param_names,
            dsp_phase_settings,
            net_phase_settings,
            resolve_dsp_settings,
            run_study,
        )
        from ds_project.reporting.optuna_info import save_study_info

        settings = replace(
            settings,
            optuna=replace(
                settings.optuna,
                study_name=make_study_name(settings, cfg_hash),
                storage=scoped_storage(settings),
            ),
        )
        attrs = {
            "product": settings.data.product,
            "turbine": settings.data.turbine,
            "data_version": settings.data.version,
            "dataset_hash": manifest.get("dataset_hash", ""),
            "run_hash": cfg_hash,
        }
        dsp_study = None
        search = search_description(settings)
        save_json(search, run_dir / "search_config.json")
        logger.info("Búsqueda Optuna: %s", search["mode_label"])
        if settings.optuna.two_phase:
            n_dsp, n_net = settings.optuna.phase_trials()
            logger.info("Optuna en dos fases: %d trials de DSP y %d de red", n_dsp, n_net)
            dsp_study = run_study(
                dataset_path,
                dsp_phase_settings(settings, n_dsp),
                {**attrs, "phase": "dsp"},
            )
            settings = resolve_dsp_settings(settings, dict(dsp_study.best_params))
            data = prepare_dataset(settings)
            labels = data["labels"]
            net_path = _save_search_dataset(
                dataset_path.with_name(f"{dataset_path.stem}_net.npz"), data, settings.dsp.has_aux
            )
            study = run_study(
                net_path, net_phase_settings(settings, n_net), {**attrs, "phase": "net"}
            )
            best = {**dsp_study.best_params, **study.best_params}
            params.update(study.best_params)
            summary_trials = len(dsp_study.trials) + len(study.trials)
        else:
            study = run_study(dataset_path, settings, attrs)
            best = dict(study.best_params)
            summary_trials = len(study.trials)
            if settings.optuna.active_dsp_space:
                settings = resolve_dsp_settings(settings, best)
                data = prepare_dataset(settings)
                labels = data["labels"]
                dsp_names = dsp_param_names(settings.optuna.active_dsp_space)
                params.update({k: v for k, v in best.items() if k not in dsp_names})
            else:
                params.update(best)
        plot_optuna_history(study, run_dir / "optuna_history.png")
        save_study_info(study, run_dir, dsp_study)
        summary = {
            "best_value": study.best_value,
            "best_params": best,
            "n_trials": summary_trials,
            "search_mode": search["mode"],
        }
        if dsp_study is not None:
            summary.update(
                dsp_phase={
                    "best_value": dsp_study.best_value,
                    "best_params": dict(dsp_study.best_params),
                    "n_trials": len(dsp_study.trials),
                },
            )
        save_json(summary, run_dir / "optuna_summary.json")
    save_json(params, run_dir / "best_params.json")

    mfe_kind, aux_kind = settings.dsp.scaler_kinds(settings.train.scaler)
    scaler = FeatureScaler(mfe_kind)
    X_train = scaler.fit_transform(data["X_train"])
    X_val = scaler.transform(data["X_val"])
    X_test = scaler.transform(data["X_test"])
    aux_scaler = None
    X_train_feats = X_val_feats = X_test_feats = None
    if settings.dsp.has_aux:
        aux_scaler = FeatureScaler(aux_kind)
        X_train_feats = aux_scaler.fit_transform(data["X_train_feats"])
        X_val_feats = aux_scaler.transform(data["X_val_feats"])
        X_test_feats = aux_scaler.transform(data["X_test_feats"])

    # La búsqueda consume aleatoriedad: se resemilla para un modelo final reproducible.
    set_global_seed(settings.data.seed, settings.train.deterministic)
    model, history = train_model(
        X_train,
        data["y_train"],
        X_val,
        data["y_val"],
        n_classes=data["n_classes"],
        params=params,
        epochs=settings.train.epochs,
        patience=settings.train.patience,
        use_class_weights=settings.train.use_class_weights,
        verbose=2,
        X_train_feats=X_train_feats,
        X_val_feats=X_val_feats,
        mfe_reshape=settings.dsp.mfe_reshape,
    )
    model.save(run_dir / "model.keras")

    tflite_path = export_tflite(
        model,
        run_dir / "model.tflite",
        settings.export.quantization,
        representative_data=[X_train, X_train_feats] if X_train_feats is not None else X_train,
        n_representative=settings.export.representative_samples,
    )

    scaler.save(run_dir / "scaler.joblib")
    scaler_dict = scaler.to_dict()
    aux_scaler_dict = None
    if aux_scaler is not None:
        aux_scaler.save(run_dir / "feature_scaler.joblib")
        aux_scaler_dict = aux_scaler.to_dict()
    export_dsp_json(
        settings.dsp,
        labels,
        run_dir / "dsp_params.json",
        scaler_dict,
        aux_scaler_dict,
        extra={
            "run_id": run_id,
            "quantization": settings.export.quantization,
            "product": settings.data.product,
            "turbine": settings.data.turbine,
            "data_version": settings.data.version,
            "dataset_hash": manifest.get("dataset_hash", ""),
        },
    )
    if settings.export.export_header:
        export_dsp_header(
            settings.dsp, labels, run_dir / "dsp_params.h", scaler_dict, aux_scaler_dict
        )

    metrics: Dict[str, Any] = {"labels": labels}
    if len(X_test):
        metrics["keras"] = evaluate_keras(model, X_test, data["y_test"], labels, X_test_feats)
        metrics["tflite"] = evaluate_tflite(
            tflite_path, X_test, data["y_test"], labels, X_test_feats
        )
        plot_confusion_matrix(
            metrics["tflite"]["confusion_matrix"],
            labels,
            run_dir / "confusion_matrix.png",
        )
    save_json(metrics, run_dir / "metrics.json")
    save_json(history, run_dir / "history.json")
    plot_training_history(history, run_dir / "training_history.png")
    save_yaml(settings, run_dir / "config_snapshot.yaml")

    storage = get_storage(settings.storage)
    if storage is not None and settings.storage.upload_artifacts:
        d = settings.data
        prefix = f"models/{d.product}/{d.turbine}" if d.product else "models"
        n = storage.upload_directory(run_dir, f"{prefix}/{run_id}")
        logger.info("Subidos %d artefactos", n)

    logger.info("Pipeline completado: %s", run_dir)
    return run_dir


if __name__ == "__main__":  # pragma: no cover
    from ds_project.logger import get_logger

    get_logger("ds_project")  # aplica configs/logging.yaml para que los INFO salgan en el log
    run_pipeline()
