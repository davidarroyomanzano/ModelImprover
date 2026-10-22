"""Resumen serializable de un estudio Optuna, guardado dentro del run."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

from ds_project.utils.io import load_json, load_yaml, save_json

logger = logging.getLogger(__name__)
PathLike = Union[str, Path]
TRIALS_FILE = "optuna_trials.json"


def study_to_dict(study) -> Dict[str, Any]:
    """Trials, mejor trial e importancia de hiperparámetros de un estudio."""
    trials = []
    for t in study.trials:
        duration = t.duration.total_seconds() if t.duration is not None else None
        trials.append(
            {
                "number": t.number,
                "value": t.value,
                "state": t.state.name,
                "duration_s": duration,
                "params": dict(t.params),
                "user_attrs": {
                    k: v for k, v in t.user_attrs.items() if isinstance(v, (int, float, str))
                },
            }
        )
    try:
        best = study.best_trial.number
    except ValueError:
        best = None
    importances: Dict[str, float] = {}
    try:
        from optuna.importance import get_param_importances

        importances = {k: float(v) for k, v in get_param_importances(study).items()}
    except Exception as exc:  # noqa: BLE001 - la importancia es opcional
        logger.info("Sin importancia de hiperparámetros: %s", exc)
    return {
        "study_name": study.study_name,
        "direction": study.direction.name,
        "best_trial": best,
        "trials": trials,
        "importances": importances,
    }


def save_study_info(study, run_dir: PathLike, dsp_study=None) -> Optional[Path]:
    """Guarda los trials en el run; el entrenamiento no debe fallar por este extra.

    En la búsqueda en dos fases ``study`` es el de la red y ``dsp_study`` el del DSP.
    """
    try:
        info = study_to_dict(study)
        if dsp_study is not None:
            info["dsp_phase"] = study_to_dict(dsp_study)
        return save_json(info, Path(run_dir) / TRIALS_FILE)
    except Exception as exc:  # noqa: BLE001
        logger.warning("No se guardaron los trials de Optuna: %s", exc)
        return None


def load_study_info(run_dir: PathLike) -> Optional[Dict[str, Any]]:
    """Lee ``optuna_trials.json``; si falta, lo reconstruye desde el almacenamiento del run."""
    run_dir = Path(run_dir)
    path = run_dir / TRIALS_FILE
    if path.exists():
        return load_json(path)
    try:
        import optuna

        optuna_cfg = (load_yaml(run_dir / "config_snapshot.yaml") or {}).get("optuna", {})
        name, storage = optuna_cfg.get("study_name"), optuna_cfg.get("storage")
        if not name or not storage:
            return None
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        info = study_to_dict(optuna.load_study(study_name=name, storage=storage))
    except Exception as exc:  # noqa: BLE001 - el informe sigue sin la sección Optuna
        logger.warning("No se pudo cargar el estudio Optuna: %s", exc)
        return None
    save_json(info, path)
    return info
