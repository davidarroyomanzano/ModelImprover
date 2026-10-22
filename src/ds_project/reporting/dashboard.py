"""Cuadro de mandos HTML autocontenido de un run (rendimiento, parámetros y factibilidad).

El fichero resultante lleva datos, estilos, código y Plotly incrustados: se puede enviar tal cual
y abrir offline en cualquier navegador. Los costes (tamaño, cómputo) son estimaciones analíticas
a partir de los hiperparámetros de cada trial; ``n_params`` real se usa si el run lo guardó.
"""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ds_project.dsp.params import PINBOLT_AUX_DIM, DSPParams
from ds_project.models.architectures import DEFAULT_MODEL_PARAMS
from ds_project.models.tuning import PHASE1_MODEL_PARAMS
from ds_project.reporting.optuna_info import load_study_info
from ds_project.utils.io import load_json, load_yaml

PathLike = Union[str, Path]
TEMPLATE = Path(__file__).with_name("dashboard_template.html")
BYTES_PER_WEIGHT = {"float32": 4.0, "float16": 2.0, "int8": 1.0, "dynamic": 1.0}
# Límites de factibilidad por defecto; el usuario los ajusta con sliders en el propio HTML.
DEFAULT_LIMITS: Dict[str, float] = {
    "max_window_s": 1.0,
    "max_size_kb": 1024.0,
    "max_mops": 150.0,
    "max_false_alarm": 0.05,
}


def estimate_network(
    dsp: DSPParams, model_params: Dict[str, Any], n_classes: int
) -> Optional[Dict[str, float]]:
    """Parámetros y MACs de la red de ``build_model`` para el perfil pinbolt (None si no aplica)."""
    reshape = dsp.mfe_reshape
    if reshape is None:
        return None
    p = {**DEFAULT_MODEL_PARAMS, **model_params}
    length, channels = reshape
    params = 0
    macs = 0
    k = int(p["kernel_size"])
    for i in range(int(p["n_conv_layers"])):
        cout = int(p["filters"]) * 2**i
        params += k * channels * cout + cout + 4 * cout  # conv + BatchNorm (4 vectores)
        macs += length * k * channels * cout
        channels = cout
        if length >= 2:
            length //= 2
    flat = length * channels
    units = int(p["units"])
    n_dense = int(p["n_dense_layers"])
    aux_out = PINBOLT_AUX_DIM
    for _ in range(n_dense):  # rama de las features auxiliares
        params += aux_out * units + units
        macs += aux_out * units
        aux_out = units
    dim = flat + aux_out
    for _ in range(n_dense):
        params += dim * units + units
        macs += dim * units
        dim = units
    params += dim * n_classes + n_classes
    macs += dim * n_classes
    return {"n_params": float(params), "macs": float(macs)}


def estimate_dsp_ops(dsp: DSPParams) -> float:
    """Operaciones aproximadas del DSP por ventana (FFT + banco de filtros por frame)."""
    total = 0.0
    for b in dsp.mfe_blocks:
        frames = b.pinbolt_n_frames(dsp.n_samples, dsp.sample_rate, dsp.zero_padding)
        n = b.fft_length
        total += frames * (5.0 * n * math.log2(n) + b.n_filters * (n / 2.0) + 3.0 * n)
    return total


def _trial_dsp(base: DSPParams, params: Dict[str, Any]) -> DSPParams:
    from dataclasses import replace

    changes: Dict[str, Any] = {}
    if "window_size" in params:
        changes["clip_duration"] = float(params["window_size"])
    blocks = []
    for i, block in enumerate(base.mfe_blocks, start=1):
        over = {
            f: params[f"{f}_{i}"]
            for f in ("frame_length", "frame_stride", "n_filters", "fft_length")
            + ("low_freq", "high_freq", "noise_floor_db")
            if f"{f}_{i}" in params
        }
        blocks.append(replace(block, **over))
    changes["mfe_blocks"] = tuple(blocks)
    return replace(base, **changes)


def derive_trial(
    trial: Dict[str, Any],
    base_dsp: DSPParams,
    net_defaults: Dict[str, Any],
    n_classes: int,
    bytes_per_weight: float,
) -> Dict[str, Any]:
    """Métricas derivadas (ventana, tamaño, cómputo) de un trial; vacías si el DSP es inválido."""
    params = trial.get("params", {})
    out: Dict[str, Any] = {}
    try:
        dsp = _trial_dsp(base_dsp, params)
    except ValueError:
        return out
    step = float(params.get("window_step", dsp.clip_duration))
    step = max(min(step, dsp.clip_duration), 1e-3)
    out["window_s"] = dsp.clip_duration
    out["step_s"] = step
    out["dsp_mops"] = estimate_dsp_ops(dsp) / step / 1e6
    net_params = {k: v for k, v in params.items() if k in DEFAULT_MODEL_PARAMS}
    est = estimate_network(dsp, {**net_defaults, **net_params}, n_classes)
    attrs = trial.get("user_attrs") or {}
    if est is not None:
        n_params = float(attrs.get("n_params", est["n_params"]))
        out["n_params"] = n_params
        out["size_kb"] = n_params * bytes_per_weight / 1024.0 + 8.0
        out["net_mops"] = est["macs"] * 2 / step / 1e6
        out["mops"] = out["dsp_mops"] + out["net_mops"]
    return out


def _read_predictions(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        pcols = [i for i, h in enumerate(header) if h.startswith("p_")]
        labels = [header[i][2:] for i in pcols]
        index = {label: i for i, label in enumerate(labels)}
        files: Dict[str, int] = {}
        seg_file: List[int] = []
        seg_true: List[int] = []
        probs: List[List[float]] = []
        for row in reader:
            seg_file.append(files.setdefault(row[0], len(files)))
            seg_true.append(index.get(row[2], -1))
            probs.append([round(float(row[i]), 3) for i in pcols])
    return {
        "labels": labels,
        "files": list(files),
        "seg_file": seg_file,
        "seg_true": seg_true,
        "probs": probs,
    }


def _trial_rows(info: Dict[str, Any], phase: str, derive) -> List[Dict[str, Any]]:
    rows = []
    for t in info.get("trials", []):
        rows.append(
            {
                "n": t["number"],
                "phase": phase,
                "state": t["state"],
                "value": t.get("value"),
                "dur": t.get("duration_s"),
                "params": t.get("params", {}),
                "attrs": t.get("user_attrs") or {},
                "d": derive(t, phase),
            }
        )
    return rows


def build_payload(run_dir: PathLike, limits: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    snapshot = load_yaml(run_dir / "config_snapshot.yaml") or {}
    dsp_json = load_json(run_dir / "dsp_params.json")
    base_dsp = DSPParams.from_dict(dsp_json["dsp"])
    metrics = load_json(run_dir / "metrics.json") if (run_dir / "metrics.json").exists() else {}
    labels = metrics.get("labels") or dsp_json.get("labels") or []
    quant = (snapshot.get("export") or {}).get("quantization", "float16")
    bytes_pw = BYTES_PER_WEIGHT.get(quant, 4.0)
    train_params = (snapshot.get("train") or {}).get("model_params") or {}
    optuna_cfg = snapshot.get("optuna") or {}
    phase1_params = {**PHASE1_MODEL_PARAMS, **(optuna_cfg.get("phase1_model_params") or {})}
    n_classes = max(len(labels), 2)

    def derive(trial: Dict[str, Any], phase: str) -> Dict[str, Any]:
        net = phase1_params if phase == "dsp" else train_params
        return derive_trial(trial, base_dsp, net, n_classes, bytes_pw)

    info = load_study_info(run_dir) or {}
    dsp_phase = info.get("dsp_phase")
    trials: List[Dict[str, Any]] = []
    if dsp_phase:
        trials += _trial_rows(dsp_phase, "dsp", derive)
        trials += _trial_rows(info, "net", derive)
    elif info:
        trials += _trial_rows(info, "joint", derive)
    best_phase = "net" if dsp_phase else "joint"
    summary = (
        load_json(run_dir / "optuna_summary.json")
        if (run_dir / "optuna_summary.json").exists()
        else {}
    )
    search = (
        load_json(run_dir / "search_config.json")
        if (run_dir / "search_config.json").exists()
        else {}
    )
    final_dsp = derive_trial({"params": {}}, base_dsp, train_params, n_classes, bytes_pw)
    tflite = run_dir / "model.tflite"
    final = {
        "window_s": base_dsp.clip_duration,
        "tflite_kb": tflite.stat().st_size / 1024.0 if tflite.exists() else None,
        "model_params": train_params,
        "dsp_mops": final_dsp.get("dsp_mops"),
        "mfe_blocks": [vars(b) for b in base_dsp.mfe_blocks],
    }
    evaluation = None
    ev_dir = run_dir / "evaluation"
    if (ev_dir / "metrics_test.json").exists():
        evaluation = {
            "summary": load_json(ev_dir / "metrics_test.json"),
            "predictions": _read_predictions(ev_dir / "predictions.csv"),
        }
    manifest = run_dir / "dataset_manifest.json"
    per_class = load_json(manifest).get("per_class", {}) if manifest.exists() else {}
    history = load_json(run_dir / "history.json") if (run_dir / "history.json").exists() else {}
    parts = run_dir.resolve().parts
    return {
        "meta": {
            "run": run_dir.name,
            "project": snapshot.get("data", {}).get("product", ""),
            "turbine": snapshot.get("data", {}).get("turbine", ""),
            "data_version": snapshot.get("data", {}).get("version", ""),
            "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "search_mode": summary.get("search_mode") or search.get("mode") or "",
            "search": search,
            "quantization": quant,
            "metric": optuna_cfg.get("metric", "val_accuracy"),
            "path": "/".join(parts[-5:]),
        },
        "limits": {**DEFAULT_LIMITS, **(limits or {})},
        "labels": labels,
        "trials": trials,
        "best_phase": best_phase,
        "best_trial": info.get("best_trial"),
        "importances": info.get("importances") or {},
        "dsp_importances": (dsp_phase or {}).get("importances") or {},
        "final": final,
        "internal": {"keras": metrics.get("keras"), "tflite": metrics.get("tflite")},
        "evaluation": evaluation,
        "train_per_class": per_class,
        "history": history,
    }


def _plotly_js() -> str:
    try:
        from plotly.offline import get_plotlyjs
    except ImportError as exc:
        raise RuntimeError(
            "El dashboard necesita plotly: instala el extra con 'poetry install -E dashboard'"
        ) from exc
    return get_plotlyjs()


def render_html(payload: Dict[str, Any]) -> str:
    data = json.dumps(payload, ensure_ascii=False, allow_nan=False, default=_json_default)
    data = data.replace("</", "<\\/").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    html = TEMPLATE.read_text(encoding="utf-8")
    title = f"{payload['meta']['project']} {payload['meta']['turbine']} · {payload['meta']['run']}"
    html = html.replace("__TITLE__", title)
    # El JS de Plotly se inserta el último: puede contener cualquier marcador de texto.
    html = html.replace("__DATA_JSON__", data)
    return html.replace("__PLOTLY_JS__", _plotly_js())


def _json_default(obj: Any) -> Any:
    if isinstance(obj, float):
        return None
    raise TypeError(f"No serializable: {type(obj)}")


def _clean(obj: Any) -> Any:
    """Sustituye NaN/inf por None para que el JSON sea válido en el navegador."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    return obj


def write_dashboard(
    run_dir: PathLike,
    output: Optional[PathLike] = None,
    limits: Optional[Dict[str, float]] = None,
) -> Path:
    """Genera ``dashboard_<run>.html`` (por defecto dentro del run)."""
    run_dir = Path(run_dir)
    payload = _clean(build_payload(run_dir, limits))
    path = Path(output) if output else run_dir / f"dashboard_{run_dir.name}.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html(payload), encoding="utf-8")
    return path


def dashboard_for_project(
    product: str = "",
    turbine: str = "",
    data_version: str = "",
    run_dir: str = "",
    output: str = "",
    limits: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Resuelve el run (por defecto, el último del ámbito) y escribe su dashboard."""
    from ds_project.config import load_settings
    from ds_project.dsp.module_export import find_latest_run

    if run_dir:
        run = Path(run_dir)
    else:
        settings = load_settings(
            product=product or None, turbine=turbine or None, data_version=data_version or None
        )
        run = find_latest_run(settings.resolved_paths().models_dir, settings.data.version or None)
    path = write_dashboard(run, output or None, limits)
    size_mb = round(path.stat().st_size / 2**20, 2)
    return {"run_dir": str(run), "dashboard": str(path), "size_mb": size_mb}
