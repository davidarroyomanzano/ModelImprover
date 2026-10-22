"""Lógica del asistente guiado (sin dependencias de interfaz).

Pasos: proyecto -> datos -> configuración -> entrenar -> evaluar -> exportar. El estado de cada
paso se deduce de lo que hay en disco, así que el asistente se puede retomar en cualquier momento.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from ds_project.config import Settings, load_settings
from ds_project.layout import OVERLAY_HEADER
from ds_project.services import (
    ERROR,
    OK,
    WARN,
    Check,
    ProjectInfo,
    RunInfo,
    count_by_class,
    list_runs,
)
from ds_project.utils.io import to_builtin

PathLike = Union[str, Path]

DONE, PENDING, ATTENTION, LOCKED = "done", "pending", "attention", "locked"
STEPS = (
    ("project", "Proyecto"),
    ("data", "Datos"),
    ("config", "Configuración"),
    ("train", "Entrenar"),
    ("evaluate", "Evaluar"),
    ("export", "Exportar"),
    ("dashboard", "Dashboard"),
)
BASE_CONFIGS = ("configs/config.yaml", "configs/optuna.yaml", "configs/dsp.yaml")
HISTORY_DIR = ".history"
IMBALANCE_RATIO = 3.0
MIN_PER_CLASS = 10
# Perfiles DSP: el export con la plantilla PinBolt exige profile=pinbolt.
DSP_PROFILES = {"default": "configs/dsp.yaml", "pinbolt": "configs/dsp_pinbolt.yaml"}
EXPORT_PROFILE = "pinbolt"
# Claves que no se comparan en el diff: las fija el ámbito, no el overlay.
_SCOPE_KEYS = {"data.product", "data.turbine", "data.version", "paths.projects_dir"}


@dataclass
class Step:
    key: str
    title: str
    status: str
    detail: str = ""


# --- project.yaml -----------------------------------------------------------------------------


def overlay_path(product: str, turbine: str, projects_dir: PathLike = "projects") -> Path:
    return Path(projects_dir) / product / turbine / "project.yaml"


def read_overlay(product: str, turbine: str, projects_dir: PathLike = "projects") -> Dict[str, Any]:
    path = overlay_path(product, turbine, projects_dir)
    if not path.is_file():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} debe ser un diccionario de secciones")
    return data


def validate_overlay(product: str, turbine: str, overlay: Dict[str, Any]) -> Settings:
    """Configuración resultante con ``overlay``; ``ValueError`` con el motivo si no es válida."""
    try:
        return load_settings(product=product, turbine=turbine, overlay=overlay)
    except ValueError:
        raise
    except (TypeError, KeyError, AttributeError) as exc:
        raise ValueError(f"Configuración no válida: {exc}") from exc


def base_config(product: str, turbine: str) -> Dict[str, Any]:
    """Configuración de ``configs/*.yaml`` sin el overlay del proyecto."""
    return to_builtin(load_settings(product=product, turbine=turbine, overlay={}))


def get_value(config: Dict[str, Any], key: str) -> Any:
    node: Any = config
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def set_override(
    overlay: Dict[str, Any], base: Dict[str, Any], key: str, value: Any
) -> Dict[str, Any]:
    """Fija ``key`` (``seccion.campo``) en el overlay, o la quita si coincide con la base."""
    section, name = key.split(".", 1)
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in overlay.items()}
    current = dict(out.get(section) or {})
    if to_builtin(value) == get_value(base, key):
        current.pop(name, None)
    else:
        current[name] = to_builtin(value)
    if current:
        out[section] = current
    else:
        out.pop(section, None)
    return out


def normalize_overlay(overlay: Dict[str, Any], base: Dict[str, Any]) -> Dict[str, Any]:
    """Quita del overlay los campos que coinciden con la configuración base."""
    out: Dict[str, Any] = {}
    for section, values in overlay.items():
        if not isinstance(values, dict):
            out[section] = values
            continue
        kept = {
            k: v for k, v in values.items() if to_builtin(v) != get_value(base, f"{section}.{k}")
        }
        if kept:
            out[section] = kept
    return out


def _flatten(data: Any, prefix: str = "") -> Dict[str, Any]:
    if isinstance(data, dict) and data:
        out: Dict[str, Any] = {}
        for key, value in data.items():
            out.update(_flatten(value, f"{prefix}.{key}" if prefix else str(key)))
        return out
    return {prefix: data}


def overlay_diff(base: Dict[str, Any], effective: Dict[str, Any]) -> List[Dict[str, str]]:
    """Valores del proyecto que difieren de la configuración base."""
    flat_base, flat_eff = _flatten(base), _flatten(effective)
    rows = []
    for key in sorted(set(flat_base) | set(flat_eff)):
        if key in _SCOPE_KEYS or flat_base.get(key) == flat_eff.get(key):
            continue
        rows.append(
            {
                "clave": key,
                "base": json.dumps(flat_base.get(key), ensure_ascii=False),
                "proyecto": json.dumps(flat_eff.get(key), ensure_ascii=False),
            }
        )
    return rows


def save_overlay(
    product: str, turbine: str, overlay: Dict[str, Any], projects_dir: PathLike = "projects"
) -> Optional[Path]:
    """Valida y guarda ``project.yaml``; devuelve la copia de seguridad anterior (si la hay)."""
    overlay = {k: v for k, v in overlay.items() if v not in (None, {}, [])}
    validate_overlay(product, turbine, overlay)
    path = overlay_path(product, turbine, projects_dir)
    backup = None
    if path.is_file():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        backup = path.parent / HISTORY_DIR / f"project_{stamp}.yaml"
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, backup)
    text = OVERLAY_HEADER.format(project=f"{product}_{turbine}")
    if overlay:
        text += yaml.safe_dump(overlay, sort_keys=False, allow_unicode=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return backup


# --- datos ------------------------------------------------------------------------------------


def data_checks(raw: Dict[str, int], test: Dict[str, int]) -> List[Check]:
    """Avisos de calidad del conjunto: desequilibrio, clases pequeñas y proporción de test."""
    checks: List[Check] = []
    filled = {c: n for c, n in raw.items() if n}
    if len(filled) < 2:
        checks.append(Check("Clases", ERROR, "se necesitan al menos 2 clases con audios en raw"))
    else:
        ratio = max(filled.values()) / min(filled.values())
        status = WARN if ratio > IMBALANCE_RATIO else OK
        checks.append(Check("Equilibrio", status, f"ratio mayor/menor = {ratio:.1f}"))
    small = sorted(c for c, n in raw.items() if n < MIN_PER_CLASS)
    if small:
        checks.append(
            Check("Tamaño", WARN, f"menos de {MIN_PER_CLASS} audios en raw: {', '.join(small)}")
        )
    if not any(test.values()):
        checks.append(Check("Test", ERROR, "data/test vacío: no se podrá evaluar"))
    else:
        share = sum(test.values()) / max(sum(test.values()) + sum(raw.values()), 1)
        status = OK if 0.05 <= share <= 0.4 else WARN
        checks.append(Check("Test", status, f"{share:.0%} de los audios están en test"))
    return checks


# --- runs, evaluación y exports ---------------------------------------------------------------


def config_mtime(product: str, turbine: str, projects_dir: PathLike = "projects") -> float:
    """Última modificación de la configuración que afecta al entrenamiento."""
    files = [Path(p) for p in BASE_CONFIGS] + [overlay_path(product, turbine, projects_dir)]
    return max((f.stat().st_mtime for f in files if f.is_file()), default=0.0)


def run_time(run: RunInfo) -> float:
    metrics = run.path / "metrics.json"
    return (metrics if metrics.is_file() else run.path).stat().st_mtime


def test_metrics(run: RunInfo) -> Dict[str, Any]:
    path = run.path / "evaluation" / "metrics_test.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _sha1(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def run_dsp_profile(run_dir: PathLike) -> Optional[str]:
    """Perfil DSP con el que se entrenó un run (``None`` si no se puede leer)."""
    try:
        data = json.loads((Path(run_dir) / "dsp_params.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    dsp = data.get("dsp") if isinstance(data, dict) else None
    return str(dsp.get("profile", "default")) if isinstance(dsp, dict) else None


def with_dsp_profile(overlay: Dict[str, Any], profile: str) -> Dict[str, Any]:
    """Overlay con la sección ``dsp`` del perfil (``default`` la elimina)."""
    if profile not in DSP_PROFILES:
        raise ValueError(f"Perfil DSP desconocido: {profile!r}")
    out = {k: v for k, v in overlay.items() if k != "dsp"}
    if profile != "default":
        dsp = yaml.safe_load(Path(DSP_PROFILES[profile]).read_text(encoding="utf-8")) or {}
        dsp.pop("version", None)
        out["dsp"] = dsp
    return out


def with_template_dsp(overlay: Dict[str, Any], constants: Dict[str, Any]) -> Dict[str, Any]:
    """Overlay con el DSP fijado a las constantes de una plantilla (y sin buscar el DSP)."""
    from ds_project.dsp.templates import dsp_overrides

    out = with_dsp_profile(overlay, EXPORT_PROFILE)
    out["dsp"] = {**out["dsp"], **dsp_overrides(constants)}
    out["optuna"] = {**(out.get("optuna") or {}), "tune_dsp": False}
    return out


# Modos de entrenamiento -> valores de optuna.enabled / optuna.tune_dsp / optuna.strategy.
TRAIN_MODES = {
    "search_all": {"enabled": True, "tune_dsp": True, "strategy": "joint"},
    "two_phase": {"enabled": True, "tune_dsp": True, "strategy": "two_phase"},
    "search_net": {"enabled": True, "tune_dsp": False, "strategy": "joint"},
    "fixed": {"enabled": False},
}
TRAIN_MODE_LABELS = {
    "search_all": "Optuna: buscar DSP y red a la vez",
    "two_phase": "Optuna en dos fases: 1) DSP con una red simple, 2) red con ese DSP",
    "search_net": "Optuna: DSP fijo (dsp.mfe_blocks), buscar solo la red",
    "fixed": "Sin Optuna: DSP y red fijos (dsp.* y train.model_params)",
}


def train_mode(config: Dict[str, Any]) -> str:
    """Modo de entrenamiento de una configuración efectiva (``to_builtin(settings)``)."""
    if not get_value(config, "optuna.enabled"):
        return "fixed"
    if get_value(config, "optuna.tune_dsp") is False:
        return "search_net"
    return "two_phase" if get_value(config, "optuna.strategy") == "two_phase" else "search_all"


def exports_of_run(project_dir: PathLike, run: RunInfo) -> List[str]:
    """Versiones de ``export/`` cuyo ``nn_*.tflite`` es el ``model.tflite`` de ``run``."""
    model = run.path / "model.tflite"
    export = Path(project_dir) / "export"
    if not model.is_file() or not export.is_dir():
        return []
    digest = _sha1(model)
    return sorted(
        d.name
        for d in export.iterdir()
        if d.is_dir() and any(_sha1(f) == digest for f in d.glob("nn_*.tflite"))
    )


def dashboards_of_run(run_path: PathLike) -> List[Path]:
    """``dashboard_*.html`` del run, el más reciente primero."""
    found = Path(run_path).glob("dashboard_*.html")
    return sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)


# --- estado del asistente ---------------------------------------------------------------------


def _check(checks: List[Check], name: str) -> Optional[Check]:
    return next((c for c in checks if c.name == name), None)


def wizard_steps(
    project: Optional[ProjectInfo],
    data_version: str,
    checks: List[Check],
    projects_dir: PathLike = "projects",
    extensions: Optional[List[str]] = None,
) -> List[Step]:
    """Estado de cada paso; un paso queda bloqueado mientras el anterior no esté hecho."""
    titles = dict(STEPS)
    status: Dict[str, Step] = {}

    def put(key: str, state: str, detail: str = "") -> None:
        status[key] = Step(key, titles[key], state, detail)

    scope = project.path / data_version if project and data_version else None
    if scope is None or not scope.is_dir():
        put("project", PENDING, "elige o crea un proyecto y una versión de datos")
    else:
        put("project", DONE, f"{project.name} / {data_version}")

    if scope is not None and scope.is_dir():
        raw = count_by_class(scope / "data" / "raw", extensions)
        test = count_by_class(scope / "data" / "test", extensions)
        problems = [c for c in data_checks(raw, test) if c.status == ERROR]
        classes = _check(checks, "Clases test")
        if classes and classes.status == ERROR:
            problems.append(classes)
        if problems:
            put("data", PENDING, "; ".join(c.detail for c in problems))
        else:
            put("data", DONE, f"{sum(raw.values())} audios raw, {sum(test.values())} test")

    config = _check(checks, "Configuración")
    template = _check(checks, "Plantilla DSP")
    if config and config.status == ERROR:
        put("config", PENDING, config.detail)
    elif template and template.status == ERROR:
        put("config", PENDING, "falta la plantilla DSP del proyecto")
    else:
        put("config", DONE, "configuración válida")

    runs = list_runs(scope / "models") if scope is not None else []
    latest = runs[0] if runs else None
    if latest is None:
        put("train", PENDING, "sin entrenamientos")
    elif latest.missing:
        put("train", ATTENTION, f"al último run le faltan: {', '.join(latest.missing)}")
    elif project and config_mtime(project.product, project.turbine, projects_dir) > run_time(
        latest
    ):
        put("train", ATTENTION, "la configuración cambió después del último run")
    else:
        put("train", DONE, latest.name)

    metrics = test_metrics(latest) if latest else {}
    verdict = metrics.get("verdict") or {}
    if not metrics:
        put("evaluate", PENDING, "el último run no está evaluado")
    elif not verdict.get("passed", False):
        put(
            "evaluate", ATTENTION, "no supera los umbrales: " + "; ".join(verdict.get("failed", []))
        )
    else:
        put("evaluate", DONE, "apto" + (" (sin umbrales)" if verdict.get("informative") else ""))

    exported = exports_of_run(project.path, latest) if project and latest else []
    if exported:
        put("export", DONE, f"exportado como {', '.join(exported)}")
    elif latest and run_dsp_profile(latest.path) not in (None, EXPORT_PROFILE):
        put(
            "export",
            ATTENTION,
            f"el último run no usa el perfil DSP {EXPORT_PROFILE}: cámbialo en Configuración y "
            "reentrena",
        )
    else:
        put("export", PENDING, "el último run no está exportado")

    dashboards = dashboards_of_run(latest.path) if latest else []
    if dashboards:
        put("dashboard", DONE, dashboards[0].name)
    else:
        put("dashboard", PENDING, "el último run no tiene dashboard")

    steps: List[Step] = []
    blocked = False
    for key, title in STEPS:
        step = status.get(key) or Step(key, title, PENDING)
        if blocked:
            step = Step(key, title, LOCKED, "completa antes los pasos anteriores")
        elif step.status != DONE:
            blocked = step.status == PENDING
        steps.append(step)
    return steps


def current_step(steps: List[Step]) -> str:
    """Primer paso sin terminar (o el último si todo está hecho)."""
    return next((s.key for s in steps if s.status in (PENDING, ATTENTION)), steps[-1].key)


# --- progreso de Optuna -----------------------------------------------------------------------


def optuna_storage(settings: Settings) -> str:
    """URL de Optuna con la SQLite colgando del ámbito (como en el entrenamiento)."""
    url = settings.optuna.storage
    prefix = "sqlite:///"
    scope = settings.scope_dir()
    if scope is None or not url.startswith(prefix):
        return url
    return prefix + (scope / url[len(prefix) :]).as_posix()


def optuna_progress(
    product: str, turbine: str, data_version: str, settings: Optional[Settings] = None
) -> Optional[Dict[str, Any]]:
    """Resumen del estudio Optuna más reciente del ámbito (``None`` si no hay)."""
    settings = settings or load_settings(
        product=product, turbine=turbine, data_version=data_version or None
    )
    url = optuna_storage(settings)
    if url.startswith("sqlite:///") and not Path(url[len("sqlite:///") :]).is_file():
        return None
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    prefix = f"{settings.optuna.study_name}__{settings.project_name}__{settings.data.version}__"
    try:
        summaries = [
            s
            for s in optuna.get_all_study_summaries(storage=url)
            if s.study_name.startswith(prefix)
        ]
    except Exception:  # noqa: BLE001 - base de datos bloqueada o de otra versión
        return None
    if not summaries:
        return None
    summary = max(summaries, key=lambda s: s.datetime_start or datetime.min)
    study = optuna.load_study(study_name=summary.study_name, storage=url)
    trials = study.get_trials(deepcopy=False)
    states: Dict[str, int] = {}
    for t in trials:
        states[t.state.name] = states.get(t.state.name, 0) + 1
    complete = [t for t in trials if t.state.name == "COMPLETE" and t.value is not None]
    best = None
    if complete:
        pick = max if study.direction.name == "MAXIMIZE" else min
        best = pick(complete, key=lambda t: t.value)
    phase = summary.study_name.rsplit("__", 1)[-1]
    phase = phase if phase in ("dsp", "net") else ""
    target = settings.optuna.n_trials
    if phase:
        target = settings.optuna.phase_trials()[0 if phase == "dsp" else 1]
    return {
        "study_name": summary.study_name,
        "phase": phase,
        "direction": study.direction.name.lower(),
        "metric": settings.optuna.metric,
        "target": target,
        "states": states,
        "finished": sum(n for s, n in states.items() if s != "RUNNING"),
        "best_value": best.value if best else None,
        "best_trial": best.number if best else None,
        "best_params": dict(best.params) if best else {},
        "values": [{"trial": t.number, "valor": t.value} for t in complete],
    }
