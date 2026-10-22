"""Consultas de solo lectura sobre proyectos, datos, runs y exports.

Capa sin dependencias de interfaz: la usa la UI (``ds_project.ui``) y puede usarse desde
scripts o menús de terminal.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ds_project.config import Settings, load_settings
from ds_project.data.source_manifest import find_source_manifests
from ds_project.dsp.module_export import resolve_template
from ds_project.utils.io import to_builtin

PathLike = Union[str, Path]

OK, WARN, ERROR = "ok", "warn", "error"
_RESERVED_DIRS = {"export", "templates"}
_ARTIFACTS = ("model.tflite", "model.keras", "dsp_params.json", "metrics.json")


@dataclass
class ProjectInfo:
    product: str
    turbine: str
    path: Path
    data_versions: List[str] = field(default_factory=list)
    export_versions: List[str] = field(default_factory=list)
    has_overlay: bool = False
    has_templates: bool = False

    @property
    def name(self) -> str:
        return f"{self.product}_{self.turbine}"


@dataclass
class RunInfo:
    name: str
    path: Path
    accuracy: Optional[float] = None
    f1_macro: Optional[float] = None
    has_evaluation: bool = False
    missing: List[str] = field(default_factory=list)
    search: Dict[str, Any] = field(default_factory=dict)  # search_config.json (vacío sin Optuna)


@dataclass
class Check:
    name: str
    status: str
    detail: str = ""


def _subdirs(path: Path) -> List[Path]:
    if not path.is_dir():
        return []
    return sorted(p for p in path.iterdir() if p.is_dir() and not p.name.startswith((".", "__")))


def list_projects(projects_dir: PathLike = "projects") -> List[ProjectInfo]:
    """Proyectos ``<projects_dir>/<producto>/<turbina>`` existentes en disco."""
    out: List[ProjectInfo] = []
    for product in _subdirs(Path(projects_dir)):
        for turbine in _subdirs(product):
            out.append(
                ProjectInfo(
                    product=product.name,
                    turbine=turbine.name,
                    path=turbine,
                    data_versions=[
                        p.name for p in _subdirs(turbine) if p.name not in _RESERVED_DIRS
                    ],
                    export_versions=[p.name for p in _subdirs(turbine / "export")],
                    has_overlay=(turbine / "project.yaml").is_file(),
                    has_templates=(turbine / "templates").is_dir(),
                )
            )
    return out


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def list_runs(models_dir: PathLike) -> List[RunInfo]:
    """Runs de ``models_dir`` (más reciente primero) con sus métricas de validación."""
    runs: List[RunInfo] = []
    for path in _subdirs(Path(models_dir)):
        keras = _read_json(path / "metrics.json").get("keras") or {}
        runs.append(
            RunInfo(
                name=path.name,
                path=path,
                accuracy=keras.get("accuracy"),
                f1_macro=keras.get("f1_macro"),
                has_evaluation=(path / "evaluation" / "report.md").is_file(),
                missing=[a for a in _ARTIFACTS if not (path / a).is_file()],
                search=_read_json(path / "search_config.json"),
            )
        )
    return sorted(runs, key=lambda r: r.name, reverse=True)


def count_by_class(data_dir: PathLike, extensions: Optional[List[str]] = None) -> Dict[str, int]:
    """Número de audios por carpeta de clase en ``data_dir/<clase>/``."""
    exts = {e.lower() for e in (extensions or [".wav"])}
    return {
        cls.name: sum(1 for f in cls.rglob("*") if f.is_file() and f.suffix.lower() in exts)
        for cls in _subdirs(Path(data_dir))
    }


def _check_split(label: str, counts: Dict[str, int], required: bool) -> Check:
    if not counts:
        status = ERROR if required else WARN
        return Check(f"Datos {label}", status, "sin carpetas de clase o directorio inexistente")
    empty = sorted(c for c, n in counts.items() if n == 0)
    detail = f"{sum(counts.values())} audios en {len(counts)} clases"
    if empty:
        return Check(f"Datos {label}", WARN, f"{detail}; clases vacías: {', '.join(empty)}")
    return Check(f"Datos {label}", OK, detail)


def check_project(
    product: str,
    turbine: str = "all",
    data_version: str = "",
    projects_dir: PathLike = "projects",
    templates_dir: PathLike = "templates",
    settings: Optional[Settings] = None,
) -> List[Check]:
    """Comprueba que el proyecto tiene lo necesario para entrenar, evaluar y exportar."""
    try:
        settings = settings or load_settings(
            product=product, turbine=turbine, data_version=data_version or None
        )
    except ValueError as exc:
        return [Check("Configuración", ERROR, str(exc))]

    checks = [Check("Configuración", OK, f"{settings.project_name or 'sin proyecto'}")]
    project = Path(projects_dir) / product / turbine
    if not project.is_dir():
        checks.append(Check("Proyecto", ERROR, f"no existe {project} (ejecuta invoke init)"))
        return checks
    checks.append(Check("Proyecto", OK, str(project)))
    overlay = project / "project.yaml"
    checks.append(
        Check(
            "project.yaml", OK if overlay.is_file() else WARN, "" if overlay.is_file() else "falta"
        )
    )

    scope = project / data_version if data_version else project
    data = scope / "data"
    if not scope.is_dir():
        checks.append(Check("Versión de datos", ERROR, f"no existe {scope}"))
        return checks
    exts = settings.data.extensions
    raw = count_by_class(data / "raw", exts)
    test = count_by_class(data / "test", exts)
    checks.append(_check_split("raw", raw, required=True))
    checks.append(_check_split("test", test, required=False))
    if raw and test:
        only_test, only_raw = sorted(set(test) - set(raw)), sorted(set(raw) - set(test))
        if only_test:
            checks.append(
                Check("Clases test", ERROR, f"desconocidas en raw: {', '.join(only_test)}")
            )
        if only_raw:
            checks.append(Check("Clases test", WARN, f"sin test: {', '.join(only_raw)}"))
        if not only_test and not only_raw:
            checks.append(Check("Clases test", OK, "mismas clases que raw"))

    manifests = find_source_manifests(data)
    checks.append(
        Check(
            "Manifests de origen",
            OK if manifests else WARN,
            f"{len(manifests)} ficheros" if manifests else "no hay (validación de origen omitida)",
        )
    )
    try:
        template = resolve_template(templates_dir, project, data_version)
        checks.append(Check("Plantilla DSP", OK, str(template)))
    except FileNotFoundError as exc:
        checks.append(Check("Plantilla DSP", ERROR, str(exc)))

    runs = list_runs(scope / "models")
    if runs:
        latest = runs[0]
        status = OK if not latest.missing else WARN
        detail = f"{len(runs)} runs; último {latest.name}"
        if latest.missing:
            detail += f" (faltan: {', '.join(latest.missing)})"
        checks.append(Check("Runs", status, detail))
    else:
        checks.append(Check("Runs", WARN, "sin entrenamientos (invoke train)"))
    exports = _subdirs(project / "export")
    checks.append(
        Check(
            "Exports",
            OK if exports else WARN,
            ", ".join(p.name for p in exports) or "sin exportaciones (invoke export)",
        )
    )
    return checks


def summarize(checks: List[Check]) -> str:
    """Peor estado de una lista de comprobaciones."""
    statuses = {c.status for c in checks}
    return ERROR if ERROR in statuses else WARN if WARN in statuses else OK


_BLOCKING = {
    "train": ("Configuración", "Proyecto", "Versión de datos", "Datos raw", "Clases test"),
    "evaluate": ("Configuración", "Proyecto", "Versión de datos", "Clases test"),
    "export": ("Configuración", "Proyecto", "Versión de datos", "Plantilla DSP"),
}
_BLOCKING["all"] = _BLOCKING["train"] + ("Plantilla DSP",)
_NEEDS_RUN = ("evaluate", "export")


def blockers(task: str, checks: List[Check], n_runs: int = 0) -> List[str]:
    """Motivos que impiden lanzar ``task`` con el estado actual (lista vacía = se puede)."""
    names = _BLOCKING.get(task, ())
    out = [f"{c.name}: {c.detail}" for c in checks if c.status == ERROR and c.name in names]
    if task in _NEEDS_RUN and n_runs == 0:
        out.append("No hay runs entrenados en esta versión de datos (ejecuta train)")
    return out


def next_version(versions: List[str]) -> str:
    """Siguiente ``v<n>`` libre a partir de las versiones exportadas (``v1`` si no hay)."""
    nums = [int(v[1:]) for v in versions if re.match(r"^v\d+$", v)]
    return f"v{max(nums) + 1}" if nums else "v1"


def effective_config(product: str, turbine: str = "all", data_version: str = "") -> Dict[str, Any]:
    """Configuración resultante (base + ``project.yaml`` + ámbito) como diccionario."""
    settings = load_settings(product=product, turbine=turbine, data_version=data_version or None)
    return to_builtin(settings)


def safe_join(base: PathLike, relative: str = "") -> Path:
    """``base/relative`` garantizando que no sale de ``base`` (evita ``..`` y rutas absolutas)."""
    root = Path(base).resolve()
    target = (root / relative).resolve()
    if target != root and root not in target.parents:
        raise ValueError(f"Ruta fuera del proyecto: {relative}")
    return target


def list_dir(base: PathLike, relative: str = "") -> List[Dict[str, Any]]:
    """Contenido de una carpeta (carpetas primero) para el explorador de la UI."""
    target = safe_join(base, relative)
    if not target.is_dir():
        return []
    rows = [
        {
            "name": p.name,
            "is_dir": p.is_dir(),
            "size": 0 if p.is_dir() else p.stat().st_size,
        }
        for p in target.iterdir()
    ]
    return sorted(rows, key=lambda r: (not r["is_dir"], r["name"].lower()))
