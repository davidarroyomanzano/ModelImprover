"""Plantillas DSP (``dsp_template.py``) por proyecto o versión de datos.

- Niveles: ``<proyecto>/templates/<datos>/`` > ``<proyecto>/templates/`` > ``templates/`` común
  (el mismo orden que usa el export).
- Comprobación: la plantilla se importa en un subproceso, se rellena como en el export y se
  compara numéricamente con el DSP de entrenamiento (perfil ``pinbolt``).
- Instalación: valida antes de sustituir y guarda la anterior en ``templates/.history/``.

CLI: ``python -m ds_project.dsp.templates {check,install,remove} ...``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

PathLike = Union[str, Path]

TEMPLATE_NAME = "dsp_template.py"
HISTORY_DIR = ".history"
LEVEL_VERSION, LEVEL_PROJECT, LEVEL_COMMON = "versión", "proyecto", "común"
INSPECT_TIMEOUT = 180
_MARKER = "@@DSP_TEMPLATE_REPORT@@"
_REQUIRED = (
    "SAMPLE_RATE",
    "WINDOW_SIZE_SEC",
    "WINDOW_STEP_SEC",
    "ZERO_PADDING",
    "PREEMPHASIS_COEFF",
    "CLASS_NAMES",
    "MFE_BLOCKS",
    "N_FILTERS_FINAL",
    "AUX_FEATURE_DIM",
    "AUX_MEAN",
    "AUX_SCALE",
    "extract_model_inputs",
)
_PARITY_CLIPS = 3
_MFE_ATOL = 1e-6
# Fracción de valores MFE que pueden diferir (redondeos en la cuantización a 1/256).
_MFE_MAX_MISMATCH = 0.001
_AUX_TOL = 1e-3


class TemplateError(ValueError):
    """La plantilla no se puede instalar."""


@dataclass
class TemplateSlot:
    level: str
    path: Path
    exists: bool = False
    active: bool = False


@dataclass
class TemplateReport:
    ok: bool = False  # el export puede rellenarla y el módulo generado se importa
    parity: Optional[bool] = None  # reproduce el DSP de entrenamiento (None: sin comprobar)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    constants: Dict[str, Any] = field(default_factory=dict)
    parity_detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TemplateReport:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class InstallResult:
    path: Path
    backup: Optional[Path]
    report: TemplateReport


# --- niveles ----------------------------------------------------------------------------------


def target_path(project_dir: PathLike, data_version: str = "") -> Path:
    base = Path(project_dir) / "templates"
    return (base / data_version if data_version else base) / TEMPLATE_NAME


def template_slots(
    project_dir: PathLike, data_version: str = "", templates_dir: PathLike = "templates"
) -> List[TemplateSlot]:
    """Niveles en orden de prioridad; ``active`` marca el que usará el export."""
    slots = []
    if data_version:
        slots.append(TemplateSlot(LEVEL_VERSION, target_path(project_dir, data_version)))
    slots.append(TemplateSlot(LEVEL_PROJECT, target_path(project_dir)))
    slots.append(TemplateSlot(LEVEL_COMMON, Path(templates_dir) / TEMPLATE_NAME))
    found = False
    for slot in slots:
        slot.exists = slot.path.is_file()
        if slot.exists and not found:
            slot.active = found = True
    return slots


def active_template(
    project_dir: PathLike, data_version: str = "", templates_dir: PathLike = "templates"
) -> Optional[Path]:
    slot = next(
        (s for s in template_slots(project_dir, data_version, templates_dir) if s.active), None
    )
    return slot.path if slot else None


def read_window_step(path: PathLike) -> Optional[float]:
    """``WINDOW_STEP_SEC`` de la plantilla, leído sin ejecutarla."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return None
    match = re.search(r"^WINDOW_STEP_SEC\s*=\s*([0-9_.eE+-]+)\s*(#.*)?$", text, re.MULTILINE)
    try:
        return float(match.group(1).replace("_", "")) if match else None
    except ValueError:
        return None


# --- comprobación (se ejecuta en un subproceso) -----------------------------------------------


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"No se puede cargar {path}")
    module = importlib.util.module_from_spec(spec)
    # Sin __pycache__ junto a la plantilla (ni del fichero temporal .upload-*).
    previous, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def _clean(value: Any) -> Any:
    value = float(value)
    return int(value) if value.is_integer() else value


def _constants(module) -> Dict[str, Any]:
    from ds_project.dsp.module_export import _MFE_KEYS

    nyquist = float(module.SAMPLE_RATE) / 2.0
    # Valores por defecto de la plantilla cuando un bloque omite la clave.
    defaults = {"LOW_FREQ": 300.0, "HIGH_FREQ": nyquist, "NOISE_FLOOR_DB": -32.0}
    blocks = []
    for block in module.MFE_BLOCKS:
        out: Dict[str, Any] = {}
        for tpl_key, key in _MFE_KEYS:
            value = block.get(tpl_key)
            value = defaults.get(tpl_key) if value is None else value
            out[key] = int(value) if key in ("n_filters", "fft_length") else _clean(value)
        blocks.append(out)
    return {
        "dsp_version": str(getattr(module, "DSP_VERSION", "")),
        "dsp_type": str(getattr(module, "DSP_TYPE", "")),
        "sample_rate": int(module.SAMPLE_RATE),
        "window_size_sec": float(module.WINDOW_SIZE_SEC),
        "window_step_sec": float(module.WINDOW_STEP_SEC),
        "zero_padding": bool(module.ZERO_PADDING),
        "pre_emphasis": float(module.PREEMPHASIS_COEFF),
        "n_filters_final": int(module.N_FILTERS_FINAL or 0),
        "aux_feature_dim": int(module.AUX_FEATURE_DIM),
        "class_names": [str(c) for c in module.CLASS_NAMES],
        "mfe_blocks": blocks,
    }


def dsp_overrides(constants: Dict[str, Any]) -> Dict[str, Any]:
    """Sección ``dsp`` de project.yaml que reproduce la plantilla en el entrenamiento."""
    return {
        "profile": "pinbolt",
        "features": ["mfe"],
        "aux_features": [],
        "sample_rate": int(constants["sample_rate"]),
        "clip_duration": float(constants["window_size_sec"]),
        "pre_emphasis": float(constants["pre_emphasis"]),
        "zero_padding": bool(constants["zero_padding"]),
        "n_filters_final": int(constants["n_filters_final"]),
        "mfe_blocks": [dict(b) for b in constants["mfe_blocks"]],
    }


def _parity(rendered, params, module, report: TemplateReport) -> None:
    import numpy as np

    from ds_project.features.audio import extract_auxiliary_features, extract_features

    if int(module.AUX_FEATURE_DIM) != params.aux_feature_dim:
        report.parity = False
        report.parity_detail = (
            f"la plantilla extrae {module.AUX_FEATURE_DIM} features auxiliares y el "
            f"entrenamiento {params.aux_feature_dim}"
        )
        return
    rng = np.random.default_rng(0)
    t = np.arange(params.n_samples) / params.sample_rate
    clips = [rng.normal(0, 0.1, params.n_samples).astype(np.float32) for _ in range(2)]
    clips.append(
        (0.3 * np.sin(2 * np.pi * 440 * t) + rng.normal(0, 0.02, t.size)).astype(np.float32)
    )
    mean = np.asarray(module.AUX_MEAN, dtype=np.float32)
    scale = np.asarray(module.AUX_SCALE, dtype=np.float32)
    mismatch, mfe_diff, aux_diff, aux_ok = 0.0, 0.0, 0.0, True
    for clip in clips[:_PARITY_CLIPS]:
        mfe, aux = rendered.extract_model_inputs(clip, params.sample_rate)
        expected_mfe = extract_features(clip, params)
        if mfe.shape != expected_mfe.shape:
            report.parity = False
            report.parity_detail = (
                f"vector MFE de {mfe.shape[0]} valores; el entrenamiento genera "
                f"{expected_mfe.shape[0]}"
            )
            return
        diff = np.abs(mfe - expected_mfe)
        mismatch = max(mismatch, float(np.mean(diff > _MFE_ATOL)))
        mfe_diff = max(mfe_diff, float(diff.max()))
        expected_aux = (extract_auxiliary_features(clip, params) - mean) / scale
        aux_diff = max(aux_diff, float(np.max(np.abs(aux - expected_aux))))
        aux_ok = aux_ok and bool(np.allclose(aux, expected_aux, rtol=_AUX_TOL, atol=_AUX_TOL))
    report.parity = mismatch <= _MFE_MAX_MISMATCH and aux_ok
    report.parity_detail = (
        f"MFE: {mismatch:.2%} de valores distintos (máx. {mfe_diff:.2e}); "
        f"auxiliares: diferencia máx. {aux_diff:.2e}"
    )


def inspect_in_process(path: PathLike) -> TemplateReport:
    """Comprobación completa en el proceso actual (ejecuta el código de la plantilla)."""
    from ds_project.dsp.export import build_dsp_artifact
    from ds_project.dsp.module_export import render_dsp_module
    from ds_project.dsp.params import DSPParams

    path = Path(path)
    report = TemplateReport()
    try:
        module = _load_module(path, f"_tpl_{uuid.uuid4().hex}")
    except Exception as exc:  # noqa: BLE001 - se informa del fallo de la plantilla
        report.errors.append(f"No se puede importar la plantilla: {type(exc).__name__}: {exc}")
        return report
    missing = [name for name in _REQUIRED if not hasattr(module, name)]
    if missing:
        report.errors.append(f"Faltan en la plantilla: {', '.join(missing)}")
        return report
    try:
        report.constants = _constants(module)
        params = DSPParams.from_dict(dsp_overrides(report.constants))
    except (TypeError, ValueError, KeyError, AttributeError) as exc:
        report.errors.append(f"Parámetros no válidos para el entrenamiento: {exc}")
        return report

    artifact = build_dsp_artifact(
        params,
        report.constants["class_names"],
        aux_scaler_dict={
            "kind": "standard",
            "offset": [float(v) for v in module.AUX_MEAN],
            "divisor": [float(v) for v in module.AUX_SCALE],
        },
    )
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "dsp_check.py"
        try:
            out.write_text(
                render_dsp_module(
                    path.read_text(encoding="utf-8"),
                    artifact,
                    report.constants["window_step_sec"],
                    allow_aux_mismatch=True,
                ),
                encoding="utf-8",
            )
            rendered = _load_module(out, f"_tpl_rendered_{uuid.uuid4().hex}")
        except Exception as exc:  # noqa: BLE001
            report.errors.append(f"El export no puede rellenar la plantilla: {exc}")
            return report
        report.ok = True
        try:
            _parity(rendered, params, module, report)
        except Exception as exc:  # noqa: BLE001
            report.parity = False
            report.parity_detail = f"error al extraer features: {type(exc).__name__}: {exc}"
    if report.parity is False:
        report.warnings.append(
            "No reproduce el DSP de entrenamiento: un modelo entrenado aquí no daría las mismas "
            "features con este dsp_*.py"
        )
    if report.constants["n_filters_final"] == 0:
        report.warnings.append(
            "N_FILTERS_FINAL = None: al exportar se fija al nº de filtros del primer bloque"
        )
    return report


def inspect_template(path: PathLike, timeout: int = INSPECT_TIMEOUT) -> TemplateReport:
    """Ejecuta :func:`inspect_in_process` en un subproceso aislado (con límite de tiempo)."""
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    cmd = [
        sys.executable,
        "-B",
        "-m",
        "ds_project.dsp.templates",
        "inspect",
        str(Path(path).resolve()),
    ]
    try:
        proc = subprocess.run(  # noqa: S603 - intérprete y módulo propios
            cmd, capture_output=True, text=True, encoding="utf-8", timeout=timeout, env=env
        )
    except subprocess.TimeoutExpired:
        return TemplateReport(errors=[f"La comprobación superó {timeout} s"])
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith(_MARKER):
            return TemplateReport.from_dict(json.loads(line[len(_MARKER) :]))
    tail = (proc.stderr or proc.stdout).strip().splitlines()[-5:]
    return TemplateReport(errors=["La comprobación falló: " + " | ".join(tail)])


def inspect_source(
    source: Union[PathLike, bytes], timeout: int = INSPECT_TIMEOUT
) -> TemplateReport:
    """Como :func:`inspect_template`, pero acepta también el contenido (p. ej. una subida)."""
    if not isinstance(source, bytes):
        return inspect_template(source, timeout)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / TEMPLATE_NAME
        path.write_bytes(source)
        return inspect_template(path, timeout)


# --- instalación ------------------------------------------------------------------------------


def _backup(path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    backup = path.parent / HISTORY_DIR / f"dsp_template_{stamp}.py"
    backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, backup)
    return backup


def install_template(
    source: Union[PathLike, bytes],
    project_dir: PathLike,
    data_version: str = "",
    force: bool = False,
) -> InstallResult:
    """Valida ``source`` y la instala como plantilla del proyecto (o de una versión de datos).

    Sin ``force`` se rechaza si no reproduce el DSP de entrenamiento.
    """
    target = target_path(project_dir, data_version)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = source if isinstance(source, bytes) else Path(source).read_bytes()
    stage = target.parent / f".upload-{uuid.uuid4().hex[:8]}.py"
    stage.write_bytes(data)
    try:
        report = inspect_template(stage)
        if not report.ok:
            raise TemplateError("; ".join(report.errors) or "plantilla no válida")
        if report.parity is False and not force:
            raise TemplateError(
                f"No reproduce el DSP de entrenamiento ({report.parity_detail}). "
                "Instálala con force si aun así la quieres usar."
            )
        backup = _backup(target) if target.is_file() else None
        stage.replace(target)
    finally:
        stage.unlink(missing_ok=True)
    return InstallResult(target, backup, report)


def remove_template(project_dir: PathLike, data_version: str = "") -> Path:
    """Retira la plantilla del nivel indicado (queda en ``.history/``); devuelve la copia."""
    target = target_path(project_dir, data_version)
    if not target.is_file():
        raise FileNotFoundError(f"No hay plantilla en {target}")
    backup = _backup(target)
    target.unlink()
    return backup


def format_report(report: TemplateReport) -> str:
    lines = [f"Export: {'OK' if report.ok else 'ERROR'}"]
    if report.parity is not None:
        lines.append(f"Paridad con el entrenamiento: {'OK' if report.parity else 'NO'}")
    if report.parity_detail:
        lines.append(f"  {report.parity_detail}")
    c = report.constants
    if c:
        lines.append(
            f"DSP_VERSION {c['dsp_version']} | {c['sample_rate']} Hz | ventana "
            f"{c['window_size_sec']} s, salto {c['window_step_sec']} s | pre-énfasis "
            f"{c['pre_emphasis']} | {len(c['mfe_blocks'])} bloques MFE | "
            f"{c['aux_feature_dim']} auxiliares"
        )
    lines += [f"ERROR: {e}" for e in report.errors]
    lines += [f"AVISO: {w}" for w in report.warnings]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ds_project.dsp.templates")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_inspect = sub.add_parser("inspect", help=argparse.SUPPRESS)
    p_inspect.add_argument("path")
    p_check = sub.add_parser("check", help="Comprueba una plantilla sin instalarla")
    p_check.add_argument("path")
    for name in ("install", "remove"):
        p = sub.add_parser(name)
        if name == "install":
            p.add_argument("path")
            p.add_argument("--force", action="store_true")
            p.add_argument("--apply-config", action="store_true")
        p.add_argument("--product", default="")
        p.add_argument("--turbine", default="")
        p.add_argument("--data-version", default="")
    args = parser.parse_args(argv)

    if args.cmd == "inspect":
        print(_MARKER + json.dumps(inspect_in_process(args.path).to_dict()))
        return 0
    if args.cmd == "check":
        report = inspect_template(args.path)
        print(format_report(report))
        return 0 if report.ok else 1

    from ds_project.config import load_settings

    settings = load_settings(product=args.product or None, turbine=args.turbine or None)
    project_dir = settings.project_dir()
    if project_dir is None:
        print("ERROR: falta el producto (--product / DS_PRODUCT)", file=sys.stderr)
        return 1
    product, turbine = settings.data.product, settings.data.turbine
    try:
        if args.cmd == "remove":
            backup = remove_template(project_dir, args.data_version)
            print(f"Plantilla retirada (copia en {backup})")
            return 0
        result = install_template(args.path, project_dir, args.data_version, args.force)
    except (TemplateError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(format_report(result.report))
    print(
        f"Instalada en {result.path}" + (f" (anterior en {result.backup})" if result.backup else "")
    )
    if args.apply_config:
        from ds_project import wizard

        projects_dir = settings.paths.projects_dir
        overlay = wizard.read_overlay(product, turbine, projects_dir)
        new = wizard.with_template_dsp(overlay, result.report.constants)
        base = wizard.base_config(product, turbine)
        wizard.save_overlay(product, turbine, wizard.normalize_overlay(new, base), projects_dir)
        print("project.yaml: DSP de la plantilla aplicado (optuna.tune_dsp = false)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
