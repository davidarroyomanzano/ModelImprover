"""Genera el paquete DSP + TFLite de PinBolt a partir de los artefactos de un entrenamiento.

Rellena las constantes de ``templates/dsp_template.py`` (sin modificar la plantilla) con los
valores de ``dsp_params.json`` y copia ``model.tflite`` con el nombre versionado.
"""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

PathLike = Union[str, Path]

_MODEL_VERSION_RE = re.compile(r"^v\d+$")
# Claves de bloque de la plantilla -> campo de MFEBlock.
DEFAULT_WINDOW_STEP = 0.3
_MFE_KEYS = (
    ("FRAME_LENGTH", "frame_length"),
    ("FRAME_STRIDE", "frame_stride"),
    ("N_FILTERS", "n_filters"),
    ("FFT_LENGTH", "fft_length"),
    ("LOW_FREQ", "low_freq"),
    ("HIGH_FREQ", "high_freq"),
    ("NOISE_FLOOR_DB", "noise_floor_db"),
)


def find_latest_run(models_dir: PathLike, data_version: Optional[str] = None) -> Path:
    """Devuelve el run más reciente (con dsp_params.json) dentro de ``models_dir``.

    Con ``data_version`` solo se consideran los runs entrenados con esa versión de datos.
    """
    runs = [p for p in Path(models_dir).glob("*") if (p / "dsp_params.json").exists()]
    if data_version:
        runs = [
            p
            for p in runs
            if json.loads((p / "dsp_params.json").read_text(encoding="utf-8"))
            .get("extra", {})
            .get("data_version")
            == data_version
        ]
    if not runs:
        suffix = f" con data_version={data_version}" if data_version else ""
        raise FileNotFoundError(f"No hay ejecuciones con dsp_params.json en {models_dir}{suffix}")
    return max(runs, key=lambda p: (p / "dsp_params.json").stat().st_mtime)


def resolve_template(
    templates_dir: PathLike, project_dir: Optional[PathLike] = None, data_version: str = ""
) -> Path:
    """Plantilla: ``<proyecto>/templates/<datos>/``, ``<proyecto>/templates/`` o la común."""
    candidates: List[Path] = []
    if project_dir:
        base = Path(project_dir) / "templates"
        if data_version:
            candidates.append(base / data_version / "dsp_template.py")
        candidates.append(base / "dsp_template.py")
    candidates.append(Path(templates_dir) / "dsp_template.py")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No hay plantilla DSP; se buscó en: {[str(c) for c in candidates]}")


def _num(value: float) -> str:
    value = float(value)
    return str(int(value)) if value.is_integer() else repr(value)


def _array(name: str, values: Sequence[float]) -> str:
    rows: List[str] = []
    vals = [f"{float(v):.7e}" for v in values]
    for i in range(0, len(vals), 4):
        rows.append("    " + ", ".join(vals[i : i + 4]) + ",")
    body = "\n".join(rows)
    return f"{name} = np.asarray([\n{body}\n], dtype=np.float32)"


def _mfe_blocks_literal(blocks: Sequence[Dict[str, Any]]) -> str:
    out = ["MFE_BLOCKS: List[Dict[str, Any]] = ["]
    for block in blocks:
        out.append("    {")
        for tpl_key, key in _MFE_KEYS:
            out.append(f'        "{tpl_key}": {_num(block[key])},')
        out.append('        "WIN_SIZE": 101,  # Edge Impulse export metadata; not used directly.')
        out.append("    },")
    out.append("]")
    return "\n".join(out)


def _labels_literal(labels: Sequence[str]) -> str:
    body = "\n".join(f'    "{label}",' for label in labels)
    return f"CLASS_NAMES = [\n{body}\n]"


def _replace(text: str, pattern: str, replacement: str) -> str:
    new, n = re.subn(pattern, lambda _m: replacement, text, count=1, flags=re.MULTILINE | re.DOTALL)
    if n != 1:
        raise ValueError(f"La plantilla no contiene el bloque esperado: {pattern}")
    return new


def render_dsp_module(
    template_text: str,
    artifact: Dict[str, Any],
    window_step_sec: float,
    dsp_version: Optional[str] = None,
    allow_aux_mismatch: bool = False,
) -> str:
    """Sustituye las constantes de la plantilla por las del entrenamiento."""
    dsp = artifact["dsp"]
    aux = artifact.get("aux_scaler")
    if not aux:
        raise ValueError("El run no tiene aux_scaler: el modelo no usa features auxiliares")
    is_pinbolt = dsp.get("profile", "default") == "pinbolt"
    if not is_pinbolt and not allow_aux_mismatch:
        raise ValueError(
            "El run no usa profile=pinbolt: sus features no reproducen la plantilla. Reentrena con "
            "DS_DSP_CONFIG=configs/dsp_pinbolt.yaml o usa allow_aux_mismatch=True."
        )
    zero_padding = bool(dsp.get("zero_padding", False))
    n_final = int(dsp.get("n_filters_final", 0)) or int(dsp["mfe_blocks"][0]["n_filters"])
    mean = np.asarray(aux["offset"], dtype=np.float32)
    scale = np.asarray(aux["divisor"], dtype=np.float32)
    aux_dim = int(mean.shape[0])

    tpl_dim = int(re.search(r"^AUX_FEATURE_DIM = (\d+)", template_text, re.MULTILINE).group(1))
    if aux_dim != tpl_dim and not allow_aux_mismatch:
        raise ValueError(
            f"El vector auxiliar del run tiene {aux_dim} valores y la plantilla extrae {tpl_dim}. "
            "El DSP generado no reproduciría el entrenamiento; alinea aux_features con la "
            "plantilla o usa allow_aux_mismatch=True bajo tu responsabilidad."
        )

    window_size_sec = float(dsp["clip_duration"])
    if not 0 < window_step_sec <= window_size_sec:
        raise ValueError("window_step_sec debe estar en (0, clip_duration]")

    text = template_text
    text = _replace(text, r"^SAMPLE_RATE = .*?$", f"SAMPLE_RATE = {int(dsp['sample_rate'])}")
    text = _replace(text, r"^WINDOW_SIZE_SEC = .*?$", f"WINDOW_SIZE_SEC = {window_size_sec!r}")
    text = _replace(
        text, r"^WINDOW_STEP_SEC = .*?$", f"WINDOW_STEP_SEC = {float(window_step_sec)!r}"
    )
    text = _replace(text, r"^ZERO_PADDING = .*?$", f"ZERO_PADDING = {bool(zero_padding)}")
    text = _replace(
        text, r"^PREEMPHASIS_COEFF = .*?$", f"PREEMPHASIS_COEFF = {float(dsp['pre_emphasis'])!r}"
    )
    text = _replace(text, r"^CLASS_NAMES = \[.*?^\]", _labels_literal(artifact["labels"]))
    text = _replace(
        text,
        r"^MFE_BLOCKS: List\[Dict\[str, Any\]\] = \[.*?^\]",
        _mfe_blocks_literal(dsp["mfe_blocks"]),
    )
    text = _replace(
        text, r"^N_FILTERS_FINAL = .*?$", f"N_FILTERS_FINAL = {n_final if is_pinbolt else None}"
    )
    text = _replace(text, r"^AUX_FEATURE_DIM = .*?$", f"AUX_FEATURE_DIM = {aux_dim}")
    text = _replace(
        text, r"^AUX_MEAN = np\.asarray\(\[.*?^\], dtype=np\.float32\)", _array("AUX_MEAN", mean)
    )
    text = _replace(
        text, r"^AUX_SCALE = np\.asarray\(\[.*?^\], dtype=np\.float32\)", _array("AUX_SCALE", scale)
    )
    if dsp_version:
        text = _replace(text, r'^DSP_VERSION = ".*?"', f'DSP_VERSION = "{dsp_version}"')
    return text


def export_pinbolt_package(
    run_dir: PathLike,
    output_dir: PathLike,
    template_path: PathLike,
    name: str = "pinbolt_g5x",
    version: str = "v150",
    window_step_sec: Optional[float] = None,
    dsp_version: Optional[str] = None,
    allow_aux_mismatch: bool = False,
) -> Dict[str, Path]:
    """Escribe ``dsp_<name>-<version>.py`` y ``nn_<name>-<version>.tflite`` en ``output_dir``.

    ``window_step_sec`` ``None`` (o 0) usa el ``WINDOW_STEP_SEC`` de la plantilla.
    """
    if not window_step_sec:
        from ds_project.dsp.templates import read_window_step

        window_step_sec = read_window_step(template_path) or DEFAULT_WINDOW_STEP
    run_dir = Path(run_dir)
    tflite = run_dir / "model.tflite"
    params_json = run_dir / "dsp_params.json"
    for required in (tflite, params_json):
        if not required.exists():
            raise FileNotFoundError(f"Falta {required}")

    artifact = json.loads(params_json.read_text(encoding="utf-8"))
    text = render_dsp_module(
        Path(template_path).read_text(encoding="utf-8"),
        artifact,
        window_step_sec,
        dsp_version,
        allow_aux_mismatch,
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dsp_path = out / f"dsp_{name}-{version}.py"
    nn_path = out / f"nn_{name}-{version}.tflite"
    dsp_path.write_text(text, encoding="utf-8")
    shutil.copyfile(tflite, nn_path)

    _validate_import(dsp_path)
    result = {"dsp": dsp_path, "tflite": nn_path}
    manifest = run_dir / "dataset_manifest.json"
    if manifest.exists():
        result["manifest"] = out / f"dataset_manifest_{name}-{version}.json"
        shutil.copyfile(manifest, result["manifest"])
    return result


def export_for_project(  # noqa: PLR0913
    product: str = "",
    turbine: str = "",
    version: str = "v150",
    run_dir: PathLike = "",
    output_dir: PathLike = "",
    template_path: PathLike = "",
    window_step_sec: Optional[float] = None,
    dsp_version: Optional[str] = None,
    allow_aux_mismatch: bool = False,
    data_version: str = "",
    templates_dir: PathLike = "templates",
) -> Dict[str, Path]:
    """Exporta con la norma ``dsp_<producto>_<turbina>-v<n>.py`` / ``nn_...-v<n>.tflite``.

    - run: el más reciente de ``<proyecto>/<datos>/models``.
    - plantilla: ``<proyecto>/templates/<datos>/``, ``<proyecto>/templates/`` o la común.
    - salida: ``<proyecto>/export/<version>/``.
    """
    from ds_project.config import load_settings

    if not _MODEL_VERSION_RE.match(version):
        raise ValueError(f"La versión del modelo debe ser v<número> (p. ej. v150): {version!r}")
    settings = load_settings(
        product=product or None, turbine=turbine or None, data_version=data_version or None
    )
    project_dir = settings.project_dir()
    if project_dir is None:
        raise ValueError("Falta el producto (--product / DS_PRODUCT)")
    data_version = settings.data.version
    models_dir = settings.resolved_paths().models_dir
    run = Path(run_dir) if run_dir else find_latest_run(models_dir, data_version or None)
    template = Path(template_path) if template_path else None
    template = template or resolve_template(templates_dir, project_dir, data_version)
    out = Path(output_dir) if output_dir else project_dir / "export" / version
    return export_pinbolt_package(
        run,
        out,
        template,
        settings.project_name,
        version,
        window_step_sec,
        dsp_version,
        allow_aux_mismatch,
    )


def _validate_import(path: Path) -> None:
    """Importa el módulo generado; la plantilla valida sus constantes al cargarse."""
    spec = importlib.util.spec_from_file_location(f"_pinbolt_{path.stem.replace('-', '_')}", path)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(importlib.util.module_from_spec(spec))
