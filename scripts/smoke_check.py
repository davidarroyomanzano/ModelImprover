"""Comprobaciones para un smoke run de la pipeline.

Uso:
    python scripts/smoke_check.py data [--data-dir DIR]
    python scripts/smoke_check.py artifacts [--models-dir DIR] [--run-id ID] [--require-optuna]
Sin rutas, usa el ámbito de projects/<producto>/<turbina>/<datos>/ (DS_PRODUCT, DS_TURBINE...).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EXPECTED_LABELS: tuple[str, ...] = (
    "Good",
    "Stopped",
    "Hydraulic_charge_start",
    "Hydraulic_charge_stop",
    "Impacts_hub",
    "Impacts_pitch",
    "Impacts_yaw",
    "Tendon_rupture",
)

REQUIRED_ARTIFACTS: tuple[str, ...] = (
    "model.tflite",
    "model.keras",
    "scaler.joblib",
    "dsp_params.json",
    "dsp_params.h",
    "metrics.json",
    "config_snapshot.yaml",
)

OPTUNA_ARTIFACTS: tuple[str, ...] = ("best_params.json", "optuna_summary.json")


def count_samples(data_dir: Path) -> dict[str, int]:
    """Cuenta los ficheros .wav por carpeta de etiqueta."""
    if not data_dir.is_dir():
        return {}
    return {
        d.name: sum(1 for _ in d.glob("*.wav")) for d in sorted(data_dir.iterdir()) if d.is_dir()
    }


def check_data(
    data_dir: Path,
    expected: tuple[str, ...] = EXPECTED_LABELS,
    min_per_class: int = 2,
) -> list[str]:
    """Devuelve una lista de problemas encontrados en el directorio de datos."""
    counts = count_samples(data_dir)
    problems: list[str] = []
    if not counts:
        return [f"No se encontraron etiquetas en {data_dir}"]
    for label in expected:
        n = counts.get(label, 0)
        if n == 0:
            problems.append(f"Falta la clase '{label}'")
        elif n < min_per_class:
            problems.append(f"Clase '{label}' con {n} muestra(s) (< {min_per_class})")
    for label in counts:
        if label not in expected:
            problems.append(f"Etiqueta inesperada '{label}'")
    return problems


def latest_run(models_dir: Path) -> Path | None:
    """Devuelve el directorio de ejecución modificado más recientemente."""
    try:
        runs = [p for p in models_dir.iterdir() if p.is_dir()]
    except (FileNotFoundError, NotADirectoryError):
        return None

    return max(runs, key=lambda p: p.stat().st_mtime) if runs else None


def _find_labels(obj: object) -> list[str] | None:
    """Busca recursivamente una lista de etiquetas en el JSON de parámetros DSP."""
    if isinstance(obj, dict):
        for key in ("labels", "class_names", "classes"):
            value = obj.get(key)
            if isinstance(value, list) and all(isinstance(v, str) for v in value):
                return value
        for value in obj.values():
            found = _find_labels(value)
            if found is not None:
                return found
    return None


def check_artifacts(
    run_dir: Path,
    expected: tuple[str, ...] = EXPECTED_LABELS,
    require_optuna: bool = False,
) -> list[str]:
    """Devuelve una lista de problemas en los artefactos de una ejecución."""
    problems: list[str] = []
    required = REQUIRED_ARTIFACTS + (OPTUNA_ARTIFACTS if require_optuna else ())
    for name in required:
        path = run_dir / name
        if not path.is_file():
            problems.append(f"Falta {name}")
        elif path.stat().st_size == 0:
            problems.append(f"{name} está vacío")

    dsp_json = run_dir / "dsp_params.json"
    if dsp_json.is_file() and dsp_json.stat().st_size > 0:
        try:
            labels = _find_labels(json.loads(dsp_json.read_text(encoding="utf-8")))
        except json.JSONDecodeError as exc:
            problems.append(f"dsp_params.json inválido: {exc}")
        else:
            if labels is None:
                problems.append("dsp_params.json no contiene lista de etiquetas")
            elif set(labels) != set(expected):
                problems.append(f"Etiquetas en dsp_params.json no coinciden: {labels}")

    metrics = run_dir / "metrics.json"
    if metrics.is_file() and metrics.stat().st_size > 0:
        try:
            json.loads(metrics.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            problems.append(f"metrics.json inválido: {exc}")
    return problems


def _report(title: str, problems: list[str]) -> int:
    if problems:
        print(f"[FAIL] {title}")
        for p in problems:
            print(f"  - {p}")
        return 1
    print(f"[OK] {title}")
    return 0


def _scoped_paths():
    from ds_project.config import load_settings

    return load_settings().resolved_paths()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_data = sub.add_parser("data", help="Valida muestras por clase")
    p_data.add_argument(
        "--data-dir", type=Path, default=None, help="Por defecto, <ámbito>/data/raw"
    )
    p_data.add_argument("--min-per-class", type=int, default=2)

    p_art = sub.add_parser("artifacts", help="Valida artefactos de una ejecución")
    p_art.add_argument("--models-dir", type=Path, default=None, help="Por defecto, <ámbito>/models")
    p_art.add_argument("--run-id", default=None)
    p_art.add_argument("--require-optuna", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "data" and args.data_dir is None:
        args.data_dir = Path(_scoped_paths().raw_dir)
    if args.command == "artifacts" and args.models_dir is None:
        args.models_dir = Path(_scoped_paths().models_dir)

    if args.command == "data":
        for label, n in count_samples(args.data_dir).items():
            print(f"  {label:<25} {n}")
        return _report(
            f"Datos en {args.data_dir}",
            check_data(args.data_dir, min_per_class=args.min_per_class),
        )

    run_dir = args.models_dir / args.run_id if args.run_id else latest_run(args.models_dir)
    if run_dir is None or not run_dir.is_dir():
        return _report("Artefactos", [f"No hay ejecuciones en {args.models_dir}"])
    return _report(
        f"Artefactos en {run_dir}",
        check_artifacts(run_dir, require_optuna=args.require_optuna),
    )


if __name__ == "__main__":
    sys.exit(main())
