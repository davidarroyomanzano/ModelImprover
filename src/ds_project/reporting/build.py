"""Orquesta la evaluación de un run sobre ``data/test`` y genera el informe (MD, HTML, PDF)."""

from __future__ import annotations

import logging
import platform
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ds_project.data.source_manifest import validate_source_manifests
from ds_project.data.test_set import discover_labeled_files
from ds_project.dsp.module_export import find_latest_run
from ds_project.reporting import figures
from ds_project.reporting.document import Document, write_all
from ds_project.reporting.evaluation import (
    misclassified,
    run_evaluation,
    save_predictions,
    summarize,
)
from ds_project.reporting.optuna_info import load_study_info
from ds_project.utils.io import load_json, save_json

logger = logging.getLogger(__name__)
PathLike = Union[str, Path]
EVAL_DIR = "evaluation"


def _pct(x: Optional[float]) -> str:
    return "n/d" if x is None else f"{100 * x:.2f} %"


def _f(x: Optional[float], nd: int = 3) -> str:
    return "n/d" if x is None else f"{x:.{nd}f}"


def _ci(ci: Dict[str, List[float]], key: str) -> str:
    return f"[{_pct(ci[key][0])}, {_pct(ci[key][1])}]" if key in ci else "n/d"


def verdict(
    summary: Dict[str, Any], min_accuracy: float = 0.0, min_f1: float = 0.0
) -> Dict[str, Any]:
    """Compara accuracy y F1 macro por segmento con los umbrales; sin umbrales es informativo."""
    seg = summary["segment"]
    checks = []
    if min_accuracy:
        checks.append(("accuracy", seg["accuracy"], min_accuracy))
    if min_f1:
        checks.append(("f1_macro", seg["f1_macro"], min_f1))
    failed = [f"{n}={v:.4f} < {t:.4f}" for n, v, t in checks if v < t]
    return {
        "thresholds": {n: t for n, _, t in checks},
        "passed": not failed,
        "failed": failed,
        "informative": not checks,
    }


def _tflite_report(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / "metrics.json"
    return (load_json(path) if path.exists() else {}).get("tflite", {})


def _data_section(doc: Document, summary, manifest_check, run_dir, fig_dir, labels) -> None:
    doc.h1("2. Datos")
    ds_path = run_dir / "dataset_manifest.json"
    train_counts: Dict[str, int] = {}
    if ds_path.exists():
        ds = load_json(ds_path)
        train_counts = ds.get("per_class", {})
        doc.p(
            f"Conjunto de entrenamiento: {ds.get('n_files', 0)} ficheros "
            f"(huella `{ds.get('dataset_hash', '')}`). Conjunto de test: {summary['n_files']} "
            f"ficheros, {summary['n_segments']} segmentos."
        )
    rows = [
        [
            lbl,
            train_counts.get(lbl, 0),
            summary["files_by_class"][lbl],
            summary["segments_by_class"][lbl],
            f"{summary['seconds_by_class'][lbl]:.1f}",
        ]
        for lbl in labels
    ]
    doc.table(
        ["Clase", "Ficheros train/val", "Ficheros test", "Segmentos test", "Segundos test"],
        rows,
        "Composición de los conjuntos por clase",
    )
    doc.image(
        figures.class_distribution(
            train_counts, summary["files_by_class"], fig_dir / "class_distribution.png"
        ),
        "Distribución de clases",
    )
    if summary["skipped_unknown_classes"]:
        skipped = summary["skipped_unknown_classes"]
        doc.p(f"Clases de test ignoradas por no existir en el modelo: {skipped}")
    if not manifest_check.get("available"):
        doc.p("No hay manifests de origen en la carpeta de datos: no se validó la coherencia.")
        return
    doc.h2("Validación contra los manifests de origen")
    state = "COHERENTE" if manifest_check["ok"] else "CON INCIDENCIAS"
    doc.p(
        f"Resultado: **{state}**. Filas en manifests: {manifest_check['n_rows']}; WAV en disco: "
        f"{manifest_check['n_wavs']}; grabaciones de origen distintas: "
        + ", ".join(f"{k} {v}" for k, v in (manifest_check.get("n_sources") or {}).items())
        + "."
    )
    keys = [
        ("missing_on_disk", "Filas del manifest sin WAV en disco"),
        ("unlisted_on_disk", "WAV en disco sin fila en el manifest"),
        ("class_mismatch", "Clase del manifest distinta a la carpeta"),
        ("duplicates", "Filas duplicadas"),
        ("shared_sources", "Grabaciones compartidas entre raw y test (fuga)"),
        ("sample_rate_mismatch", "Frecuencia de muestreo distinta a la del DSP"),
        ("duration_mismatch", "Duración distinta a la del manifest"),
        ("unreadable", "WAV ilegibles"),
    ]
    rows = []
    for key, text in keys:
        value = manifest_check.get(key)
        if isinstance(value, dict):
            value = value.get("count", 0)
        rows.append([text, value])
    doc.table(["Comprobación", "Casos"], rows)
    if manifest_check.get("warnings"):
        doc.bullets([str(w) for w in manifest_check["warnings"]])
    roles = manifest_check.get("roles_by_split") or {}
    if roles:
        names = sorted({r for split in roles.values() for r in split})
        doc.table(
            ["Conjunto"] + names,
            [[split] + [roles[split].get(n, 0) for n in names] for split in roles],
            "Rol de cada muestra según el manifest",
        )


def _results_section(doc, summary, ev, fig_dir, run_dir) -> None:
    labels = summary["labels"]
    seg, fil, ci = summary["segment"], summary["file"], summary["bootstrap_ci_95"]
    doc.h1("4. Resultados sobre data/test")
    doc.table(
        ["Métrica", "Por segmento", "Por fichero"],
        [
            ["Accuracy", _pct(seg["accuracy"]), _pct(fil["accuracy"])],
            ["Balanced accuracy", _pct(seg["balanced_accuracy"]), _pct(fil["balanced_accuracy"])],
            ["F1 macro", _f(seg["f1_macro"]), _f(fil["f1_macro"])],
            ["Precisión macro", _f(seg["precision_macro"]), _f(fil["precision_macro"])],
            ["Recall macro", _f(seg["recall_macro"]), _f(fil["recall_macro"])],
            ["ROC-AUC macro", _f(seg.get("roc_auc_macro")), ""],
        ],
        "Métricas globales (el fichero agrega la media de probabilidades de sus segmentos)",
    )
    doc.p(
        f"IC 95 % (bootstrap por ficheros): accuracy {_ci(ci, 'accuracy')}, "
        f"F1 macro {_ci(ci, 'f1_macro')}."
    )
    report = seg["report"]
    rows = [
        [
            lbl,
            _f(report.get(lbl, {}).get("precision")),
            _f(report.get(lbl, {}).get("recall")),
            _f(report.get(lbl, {}).get("f1-score")),
            int(report.get(lbl, {}).get("support", 0)),
            _f(seg["roc_auc"].get(lbl)),
            _f(seg["average_precision"].get(lbl)),
        ]
        for lbl in labels
    ]
    doc.table(
        ["Clase", "Precisión", "Recall", "F1", "Soporte", "ROC-AUC", "AP"],
        rows,
        "Métricas por clase (segmentos)",
    )
    doc.image(
        figures.per_class(report, labels, fig_dir / "per_class.png"), "Precisión, recall y F1"
    )
    doc.image(
        figures.confusion(seg["confusion_matrix"], labels, fig_dir / "confusion.png"),
        "Matriz de confusión por segmento",
    )
    doc.image(
        figures.roc_pr(ev.y_true, ev.probs, labels, fig_dir / "roc_pr.png"), "Curvas ROC y PR"
    )
    doc.h2("Calibración")
    doc.p(
        f"Error de calibración esperado (ECE): {_f(seg['calibration']['ece'], 4)}. "
        "Una confianza alta en errores indica que el umbral de decisión necesita revisión."
    )
    doc.image(
        figures.confidence(ev.probs, ev.y_true, seg["calibration"], fig_dir / "confidence.png"),
        "Confianza y calibración",
    )
    errors = misclassified(ev, 20)
    doc.h2("Errores más confiados")
    if errors:
        doc.table(
            ["Fichero", "Real", "Predicha", "Confianza", "P(real)"],
            [
                [e["file"], e["true"], e["pred"], _f(e["confidence"]), _f(e["p_true"])]
                for e in errors
            ],
            "Segmentos mal clasificados (hasta 20)",
        )
    else:
        doc.p("No hay segmentos mal clasificados.")


def _optuna_section(doc, info, fig_dir) -> None:
    doc.h1("5. Búsqueda de hiperparámetros (Optuna)")
    if not info:
        doc.p("No hay información del estudio Optuna disponible para este run.")
        return
    dsp_phase = info.get("dsp_phase")
    if dsp_phase:
        doc.p(
            "Búsqueda en dos fases: primero el DSP con una red fija pequeña y después la red "
            "con el mejor DSP."
        )
        doc.h2("5.1 Fase 1: DSP")
        _study_block(doc, dsp_phase, fig_dir / "optuna_dsp.png")
        doc.h2("5.2 Fase 2: red")
    _study_block(doc, info, fig_dir / "optuna.png")


def _study_block(doc, info, chart_path) -> None:
    trials = info.get("trials", [])
    states: Dict[str, int] = {}
    for t in trials:
        states[t["state"]] = states.get(t["state"], 0) + 1
    durations = [t["duration_s"] for t in trials if t.get("duration_s") is not None]
    doc.p(
        f"Estudio `{info.get('study_name')}` ({info.get('direction')}): {len(trials)} trials "
        f"({', '.join(f'{k}: {v}' for k, v in sorted(states.items()))}). "
        f"Mejor trial: #{info.get('best_trial')}."
        + (f" Duración total {sum(durations) / 60:.1f} min." if durations else "")
    )
    chart = figures.optuna_overview(info, chart_path)
    if chart:
        doc.image(chart, "Historia de la búsqueda e importancia de hiperparámetros")
    done = [t for t in trials if t.get("value") is not None]
    done.sort(key=lambda t: t["value"], reverse=info.get("direction") == "MAXIMIZE")
    best = done[:10]
    if best:
        keys = list(best[0]["params"])
        doc.table(
            ["Parámetro"] + [f"#{t['number']}" for t in best[:5]],
            [
                [k]
                + [
                    _f(t["params"].get(k), 4)
                    if isinstance(t["params"].get(k), float)
                    else str(t["params"].get(k, ""))
                    for t in best[:5]
                ]
                for k in keys
            ],
            "Hiperparámetros de los 5 mejores trials",
        )
        doc.table(
            ["Trial", "Valor", "Duración (s)"],
            [[f"#{t['number']}", _f(t["value"], 4), _f(t.get("duration_s"), 1)] for t in best],
            "Top 10 trials",
        )


def _model_section(doc, run_dir, dsp, metrics) -> None:
    doc.h1("3. Modelo")
    d = dsp.get("dsp", {})
    extra = dsp.get("extra", {})
    doc.table(
        ["Campo", "Valor"],
        [
            ["Run", extra.get("run_id", run_dir.name)],
            ["Cuantización", extra.get("quantization", "")],
            ["Frecuencia de muestreo", d.get("sample_rate")],
            ["Ventana (s)", d.get("clip_duration")],
            ["Modo", d.get("mode")],
            ["Bloques MFE", len(d.get("mfe_blocks", []))],
            ["Features auxiliares", ", ".join(d.get("aux_features", [])) or "ninguna"],
        ],
    )
    blocks = d.get("mfe_blocks", [])
    if blocks:
        keys = list(blocks[0])
        doc.table(
            ["Bloque"] + keys,
            [[i + 1] + [b.get(k) for k in keys] for i, b in enumerate(blocks)],
            "Parámetros de los bloques MFE",
        )
    if metrics:
        doc.p(
            "Validación interna del entrenamiento (partición de raw, TFLite): "
            f"accuracy {_pct(metrics.get('accuracy'))}, F1 macro {_f(metrics.get('f1_macro'))}. "
            "La evaluación externa de este informe usa data/test."
        )


def _repro_section(doc, run_dir, project) -> None:
    doc.h1("6. Reproducibilidad")
    versions = {"python": platform.python_version()}
    for mod in ("numpy", "tensorflow", "sklearn", "optuna", "librosa"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            versions[mod] = "n/d"
    doc.table(["Componente", "Versión"], [[k, v] for k, v in versions.items()])
    doc.bullets(
        [
            f"Run: `{run_dir}`",
            f"Proyecto: `{project}`",
            "Reproducir: `poetry run invoke train` y `poetry run invoke evaluate`",
        ]
    )


def evaluate_for_project(  # noqa: PLR0913
    product: str = "",
    turbine: str = "",
    data_version: str = "",
    run_dir: PathLike = "",
    min_accuracy: float = 0.0,
    min_f1: float = 0.0,
    n_boot: int = 300,
) -> Dict[str, Any]:
    """Evalúa el run (el último del ámbito, salvo ``run_dir``) y escribe ``<run>/evaluation/``."""
    from ds_project.config import load_settings

    settings = load_settings(
        product=product or None, turbine=turbine or None, data_version=data_version or None
    )
    paths = settings.resolved_paths()
    run = (
        Path(run_dir)
        if run_dir
        else find_latest_run(paths.models_dir, settings.data.version or None)
    )
    out = run / EVAL_DIR
    try:
        discover_labeled_files(Path(paths.test_dir))
    except (FileNotFoundError, ValueError):
        msg = f"Sin audios en {paths.test_dir}: evaluación omitida"
        logger.warning(msg)
        return {
            "run_dir": run,
            "skipped": msg,
            "verdict": {"passed": True, "informative": True, "failed": []},
        }
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    ev = run_evaluation(run, paths.test_dir)
    summary = summarize(ev, n_boot=n_boot)
    dsp = load_json(run / "dsp_params.json") if (run / "dsp_params.json").exists() else {}
    sample_rate = dsp.get("dsp", {}).get("sample_rate")
    manifest_check = validate_source_manifests(
        Path(paths.raw_dir).parent, paths.raw_dir, paths.test_dir, sample_rate=sample_rate
    )
    info = load_study_info(run)
    result = verdict(summary, min_accuracy, min_f1)

    save_predictions(ev, out / "predictions.csv")
    save_json({**summary, "verdict": result}, out / "metrics_test.json")
    save_json(manifest_check, out / "source_manifest_check.json")

    project = settings.project_name or run.parent.parent.name
    doc = Document(
        f"Informe de evaluación: {project}",
        f"Run {run.name} · {datetime.now():%Y-%m-%d %H:%M}",
    )
    doc.h1("1. Resumen")
    seg = summary["segment"]
    if result["informative"]:
        state = "Informativo (sin umbrales de aceptación)"
    else:
        state = "APTO" if result["passed"] else "NO APTO: " + "; ".join(result["failed"])
    doc.bullets(
        [
            f"Veredicto: **{state}**",
            f"Accuracy por segmento: {_pct(seg['accuracy'])}; F1 macro: {_f(seg['f1_macro'])}",
            f"Accuracy por fichero: {_pct(summary['file']['accuracy'])}",
            f"Test: {summary['n_files']} ficheros, {summary['n_segments']} segmentos",
        ]
    )
    _data_section(doc, summary, manifest_check, run, fig_dir, summary["labels"])
    _model_section(doc, run, dsp, _tflite_report(run))
    _results_section(doc, summary, ev, fig_dir, run)
    _optuna_section(doc, info, fig_dir)
    _repro_section(doc, run, project)
    files = write_all(doc, out, "report")
    return {
        "run_dir": run,
        "output_dir": out,
        "verdict": result,
        "accuracy": seg["accuracy"],
        "f1_macro": seg["f1_macro"],
        **{k: v for k, v in files.items()},
    }
