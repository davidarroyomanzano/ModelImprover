"""Reevaluación de un run sobre ``data/test`` (conjunto externo, nunca usado para entrenar)."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from ds_project.data.process import load_audio, segment_signal
from ds_project.data.test_set import discover_labeled_files
from ds_project.features.build import build_auxiliary_features, build_features
from ds_project.models.evaluate import compute_metrics, tflite_predict_proba
from ds_project.models.predict import AudioClassifier
from ds_project.utils.io import load_yaml

PathLike = Union[str, Path]


@dataclass
class Evaluation:
    run_dir: Path
    labels: List[str]
    files: List[str]
    file_labels: List[int]
    seconds_by_class: Dict[str, float]
    seg_file: np.ndarray
    y_true: np.ndarray
    probs: np.ndarray
    skipped: Dict[str, int] = field(default_factory=dict)

    @property
    def y_pred(self) -> np.ndarray:
        return np.argmax(self.probs, axis=-1)


def _segmenting(run_dir: Path) -> Dict[str, float]:
    try:
        data = (load_yaml(run_dir / "config_snapshot.yaml") or {}).get("data", {})
    except OSError:
        data = {}
    return {
        "overlap": float(data.get("segment_overlap", 0.0)),
        "min_ratio": float(data.get("min_segment_ratio", 0.5)),
    }


def run_evaluation(run_dir: PathLike, test_dir: PathLike) -> Evaluation:
    """Segmenta ``test_dir/<clase>/*.wav`` como en el entrenamiento y clasifica con el TFLite."""
    run_dir = Path(run_dir)
    clf = AudioClassifier.from_run_dir(run_dir)
    index = {lbl: i for i, lbl in enumerate(clf.labels)}
    seg = _segmenting(run_dir)
    params = clf.params

    files: List[str] = []
    file_labels: List[int] = []
    seconds: Dict[str, float] = {lbl: 0.0 for lbl in clf.labels}
    skipped: Dict[str, int] = {}
    signals: List[np.ndarray] = []
    seg_file: List[int] = []
    for item in discover_labeled_files(Path(test_dir)):
        if item.label not in index:
            skipped[item.label] = skipped.get(item.label, 0) + 1
            continue
        audio = load_audio(item.path, params.sample_rate)
        seconds[item.label] += len(audio) / params.sample_rate
        segs = segment_signal(audio, params.n_samples, seg["overlap"], seg["min_ratio"])
        files.append(item.path.name)
        file_labels.append(index[item.label])
        seg_file += [len(files) - 1] * len(segs)
        signals += segs

    if not signals:
        raise ValueError(f"No hay audios evaluables en {test_dir} para las clases {clf.labels}")
    X = clf.scaler.transform(build_features(signals, params))
    aux = None
    if clf.feature_scaler is not None:
        aux = clf.feature_scaler.transform(build_auxiliary_features(signals, params))
    probs = tflite_predict_proba(clf.model_path, X, aux)
    seg_file_arr = np.asarray(seg_file, dtype=np.int64)
    y_true = np.asarray(file_labels, dtype=np.int64)[seg_file_arr]
    return Evaluation(
        run_dir, list(clf.labels), files, file_labels, seconds, seg_file_arr, y_true, probs, skipped
    )


def file_level(ev: Evaluation) -> Dict[str, np.ndarray]:
    """Agrega por fichero: media de probabilidades de sus segmentos."""
    n, k = len(ev.files), len(ev.labels)
    acc = np.zeros((n, k), dtype=np.float64)
    np.add.at(acc, ev.seg_file, ev.probs)
    counts = np.bincount(ev.seg_file, minlength=n).reshape(-1, 1)
    return {"y_true": np.asarray(ev.file_labels), "probs": acc / np.maximum(counts, 1)}


def _auc_ap(y_true: np.ndarray, probs: np.ndarray, labels: List[str]) -> Dict[str, Any]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    roc: Dict[str, Optional[float]] = {}
    ap: Dict[str, Optional[float]] = {}
    for k, lbl in enumerate(labels):
        pos = y_true == k
        if pos.any() and not pos.all():
            roc[lbl] = float(roc_auc_score(pos, probs[:, k]))
            ap[lbl] = float(average_precision_score(pos, probs[:, k]))
        else:
            roc[lbl] = ap[lbl] = None
    valid = [v for v in roc.values() if v is not None]
    return {
        "roc_auc": roc,
        "average_precision": ap,
        "roc_auc_macro": float(np.mean(valid)) if valid else None,
    }


def _calibration(probs: np.ndarray, y_true: np.ndarray, bins: int = 10) -> Dict[str, Any]:
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece, rows = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
            rows.append(
                {
                    "bin": f"{lo:.1f}-{hi:.1f}",
                    "n": int(mask.sum()),
                    "confidence": float(conf[mask].mean()),
                    "accuracy": float(correct[mask].mean()),
                }
            )
    return {"ece": float(ece), "bins": rows}


def _percentiles(values: List[float]) -> List[float]:
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def _bootstrap_ci(ev: Evaluation, n_boot: int, seed: int) -> Dict[str, List[float]]:
    """IC 95 % remuestreando ficheros (los segmentos de un fichero no son independientes)."""
    from sklearn.metrics import f1_score

    rng = np.random.default_rng(seed)
    idx = list(range(len(ev.labels)))
    y_pred, y_true = ev.y_pred, ev.y_true
    n_files = len(ev.files)
    acc: List[float] = []
    f1: List[float] = []
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, n_files, n_files), minlength=n_files)[ev.seg_file]
        if w.sum() == 0:
            continue
        acc.append(float(np.average(y_pred == y_true, weights=w)))
        score = f1_score(
            y_true, y_pred, labels=idx, average="macro", zero_division=0, sample_weight=w
        )
        f1.append(float(score))
    return {"accuracy": _percentiles(acc), "f1_macro": _percentiles(f1)} if acc else {}


def summarize(ev: Evaluation, n_boot: int = 300, seed: int = 42) -> Dict[str, Any]:
    """Métricas por segmento y por fichero, AUC/AP, calibración e IC bootstrap por fichero."""
    from sklearn.metrics import balanced_accuracy_score

    seg = compute_metrics(ev.y_true, ev.y_pred, ev.labels)
    seg["balanced_accuracy"] = float(balanced_accuracy_score(ev.y_true, ev.y_pred))
    seg.update(_auc_ap(ev.y_true, ev.probs, ev.labels))
    seg["calibration"] = _calibration(ev.probs, ev.y_true)
    fl = file_level(ev)
    fpred = fl["probs"].argmax(axis=1)
    per_file = compute_metrics(fl["y_true"], fpred, ev.labels)
    per_file["balanced_accuracy"] = float(balanced_accuracy_score(fl["y_true"], fpred))
    support = np.bincount(ev.y_true, minlength=len(ev.labels))
    file_labels = np.asarray(ev.file_labels)
    return {
        "labels": ev.labels,
        "n_files": len(ev.files),
        "n_segments": int(len(ev.y_true)),
        "seconds_by_class": {k: round(v, 2) for k, v in ev.seconds_by_class.items()},
        "files_by_class": {lbl: int(np.sum(file_labels == i)) for i, lbl in enumerate(ev.labels)},
        "segments_by_class": {lbl: int(support[i]) for i, lbl in enumerate(ev.labels)},
        "skipped_unknown_classes": ev.skipped,
        "segment": seg,
        "file": per_file,
        "bootstrap_ci_95": _bootstrap_ci(ev, n_boot, seed) if n_boot else {},
    }


def misclassified(ev: Evaluation, limit: int = 25) -> List[Dict[str, Any]]:
    """Segmentos mal clasificados, de mayor a menor confianza en la clase errónea."""
    pred = ev.y_pred
    rows = []
    for i in np.flatnonzero(pred != ev.y_true):
        p = ev.probs[i]
        rows.append(
            {
                "file": ev.files[int(ev.seg_file[i])],
                "true": ev.labels[int(ev.y_true[i])],
                "pred": ev.labels[int(pred[i])],
                "confidence": float(p[pred[i]]),
                "p_true": float(p[ev.y_true[i]]),
            }
        )
    rows.sort(key=lambda r: -r["confidence"])
    return rows[:limit] if limit else rows


def save_predictions(ev: Evaluation, path: PathLike) -> Path:
    """CSV con una fila por segmento: fichero, real, predicho y probabilidades."""
    path = Path(path)
    pred = ev.y_pred
    header = ["file", "segment", "true", "pred", "confidence"] + [f"p_{x}" for x in ev.labels]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        seen: Dict[int, int] = {}
        for i, f in enumerate(ev.seg_file):
            seen[int(f)] = seen.get(int(f), -1) + 1
            w.writerow(
                [
                    ev.files[int(f)],
                    seen[int(f)],
                    ev.labels[int(ev.y_true[i])],
                    ev.labels[int(pred[i])],
                    f"{float(ev.probs[i][pred[i]]):.5f}",
                ]
                + [f"{float(p):.5f}" for p in ev.probs[i]]
            )
    return path
