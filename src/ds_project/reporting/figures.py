"""Figuras del informe de evaluación (matplotlib, backend Agg)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

PathLike = Union[str, Path]
_BLUE, _RED, _GREY = "#2f6fdb", "#d64545", "#9aa5b1"


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save(fig, path: PathLike) -> Path:
    path = Path(path)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    _plt().close(fig)
    return path


def confusion(cm: Sequence[Sequence[int]], labels: List[str], path: PathLike) -> Path:
    """Matriz de confusión en conteos (izquierda) y normalizada por clase real (derecha)."""
    plt = _plt()
    arr = np.asarray(cm, dtype=float)
    norm = arr / np.maximum(arr.sum(axis=1, keepdims=True), 1)
    size = max(5.0, 0.55 * len(labels) + 3)
    fig, axes = plt.subplots(1, 2, figsize=(size * 2, size))
    panels = ((axes[0], arr, "Conteos", "{:.0f}"), (axes[1], norm, "Recall", "{:.2f}"))
    for ax, data, title, fmt in panels:
        top = data.max() or 1
        ax.imshow(data, cmap="Blues", vmin=0, vmax=top)
        ax.set_title(title)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Predicha")
        ax.set_ylabel("Real")
        for i in range(len(labels)):
            for j in range(len(labels)):
                if data[i, j] > 0:
                    color = "white" if data[i, j] > 0.6 * top else "black"
                    ax.text(
                        j,
                        i,
                        fmt.format(data[i, j]),
                        ha="center",
                        va="center",
                        fontsize=7,
                        color=color,
                    )
    fig.tight_layout()
    return _save(fig, path)


def class_distribution(
    train_counts: Dict[str, int], test_counts: Dict[str, int], path: PathLike
) -> Path:
    plt = _plt()
    labels = sorted(set(train_counts) | set(test_counts))
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar(
        x - 0.2, [train_counts.get(k, 0) for k in labels], 0.4, label="raw (train/val)", color=_BLUE
    )
    ax.bar(x + 0.2, [test_counts.get(k, 0) for k in labels], 0.4, label="test", color="#f0a030")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Ficheros (escala log)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, path)


def per_class(report: Dict[str, Any], labels: List[str], path: PathLike) -> Path:
    plt = _plt()
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 3.8))
    bars = ((-0.27, "precision", _BLUE), (0.0, "recall", "#f0a030"), (0.27, "f1-score", "#3fa66b"))
    for off, key, color in bars:
        ax.bar(
            x + off, [report.get(k, {}).get(key, 0.0) for k in labels], 0.27, label=key, color=color
        )
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax.legend(ncol=3, loc="lower center")
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, path)


def roc_pr(y_true: np.ndarray, probs: np.ndarray, labels: List[str], path: PathLike) -> Path:
    from sklearn.metrics import precision_recall_curve, roc_curve

    plt = _plt()
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4))
    for k, lbl in enumerate(labels):
        pos = y_true == k
        if not pos.any() or pos.all():
            continue
        fpr, tpr, _ = roc_curve(pos, probs[:, k])
        prec, rec, _ = precision_recall_curve(pos, probs[:, k])
        a.plot(fpr, tpr, label=lbl, lw=1.2)
        b.plot(rec, prec, label=lbl, lw=1.2)
    a.plot([0, 1], [0, 1], color=_GREY, ls="--", lw=0.8)
    a.set(title="ROC (uno contra todos)", xlabel="FPR", ylabel="TPR")
    b.set(title="Precisión-Recall", xlabel="Recall", ylabel="Precisión")
    a.legend(fontsize=7, loc="lower right")
    for ax in (a, b):
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, path)


def confidence(
    probs: np.ndarray, y_true: np.ndarray, calibration: Dict[str, Any], path: PathLike
) -> Path:
    plt = _plt()
    conf = probs.max(axis=1)
    ok = probs.argmax(axis=1) == y_true
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8))
    bins = np.linspace(0, 1, 21)
    a.hist(conf[ok], bins=bins, alpha=0.8, label="acierto", color=_BLUE)
    if (~ok).any():
        a.hist(conf[~ok], bins=bins, alpha=0.8, label="error", color=_RED)
    a.set(title="Confianza de la predicción", xlabel="Confianza", ylabel="Segmentos")
    a.set_yscale("log")
    a.legend()
    rows = calibration.get("bins", [])
    b.plot([0, 1], [0, 1], color=_GREY, ls="--", lw=0.8)
    if rows:
        b.plot([r["confidence"] for r in rows], [r["accuracy"] for r in rows], "o-", color=_BLUE)
    b.set(
        title=f"Calibración (ECE={calibration.get('ece', 0):.3f})",
        xlabel="Confianza media",
        ylabel="Acierto",
    )
    for ax in (a, b):
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, path)


def optuna_overview(info: Dict[str, Any], path: PathLike) -> Optional[Path]:
    """Valor por trial con el mejor acumulado, e importancia de hiperparámetros."""
    done = [t for t in info.get("trials", []) if t.get("value") is not None]
    if not done:
        return None
    plt = _plt()
    maximize = info.get("direction", "MAXIMIZE") == "MAXIMIZE"
    values = [t["value"] for t in done]
    best = np.maximum.accumulate(values) if maximize else np.minimum.accumulate(values)
    imp = info.get("importances") or {}
    fig, axes = plt.subplots(1, 2 if imp else 1, figsize=(11 if imp else 6, 4), squeeze=False)
    a = axes[0][0]
    a.scatter([t["number"] for t in done], values, s=12, color=_GREY, label="trial")
    a.plot([t["number"] for t in done], best, color=_BLUE, lw=1.5, label="mejor acumulado")
    a.set(title="Historia de la búsqueda", xlabel="Trial", ylabel="Valor objetivo")
    a.legend()
    a.grid(alpha=0.3)
    if imp:
        top = list(imp.items())[:12][::-1]
        b = axes[0][1]
        b.barh([k for k, _ in top], [v for _, v in top], color=_BLUE)
        b.set(title="Importancia de hiperparámetros", xlabel="Importancia relativa")
        b.tick_params(axis="y", labelsize=7)
        b.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return _save(fig, path)
