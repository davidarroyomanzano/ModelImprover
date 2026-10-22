"""Gráficas: historial de entrenamiento, matriz de confusión e historial de Optuna."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PathLike = Union[str, Path]


def _save(fig, path: PathLike) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


def plot_training_history(history: Dict[str, List[float]], path: PathLike) -> Path:
    metrics = sorted({k.replace("val_", "") for k in history if k != "lr"})
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4), squeeze=False)
    for ax, m in zip(axes[0], metrics):
        if m in history:
            ax.plot(history[m], label=f"train {m}")
        if f"val_{m}" in history:
            ax.plot(history[f"val_{m}"], label=f"val {m}")
        ax.set_title(m)
        ax.set_xlabel("época")
        ax.legend()
    return _save(fig, path)


def plot_confusion_matrix(
    cm: Sequence[Sequence[int]], labels: Sequence[str], path: PathLike
) -> Path:
    cm_arr = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(max(4, len(labels)), max(4, len(labels))))
    im = ax.imshow(cm_arr, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    thresh = cm_arr.max() / 2 if cm_arr.size else 0
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            ax.text(
                j,
                i,
                int(cm_arr[i, j]),
                ha="center",
                va="center",
                color="white" if cm_arr[i, j] > thresh else "black",
            )
    return _save(fig, path)


def plot_optuna_history(study, path: PathLike) -> Path:
    from optuna.trial import TrialState

    trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    values = [t.value for t in trials]
    numbers = [t.number for t in trials]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(numbers, values, label="trial", alpha=0.7)
    if values:
        best = (
            np.maximum.accumulate(values)
            if study.direction.name == "MAXIMIZE"
            else np.minimum.accumulate(values)
        )
        ax.plot(numbers, best, color="red", label="mejor")
    ax.set_xlabel("trial")
    ax.set_ylabel("objetivo")
    ax.legend()
    return _save(fig, path)
