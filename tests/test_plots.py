from types import SimpleNamespace

import matplotlib.pyplot as plt
import pytest

from ds_project.visualization.plots import (
    plot_confusion_matrix,
    plot_optuna_history,
    plot_training_history,
)


def _ok(path):
    return path.exists() and path.stat().st_size > 0


def test_training_history(tmp_path):
    hist = {
        "loss": [1, 0.5],
        "val_loss": [1.1, 0.6],
        "accuracy": [0.5, 0.8],
        "val_accuracy": [0.4, 0.7],
        "lr": [1e-3, 1e-3],
    }
    assert _ok(plot_training_history(hist, tmp_path / "a" / "h.png"))


def test_confusion_matrix(tmp_path):
    assert _ok(plot_confusion_matrix([[3, 1], [0, 4]], ["a", "b"], tmp_path / "cm.png"))


def test_figures_are_closed(tmp_path):
    before = len(plt.get_fignums())
    plot_confusion_matrix([[1]], ["a"], tmp_path / "cm.png")
    plot_training_history({"loss": [1.0]}, tmp_path / "h.png")
    assert len(plt.get_fignums()) == before


@pytest.mark.parametrize("direction", ["MAXIMIZE", "MINIMIZE"])
@pytest.mark.parametrize("values", [[0.5, 0.7, 0.6], []])
def test_optuna_history(tmp_path, direction, values):
    TrialState = pytest.importorskip("optuna.trial").TrialState
    trials = [
        SimpleNamespace(state=TrialState.COMPLETE, value=v, number=i) for i, v in enumerate(values)
    ]
    trials.append(SimpleNamespace(state=TrialState.PRUNED, value=None, number=99))
    study = SimpleNamespace(trials=trials, direction=SimpleNamespace(name=direction))
    assert _ok(plot_optuna_history(study, tmp_path / "o.png"))
