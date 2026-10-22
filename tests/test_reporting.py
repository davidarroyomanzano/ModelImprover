"""Informe de evaluación: documento (md/html/pdf), resumen de métricas, veredicto y Optuna."""

from __future__ import annotations

import numpy as np
import pytest

from ds_project.reporting.build import verdict
from ds_project.reporting.document import Document, write_all
from ds_project.reporting.evaluation import Evaluation, misclassified, summarize
from ds_project.reporting.optuna_info import load_study_info, save_study_info, study_to_dict


def _evaluation(tmp_path) -> Evaluation:
    labels = ["a", "b"]
    seg_file = np.array([0, 1, 2, 2, 3, 3])
    file_labels = [0, 0, 1, 1]
    probs = np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.2, 0.8], [0.4, 0.6], [0.3, 0.7]])
    y_true = np.asarray(file_labels)[seg_file]
    return Evaluation(
        tmp_path, labels, ["f0", "f1", "f2", "f3"], file_labels, {"a": 2.0, "b": 3.0},
        seg_file, y_true, probs,
    )  # fmt: skip


def test_document_writes_three_formats(tmp_path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([0, 1])
    png = tmp_path / "x.png"
    fig.savefig(png)
    plt.close(fig)
    doc = Document("Título", "sub")
    doc.h1("Sección")
    doc.p("Texto **negrita** y `código`")
    doc.bullets(["uno", "dos"])
    doc.table(["A", "B"], [[1, 2.5]], "tabla")
    doc.image(png, "figura")
    files = write_all(doc, tmp_path / "out")
    assert set(files) == {"markdown", "html", "pdf"}
    assert files["pdf"].read_bytes().startswith(b"%PDF")
    html = files["html"].read_text(encoding="utf-8")
    assert "<b>negrita</b>" in html and "data:image/png;base64" in html
    assert "| A | B |" in files["markdown"].read_text(encoding="utf-8")


def test_summarize_file_and_segment_levels(tmp_path) -> None:
    summary = summarize(_evaluation(tmp_path), n_boot=20)
    assert summary["n_files"] == 4 and summary["n_segments"] == 6
    assert summary["segment"]["accuracy"] == pytest.approx(5 / 6)
    assert summary["file"]["accuracy"] == pytest.approx(1.0)
    assert summary["files_by_class"] == {"a": 2, "b": 2}
    assert set(summary["bootstrap_ci_95"]) == {"accuracy", "f1_macro"}


def test_misclassified_lists_wrong_segments(tmp_path) -> None:
    rows = misclassified(_evaluation(tmp_path))
    assert [(r["file"], r["pred"]) for r in rows] == [("f2", "a")]


def test_verdict_thresholds(tmp_path) -> None:
    summary = summarize(_evaluation(tmp_path), n_boot=0)
    assert verdict(summary)["informative"]
    assert verdict(summary, min_accuracy=0.8)["passed"]
    failed = verdict(summary, min_accuracy=0.9, min_f1=0.5)
    assert not failed["passed"] and len(failed["failed"]) == 1


def test_optuna_trials_are_frozen_in_run(tmp_path) -> None:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", study_name="s")
    study.optimize(lambda t: t.suggest_float("x", 0, 1), n_trials=4)
    info = study_to_dict(study)
    assert info["direction"] == "MAXIMIZE" and len(info["trials"]) == 4
    save_study_info(study, tmp_path)
    assert load_study_info(tmp_path)["study_name"] == "s"


def test_load_study_info_without_storage_is_none(tmp_path) -> None:
    assert load_study_info(tmp_path) is None
