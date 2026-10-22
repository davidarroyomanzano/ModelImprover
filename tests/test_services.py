"""Consultas de solo lectura (services) y humo de la UI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ds_project import services
from ds_project.layout import init_project


def _wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"RIFF")


@pytest.fixture()
def workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DS_CONFIG", str(tmp_path / "sin_config.yaml"))
    monkeypatch.delenv("DS_PROJECTS_DIR", raising=False)
    init_project("pinbolt", "g5x", "v1", tmp_path, "v150")
    (tmp_path / "templates").mkdir()
    (tmp_path / "templates" / "dsp_template.py").write_text("# plantilla", encoding="utf-8")
    return tmp_path


def _status(checks, name):
    return next(c for c in checks if c.name == name)


def test_list_projects(workspace):
    (projects,) = services.list_projects("projects")
    assert projects.name == "pinbolt_g5x"
    assert projects.data_versions == ["v1"]
    assert projects.export_versions == ["v150"]
    assert projects.has_overlay and projects.has_templates


def test_list_projects_missing_dir(tmp_path):
    assert services.list_projects(tmp_path / "nada") == []


def test_list_runs_reads_metrics_and_orders_latest_first(tmp_path):
    models = tmp_path / "models"
    for name, acc in (("20260101_000000_aaa", 0.5), ("20260202_000000_bbb", 0.9)):
        run = models / name
        run.mkdir(parents=True)
        (run / "metrics.json").write_text(
            json.dumps({"keras": {"accuracy": acc, "f1_macro": 0.8}}), encoding="utf-8"
        )
    (models / "20260202_000000_bbb" / "evaluation").mkdir()
    (models / "20260202_000000_bbb" / "evaluation" / "report.md").write_text("x", encoding="utf-8")
    (models / "optuna.db").write_text("", encoding="utf-8")
    (models / "20260202_000000_bbb" / "search_config.json").write_text(
        json.dumps({"mode": "two_phase"}), encoding="utf-8"
    )
    runs = services.list_runs(models)
    assert [r.name for r in runs] == ["20260202_000000_bbb", "20260101_000000_aaa"]
    assert runs[0].accuracy == 0.9 and runs[0].has_evaluation and not runs[1].has_evaluation
    assert runs[0].search == {"mode": "two_phase"} and runs[1].search == {}
    assert "model.tflite" in runs[0].missing


def test_list_runs_tolerates_corrupt_metrics(tmp_path):
    run = tmp_path / "models" / "r1"
    run.mkdir(parents=True)
    (run / "metrics.json").write_text("{no json", encoding="utf-8")
    assert services.list_runs(tmp_path / "models")[0].accuracy is None


def test_count_by_class(tmp_path):
    _wav(tmp_path / "A" / "1.wav")
    _wav(tmp_path / "A" / "2.WAV")
    _wav(tmp_path / "B" / "x.txt")
    assert services.count_by_class(tmp_path) == {"A": 2, "B": 0}


def test_check_project_reports_missing_data_and_runs(workspace):
    checks = services.check_project("pinbolt", "g5x", "v1")
    assert _status(checks, "Datos raw").status == services.ERROR
    assert _status(checks, "Datos test").status == services.WARN
    assert _status(checks, "Plantilla DSP").status == services.OK
    assert _status(checks, "Runs").status == services.WARN
    assert services.summarize(checks) == services.ERROR


def test_check_project_ok_and_class_mismatch(workspace):
    data = workspace / "projects" / "pinbolt" / "g5x" / "v1" / "data"
    _wav(data / "raw" / "Good" / "a.wav")
    _wav(data / "test" / "Good" / "b.wav")
    checks = services.check_project("pinbolt", "g5x", "v1")
    assert _status(checks, "Datos raw").status == services.OK
    assert _status(checks, "Clases test").status == services.OK
    _wav(data / "test" / "Bad" / "c.wav")
    checks = services.check_project("pinbolt", "g5x", "v1")
    assert _status(checks, "Clases test").status == services.ERROR


def test_check_project_unknown_and_invalid(workspace):
    assert services.check_project("otro", "all", "v1")[-1].status == services.ERROR
    assert services.check_project("Bad_Name", "all")[0].status == services.ERROR


def test_check_project_without_template(workspace):
    (workspace / "templates" / "dsp_template.py").unlink()
    checks = services.check_project("pinbolt", "g5x", "v1")
    assert _status(checks, "Plantilla DSP").status == services.ERROR


def test_safe_join_blocks_traversal(tmp_path):
    (tmp_path / "a").mkdir()
    assert services.safe_join(tmp_path, "a") == (tmp_path / "a").resolve()
    for bad in ("..", "a/../..", str(tmp_path.parent)):
        with pytest.raises(ValueError):
            services.safe_join(tmp_path, bad)


def test_list_dir_sorts_dirs_first(tmp_path):
    (tmp_path / "b").mkdir()
    (tmp_path / "a.txt").write_text("hola", encoding="utf-8")
    rows = services.list_dir(tmp_path)
    assert [(r["name"], r["is_dir"]) for r in rows] == [("b", True), ("a.txt", False)]
    assert rows[1]["size"] == 4
    assert services.list_dir(tmp_path, "no-existe") == []


def test_ui_smoke(workspace, monkeypatch):
    testing = pytest.importorskip("streamlit.testing.v1")
    monkeypatch.setenv("DS_PROJECTS_DIR", str(workspace / "projects"))
    monkeypatch.setenv("DS_REPO_ROOT", str(workspace))
    app = Path(services.__file__).parent / "ui" / "app.py"
    at = testing.AppTest.from_file(str(app), default_timeout=30).run()
    assert not at.exception
    assert at.title[0].value == "ModelosPinbolt"
    assert [t.label for t in at.tabs] == [
        "Asistente",
        "Estado",
        "Ejecutar",
        "Datos",
        "Runs",
        "Exports",
        "Archivos",
    ]
    commands = [c.value for c in at.code if c.value.startswith("poetry run invoke")]
    assert commands == [
        "poetry run invoke init-project --product pinbolt --turbine g5x --data-version v1"
    ]
    at.radio(key="action").set_value("Exportar").run()
    assert not at.exception
    assert any(
        "export --product pinbolt" in c.value and "--version v151" in c.value for c in at.code
    )


def test_ui_without_projects_offers_init(tmp_path, monkeypatch):
    testing = pytest.importorskip("streamlit.testing.v1")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DS_PROJECTS_DIR", str(tmp_path / "vacio"))
    monkeypatch.setenv("DS_REPO_ROOT", str(tmp_path))
    app = Path(services.__file__).parent / "ui" / "app.py"
    at = testing.AppTest.from_file(str(app), default_timeout=30).run()
    assert not at.exception
    assert at.warning and "No hay proyectos" in at.warning[0].value
    assert at.error[0].value == "Falta el producto"
    assert at.button[0].disabled
