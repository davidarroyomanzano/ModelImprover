"""Lógica del asistente (``ds_project.wizard``) y su pestaña en la UI."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest

from ds_project import services, wizard
from ds_project.config import load_settings
from ds_project.layout import init_project


def _wav(path: Path, n: int = 1) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (path / f"{i}.wav").write_bytes(b"RIFF")


@pytest.fixture()
def workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DS_CONFIG", str(tmp_path / "sin_config.yaml"))
    monkeypatch.delenv("DS_PROJECTS_DIR", raising=False)
    init_project("pinbolt", "g5x", "v1", tmp_path, "v150")
    (tmp_path / "templates").mkdir()
    (tmp_path / "templates" / "dsp_template.py").write_text("# plantilla", encoding="utf-8")
    return tmp_path


def _scope(workspace: Path) -> Path:
    return workspace / "projects" / "pinbolt" / "g5x" / "v1"


def _steps(workspace: Path):
    (project,) = services.list_projects("projects")
    checks = services.check_project("pinbolt", "g5x", "v1")
    return {s.key: s for s in wizard.wizard_steps(project, "v1", checks)}


def _make_run(scope: Path, name: str = "20260101_000000_abc", verdict=None) -> Path:
    run = scope / "models" / name
    run.mkdir(parents=True)
    for f in ("model.tflite", "model.keras", "dsp_params.json", "config_snapshot.yaml"):
        (run / f).write_bytes(b"modelo")
    (run / "metrics.json").write_text(json.dumps({"keras": {"accuracy": 0.9}}), encoding="utf-8")
    if verdict is not None:
        (run / "evaluation").mkdir()
        (run / "evaluation" / "metrics_test.json").write_text(
            json.dumps({"verdict": verdict, "segment": {"accuracy": 0.9}}), encoding="utf-8"
        )
    return run


def test_load_settings_overlay_replaces_project_yaml(workspace):
    settings = load_settings(product="pinbolt", turbine="g5x", overlay={"train": {"epochs": 3}})
    assert settings.train.epochs == 3
    with pytest.raises(ValueError):
        load_settings(product="pinbolt", turbine="g5x", overlay=["no"])  # type: ignore[arg-type]


def test_validate_overlay_rejects_unknown_keys(workspace):
    with pytest.raises(ValueError):
        wizard.validate_overlay("pinbolt", "g5x", {"train": {"nope": 1}})
    with pytest.raises(ValueError):
        wizard.validate_overlay("pinbolt", "g5x", {"train": {"scaler": "raro"}})


def test_set_override_sets_and_removes(workspace):
    base = wizard.base_config("pinbolt", "g5x")
    epochs = base["train"]["epochs"]
    overlay = wizard.set_override({}, base, "train.epochs", epochs + 1)
    assert overlay == {"train": {"epochs": epochs + 1}}
    assert wizard.set_override(overlay, base, "train.epochs", epochs) == {}


def test_overlay_diff_ignores_scope_keys(workspace):
    base = wizard.base_config("pinbolt", "g5x")
    effective = services.to_builtin(
        wizard.validate_overlay("pinbolt", "g5x", {"train": {"patience": 99}})
    )
    effective["data"]["version"] = "v9"
    assert wizard.overlay_diff(base, effective) == [
        {"clave": "train.patience", "base": str(base["train"]["patience"]), "proyecto": "99"}
    ]


def test_save_overlay_validates_and_backs_up(workspace):
    path = wizard.overlay_path("pinbolt", "g5x")
    original = path.read_text(encoding="utf-8")
    with pytest.raises(ValueError):
        wizard.save_overlay("pinbolt", "g5x", {"train": {"nope": 1}})
    assert path.read_text(encoding="utf-8") == original
    backup = wizard.save_overlay("pinbolt", "g5x", {"train": {"epochs": 7}})
    assert backup is not None and backup.parent.name == wizard.HISTORY_DIR
    assert backup.read_text(encoding="utf-8") == original
    assert wizard.read_overlay("pinbolt", "g5x") == {"train": {"epochs": 7}}
    assert path.read_text(encoding="utf-8").startswith("#")
    assert load_settings(product="pinbolt", turbine="g5x").train.epochs == 7


def test_data_checks():
    status = {c.name: c.status for c in wizard.data_checks({"A": 50}, {})}
    assert status == {"Clases": services.ERROR, "Test": services.ERROR}
    checks = wizard.data_checks({"A": 100, "B": 5}, {"A": 20, "B": 2})
    status = {c.name: c.status for c in checks}
    assert status["Equilibrio"] == services.WARN and status["Tamaño"] == services.WARN
    assert status["Test"] == services.OK


def test_exports_of_run_matches_by_content(workspace):
    run = _make_run(_scope(workspace))
    (info,) = services.list_runs(_scope(workspace) / "models")
    project = workspace / "projects" / "pinbolt" / "g5x"
    assert wizard.exports_of_run(project, info) == []
    shutil.copyfile(
        run / "model.tflite", project / "export" / "v150" / "nn_pinbolt_g5x-v150.tflite"
    )
    assert wizard.exports_of_run(project, info) == ["v150"]


def test_wizard_steps_progression(workspace):
    steps = _steps(workspace)
    assert steps["project"].status == wizard.DONE
    assert steps["data"].status == wizard.PENDING
    assert steps["config"].status == wizard.LOCKED
    assert wizard.current_step(list(steps.values())) == "data"

    scope = _scope(workspace)
    for cls in ("A", "B"):
        _wav(scope / "data" / "raw" / cls, 20)
        _wav(scope / "data" / "test" / cls, 4)
    steps = _steps(workspace)
    assert steps["data"].status == steps["config"].status == wizard.DONE
    assert steps["train"].status == wizard.PENDING
    assert steps["evaluate"].status == wizard.LOCKED

    run = _make_run(scope, verdict={"passed": False, "failed": ["accuracy < 0.95"]})
    future = wizard.config_mtime("pinbolt", "g5x") + 10
    os.utime(run / "metrics.json", (future, future))
    steps = _steps(workspace)
    assert steps["train"].status == wizard.DONE
    assert steps["evaluate"].status == wizard.ATTENTION
    assert steps["export"].status == wizard.PENDING
    assert steps["dashboard"].status == wizard.LOCKED
    assert wizard.current_step(list(steps.values())) == "evaluate"
    (run / "dashboard_x.html").write_text("<html></html>")
    assert [p.name for p in wizard.dashboards_of_run(run)] == ["dashboard_x.html"]
    (run / "dashboard_x.html").unlink()

    wizard.save_overlay("pinbolt", "g5x", {"train": {"epochs": 2}})
    os.utime(wizard.overlay_path("pinbolt", "g5x"), (future + 10, future + 10))
    assert _steps(workspace)["train"].status == wizard.ATTENTION


def test_wizard_steps_without_project():
    steps = wizard.wizard_steps(None, "", [])
    assert steps[0].status == wizard.PENDING
    assert all(s.status == wizard.LOCKED for s in steps[1:])
    assert wizard.current_step(steps) == "project"


def test_dsp_profile_helpers(workspace):
    repo = Path(__file__).resolve().parents[1]
    (workspace / "configs").mkdir()
    shutil.copyfile(
        repo / "configs" / "dsp_pinbolt.yaml", workspace / "configs" / "dsp_pinbolt.yaml"
    )
    overlay = wizard.with_dsp_profile({"train": {"epochs": 3}}, "pinbolt")
    assert overlay["train"] == {"epochs": 3} and overlay["dsp"]["profile"] == "pinbolt"
    assert "version" not in overlay["dsp"]
    settings = wizard.validate_overlay("pinbolt", "g5x", overlay)
    assert settings.dsp.profile == "pinbolt"
    assert wizard.with_dsp_profile(overlay, "default") == {"train": {"epochs": 3}}
    with pytest.raises(ValueError):
        wizard.with_dsp_profile({}, "raro")

    scope = _scope(workspace)
    for cls in ("A", "B"):
        _wav(scope / "data" / "raw" / cls, 20)
        _wav(scope / "data" / "test" / cls, 4)
    run = _make_run(scope, verdict={"passed": True, "failed": []})
    future = wizard.config_mtime("pinbolt", "g5x") + 10
    os.utime(run / "metrics.json", (future, future))
    assert wizard.run_dsp_profile(run) is None
    (run / "dsp_params.json").write_text(json.dumps({"dsp": {}}), encoding="utf-8")
    assert wizard.run_dsp_profile(run) == "default"
    assert _steps(workspace)["export"].status == wizard.ATTENTION
    (run / "dsp_params.json").write_text(
        json.dumps({"dsp": {"profile": "pinbolt"}}), encoding="utf-8"
    )
    assert _steps(workspace)["export"].status == wizard.PENDING


def test_optuna_progress(workspace):
    optuna = pytest.importorskip("optuna")
    assert wizard.optuna_progress("pinbolt", "g5x", "v1") is None
    settings = load_settings(product="pinbolt", turbine="g5x", data_version="v1")
    url = wizard.optuna_storage(settings)
    assert url.startswith("sqlite:///") and "pinbolt/g5x/v1" in url
    Path(url[len("sqlite:///") :]).parent.mkdir(parents=True, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    other = optuna.create_study(study_name="otro__x", storage=url, direction="maximize")
    other.optimize(lambda t: 5.0, n_trials=1)
    name = f"{settings.optuna.study_name}__{settings.project_name}__v1__abc"
    study = optuna.create_study(study_name=name, storage=url, direction="maximize")
    study.optimize(lambda t: t.suggest_float("x", 0, 1), n_trials=3)
    progress = wizard.optuna_progress("pinbolt", "g5x", "v1")
    assert progress["study_name"] == name
    assert progress["finished"] == 3 and progress["states"] == {"COMPLETE": 3}
    assert progress["best_value"] == study.best_value
    assert set(progress["best_params"]) == {"x"} and len(progress["values"]) == 3


def _step_radio(at):
    return next(r for r in at.radio if (r.key or "").startswith("wz-radio-"))


def test_ui_wizard_navigation(workspace, monkeypatch):
    testing = pytest.importorskip("streamlit.testing.v1")
    monkeypatch.setenv("DS_PROJECTS_DIR", str(workspace / "projects"))
    monkeypatch.setenv("DS_REPO_ROOT", str(workspace))
    app = Path(services.__file__).parent / "ui" / "app.py"
    at = testing.AppTest.from_file(str(app), default_timeout=30).run()
    assert not at.exception
    assert _step_radio(at).value == "data"
    _step_radio(at).set_value("config").run()
    assert not at.exception
    assert any("Completa antes" in i.value for i in at.info)
    _step_radio(at).set_value("project").run()
    assert not at.exception
    at.button(key="wz-next").click().run()
    assert _step_radio(at).value == "data"
    # Si cambia el estado de un paso (iconos), el paso elegido se mantiene.
    scope = _scope(workspace)
    for cls in ("A", "B"):
        _wav(scope / "data" / "raw" / cls, 20)
        _wav(scope / "data" / "test" / cls, 4)
    at.run()
    assert not at.exception
    assert _step_radio(at).value == "data"
    assert at.session_state["wz-step"] == "data"


def test_ui_wizard_config_saves_overlay(workspace, monkeypatch):
    testing = pytest.importorskip("streamlit.testing.v1")
    scope = _scope(workspace)
    for cls in ("A", "B"):
        _wav(scope / "data" / "raw" / cls, 20)
        _wav(scope / "data" / "test" / cls, 4)
    monkeypatch.setenv("DS_PROJECTS_DIR", str(workspace / "projects"))
    monkeypatch.setenv("DS_REPO_ROOT", str(workspace))
    app = Path(services.__file__).parent / "ui" / "app.py"
    at = testing.AppTest.from_file(str(app), default_timeout=30).run()
    _step_radio(at).set_value("config").run()
    assert not at.exception
    assert at.button(key="wz-save").disabled
    at.number_input(key="wz-cfg-pinbolt_g5x-ep").set_value(5).run()
    assert not at.exception
    assert any("sin guardar" in w.value for w in at.warning)
    at.button(key="wz-save").click().run()
    assert not at.exception
    assert wizard.read_overlay("pinbolt", "g5x") == {"train": {"epochs": 5}}
    assert any("project.yaml guardado" in s.value for s in at.success)


def test_ui_wizard_applies_dsp_profile(workspace, monkeypatch):
    testing = pytest.importorskip("streamlit.testing.v1")
    repo = Path(__file__).resolve().parents[1]
    (workspace / "configs").mkdir()
    shutil.copyfile(
        repo / "configs" / "dsp_pinbolt.yaml", workspace / "configs" / "dsp_pinbolt.yaml"
    )
    scope = _scope(workspace)
    for cls in ("A", "B"):
        _wav(scope / "data" / "raw" / cls, 20)
        _wav(scope / "data" / "test" / cls, 4)
    monkeypatch.setenv("DS_PROJECTS_DIR", str(workspace / "projects"))
    monkeypatch.setenv("DS_REPO_ROOT", str(workspace))
    app = Path(services.__file__).parent / "ui" / "app.py"
    at = testing.AppTest.from_file(str(app), default_timeout=30).run()
    _step_radio(at).set_value("config").run()
    assert any("no se podrá exportar" in w.value for w in at.warning)
    at.selectbox(key="wz-profile-pinbolt_g5x").set_value("pinbolt").run()
    at.button(key="wz-apply-profile").click().run()
    assert not at.exception
    assert wizard.read_overlay("pinbolt", "g5x")["dsp"]["profile"] == "pinbolt"
    assert not any("no se podrá exportar" in w.value for w in at.warning)
    assert not any("sin guardar" in w.value for w in at.warning)
    assert at.button(key="wz-save").disabled
