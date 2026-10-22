"""Plantillas DSP por proyecto: comprobación, instalación, modos de entrenamiento y UI."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from ds_project import services, wizard
from ds_project.config import OptunaCfg, load_settings
from ds_project.dsp import templates as tpl
from ds_project.dsp.module_export import export_pinbolt_package
from ds_project.layout import init_project

REPO = Path(__file__).resolve().parents[1]
COMMON = REPO / "templates" / "dsp_template.py"
MATH_LINE = "mfe_db = 10.0 * np.log10("


def _mutated() -> bytes:
    text = COMMON.read_text(encoding="utf-8")
    assert MATH_LINE in text
    return text.replace(MATH_LINE, "mfe_db = 20.0 * np.log10(").encode("utf-8")


@pytest.fixture()
def workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DS_CONFIG", str(tmp_path / "sin_config.yaml"))
    monkeypatch.delenv("DS_PROJECTS_DIR", raising=False)
    init_project("pinbolt", "g5x", "v1", tmp_path)
    (tmp_path / "templates").mkdir()
    shutil.copyfile(COMMON, tmp_path / "templates" / "dsp_template.py")
    (tmp_path / "configs").mkdir()
    shutil.copyfile(
        REPO / "configs" / "dsp_pinbolt.yaml", tmp_path / "configs" / "dsp_pinbolt.yaml"
    )
    return tmp_path


def _project(workspace: Path) -> Path:
    return workspace / "projects" / "pinbolt" / "g5x"


def test_common_template_matches_training_dsp():
    report = tpl.inspect_template(COMMON)
    assert report.ok and report.parity, report
    c = report.constants
    assert c["window_step_sec"] == pytest.approx(0.3)
    assert c["sample_rate"] == 16000 and len(c["mfe_blocks"]) >= 1
    assert set(c["mfe_blocks"][0]) == {
        "frame_length",
        "frame_stride",
        "n_filters",
        "fft_length",
        "low_freq",
        "high_freq",
        "noise_floor_db",
    }
    # Las constantes se traducen a un DSP de entrenamiento válido.
    overrides = tpl.dsp_overrides(c)
    assert overrides["profile"] == "pinbolt" and overrides["aux_features"] == []


def test_broken_template_is_rejected(workspace):
    project = _project(workspace)
    with pytest.raises(tpl.TemplateError, match="importar"):
        tpl.install_template(b"raise RuntimeError('rota')\n", project)
    assert not tpl.target_path(project).exists()
    assert not list(tpl.target_path(project).parent.glob(".upload-*"))


def test_template_without_required_names_is_rejected():
    report = tpl.inspect_source(b"SAMPLE_RATE = 16000\n")
    assert not report.ok and "Faltan" in report.errors[0]


def test_modified_math_needs_force_and_keeps_history(workspace):
    project = _project(workspace)
    with pytest.raises(tpl.TemplateError, match="force"):
        tpl.install_template(_mutated(), project)
    assert not tpl.target_path(project).exists()

    first = tpl.install_template(_mutated(), project, force=True)
    assert first.report.ok and first.report.parity is False and first.backup is None
    second = tpl.install_template(COMMON, project, "v1")
    assert second.path == tpl.target_path(project, "v1") and second.report.parity
    third = tpl.install_template(COMMON, project)
    assert third.backup is not None and third.backup.read_bytes() == _mutated()
    assert third.backup.parent.name == tpl.HISTORY_DIR
    assert not list(project.rglob("__pycache__")), "la comprobación no debe dejar .pyc"


def test_template_slots_follow_export_order(workspace):
    project = _project(workspace)
    slots = tpl.template_slots(project, "v1", "templates")
    assert [s.level for s in slots] == [tpl.LEVEL_VERSION, tpl.LEVEL_PROJECT, tpl.LEVEL_COMMON]
    assert [s.active for s in slots] == [False, False, True]
    for level_dir in (project / "templates", project / "templates" / "v1"):
        level_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(COMMON, level_dir / tpl.TEMPLATE_NAME)
    slots = tpl.template_slots(project, "v1", "templates")
    assert [s.active for s in slots] == [True, False, False]
    assert all(s.exists for s in slots)
    backup = tpl.remove_template(project, "v1")
    assert backup.is_file() and not (project / "templates" / "v1" / tpl.TEMPLATE_NAME).exists()
    assert tpl.active_template(project, "v1", "templates") == tpl.target_path(project)
    with pytest.raises(FileNotFoundError):
        tpl.remove_template(project, "v1")


def test_read_window_step(tmp_path):
    path = tmp_path / "t.py"
    path.write_text("x = 1\nWINDOW_STEP_SEC = 0.45  # salto\n", encoding="utf-8")
    assert tpl.read_window_step(path) == 0.45
    path.write_text("WINDOW_STEP_SEC = calcula()\n", encoding="utf-8")
    assert tpl.read_window_step(path) is None
    assert tpl.read_window_step(tmp_path / "no.py") is None


def test_export_uses_template_window_step(tmp_path):
    template = tmp_path / "dsp_template.py"
    template.write_text(
        COMMON.read_text(encoding="utf-8").replace(
            "WINDOW_STEP_SEC = 0.3", "WINDOW_STEP_SEC = 0.45", 1
        ),
        encoding="utf-8",
    )
    assert tpl.read_window_step(template) == 0.45
    run = tmp_path / "run"
    run.mkdir()
    block = {
        "frame_length": 0.025,
        "frame_stride": 0.4,
        "n_filters": 8,
        "fft_length": 2048,
        "low_freq": 550,
        "high_freq": 3350,
        "noise_floor_db": -30,
    }
    artifact = {
        "dsp": {
            "sample_rate": 16000,
            "clip_duration": 0.9,
            "pre_emphasis": 0.98,
            "profile": "pinbolt",
            "zero_padding": True,
            "n_filters_final": 0,
            "mfe_blocks": [block],
        },
        "labels": ["a", "b"],
        "aux_scaler": {"kind": "standard", "offset": [0.0] * 25, "divisor": [1.0] * 25},
    }
    (run / "dsp_params.json").write_text(json.dumps(artifact), encoding="utf-8")
    (run / "model.tflite").write_bytes(b"tflite")
    out = export_pinbolt_package(run, tmp_path / "out", template, "demo", "v1")
    text = out["dsp"].read_text(encoding="utf-8")
    assert "WINDOW_STEP_SEC = 0.45" in text
    out = export_pinbolt_package(run, tmp_path / "out2", template, "demo", "v1", 0.6)
    assert "WINDOW_STEP_SEC = 0.6" in out["dsp"].read_text(encoding="utf-8")


def test_with_template_dsp_fixes_dsp_and_disables_dsp_search(workspace):
    report = tpl.inspect_template(COMMON)
    overlay = wizard.with_template_dsp({"train": {"epochs": 3}}, report.constants)
    assert overlay["train"] == {"epochs": 3}
    assert overlay["optuna"] == {"tune_dsp": False}
    settings = load_settings(product="pinbolt", turbine="g5x", overlay=overlay)
    assert settings.dsp.profile == "pinbolt"
    assert len(settings.dsp.mfe_blocks) == len(report.constants["mfe_blocks"])
    assert settings.dsp.pre_emphasis == pytest.approx(report.constants["pre_emphasis"])
    assert settings.optuna.active_dsp_space == {}
    assert settings.optuna.tune_dsp is False
    assert wizard.train_mode(services.to_builtin(settings)) == "search_net"


def test_train_modes():
    assert OptunaCfg().active_dsp_space == OptunaCfg().dsp_search_space
    assert OptunaCfg(tune_dsp=False).active_dsp_space == {}
    base = {"optuna": {"enabled": True, "tune_dsp": True}}
    assert wizard.train_mode(base) == "search_all"
    assert wizard.train_mode({"optuna": {"enabled": True, "tune_dsp": False}}) == "search_net"
    assert wizard.train_mode({"optuna": {"enabled": False, "tune_dsp": True}}) == "fixed"
    assert set(wizard.TRAIN_MODES) == set(wizard.TRAIN_MODE_LABELS)


def test_cli_install_applies_config(workspace, capsys):
    rc = tpl.main(
        ["install", str(COMMON), "--product", "pinbolt", "--turbine", "g5x", "--apply-config"]
    )
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "Instalada en" in out and tpl.target_path(_project(workspace)).is_file()
    overlay = wizard.read_overlay("pinbolt", "g5x")
    assert overlay["dsp"]["profile"] == "pinbolt"
    assert overlay["optuna"] == {"tune_dsp": False}
    assert tpl.main(["remove", "--product", "pinbolt", "--turbine", "g5x"]) == 0
    assert tpl.main(["check", str(COMMON)]) == 0


def test_cli_rejects_mutated_template(workspace, capsys):
    bad = workspace / "malo.py"
    bad.write_bytes(_mutated())
    assert tpl.main(["install", str(bad), "--product", "pinbolt", "--turbine", "g5x"]) == 1
    assert "force" in capsys.readouterr().err


def _step_radio(at):
    return next(r for r in at.radio if (r.key or "").startswith("wz-radio-"))


def _wav(path: Path, n: int) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (path / f"{i}.wav").write_bytes(b"RIFF")


def test_ui_installs_template_and_sets_fixed_mode(workspace, monkeypatch):
    testing = pytest.importorskip("streamlit.testing.v1")
    scope = _project(workspace) / "v1"
    for cls in ("A", "B"):
        _wav(scope / "data" / "raw" / cls, 20)
        _wav(scope / "data" / "test" / cls, 4)
    monkeypatch.setenv("DS_PROJECTS_DIR", str(workspace / "projects"))
    monkeypatch.setenv("DS_REPO_ROOT", str(workspace))
    app = Path(services.__file__).parent / "ui" / "app.py"
    at = testing.AppTest.from_file(str(app), default_timeout=180).run()
    _step_radio(at).set_value("config").run()
    assert not at.exception
    at.radio(key="wz-tpl-mode").set_value("Ruta local").run()
    at.text_input(key="wz-tpl-path").input(f'"{COMMON}"').run()
    assert not at.exception
    assert any("Reproduce el DSP" in s.value for s in at.success)
    at.radio(key="wz-tpl-level").set_value("Solo la versión de datos v1").run()
    at.button(key="wz-tpl-install").click().run()
    assert not at.exception
    assert tpl.target_path(_project(workspace), "v1").is_file()
    overlay = wizard.read_overlay("pinbolt", "g5x")
    assert overlay["dsp"]["profile"] == "pinbolt" and overlay["optuna"]["tune_dsp"] is False
    assert any("Plantilla instalada" in s.value for s in at.success)

    # Modo sin Optuna: aparecen los hiperparámetros fijos de la red y se guardan.
    at.radio(key="wz-cfg-pinbolt_g5x-mode").set_value("fixed").run()
    assert not at.exception
    at.number_input(key="wz-cfg-pinbolt_g5x-mp-units").set_value(48).run()
    at.button(key="wz-save").click().run()
    assert not at.exception
    overlay = wizard.read_overlay("pinbolt", "g5x")
    assert overlay["optuna"]["enabled"] is False
    assert overlay["train"]["model_params"] == {"units": 48}
    settings = load_settings(product="pinbolt", turbine="g5x")
    assert settings.train.model_params == {"units": 48}
