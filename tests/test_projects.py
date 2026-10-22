"""Proyectos y versiones: config por proyecto, rutas, manifiesto, estudio Optuna y export."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from ds_project import pipeline
from ds_project.config import DataCfg, PathsCfg, Settings, load_settings
from ds_project.data.manifest import build_manifest, diff_manifests
from ds_project.dsp.module_export import (
    export_for_project,
    export_pinbolt_package,
    find_latest_run,
    resolve_template,
)
from ds_project.models import tuning
from tests.test_module_export import TEMPLATE, _make_run


def _w(path: Path, data: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


# ---------- config ----------
def test_resolved_paths_without_project_is_identity():
    s = Settings()
    assert s.resolved_paths() is s.paths


def test_resolved_paths_with_project():
    s = Settings(data=DataCfg(product="pinbolt", turbine="g5x"))
    p = s.resolved_paths()
    scope = Path("projects/pinbolt/g5x") / s.data.version
    assert Path(p.raw_dir) == scope / "data/raw"
    assert Path(p.test_dir) == scope / "data/test"
    assert Path(p.models_dir) == scope / "models"
    assert s.project_name == "pinbolt_g5x"


def test_product_without_turbine_defaults_to_all():
    s = Settings(data=DataCfg(product="pinbolt"))
    assert s.data.turbine == "all" and s.project_name == "pinbolt_all"


def test_resolved_paths_with_project_and_version():
    s = Settings(data=DataCfg(product="pinbolt", turbine="g5x", version="v3"))
    p = s.resolved_paths()
    assert Path(p.raw_dir) == Path("projects/pinbolt/g5x/v3/data/raw")
    assert Path(p.models_dir) == Path("projects/pinbolt/g5x/v3/models")


def test_init_project_creates_layout(tmp_path, monkeypatch):
    from ds_project.layout import init_project

    monkeypatch.setenv("DS_CONFIG", str(tmp_path / "sin_config.yaml"))
    created = init_project("pinbolt", "g5x", "v3", tmp_path, "v150")
    base = "projects/pinbolt/g5x"
    for rel in (
        f"{base}/v3/data/raw",
        f"{base}/v3/data/test",
        f"{base}/v3/models",
        f"{base}/export/v150",
        f"{base}/templates",
        f"{base}/project.yaml",
    ):
        assert (tmp_path / rel).exists(), rel
    assert init_project("pinbolt", "g5x", "v3", tmp_path, "v150") == []
    assert created


@pytest.mark.parametrize("bad", ["../x", "a/b", "a b", "-x", "A", "g_5x"])
def test_product_and_turbine_are_validated(bad):
    with pytest.raises(ValueError):
        DataCfg(product=bad)
    with pytest.raises(ValueError):
        DataCfg(product="pinbolt", turbine=bad)


@pytest.mark.parametrize("bad", ["../x", "a/b", "a b", "-x"])
def test_version_is_validated(bad):
    with pytest.raises(ValueError):
        DataCfg(version=bad)


def test_project_overlay_and_precedence(tmp_path, monkeypatch):
    base = _w(tmp_path / "c.yaml", {"data": {"seed": 1, "version": "base"}})
    opt = _w(tmp_path / "o.yaml", {"n_trials": 10, "seed": 3})
    dsp = _w(tmp_path / "d.yaml", {"sample_rate": 8000})
    _w(
        tmp_path / "projects" / "demo" / "g5x" / "project.yaml",
        {"data": {"seed": 7}, "optuna": {"n_trials": 99}, "dsp": {"sample_rate": 16000}},
    )
    monkeypatch.setenv("DS_PROJECTS_DIR", str(tmp_path / "projects"))
    s = load_settings(base, opt, dsp, product="demo", turbine="g5x", data_version="v3")
    assert (s.data.product, s.data.turbine, s.data.version) == ("demo", "g5x", "v3")
    assert s.data.seed == 7
    assert (s.optuna.n_trials, s.optuna.seed) == (99, 3)
    assert s.dsp.sample_rate == 16000


def test_project_from_env_and_missing_overlay(tmp_path, monkeypatch):
    monkeypatch.setenv("DS_PRODUCT", "nuevo")
    monkeypatch.setenv("DS_TURBINE", "n155")
    monkeypatch.setenv("DS_DATA_VERSION", "v9")
    monkeypatch.setenv("DS_PROJECTS_DIR", str(tmp_path / "nada"))
    s = load_settings(tmp_path / "a.yaml", tmp_path / "b.yaml", tmp_path / "c.yaml")
    assert (s.data.product, s.data.turbine, s.data.version) == ("nuevo", "n155", "v9")


def test_project_overlay_unknown_section(tmp_path, monkeypatch):
    _w(tmp_path / "projects" / "demo" / "all" / "project.yaml", {"otra": {}})
    monkeypatch.setenv("DS_PROJECTS_DIR", str(tmp_path / "projects"))
    with pytest.raises(ValueError, match="demo"):
        load_settings(tmp_path / "a", tmp_path / "b", tmp_path / "c", product="demo")


# ---------- manifiesto ----------
def _files(tmp_path: Path) -> dict:
    out = {}
    for label, n in (("a", 2), ("b", 1)):
        d = tmp_path / label
        d.mkdir()
        out[label] = []
        for i in range(n):
            f = d / f"{i}.wav"
            f.write_bytes(f"{label}{i}".encode())
            out[label].append(f)
    return out


def test_manifest_is_deterministic_and_cached(tmp_path):
    files = _files(tmp_path)
    cache = tmp_path / "cache.json"
    m1 = build_manifest(files, tmp_path, cache)
    assert cache.exists()
    m2 = build_manifest(files, tmp_path, cache)
    assert m1 == m2 and m1["n_files"] == 3 and m1["per_class"] == {"a": 2, "b": 1}
    assert m1["files"][0]["path"] == "a/0.wav"


def test_manifest_hash_changes_with_data(tmp_path):
    files = _files(tmp_path)
    old = build_manifest(files, tmp_path)
    (tmp_path / "a" / "0.wav").write_bytes(b"cambiado")
    new_file = tmp_path / "b" / "9.wav"
    new_file.write_bytes(b"nuevo")
    files["b"].append(new_file)
    new = build_manifest(files, tmp_path)
    assert old["dataset_hash"] != new["dataset_hash"]
    assert diff_manifests(old, new) == {
        "added": ["b/9.wav"],
        "removed": [],
        "changed": ["a/0.wav"],
    }


def test_manifest_tolerates_missing_files(tmp_path):
    m = build_manifest({"a": [Path("no_existe.wav")]}, tmp_path)
    assert m["files"][0]["size"] == -1


# ---------- pipeline / optuna ----------
def test_study_name_reflects_project_version_and_hash():
    s = Settings(data=DataCfg(product="pinbolt", turbine="g5x", version="v2"))
    assert pipeline.make_study_name(s, "abc") == f"{s.optuna.study_name}__pinbolt_g5x__v2__abc"
    assert pipeline.make_study_name(Settings(), "abc").endswith("__default__nover__abc")


def test_create_study_stores_user_attrs(monkeypatch):
    opt = MagicMock()
    monkeypatch.setitem(__import__("sys").modules, "optuna", opt)
    monkeypatch.setattr(tuning, "_make_storage", lambda url: url)
    study = tuning.create_study(Settings().optuna, {"product": "pinbolt", "dataset_hash": "h"})
    study.set_user_attr.assert_any_call("product", "pinbolt")
    study.set_user_attr.assert_any_call("dataset_hash", "h")


def test_run_pipeline_scopes_paths_study_and_hash(make_settings, tmp_path, monkeypatch):
    from tests.test_pipeline import _data

    manifest = {"dataset_hash": "d1", "n_files": 2, "per_class": {}, "files": []}

    def settings(turbine, tune=False):
        paths = PathsCfg(projects_dir=str(tmp_path / "projects"))
        data = DataCfg(product="pinbolt", turbine=turbine, version="v1")
        return make_settings(paths=paths, data=data)

    _stub_heavy(monkeypatch, {**_data(), "manifest": manifest})
    run_a = pipeline.run_pipeline(settings("g5x"), tune=False)
    run_b = pipeline.run_pipeline(settings("g114"), tune=False)
    assert run_a.parent == tmp_path / "projects" / "pinbolt" / "g5x" / "v1" / "models"
    assert run_b.parent == tmp_path / "projects" / "pinbolt" / "g114" / "v1" / "models"
    assert json.loads((run_a / "dsp_params.json").read_text())["extra"] == {
        "run_id": run_a.name,
        "quantization": "float16",
        "product": "pinbolt",
        "turbine": "g5x",
        "data_version": "v1",
        "dataset_hash": "d1",
    }
    assert (run_a / "dataset_manifest.json").exists()

    manifest["dataset_hash"] = "d2"
    run_c = pipeline.run_pipeline(settings("g5x"), tune=False)
    assert run_c != run_a and run_c.parent == run_a.parent


def test_run_pipeline_tuning_uses_scoped_study(make_settings, monkeypatch):
    from tests.test_pipeline import _data

    _stub_heavy(monkeypatch, {**_data(), "manifest": {"dataset_hash": "d1"}})
    study = MagicMock(best_params={"units": 16}, best_value=0.9, trials=[1])
    run_study = MagicMock(return_value=study)
    monkeypatch.setattr("ds_project.models.tuning.run_study", run_study)
    s = make_settings(data=DataCfg(product="pinbolt", turbine="g5x", version="v4"))
    pipeline.run_pipeline(s, tune=True)
    used, attrs = run_study.call_args.args[1], run_study.call_args.args[2]
    assert used.optuna.study_name.split("__")[1:3] == ["pinbolt_g5x", "v4"]
    assert attrs["product"] == "pinbolt" and attrs["dataset_hash"] == "d1"


def _stub_heavy(monkeypatch, data):
    model = MagicMock()
    metrics = {"accuracy": 1.0, "confusion_matrix": [[2, 0], [0, 2]]}
    monkeypatch.setattr(pipeline, "prepare_dataset", lambda s: dict(data))
    monkeypatch.setattr(
        "ds_project.models.train.train_model",
        MagicMock(return_value=(model, {"loss": [1.0], "val_loss": [0.9]})),
    )
    monkeypatch.setattr("ds_project.models.train.set_global_seed", MagicMock())
    monkeypatch.setattr(
        "ds_project.models.export.export_tflite",
        MagicMock(side_effect=lambda m, path, *a, **k: Path(path)),
    )
    monkeypatch.setattr(
        "ds_project.models.evaluate.evaluate_keras", MagicMock(return_value=metrics)
    )
    monkeypatch.setattr(
        "ds_project.models.evaluate.evaluate_tflite", MagicMock(return_value=metrics)
    )
    for name in ("plot_training_history", "plot_confusion_matrix", "plot_optuna_history"):
        monkeypatch.setattr(f"ds_project.visualization.plots.{name}", MagicMock())


# ---------- export por proyecto ----------
def test_resolve_template_prefers_project_and_version(tmp_path):
    common = tmp_path / "dsp_template.py"
    common.write_text("c")
    project = tmp_path / "projects" / "pinbolt" / "g5x"
    assert resolve_template(tmp_path, project, "v1") == common
    proj = project / "templates" / "dsp_template.py"
    proj.parent.mkdir(parents=True)
    proj.write_text("p")
    assert resolve_template(tmp_path, project, "v1") == proj
    ver = project / "templates" / "v1" / "dsp_template.py"
    ver.parent.mkdir()
    ver.write_text("v")
    assert resolve_template(tmp_path, project, "v1") == ver
    assert resolve_template(tmp_path, project, "v2") == proj
    assert resolve_template(tmp_path, tmp_path / "projects" / "pinbolt" / "n155", "v1") == common
    with pytest.raises(FileNotFoundError):
        resolve_template(tmp_path / "vacio", tmp_path / "projects" / "otro" / "x", "v1")


def test_find_latest_run_filters_by_data_version(tmp_path):
    for name, version in (("r1", "v1"), ("r2", "v2")):
        run = tmp_path / name
        run.mkdir()
        (run / "dsp_params.json").write_text(
            json.dumps({"extra": {"data_version": version}}), encoding="utf-8"
        )
    assert find_latest_run(tmp_path, "v1") == tmp_path / "r1"
    with pytest.raises(FileNotFoundError, match="v3"):
        find_latest_run(tmp_path, "v3")


def test_export_copies_dataset_manifest(tmp_path):
    run = _make_run(tmp_path)
    (run / "dataset_manifest.json").write_text('{"dataset_hash": "x"}', encoding="utf-8")
    out = export_pinbolt_package(run, tmp_path / "out", TEMPLATE, "demo", "v1")
    assert out["manifest"].name == "dataset_manifest_demo-v1.json"
    assert json.loads(out["manifest"].read_text())["dataset_hash"] == "x"


def test_export_for_project_layout(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DS_CONFIG", str(tmp_path / "sin_config.yaml"))
    project = tmp_path / "projects" / "pinbolt" / "g5x"
    run = project / "v1" / "models" / "run"
    run.parent.mkdir(parents=True)
    _make_run(tmp_path).rename(run)
    params = json.loads((run / "dsp_params.json").read_text(encoding="utf-8"))
    params.setdefault("extra", {})["data_version"] = "v1"
    (run / "dsp_params.json").write_text(json.dumps(params), encoding="utf-8")
    tpl = project / "templates" / "dsp_template.py"
    tpl.parent.mkdir(parents=True)
    tpl.write_text(TEMPLATE.read_text(encoding="utf-8"))
    out = export_for_project("pinbolt", "g5x", "v7", data_version="v1", templates_dir=tmp_path)
    assert out["dsp"] == Path("projects/pinbolt/g5x/export/v7/dsp_pinbolt_g5x-v7.py")
    assert out["tflite"].name == "nn_pinbolt_g5x-v7.tflite"


def test_export_for_project_with_data_version(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DS_CONFIG", str(tmp_path / "sin_config.yaml"))
    project = tmp_path / "projects" / "pinbolt" / "g5x"
    run = project / "v2" / "models" / "run"
    run.parent.mkdir(parents=True)
    _make_run(tmp_path).rename(run)
    params = json.loads((run / "dsp_params.json").read_text(encoding="utf-8"))
    params.setdefault("extra", {})["data_version"] = "v2"
    (run / "dsp_params.json").write_text(json.dumps(params), encoding="utf-8")
    tpl = project / "templates" / "v2" / "dsp_template.py"
    tpl.parent.mkdir(parents=True)
    tpl.write_text(TEMPLATE.read_text(encoding="utf-8"))
    out = export_for_project(
        "pinbolt", "g5x", "v7", data_version="v2", templates_dir=tmp_path / "templates"
    )
    assert out["dsp"] == Path("projects/pinbolt/g5x/export/v7/dsp_pinbolt_g5x-v7.py")


@pytest.mark.parametrize("bad", ["150", "v1.5", "final", "V150"])
def test_export_for_project_rejects_invalid_version(bad):
    with pytest.raises(ValueError, match="v<número>"):
        export_for_project("pinbolt", "g5x", bad)


def test_scoped_storage():
    s = Settings(data=DataCfg(product="pinbolt", turbine="g5x", version="v2"))
    assert pipeline.scoped_storage(s) == "sqlite:///projects/pinbolt/g5x/v2/models/optuna.db"
    assert pipeline.scoped_storage(Settings()) == Settings().optuna.storage
    pg = Settings(
        data=DataCfg(product="pinbolt"),
        optuna=replace(Settings().optuna, storage="postgresql://x"),
    )
    assert pipeline.scoped_storage(pg) == "postgresql://x"
