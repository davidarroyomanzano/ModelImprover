"""Importación de un .zip de datos (``ds_project.data.importer``) y su panel en la UI."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from ds_project import services
from ds_project.data import importer
from ds_project.layout import init_project

WAV = b"RIFF0000WAVE"


def _zip_bytes(entries: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return buf.getvalue()


def _dataset_zip(path: Path, root: str = "N155_OUT/", nested: bool = True) -> Path:
    entries = {
        f"{root}impact_extractor_v2_manifest1.csv": "source_file,output_file\n",
        f"{root}training/Good/a.wav": WAV,
        f"{root}training/Good/b.wav": WAV,
        f"{root}training/Impacts_hub/c.wav": WAV,
        f"{root}testing/Good/d.wav": WAV,
        f"{root}testing/Impacts_hub/e.wav": WAV,
        "__MACOSX/._a.wav": b"x",
    }
    if nested:
        entries[f"{root}training/Tendon_rupture_synthetic.zip"] = _zip_bytes(
            {
                "README.txt": "sintético",
                "synthetic_manifest.csv": "x\n",
                "training/Tendon_rupture/s1.wav": WAV,
                "training/Tendon_rupture/s2.wav": WAV,
            }
        )
    path.write_bytes(_zip_bytes(entries))
    return path


def _files(base: Path) -> set:
    return {p.relative_to(base).as_posix() for p in base.rglob("*") if p.is_file()}


def test_plan_detects_root_and_splits(tmp_path):
    with zipfile.ZipFile(_dataset_zip(tmp_path / "d.zip")) as zf:
        plan = importer.plan_zip(zf)
    assert plan.root == "N155_OUT"
    assert plan.counts() == {
        "raw": {"Good": 2, "Impacts_hub": 1},
        "test": {"Good": 1, "Impacts_hub": 1},
    }
    assert [d for _, d in plan.extras] == ["impact_extractor_v2_manifest1.csv"]
    assert plan.nested == ["N155_OUT/training/Tendon_rupture_synthetic.zip"]


@pytest.mark.parametrize("root", ["", "N155_OUT/"])
def test_import_maps_training_testing_and_manifests(tmp_path, root):
    data = tmp_path / "data"
    result = importer.import_zip(_dataset_zip(tmp_path / "d.zip", root), data)
    assert _files(data) == {
        "raw/Good/a.wav",
        "raw/Good/b.wav",
        "raw/Impacts_hub/c.wav",
        "test/Good/d.wav",
        "test/Impacts_hub/e.wav",
        "impact_extractor_v2_manifest1.csv",
        "Tendon_rupture_synthetic.zip",
    }
    assert result.counts["raw"] == {"Good": 2, "Impacts_hub": 1}
    assert not result.replaced and not list(data.glob(".import-*"))


def test_import_extracts_nested_zip(tmp_path):
    data = tmp_path / "data"
    result = importer.import_zip(_dataset_zip(tmp_path / "d.zip"), data, extract_nested=True)
    files = _files(data)
    assert {"raw/Tendon_rupture/s1.wav", "raw/Tendon_rupture/s2.wav"} <= files
    assert {
        "Tendon_rupture_synthetic/README.txt",
        "Tendon_rupture_synthetic/synthetic_manifest.csv",
    } <= files
    assert "Tendon_rupture_synthetic.zip" not in files
    assert result.counts["raw"]["Tendon_rupture"] == 2


def test_import_refuses_existing_data_unless_replace(tmp_path):
    data = tmp_path / "data"
    (data / "raw" / "Old").mkdir(parents=True)
    (data / "raw" / "Old" / "x.wav").write_bytes(WAV)
    (data / "old_manifest.csv").write_text("x", encoding="utf-8")
    src = _dataset_zip(tmp_path / "d.zip", nested=False)
    with pytest.raises(importer.ZipImportError, match="ya contiene 1 audios"):
        importer.import_zip(src, data)
    assert _files(data) == {"raw/Old/x.wav", "old_manifest.csv"}
    result = importer.import_zip(src, data, replace=True)
    assert result.replaced
    assert "raw/Old/x.wav" not in _files(data) and "old_manifest.csv" not in _files(data)
    assert "raw/Good/a.wav" in _files(data)


def test_import_failure_leaves_data_untouched(tmp_path, monkeypatch):
    data = tmp_path / "data"
    (data / "raw" / "Old").mkdir(parents=True)
    (data / "raw" / "Old" / "x.wav").write_bytes(WAV)

    def boom(*_a, **_k):
        raise OSError("disco lleno")

    monkeypatch.setattr(importer, "_move", boom)
    with pytest.raises(OSError):
        importer.import_zip(_dataset_zip(tmp_path / "d.zip"), data, replace=True)
    assert _files(data) == {"raw/Old/x.wav"}


@pytest.mark.parametrize(
    ("entries", "message"),
    [
        ({"otros/x.wav": WAV}, "no contiene carpetas training"),
        ({"a/training/G/x.wav": WAV, "b/testing/G/y.wav": WAV}, "varias raíces"),
        ({"training/../x.wav": WAV}, "Nombre no válido"),
        ({"training/readme.txt": "x"}, "no contiene audios"),
        ({"training/G/x.wav": WAV, "training/g/X.WAV": WAV}, "colisionan"),
    ],
)
def test_import_rejects_bad_zips(tmp_path, entries, message):
    src = tmp_path / "bad.zip"
    src.write_bytes(_zip_bytes(entries))
    with pytest.raises(importer.ZipImportError, match=message):
        importer.import_zip(src, tmp_path / "data")
    assert not (tmp_path / "data").exists() or not _files(tmp_path / "data")


def test_import_cli_uses_project_scope(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DS_CONFIG", str(tmp_path / "sin_config.yaml"))
    monkeypatch.delenv("DS_PROJECTS_DIR", raising=False)
    init_project("pinbolt", "n155", "v2", tmp_path)
    src = _dataset_zip(tmp_path / "d.zip", nested=False)
    importer.main([str(src), "--product", "pinbolt", "--turbine", "n155", "--data-version", "v2"])
    data = tmp_path / "projects" / "pinbolt" / "n155" / "v2" / "data"
    assert (data / "raw" / "Good" / "a.wav").is_file()
    assert (data / "impact_extractor_v2_manifest1.csv").is_file()
    assert "data/raw: 3 audios" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="ya contiene"):
        importer.main(
            [str(src), "--product", "pinbolt", "--turbine", "n155", "--data-version", "v2"]
        )


def test_ui_imports_zip_from_local_path(tmp_path, monkeypatch):
    testing = pytest.importorskip("streamlit.testing.v1")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DS_CONFIG", str(tmp_path / "sin_config.yaml"))
    monkeypatch.setenv("DS_PROJECTS_DIR", str(tmp_path / "projects"))
    monkeypatch.setenv("DS_REPO_ROOT", str(tmp_path))
    init_project("pinbolt", "n155", "v1", tmp_path)
    src = _dataset_zip(tmp_path / "d.zip")
    app = Path(services.__file__).parent / "ui" / "app.py"
    at = testing.AppTest.from_file(str(app), default_timeout=30).run()
    assert not at.exception
    at.radio(key="wz-zip-mode").set_value("Ruta local").run()
    at.text_input(key="wz-zip-path").input(f'"{src}"').run()
    assert not at.exception and not at.error
    at.checkbox(key="wz-zip-nested").check().run()
    at.button(key="wz-zip-go").click().run()
    assert not at.exception
    assert any("Importación completada" in s.value for s in at.success)
    data = tmp_path / "projects" / "pinbolt" / "n155" / "v1" / "data"
    assert (data / "raw" / "Tendon_rupture" / "s1.wav").is_file()
    # Con datos ya importados, el botón queda bloqueado hasta marcar «Sustituir».
    at.radio(key="wz-zip-mode").set_value("Ruta local").run()
    at.text_input(key="wz-zip-path").input(str(src)).run()
    assert at.button(key="wz-zip-go").disabled
    at.checkbox(key="wz-zip-replace").check().run()
    assert not at.button(key="wz-zip-go").disabled
