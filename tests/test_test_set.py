"""Tests del conjunto de evaluación externa (data/test)."""

from pathlib import Path

import pytest

from ds_project.data.test_set import (
    LabeledFile,
    TestSetReport,
    discover_labeled_files,
    load_test_set,
    source_id,
    validate_test_set,
)

TestSetReport.__test__ = False  # evita que pytest intente recolectarla


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"RIFF")
    return path


@pytest.fixture
def dirs(tmp_path: Path) -> tuple[Path, Path]:
    train, test = tmp_path / "raw", tmp_path / "test"
    _touch(train / "Good" / "Good.ES1Y_012.Impacts_hub.1781181528.wav.wav.x.wav")
    _touch(train / "Stopped" / "Stopped.ES78_005.Impacts_hub.1786604846.wav.wav.y.wav")
    _touch(test / "Good" / "Good.ES1Y_012.Impacts_hub.1785564539.wav.wav.z.wav")
    _touch(test / "Stopped" / "Stopped.ES78_005.Impacts_hub.1786607651.wav.wav.w.wav")
    return train, test


# --- source_id -------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        (
            "Good.ES1Y_012.Impacts_hub.1781181528.wav.wav.70ra.wav",
            "ES1Y_012:1781181528",
        ),
        (
            "Impacts_hub.ES78_015.onrecord.1736855784.s1.json.x.wav",
            "ES78_015:1736855784",
        ),
        (
            "Impacts_pitch.ES78_015.autorecord.1736943869.json.y.wav",
            "ES78_015:1736943869",
        ),
        ("Tendon_rupture.synthetic001.wav", None),
        ("Impacts_hub.Impacts_hub_sintetic_005.wav.6lp5ifu3.event001.wav", None),
        ("Good.ES1Y_012.Impacts_hub.123.wav", None),  # timestamp no tiene 10 dígitos
    ],
)
def test_source_id(name: str, expected: str | None) -> None:
    assert source_id(Path(name)) == expected


def test_source_id_ignores_parent_dirs() -> None:
    path = Path("/a/1781181528/Good.ES1Y_012.Impacts_hub.1781181528.wav.wav.x.wav")
    assert source_id(path) == "ES1Y_012:1781181528"


# --- discover_labeled_files -----------------------------------------------


def test_discover_filters_and_sorts(tmp_path: Path) -> None:
    _touch(tmp_path / "B" / "b.wav")
    _touch(tmp_path / "A" / "a.WAV")  # extensión en mayúsculas
    _touch(tmp_path / "A" / "notes.txt")  # extensión no admitida
    _touch(tmp_path / "root.wav")  # fuera de una carpeta de etiqueta
    _touch(tmp_path / "A" / "sub" / "deep.wav")  # más de un nivel de profundidad
    (tmp_path / "A" / "dir.wav").mkdir()  # directorio con sufijo .wav

    files = discover_labeled_files(tmp_path)

    assert [(f.path.name, f.label) for f in files] == [("a.WAV", "A"), ("b.wav", "B")]
    assert all(isinstance(f, LabeledFile) for f in files)


def test_discover_accepts_str(tmp_path: Path) -> None:
    _touch(tmp_path / "A" / "a.wav")
    assert len(discover_labeled_files(str(tmp_path))) == 1


def test_discover_missing_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No existe"):
        discover_labeled_files(tmp_path / "nope")


def test_discover_path_is_file_raises(tmp_path: Path) -> None:
    f = _touch(tmp_path / "file.wav")
    with pytest.raises(FileNotFoundError):
        discover_labeled_files(f)


def test_discover_empty_raises(tmp_path: Path) -> None:
    (tmp_path / "Good").mkdir()
    with pytest.raises(ValueError, match="No se encontraron"):
        discover_labeled_files(tmp_path)


# --- load_test_set ---------------------------------------------------------


def test_load_test_set_aligned(dirs) -> None:
    _, test = dirs
    paths, labels = load_test_set(test)
    assert labels == ["Good", "Stopped"]
    assert [p.parent.name for p in paths] == labels
    assert all(p.exists() for p in paths)


# --- validate_test_set -----------------------------------------------------


def test_validate_ok(dirs) -> None:
    report = validate_test_set(*dirs)
    assert report.is_valid
    assert report.n_files == 2
    assert report.class_counts == {"Good": 1, "Stopped": 1}
    assert report.missing_in_test == report.unknown_in_test == report.leaked_sources == []


def test_validate_missing_class_still_valid(dirs) -> None:
    train, test = dirs
    _touch(train / "Tendon_rupture" / "Tendon_rupture.synthetic001.wav")
    report = validate_test_set(train, test)
    assert report.missing_in_test == ["Tendon_rupture"]
    assert report.is_valid


def test_validate_detects_leak_and_unknown(dirs) -> None:
    train, test = dirs
    _touch(test / "Good" / "Good.ES1Y_012.Impacts_hub.1781181528.wav.wav.dup.wav")
    _touch(test / "Nuevo" / "Nuevo.a.wav")
    report = validate_test_set(train, test)
    assert not report.is_valid
    assert report.leaked_sources == ["ES1Y_012:1781181528"]
    assert report.unknown_in_test == ["Nuevo"]
    assert report.class_counts == {"Good": 2, "Nuevo": 1, "Stopped": 1}


def test_validate_synthetic_files_never_leak(tmp_path: Path) -> None:
    train, test = tmp_path / "raw", tmp_path / "test"
    _touch(train / "Tendon_rupture" / "Tendon_rupture.synthetic001.wav")
    _touch(test / "Tendon_rupture" / "Tendon_rupture.synthetic001.wav")
    assert validate_test_set(train, test).leaked_sources == []


def test_validate_class_counts_sorted(dirs) -> None:
    train, test = dirs
    _touch(train / "A" / "a.wav")
    _touch(test / "A" / "a2.wav")
    assert list(validate_test_set(train, test).class_counts) == ["A", "Good", "Stopped"]


def test_validate_propagates_missing_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        validate_test_set(tmp_path / "raw", tmp_path / "test")


@pytest.mark.parametrize(
    ("unknown", "leaked", "valid"),
    [([], [], True), (["X"], [], False), ([], ["s"], False)],
)
def test_report_is_valid(unknown, leaked, valid) -> None:
    report = TestSetReport(1, {}, ["M"], unknown, leaked)
    assert report.is_valid is valid
