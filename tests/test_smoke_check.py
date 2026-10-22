"""Tests de scripts/smoke_check.py."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from unittest import mock

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "smoke_check.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("smoke_check", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sc = _load_module()
ALL_LABELS = list(sc.EXPECTED_LABELS)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def make_data(tmp_path: Path):
    def _make(counts: dict[str, int]) -> Path:
        for label, n in counts.items():
            d = tmp_path / label
            d.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                (d / f"{label}.{i}.wav").write_bytes(b"RIFF")
        return tmp_path

    return _make


@pytest.fixture
def make_run(tmp_path: Path):
    def _make(
        name: str = "run1",
        labels: list[str] | None = None,
        optuna: bool = False,
    ) -> Path:
        run = tmp_path / name
        run.mkdir(parents=True)
        names = list(sc.REQUIRED_ARTIFACTS) + (list(sc.OPTUNA_ARTIFACTS) if optuna else [])
        for n in names:
            (run / n).write_text("x", encoding="utf-8")
        (run / "dsp_params.json").write_text(
            json.dumps({"labels": labels if labels is not None else ALL_LABELS}),
            encoding="utf-8",
        )
        (run / "metrics.json").write_text(json.dumps({"accuracy": 1.0}), encoding="utf-8")
        return run

    return _make


# --------------------------------------------------------------------------- #
# count_samples / check_data
# --------------------------------------------------------------------------- #
def test_count_samples_ignores_non_wav_and_files(make_data) -> None:
    root = make_data({"Good": 2})
    (root / "Good" / "notes.txt").write_text("x")
    (root / "README.md").write_text("x")
    assert sc.count_samples(root) == {"Good": 2}


def test_count_samples_missing_dir(tmp_path: Path) -> None:
    assert sc.count_samples(tmp_path / "nope") == {}


def test_check_data_ok(make_data) -> None:
    root = make_data({label: 2 for label in ALL_LABELS})
    assert sc.check_data(root) == []


def test_check_data_missing_class(make_data) -> None:
    root = make_data({label: 2 for label in ALL_LABELS[:-1]})
    assert sc.check_data(root) == ["Falta la clase 'Tendon_rupture'"]


def test_check_data_below_minimum(make_data) -> None:
    counts = {label: 2 for label in ALL_LABELS}
    counts["Impacts_yaw"] = 1
    problems = sc.check_data(make_data(counts))
    assert problems == ["Clase 'Impacts_yaw' con 1 muestra(s) (< 2)"]


def test_check_data_unexpected_label(make_data) -> None:
    counts = {label: 2 for label in ALL_LABELS}
    counts["Unknown"] = 3
    assert sc.check_data(make_data(counts)) == ["Etiqueta inesperada 'Unknown'"]


def test_check_data_custom_minimum(make_data) -> None:
    root = make_data({label: 2 for label in ALL_LABELS})
    assert len(sc.check_data(root, min_per_class=3)) == len(ALL_LABELS)


def test_check_data_empty_dir(tmp_path: Path) -> None:
    assert sc.check_data(tmp_path / "nope") == [
        f"No se encontraron etiquetas en {tmp_path / 'nope'}"
    ]


# --------------------------------------------------------------------------- #
# latest_run
# --------------------------------------------------------------------------- #
def test_latest_run_missing_dir(tmp_path: Path) -> None:
    assert sc.latest_run(tmp_path / "nope") is None


def test_latest_run_empty_dir(tmp_path: Path) -> None:
    (tmp_path / "file.txt").write_text("x")
    assert sc.latest_run(tmp_path) is None


def test_latest_run_picks_most_recent(tmp_path: Path) -> None:
    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir()
    new.mkdir()
    mtimes = {old: 100.0, new: 200.0}
    original_stat = Path.stat

    def fake_stat(self: Path, *args, **kwargs):
        if self in mtimes:
            result = original_stat(self, *args, **kwargs)
            return mock.Mock(
                st_mode=result.st_mode,
                st_mtime=mtimes[self],
            )

        return original_stat(self, *args, **kwargs)

    with mock.patch.object(Path, "stat", fake_stat):
        assert sc.latest_run(tmp_path) == new


# --------------------------------------------------------------------------- #
# _find_labels
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"labels": ["a", "b"]}, ["a", "b"]),
        ({"classes": ["a"]}, ["a"]),
        ({"meta": {"class_names": ["a", "b"]}}, ["a", "b"]),
        ({"labels": [1, 2]}, None),
        ({"x": 1}, None),
        (["labels"], None),
    ],
)
def test_find_labels(payload: object, expected: list[str] | None) -> None:
    assert sc._find_labels(payload) == expected


# --------------------------------------------------------------------------- #
# check_artifacts
# --------------------------------------------------------------------------- #
def test_check_artifacts_ok(make_run) -> None:
    assert sc.check_artifacts(make_run(optuna=True), require_optuna=True) == []


def test_check_artifacts_optuna_not_required(make_run) -> None:
    assert sc.check_artifacts(make_run(optuna=False)) == []


@pytest.mark.parametrize("name", sc.REQUIRED_ARTIFACTS)
def test_check_artifacts_missing_required(make_run, name: str) -> None:
    run = make_run()
    (run / name).unlink()
    assert f"Falta {name}" in sc.check_artifacts(run)


@pytest.mark.parametrize("name", sc.OPTUNA_ARTIFACTS)
def test_check_artifacts_missing_optuna(make_run, name: str) -> None:
    assert f"Falta {name}" in sc.check_artifacts(make_run(), require_optuna=True)


def test_check_artifacts_empty_file(make_run) -> None:
    run = make_run()
    (run / "model.tflite").write_bytes(b"")
    assert sc.check_artifacts(run) == ["model.tflite está vacío"]


def test_check_artifacts_label_mismatch(make_run) -> None:
    problems = sc.check_artifacts(make_run(labels=["Good"]))
    assert len(problems) == 1
    assert problems[0].startswith("Etiquetas en dsp_params.json no coinciden")


def test_check_artifacts_without_labels(make_run) -> None:
    run = make_run()
    (run / "dsp_params.json").write_text(json.dumps({"n_mfcc": 13}), encoding="utf-8")
    assert sc.check_artifacts(run) == ["dsp_params.json no contiene lista de etiquetas"]


@pytest.mark.parametrize("name", ["dsp_params.json", "metrics.json"])
def test_check_artifacts_invalid_json(make_run, name: str) -> None:
    run = make_run()
    (run / name).write_text("{bad", encoding="utf-8")
    problems = sc.check_artifacts(run)
    assert len(problems) == 1
    assert problems[0].startswith(f"{name} inválido")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def test_main_data_ok(make_data, capsys) -> None:
    root = make_data({label: 2 for label in ALL_LABELS})
    assert sc.main(["data", "--data-dir", str(root)]) == 0
    assert "[OK]" in capsys.readouterr().out


def test_main_data_fail(tmp_path: Path, capsys) -> None:
    assert sc.main(["data", "--data-dir", str(tmp_path)]) == 1
    assert "[FAIL]" in capsys.readouterr().out


def test_main_artifacts_no_runs(tmp_path: Path, capsys) -> None:
    assert sc.main(["artifacts", "--models-dir", str(tmp_path)]) == 1
    assert "No hay ejecuciones" in capsys.readouterr().out


def test_main_artifacts_latest_ok(make_run, tmp_path: Path) -> None:
    make_run()
    assert sc.main(["artifacts", "--models-dir", str(tmp_path)]) == 0


def test_main_artifacts_explicit_run_id(make_run, tmp_path: Path) -> None:
    make_run("good", optuna=True)
    make_run("bad", labels=["Good"])
    args = ["artifacts", "--models-dir", str(tmp_path), "--require-optuna"]
    assert sc.main([*args, "--run-id", "good"]) == 0
    assert sc.main([*args, "--run-id", "bad"]) == 1


def test_main_artifacts_unknown_run_id(tmp_path: Path) -> None:
    assert sc.main(["artifacts", "--models-dir", str(tmp_path), "--run-id", "x"]) == 1


def test_main_requires_subcommand() -> None:
    with pytest.raises(SystemExit):
        sc.main([])
