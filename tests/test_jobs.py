"""Lanzamiento y seguimiento de trabajos en segundo plano (ds_project.jobs)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

from ds_project import jobs, services


def _wait(base: Path, job_id: str, timeout: float = 30.0) -> dict:
    end = time.time() + timeout
    while time.time() < end:
        job = jobs.get_job(base, job_id)
        if job.get("status") != jobs.RUNNING:
            return job
        time.sleep(0.1)
    raise AssertionError(f"El trabajo {job_id} no terminó")


def _py(code: str) -> list:
    return [sys.executable, "-c", code]


def test_build_args_train_and_init():
    assert jobs.build_args(
        "train", {"product": "pinbolt", "turbine": "g5x", "data_version": "v3"}
    ) == [
        "train",
        "--product",
        "pinbolt",
        "--turbine",
        "g5x",
        "--data-version",
        "v3",
    ]
    args = jobs.build_args("init-project", {"product": "pinbolt", "version": "v2"})
    assert args[-2:] == ["--export-version", "v2"] and "--turbine" in args
    assert "--version" not in jobs.build_args("train", {"product": "x", "version": "v9"})


def test_build_args_evaluate_and_export_options():
    args = jobs.build_args(
        "evaluate",
        {"product": "p", "turbine": "all", "run_dir": "/r", "min_accuracy": 0.9, "min_f1": 0},
    )
    assert args[-4:] == ["--run-dir", "/r", "--min-accuracy", "0.9"]
    args = jobs.build_args(
        "export",
        {
            "product": "p",
            "version": "v7",
            "window_step_sec": 0.5,
            "dsp_version": "2",
            "allow_aux_mismatch": True,
        },
    )
    assert args[5:] == [
        "--version",
        "v7",
        "--window-step-sec",
        "0.5",
        "--dsp-version",
        "2",
        "--allow-aux-mismatch",
    ]


@pytest.mark.parametrize(
    ("task", "params"),
    [
        ("rm", {"product": "p"}),
        ("train", {}),
        ("train", {"product": "Pin_Bolt"}),
        ("train", {"product": "p", "turbine": "g 5"}),
        ("train", {"product": "p", "data_version": "../x"}),
        ("export", {"product": "p", "version": "150"}),
        ("all", {"product": "p"}),
        ("evaluate", {"product": "p", "min_f1": 2}),
        ("export", {"product": "p", "version": "v1", "window_step_sec": -0.1}),
    ],
)
def test_build_args_rejects_invalid(task, params):
    with pytest.raises(ValueError):
        jobs.build_args(task, params)


def test_display_command_quotes_spaces():
    cmd = jobs.display_command(["evaluate", "--run-dir", "C:/a b"])
    assert cmd == "poetry run invoke evaluate --run-dir 'C:/a b'"


def test_job_success_and_log(tmp_path):
    job = jobs.start_job(_py("print('hola ñ')"), {"task": "train"}, tmp_path)
    assert job["status"] == jobs.RUNNING and job["id"].endswith("_train")
    done = _wait(tmp_path, job["id"])
    assert done["status"] == jobs.DONE and done["exit_code"] == 0 and done["finished"]
    assert "hola ñ" in jobs.read_log(tmp_path, job["id"])
    assert jobs.list_jobs(tmp_path)[0]["id"] == job["id"]


def test_job_failure(tmp_path):
    job = jobs.start_job(_py("import sys; sys.exit(3)"), {"task": "export"}, tmp_path)
    done = _wait(tmp_path, job["id"])
    assert done["status"] == jobs.FAILED and done["exit_code"] == 3


def test_single_active_job_and_cancel(tmp_path):
    job = jobs.start_job(_py("import time; time.sleep(60)"), {"task": "train"}, tmp_path)
    assert jobs.active_job(tmp_path)["id"] == job["id"]
    with pytest.raises(RuntimeError):
        jobs.start_job(_py("pass"), {"task": "export"}, tmp_path)
    assert jobs.cancel_job(tmp_path, job["id"])
    done = _wait(tmp_path, job["id"])
    assert done["status"] == jobs.CANCELLED
    assert jobs.active_job(tmp_path) is None
    assert not jobs.cancel_job(tmp_path, job["id"])


def test_orphan_job_is_reported_as_lost(tmp_path):
    orphan = tmp_path / "20260101_000000_train"
    orphan.mkdir()
    meta = {"id": orphan.name, "task": "train", "status": jobs.RUNNING}
    (orphan / "job.json").write_text(json.dumps(meta), encoding="utf-8")
    assert jobs.get_job(tmp_path, orphan.name)["status"] == jobs.LOST
    assert jobs.active_job(tmp_path) is None


def test_read_log_tail(tmp_path):
    (tmp_path / "j").mkdir()
    (tmp_path / "j" / "output.log").write_text("\n".join(map(str, range(1000))), encoding="utf-8")
    tail = jobs.read_log(tmp_path, "j", max_bytes=50)
    assert tail.startswith("…") and tail.endswith("999")
    assert jobs.read_log(tmp_path, "nada") == ""


def test_launch_runs_invoke_from_repo_root(tmp_path, monkeypatch):
    calls = {}

    def fake_start(command, meta, base_dir, cwd=None, env=None):
        calls.update(command=command, meta=meta, base=base_dir, cwd=cwd, env=env)
        return meta

    monkeypatch.setattr(jobs, "start_job", fake_start)
    (tmp_path / "tasks.py").write_text("", encoding="utf-8")
    meta = jobs.launch("train", {"product": "pinbolt", "turbine": "g5x"}, "otros", root=tmp_path)
    assert calls["command"][1:4] == ["-m", "invoke", "train"]
    assert calls["cwd"] == tmp_path.resolve()
    assert calls["base"] == tmp_path.resolve() / "otros" / ".jobs"
    assert calls["env"] == {"DS_PROJECTS_DIR": "otros"}
    assert meta["command"].startswith("poetry run invoke train --product pinbolt")


def test_repo_root_finds_tasks_py(tmp_path, monkeypatch):
    monkeypatch.delenv("DS_REPO_ROOT", raising=False)
    (tmp_path / "tasks.py").write_text("", encoding="utf-8")
    (tmp_path / "a" / "b").mkdir(parents=True)
    assert jobs.repo_root(tmp_path / "a" / "b") == tmp_path.resolve()


def test_blockers_and_next_version():
    checks = [
        services.Check("Datos raw", services.ERROR, "vacío"),
        services.Check("Plantilla DSP", services.OK, "ok"),
    ]
    assert services.blockers("train", checks) == ["Datos raw: vacío"]
    assert services.blockers("export", checks, n_runs=1) == []
    assert "No hay runs" in services.blockers("evaluate", [], n_runs=0)[0]
    assert services.blockers("init-project", checks) == []
    assert services.next_version([]) == "v1"
    assert services.next_version(["v2", "v150", "borrador"]) == "v151"
