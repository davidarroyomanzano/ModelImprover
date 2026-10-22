"""Lanzamiento y seguimiento de tareas ``invoke`` en segundo plano (sin dependencias de UI).

Cada trabajo se guarda en ``<projects_dir>/.jobs/<id>/`` con ``job.json`` (estado) y
``output.log`` (salida). Solo puede haber un trabajo activo a la vez: el entrenamiento ocupa la
GPU y la base de datos de Optuna.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ds_project.config import _check_scope

PathLike = Union[str, Path]

RUNNING, DONE, FAILED, CANCELLED, LOST = "running", "done", "failed", "cancelled", "lost"
TASKS = ("init-project", "train", "evaluate", "export", "all")
_VERSION_RE = re.compile(r"^v\d+$")
_LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_PROCS: Dict[str, subprocess.Popen] = {}
_LOCK = threading.Lock()


def repo_root(start: Optional[PathLike] = None) -> Path:
    """Primer directorio (hacia arriba) que contiene ``tasks.py``; si no, el de partida."""
    here = Path(start or os.getenv("DS_REPO_ROOT") or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "tasks.py").is_file():
            return candidate
    return here


def jobs_dir(projects_dir: PathLike = "projects", root: Optional[PathLike] = None) -> Path:
    base = Path(projects_dir)
    if not base.is_absolute():
        base = repo_root(root) / base
    return base / ".jobs"


def _flag(name: str) -> str:
    return "--" + name.replace("_", "-")


def build_args(task: str, params: Dict[str, Any]) -> List[str]:
    """Argumentos de ``invoke`` validados para ``task`` (sin el ejecutable)."""
    if task not in TASKS:
        raise ValueError(f"Tarea no permitida: {task}")
    product = str(params.get("product", ""))
    turbine = str(params.get("turbine", "")) or ("all" if product else "")
    data_version = str(params.get("data_version", ""))
    if not product:
        raise ValueError("Falta el producto")
    _check_scope(product, turbine)
    if data_version and not _LABEL_RE.match(data_version):
        raise ValueError("Versión de datos no válida (letras, números, '_', '-', '.')")
    args = [task, "--product", product, "--turbine", turbine]
    if data_version:
        args += ["--data-version", data_version]

    version = str(params.get("version") or "")
    if task in ("export", "all") or (task == "init-project" and version):
        if not _VERSION_RE.match(version):
            raise ValueError("La versión del modelo debe ser v<número> (p. ej. v150)")
        args += [_flag("export_version" if task == "init-project" else "version"), version]
    if task in ("evaluate", "export") and params.get("run_dir"):
        args += ["--run-dir", str(params["run_dir"])]
    if task == "evaluate":
        for key in ("min_accuracy", "min_f1"):
            value = float(params.get(key) or 0.0)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{key} debe estar en [0, 1]")
            if value:
                args += [_flag(key), str(value)]
    if task == "export":
        step = params.get("window_step_sec")
        if step not in (None, ""):
            if float(step) < 0:
                raise ValueError("window_step_sec debe ser >= 0 (0 = el de la plantilla)")
            if float(step) > 0:
                args += ["--window-step-sec", str(float(step))]
        if params.get("dsp_version"):
            args += ["--dsp-version", str(params["dsp_version"])]
        if params.get("allow_aux_mismatch"):
            args.append("--allow-aux-mismatch")
    return args


def display_command(args: List[str]) -> str:
    """Comando equivalente para reproducirlo desde el terminal."""
    return "poetry run invoke " + " ".join(shlex.quote(a) for a in args)


def _write(job_path: Path, meta: Dict[str, Any]) -> None:
    tmp = job_path / "job.json.tmp"
    tmp.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(job_path / "job.json")


def _read(job_path: Path) -> Dict[str, Any]:
    try:
        return json.loads((job_path / "job.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _watch(job_path: Path, proc: subprocess.Popen, log) -> None:
    code = proc.wait()
    log.close()
    with _LOCK:
        meta = _read(job_path)
        if meta.get("status") == RUNNING:
            meta["status"] = DONE if code == 0 else FAILED
        meta["exit_code"] = code
        meta["finished"] = datetime.now().isoformat(timespec="seconds")
        _write(job_path, meta)
        _PROCS.pop(meta.get("id", ""), None)


def start_job(
    command: List[str],
    meta: Dict[str, Any],
    base_dir: PathLike,
    cwd: Optional[PathLike] = None,
    env: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Lanza ``command`` en segundo plano y devuelve sus metadatos. Falla si hay otro activo."""
    base = Path(base_dir)
    active = active_job(base)
    if active:
        raise RuntimeError(f"Ya hay un trabajo en curso: {active['id']} ({active['task']})")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_id = f"{stamp}_{meta.get('task', 'job')}"
    job_path = base / job_id
    suffix = 1
    while job_path.exists():
        suffix += 1
        job_path = base / f"{job_id}_{suffix}"
    job_path.mkdir(parents=True)
    full_env = {**os.environ, **(env or {}), "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    kwargs: Dict[str, Any] = {}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    log = (job_path / "output.log").open("w", encoding="utf-8")
    proc = subprocess.Popen(
        command,
        cwd=str(cwd) if cwd else None,
        env=full_env,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        **kwargs,
    )
    meta = {
        **meta,
        "id": job_path.name,
        "pid": proc.pid,
        "status": RUNNING,
        "started": datetime.now().isoformat(timespec="seconds"),
        "finished": None,
        "exit_code": None,
    }
    with _LOCK:
        _write(job_path, meta)
        _PROCS[meta["id"]] = proc
    threading.Thread(target=_watch, args=(job_path, proc, log), daemon=True).start()
    return meta


def launch(
    task: str,
    params: Dict[str, Any],
    projects_dir: PathLike = "projects",
    root: Optional[PathLike] = None,
) -> Dict[str, Any]:
    """Valida los parámetros y lanza ``invoke <task>`` desde la raíz del repositorio."""
    args = build_args(task, params)
    root_path = repo_root(root)
    env = {}
    if str(projects_dir) != "projects":
        env["DS_PROJECTS_DIR"] = str(projects_dir)
    meta = {
        "task": task,
        "params": {k: v for k, v in params.items() if v not in (None, "")},
        "command": display_command(args),
    }
    command = [sys.executable, "-m", "invoke", *args]
    return start_job(command, meta, jobs_dir(projects_dir, root_path), cwd=root_path, env=env)


def _refresh(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Un trabajo 'running' que este proceso no controla quedó huérfano (UI reiniciada)."""
    if meta.get("status") == RUNNING and meta.get("id") not in _PROCS:
        return {**meta, "status": LOST}
    return meta


def list_jobs(base_dir: PathLike) -> List[Dict[str, Any]]:
    """Trabajos guardados, el más reciente primero."""
    base = Path(base_dir)
    if not base.is_dir():
        return []
    jobs = [_refresh(_read(p)) for p in base.iterdir() if (p / "job.json").is_file()]
    return sorted((j for j in jobs if j), key=lambda j: j.get("id", ""), reverse=True)


def active_job(base_dir: PathLike) -> Optional[Dict[str, Any]]:
    return next((j for j in list_jobs(base_dir) if j.get("status") == RUNNING), None)


def get_job(base_dir: PathLike, job_id: str) -> Dict[str, Any]:
    return _refresh(_read(Path(base_dir) / job_id))


def read_log(base_dir: PathLike, job_id: str, max_bytes: int = 64_000) -> str:
    """Final del log del trabajo (``max_bytes`` como máximo)."""
    path = Path(base_dir) / job_id / "output.log"
    if not path.is_file():
        return ""
    with path.open("rb") as fh:
        fh.seek(0, os.SEEK_END)
        size = fh.tell()
        fh.seek(max(size - max_bytes, 0))
        data = fh.read()
    text = data.decode("utf-8", errors="replace")
    return text if size <= max_bytes else "…\n" + text.split("\n", 1)[-1]


def cancel_job(base_dir: PathLike, job_id: str) -> bool:
    """Detiene el trabajo y sus subprocesos. Devuelve ``False`` si no estaba en curso."""
    proc = _PROCS.get(job_id)
    if proc is None or proc.poll() is not None:
        return False
    job_path = Path(base_dir) / job_id
    with _LOCK:
        meta = _read(job_path)
        meta["status"] = CANCELLED
        _write(job_path, meta)
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    else:
        os.killpg(proc.pid, signal.SIGTERM)
    proc.wait(timeout=30)
    return True
