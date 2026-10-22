"""Pestaña «Ejecutar»: formularios, comprobaciones previas y seguimiento de trabajos."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st
import yaml

from ds_project import jobs, services
from ds_project.ui import common

ACTIONS = {
    "Inicializar proyecto": "init-project",
    "Entrenar": "train",
    "Evaluar": "evaluate",
    "Exportar": "export",
    "Pipeline completo": "all",
}
STATUS = {
    jobs.RUNNING: "⏳ En curso",
    jobs.DONE: "✅ Terminado",
    jobs.FAILED: "❌ Con errores",
    jobs.CANCELLED: "⏹ Detenido",
    jobs.LOST: "❔ Sin seguimiento (UI reiniciada)",
}
LOG_LINES = 80
LATEST = "(último run)"


def _tail(text: str, lines: int = LOG_LINES) -> str:
    return "\n".join(text.splitlines()[-lines:]) or "(sin salida todavía)"


def _job_header(job: Dict[str, Any]) -> None:
    st.markdown(f"**{STATUS.get(job.get('status'), job.get('status'))}** · `{job.get('id')}`")
    st.code(job.get("command", ""), language="bash")
    if job.get("status") == jobs.FAILED:
        st.error(f"Código de salida {job.get('exit_code')}. Revisa el log.")


@st.fragment(run_every=2)
def _live_job(base: Path, job_id: str, prefix: str = "") -> None:
    job = jobs.get_job(base, job_id)
    _job_header(job)
    progress = common.optuna_for_job(job)
    if progress:
        common.render_optuna(progress)
    st.code(_tail(jobs.read_log(base, job_id)), language="text")
    if job.get("status") != jobs.RUNNING:
        st.rerun()
    if st.button("⏹ Detener", key=f"{prefix}cancel-{job_id}"):
        jobs.cancel_job(base, job_id)
        st.rerun()


def render_monitor(base: Path, prefix: str = "") -> Optional[Dict[str, Any]]:
    """Trabajo activo (en vivo) o el último lanzado en esta sesión; devuelve el activo."""
    active = jobs.active_job(base)
    if active:
        st.subheader("Trabajo en curso")
        _live_job(base, active["id"], prefix)
        return active
    last = st.session_state.get("last_job")
    if last:
        job = jobs.get_job(base, last)
        if job:
            st.subheader("Último trabajo")
            _job_header(job)
            with st.expander("Log", expanded=job.get("status") != jobs.DONE):
                st.code(_tail(jobs.read_log(base, last)), language="text")
    return None


def launch_button(
    task: str,
    params: Dict[str, Any],
    projects_dir: str,
    problems: List[str],
    active: Optional[Dict[str, Any]],
    key: str,
    label: str = "▶ Ejecutar",
    allowed: bool = True,
) -> None:
    """Muestra el comando equivalente, los bloqueos y el botón que lanza el trabajo."""
    problems = list(problems)
    try:
        args = jobs.build_args(task, params)
    except ValueError as exc:
        problems.append(str(exc))
        args = []
    if args:
        st.markdown("Comando equivalente:")
        st.code(jobs.display_command(args), language="bash")
    for problem in problems:
        st.error(problem)
    if active:
        st.info("Hay un trabajo en curso; espera a que termine o detenlo.")
    disabled = bool(problems or active or not allowed)
    if st.button(label, type="primary", disabled=disabled, key=key):
        try:
            job = jobs.launch(task, params, projects_dir)
        except (RuntimeError, ValueError) as exc:
            st.error(str(exc))
        else:
            st.session_state["last_job"] = job["id"]
            st.rerun()


def _render_history(base: Path) -> None:
    history = jobs.list_jobs(base)
    if not history:
        return
    with st.expander(f"Historial ({len(history)})"):
        rows = [
            {
                "id": j.get("id"),
                "tarea": j.get("task"),
                "estado": STATUS.get(j.get("status"), j.get("status")),
                "inicio": j.get("started"),
                "fin": j.get("finished"),
            }
            for j in history
        ]
        st.dataframe(rows, hide_index=True, use_container_width=True)
        job_id = st.selectbox("Ver log", [j["id"] for j in history], key="history-job")
        st.code(_tail(jobs.read_log(base, job_id), 400), language="text")


def _scope_inputs(task: str, project, version: str) -> Dict[str, str]:
    if task != "init-project" or project is None:
        return {
            "product": project.product,
            "turbine": project.turbine,
            "data_version": version,
        }
    c1, c2, c3 = st.columns(3)
    return {
        "product": c1.text_input("Producto", project.product, help="pinbolt, bathear..."),
        "turbine": c2.text_input("Turbina", project.turbine, help="g5x, n155, all..."),
        "data_version": c3.text_input("Versión de datos", version or "v1"),
    }


def _run_choice(runs: List[services.RunInfo], key: str) -> str:
    choice = st.selectbox("Run", [LATEST] + [r.name for r in runs], key=key)
    if choice == LATEST:
        return ""
    return str(next(r.path for r in runs if r.name == choice).resolve())


def _task_params(task: str, project, version: str, runs) -> Dict[str, Any]:
    if project is None:
        c1, c2, c3 = st.columns(3)
        params: Dict[str, Any] = {
            "product": c1.text_input("Producto", "", help="pinbolt, bathear..."),
            "turbine": c2.text_input("Turbina", "all", help="g5x, n155, all..."),
            "data_version": c3.text_input("Versión de datos", "v1"),
        }
    else:
        params = _scope_inputs(task, project, version)
    exported = project.export_versions if project else []
    suggested = services.next_version(exported)

    if task == "init-project":
        params["version"] = st.text_input(
            "Versión del modelo a preparar en export/ (opcional)", "", placeholder=suggested
        )
    elif task == "train":
        st.caption(
            "Los hiperparámetros salen de configs/*.yaml y project.yaml (ver Configuración "
            "efectiva). Si ya existe un run con la misma configuración y datos, se reutiliza."
        )
    elif task == "evaluate":
        params["run_dir"] = _run_choice(runs, "eval-run")
        c1, c2 = st.columns(2)
        params["min_accuracy"] = c1.number_input("Accuracy mínima", 0.0, 1.0, 0.0, 0.01)
        params["min_f1"] = c2.number_input("F1 macro mínimo", 0.0, 1.0, 0.0, 0.01)
        st.caption("Con umbrales a 0 el veredicto es informativo.")
    elif task in ("export", "all"):
        if task == "export":
            params["run_dir"] = _run_choice(runs, "export-run")
        params["version"] = st.text_input(
            "Versión del modelo", suggested, help="v<número>; norma ClaudIA", key=f"v-{task}"
        )
        if task == "export":
            with st.expander("Opciones avanzadas"):
                params["window_step_sec"] = st.number_input(
                    "Salto entre inferencias (s)",
                    0.0,
                    10.0,
                    0.0,
                    0.05,
                    help="0 = WINDOW_STEP_SEC de la plantilla DSP activa",
                )
                params["dsp_version"] = st.text_input("DSP_VERSION (vacío = plantilla)", "")
                params["allow_aux_mismatch"] = st.checkbox(
                    "Permitir vector auxiliar distinto de la plantilla", False
                )
        else:
            st.caption("lint + test + train + evaluate + export.")
    return params


def overwrite_ok(task: str, params: Dict[str, Any], project, prefix: str = "") -> bool:
    version = params.get("version")
    if task not in ("export", "all") or not project or version not in project.export_versions:
        return True
    st.warning(
        f"La versión {version} ya existe en export/. La norma de nombres considera inmutables "
        "las versiones publicadas: usa una nueva salvo que sepas lo que haces."
    )
    return st.checkbox(f"Sobrescribir {version}", False, key=f"{prefix}ow-{task}")


def render(project, version: str, projects_dir: str, checks: List[services.Check]) -> None:
    """Pestaña de ejecución; ``project`` es ``None`` si aún no hay proyectos."""
    base = jobs.jobs_dir(projects_dir)
    active = render_monitor(base)
    st.divider()

    labels = list(ACTIONS) if project else ["Inicializar proyecto"]
    task = ACTIONS[st.radio("Acción", labels, horizontal=True, key="action")]
    runs = services.list_runs(project.path / version / "models") if project and version else []
    params = _task_params(task, project, version, runs)

    problems = [] if task == "init-project" else services.blockers(task, checks, len(runs))
    allowed = overwrite_ok(task, params, project)
    launch_button(task, params, projects_dir, problems, active, "run-action", allowed=allowed)

    if project:
        with st.expander("Configuración efectiva"):
            try:
                cfg = services.effective_config(project.product, project.turbine, version)
                st.code(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), language="yaml")
            except ValueError as exc:
                st.error(str(exc))
    _render_history(base)
