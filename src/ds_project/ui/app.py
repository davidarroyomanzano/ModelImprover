"""Aplicación Streamlit sobre ``ds_project.services`` y ``ds_project.jobs``."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from ds_project import jobs, services
from ds_project.ui import assistant, common, execute

ICONS = common.ICONS
MAX_ENTRIES = 200
UI_MODULES = (common, execute, assistant)


def reload_changed_modules() -> list:
    """Recarga los módulos de ``ds_project`` fuera de ``ui/`` modificados desde su carga.

    Streamlit solo vigila la carpeta de ``app.py``: sin esto, editar p. ej. ``wizard.py`` con la
    UI abierta mezclaría módulos nuevos de ``ui/`` con versiones antiguas del resto.
    """
    changed = []
    for name, mod in sorted(sys.modules.items()):
        path = getattr(mod, "__file__", None)
        if not name.startswith("ds_project.") or name.startswith("ds_project.ui") or not path:
            continue
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            continue
        if mtime > mod.__dict__.setdefault("__ui_mtime__", mtime):
            importlib.reload(mod)
            mod.__ui_mtime__ = mtime
            changed.append(name)
    if changed:
        for mod in UI_MODULES:
            importlib.reload(mod)
    return changed


def _render_status(checks: list) -> None:
    overall = services.summarize(checks)
    st.subheader(f"{ICONS[overall]} Estado del proyecto")
    for c in checks:
        st.markdown(f"{ICONS[c.status]} **{c.name}** — {c.detail}")


def _render_data(project, version: str, active) -> None:
    scope = project.path / version if version else project.path
    if version:
        try:
            cfg = services.effective_config(project.product, project.turbine, version)
        except ValueError:
            cfg = {}
        exts = cfg.get("data", {}).get("extensions")
        common.render_zip_import(scope / "data", exts, active, "tab")
    for split in ("raw", "test"):
        common.render_counts(services.count_by_class(scope / "data" / split), f"data/{split}")


def _render_runs(scope: Path) -> None:
    runs = services.list_runs(scope / "models")
    if not runs:
        st.info("Sin runs. Ejecuta `invoke train`.")
        return
    table = pd.DataFrame(
        {
            "run": [r.name for r in runs],
            "accuracy": [r.accuracy for r in runs],
            "f1_macro": [r.f1_macro for r in runs],
            "evaluación": [r.has_evaluation for r in runs],
            "búsqueda": [r.search.get("mode", "") for r in runs],
        }
    )
    st.dataframe(table, hide_index=True, use_container_width=True)
    name = st.selectbox("Run", [r.name for r in runs])
    common.render_run(next(r for r in runs if r.name == name))


def _render_exports(project: Path) -> None:
    versions = [e["name"] for e in services.list_dir(project, "export") if e["is_dir"]]
    if not versions:
        st.info("Sin exportaciones. Ejecuta `invoke export`.")
        return
    version = st.selectbox("Versión exportada", versions)
    rows = services.list_dir(project, f"export/{version}")
    if not rows:
        st.info("Carpeta vacía.")
        return
    st.dataframe(
        pd.DataFrame(rows).drop(columns=["is_dir"]), hide_index=True, use_container_width=True
    )


def _render_files(project: Path) -> None:
    rel = st.session_state.get("rel", "")
    parts = [p for p in rel.split("/") if p]
    st.markdown(" / ".join(["`" + project.name + "`"] + [f"`{p}`" for p in parts]))
    rows = services.list_dir(project, rel)
    if parts and st.button("⬆ Subir"):
        st.session_state["rel"] = "/".join(parts[:-1])
        st.rerun()
    for row in rows[:MAX_ENTRIES]:
        if row["is_dir"]:
            if st.button(f"📁 {row['name']}", key=f"d-{rel}-{row['name']}"):
                st.session_state["rel"] = "/".join(parts + [row["name"]])
                st.rerun()
        else:
            st.text(f"📄 {row['name']}  ({row['size'] / 1024:.1f} KiB)")
    if len(rows) > MAX_ENTRIES:
        st.caption(f"Mostrando {MAX_ENTRIES} de {len(rows)} entradas.")


def main() -> None:
    st.set_page_config(page_title="ModelosPinbolt", layout="wide")
    os.chdir(jobs.repo_root())
    st.title("ModelosPinbolt")
    default_dir = os.getenv("DS_PROJECTS_DIR", "projects")
    projects_dir = st.sidebar.text_input("Carpeta de proyectos", default_dir)
    projects = services.list_projects(projects_dir)
    if not projects:
        st.warning(
            f"No hay proyectos en `{projects_dir}`. El asistente te guía para crear el primero:"
        )
        assistant.render(None, "", projects_dir, [])
        return
    by_name = {p.name: p for p in projects}
    if st.session_state.get(assistant.SELECT_PROJECT) not in by_name:
        st.session_state.pop(assistant.SELECT_PROJECT, None)
    project = by_name[st.sidebar.selectbox("Proyecto", list(by_name), key=assistant.SELECT_PROJECT)]
    versions = project.data_versions or [""]
    if st.session_state.get(assistant.SELECT_VERSION) not in versions:
        st.session_state.pop(assistant.SELECT_VERSION, None)
    version = st.sidebar.selectbox("Versión de datos", versions, key=assistant.SELECT_VERSION)
    if st.session_state.get("scope") != (project.name, version):
        st.session_state["scope"] = (project.name, version)
        st.session_state["rel"] = ""
        if st.session_state.get(assistant.STEP_KEY) != "data":
            st.session_state.pop(assistant.STEP_KEY, None)
    scope = project.path / version if version else project.path
    st.sidebar.caption(f"Ámbito: `{scope}`")
    active = jobs.active_job(jobs.jobs_dir(projects_dir))
    if active:
        st.sidebar.info(f"⏳ En curso: {active['task']} ({active['id']})")

    checks = services.check_project(project.product, project.turbine, version, projects_dir)
    tabs = st.tabs(["Asistente", "Estado", "Ejecutar", "Datos", "Runs", "Exports", "Archivos"])
    with tabs[0]:
        assistant.render(project, version, projects_dir, checks)
    with tabs[1]:
        _render_status(checks)
    with tabs[2]:
        execute.render(project, version, projects_dir, checks)
    with tabs[3]:
        _render_data(project, version, active)
    with tabs[4]:
        _render_runs(scope)
    with tabs[5]:
        _render_exports(project.path)
    with tabs[6]:
        _render_files(project.path)


reload_changed_modules()
main()
