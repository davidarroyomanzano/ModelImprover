"""Pestaña «Asistente»: proyecto → datos → configuración → train → evaluate → export."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import yaml

from ds_project import jobs, services
from ds_project import wizard as wz
from ds_project.config import VALID_QUANT, VALID_SCALERS, phase_trial_counts
from ds_project.dsp.params import VALID_FEATURES
from ds_project.ui import common, execute

STEP_KEY = "wz-step"
RADIO_PREFIX = "wz-radio-"
SELECT_PROJECT = "sb-project"
SELECT_VERSION = "sb-version"
STEP_ICONS = {wz.DONE: "✅", wz.PENDING: "🔵", wz.ATTENTION: "⚠️", wz.LOCKED: "🔒"}
QUANTIZATIONS = VALID_QUANT
SCALERS = VALID_SCALERS
MFE_COLUMNS = (
    "frame_length",
    "frame_stride",
    "n_filters",
    "fft_length",
    "low_freq",
    "high_freq",
    "noise_floor_db",
)


def _goto(step: str) -> None:
    st.session_state[STEP_KEY] = step


def _select(name: str, version: str) -> None:
    st.session_state[SELECT_PROJECT] = name
    st.session_state[SELECT_VERSION] = version
    st.session_state[STEP_KEY] = "data"


def _effective(project, version: str) -> Dict[str, Any]:
    try:
        return services.effective_config(project.product, project.turbine, version)
    except ValueError:
        return {}


def _runs(project, version: str) -> List[services.RunInfo]:
    if project is None or not version:
        return []
    return services.list_runs(project.path / version / "models")


# --- pasos ------------------------------------------------------------------------------------


def _step_project(project, version, projects_dir, checks, active) -> None:
    if project is not None:
        if version:
            st.success(f"Proyecto **{project.name}**, versión de datos **{version}**.")
        else:
            st.warning(
                f"El proyecto **{project.name}** no tiene versiones de datos: créala abajo "
                f"(producto {project.product}, turbina {project.turbine})."
            )
        st.caption("Cambia de proyecto o versión en la barra lateral.")
        for c in checks:
            st.markdown(f"{common.ICONS[c.status]} **{c.name}** — {c.detail}")
    with st.expander("Crear un proyecto o versión de datos", expanded=not version):
        c1, c2, c3 = st.columns(3)
        params = {
            "product": c1.text_input("Producto", "", help="pinbolt, bathear...", key="wz-prod"),
            "turbine": c2.text_input("Turbina", "all", help="g5x, n155, all...", key="wz-turb"),
            "data_version": c3.text_input("Versión de datos", "v1", key="wz-dv"),
        }
        st.caption(
            "Crea projects/<producto>/<turbina>/<versión>/{data,models}, export/, templates/ y "
            "un project.yaml vacío. Norma ClaudIA: minúsculas y sin espacios."
        )
        execute.launch_button(
            "init-project", params, projects_dir, [], active, "wz-init", "▶ Crear proyecto"
        )
    last_id = st.session_state.get("last_job")
    last = jobs.get_job(jobs.jobs_dir(projects_dir), last_id) if last_id else {}
    if last and last.get("task") == "init-project" and last.get("status") == jobs.DONE:
        p = last.get("params", {})
        name = f"{p.get('product')}_{p.get('turbine', 'all')}"
        st.button(
            f"➡ Trabajar con {name} / {p.get('data_version')}",
            on_click=_select,
            args=(name, p.get("data_version", "")),
            key="wz-open",
        )


def _open_folder(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if hasattr(os, "startfile"):
        os.startfile(str(path))  # noqa: S606 - solo abre el explorador local


def _step_data(project, version, projects_dir, checks, active) -> None:
    scope = project.path / version
    exts = _effective(project, version).get("data", {}).get("extensions")
    st.markdown(
        "Copia los audios en una carpeta por clase (el nombre de la carpeta es la etiqueta). "
        "`raw` se usa para entrenar y validar; `test` solo para el informe de evaluación."
    )
    cols = st.columns(2)
    for col, split in zip(cols, ("raw", "test")):
        folder = scope / "data" / split
        col.code(str(folder.resolve()), language="text")
        if hasattr(os, "startfile") and col.button(
            f"📂 Abrir data/{split}", key=f"wz-open-{split}"
        ):
            _open_folder(folder)
    if st.button("🔄 Revalidar", key="wz-recheck"):
        st.rerun()
    common.render_zip_import(scope / "data", exts, active, "wz")
    raw = services.count_by_class(scope / "data" / "raw", exts)
    test = services.count_by_class(scope / "data" / "test", exts)
    st.markdown("#### Comprobaciones")
    names = ("Datos raw", "Datos test", "Clases test", "Manifests de origen")
    for c in [c for c in checks if c.name in names] + wz.data_checks(raw, test):
        st.markdown(f"{common.ICONS[c.status]} **{c.name}** — {c.detail}")
    common.render_counts(raw, "data/raw")
    common.render_counts(test, "data/test")


def _num(col, label: str, value: Any, key: str, **kw: Any) -> Any:
    if isinstance(value, bool):
        return col.checkbox(label, value, key=key, **kw)
    if isinstance(value, int):
        return int(col.number_input(label, value=value, step=1, key=key, **kw))
    return float(col.number_input(label, value=float(value), key=key, format="%.4g", **kw))


def _config_form(project, effective: Dict[str, Any]) -> Dict[str, Any]:
    """Campos habituales; devuelve ``{"seccion.campo": valor}``."""
    values: Dict[str, Any] = {}
    g = wz.get_value
    k = f"wz-cfg-{project.name}"

    st.markdown("##### Datos")
    c1, c2, c3 = st.columns(3)
    values["data.test_size"] = _num(c1, "test_size", g(effective, "data.test_size"), f"{k}-ts")
    values["data.val_size"] = _num(c2, "val_size", g(effective, "data.val_size"), f"{k}-vs")
    values["data.segment_overlap"] = _num(
        c3, "segment_overlap", g(effective, "data.segment_overlap"), f"{k}-so"
    )

    st.markdown("##### Entrenamiento")
    c1, c2, c3, c4 = st.columns(4)
    values["train.epochs"] = _num(c1, "epochs", g(effective, "train.epochs"), f"{k}-ep")
    values["train.patience"] = _num(c2, "patience", g(effective, "train.patience"), f"{k}-pa")
    scaler = g(effective, "train.scaler")
    options = list(dict.fromkeys(SCALERS + (scaler,)))
    values["train.scaler"] = c3.selectbox("scaler", options, options.index(scaler), key=f"{k}-sc")
    values["train.use_class_weights"] = c4.checkbox(
        "use_class_weights", g(effective, "train.use_class_weights"), key=f"{k}-cw"
    )

    st.markdown("##### Modo de entrenamiento")
    modes = list(wz.TRAIN_MODES)
    mode = st.radio(
        "Modo de entrenamiento",
        modes,
        modes.index(wz.train_mode(effective)),
        format_func=wz.TRAIN_MODE_LABELS.get,
        key=f"{k}-mode",
        label_visibility="collapsed",
    )
    for name, value in wz.TRAIN_MODES[mode].items():
        values[f"optuna.{name}"] = value

    if mode == "fixed":
        from ds_project.models.architectures import DEFAULT_MODEL_PARAMS

        st.markdown("##### Red (train.model_params)")
        st.caption("Hiperparámetros fijos de la red; Optuna no se ejecuta.")
        current = dict(g(effective, "train.model_params") or {})
        model_params = {**DEFAULT_MODEL_PARAMS, **current}
        cols = st.columns(4)
        for i, (name, value) in enumerate(list(model_params.items())):
            if isinstance(value, (int, float)):
                model_params[name] = _num(cols[i % 4], name, value, f"{k}-mp-{name}")
        # Solo se guardan los ya fijados y los que cambian respecto al valor por defecto.
        values["train.model_params"] = {
            name: value
            for name, value in model_params.items()
            if name in current or value != DEFAULT_MODEL_PARAMS.get(name)
        }
    else:
        st.markdown("##### Optuna")
        c1, c2, c3 = st.columns(3)
        values["optuna.n_trials"] = _num(c1, "n_trials", g(effective, "optuna.n_trials"), f"{k}-nt")
        values["optuna.trial_epochs"] = _num(
            c2, "trial_epochs", g(effective, "optuna.trial_epochs"), f"{k}-te"
        )
        timeout = int(
            c3.number_input(
                "timeout (s, 0 = sin límite)",
                min_value=0,
                value=int(g(effective, "optuna.timeout") or 0),
                step=60,
                key=f"{k}-to",
            )
        )
        values["optuna.timeout"] = timeout or None
        if mode == "two_phase":
            c1, c2, c3 = st.columns(3)
            p1 = int(
                c1.number_input(
                    "trials de la fase DSP (0 = n_trials)",
                    min_value=0,
                    value=int(g(effective, "optuna.phase1_trials") or 0),
                    step=1,
                    key=f"{k}-p1t",
                )
            )
            p2 = int(
                c2.number_input(
                    "trials de la fase red (0 = n_trials)",
                    min_value=0,
                    value=int(g(effective, "optuna.phase2_trials") or 0),
                    step=1,
                    key=f"{k}-p2t",
                )
            )
            values["optuna.phase1_trials"] = p1 or None
            values["optuna.phase2_trials"] = p2 or None
            values["optuna.phase1_epochs"] = _num(
                c3, "phase1_epochs", g(effective, "optuna.phase1_epochs"), f"{k}-p1e"
            )
            n1, n2 = phase_trial_counts(int(values["optuna.n_trials"]), p1, p2)
            st.caption(
                f"Fase 1 (DSP, red mínima): {n1} trials · fase 2 (red, DSP fijo): {n2} trials. "
                "La red de la fase 1 se ajusta en `optuna.phase1_model_params`."
            )

    st.markdown("##### Export")
    quant = g(effective, "export.quantization")
    options = list(dict.fromkeys(QUANTIZATIONS + (quant,)))
    values["export.quantization"] = st.selectbox(
        "quantization", options, options.index(quant), key=f"{k}-qu"
    )

    st.markdown("##### DSP")
    if mode in ("search_all", "two_phase"):
        st.caption(
            "En este modo Optuna sustituye estos valores por los de `dsp_search_space`; "
            "se usan tal cual con «DSP fijo» o «Sin Optuna»."
        )
    c1, c2, c3, c4 = st.columns(4)
    values["dsp.sample_rate"] = _num(c1, "sample_rate", g(effective, "dsp.sample_rate"), f"{k}-sr")
    values["dsp.clip_duration"] = _num(
        c2, "clip_duration", g(effective, "dsp.clip_duration"), f"{k}-cd"
    )
    values["dsp.pre_emphasis"] = _num(
        c3, "pre_emphasis", float(g(effective, "dsp.pre_emphasis") or 0.0), f"{k}-pe"
    )
    values["dsp.zero_padding"] = c4.checkbox(
        "zero_padding", bool(g(effective, "dsp.zero_padding")), key=f"{k}-zp"
    )
    aux = list(g(effective, "dsp.aux_features") or [])
    values["dsp.aux_features"] = st.multiselect(
        "aux_features (rama auxiliar)",
        list(dict.fromkeys(list(VALID_FEATURES) + aux)),
        aux,
        key=f"{k}-aux",
    )
    blocks = pd.DataFrame(g(effective, "dsp.mfe_blocks") or [], columns=list(MFE_COLUMNS))
    edited = st.data_editor(blocks, num_rows="dynamic", use_container_width=True, key=f"{k}-mfe")
    values["dsp.mfe_blocks"] = json.loads(edited.to_json(orient="records"))
    return values


def _save(project, overlay: Dict[str, Any], projects_dir: str) -> None:
    try:
        backup = wz.save_overlay(project.product, project.turbine, overlay, projects_dir)
    except ValueError as exc:
        st.error(f"No se guardó: {exc}")
        return
    st.session_state["wz-saved"] = str(backup) if backup else "(sin copia previa)"
    st.rerun()


def _profile_picker(project, overlay, effective, projects_dir: str) -> None:
    current = wz.get_value(effective, "dsp.profile") or "default"
    st.markdown("##### Perfil DSP")
    profiles = list(dict.fromkeys(list(wz.DSP_PROFILES) + [current]))
    c1, c2 = st.columns([3, 1])
    chosen = c1.selectbox(
        "Perfil DSP",
        profiles,
        profiles.index(current),
        format_func=lambda p: f"{p} ({wz.DSP_PROFILES.get(p, 'project.yaml')})",
        key=f"wz-profile-{project.name}",
        help="El export con la plantilla PinBolt (dsp_*.py) exige el perfil pinbolt.",
    )
    if current != wz.EXPORT_PROFILE:
        st.warning(
            f"Con el perfil **{current}** no se podrá exportar con la plantilla DSP: "
            f"elige **{wz.EXPORT_PROFILE}** antes de entrenar."
        )
    if chosen != current:
        st.caption("Sustituye la sección `dsp` de project.yaml por la del perfil elegido.")
        if c2.button(f"Aplicar {chosen}", key="wz-apply-profile"):
            base = wz.base_config(project.product, project.turbine)
            new = wz.normalize_overlay(wz.with_dsp_profile(overlay, chosen), base)
            _save(project, new, projects_dir)


def _template_report(source: Any) -> Any:
    """Comprueba una plantilla (bytes o ruta) reutilizando el resultado si no ha cambiado."""
    import hashlib

    from ds_project.dsp import templates as tpl

    data = source if isinstance(source, bytes) else Path(source).read_bytes()
    digest = hashlib.sha1(data).hexdigest()  # noqa: S324 - solo como clave de caché
    cache = st.session_state.setdefault("wz-tpl-cache", {})
    if digest not in cache:
        with st.spinner("Comprobando la plantilla (export + paridad con el entrenamiento)..."):
            cache[digest] = tpl.inspect_source(data).to_dict()
    return tpl.TemplateReport.from_dict(cache[digest])


def _show_template_report(report) -> None:
    if not report.ok:
        for err in report.errors:
            st.error(err)
        return
    if report.parity:
        st.success(f"Reproduce el DSP de entrenamiento. {report.parity_detail}")
    else:
        st.warning(f"No reproduce el DSP de entrenamiento: {report.parity_detail}")
    for warning in report.warnings:
        if report.parity or "No reproduce" not in warning:
            st.caption(f"⚠️ {warning}")
    c = report.constants
    st.dataframe(
        [
            {
                "constante": "DSP_VERSION / DSP_TYPE",
                "valor": f"{c['dsp_version']} / {c['dsp_type']}",
            },
            {"constante": "SAMPLE_RATE", "valor": str(c["sample_rate"])},
            {"constante": "WINDOW_SIZE_SEC", "valor": str(c["window_size_sec"])},
            {"constante": "WINDOW_STEP_SEC", "valor": str(c["window_step_sec"])},
            {"constante": "PREEMPHASIS_COEFF", "valor": str(c["pre_emphasis"])},
            {"constante": "ZERO_PADDING", "valor": str(c["zero_padding"])},
            {"constante": "AUX_FEATURE_DIM", "valor": str(c["aux_feature_dim"])},
            {"constante": "CLASS_NAMES", "valor": ", ".join(c["class_names"])},
        ],
        hide_index=True,
        use_container_width=True,
    )
    st.dataframe(pd.DataFrame(c["mfe_blocks"]), use_container_width=True)


def _apply_template(project, overlay, constants, projects_dir: str) -> None:
    base = wz.base_config(project.product, project.turbine)
    new = wz.normalize_overlay(wz.with_template_dsp(overlay, constants), base)
    _save(project, new, projects_dir)


def _template_section(project, version, projects_dir, overlay, active) -> None:
    from ds_project.dsp import templates as tpl

    st.markdown("##### Plantilla DSP (dsp_template.py)")
    done = st.session_state.pop("wz-tpl-done", None)
    if done:
        st.success(done)
    slots = tpl.template_slots(project.path, version, "templates")
    st.dataframe(
        [
            {
                "nivel": s.level,
                "ruta": s.path.as_posix(),
                "estado": "✅ activa" if s.active else ("existe" if s.exists else "—"),
            }
            for s in slots
        ],
        hide_index=True,
        use_container_width=True,
    )
    st.caption(
        "El export usa la primera que exista: versión de datos → proyecto → común. "
        "Aplicar su DSP fija en project.yaml los bloques MFE, ventana y pre-énfasis de la "
        "plantilla y desactiva la búsqueda del DSP (Optuna solo ajusta la red)."
    )
    current = next((s for s in slots if s.active), None)
    if current is not None:
        c1, c2 = st.columns(2)
        if c1.toggle("🔍 Ver la plantilla activa", key="wz-tpl-show"):
            report = _template_report(current.path)
            _show_template_report(report)
            if report.ok and st.button(
                "Aplicar el DSP de la plantilla activa a project.yaml",
                disabled=bool(active),
                key="wz-tpl-apply",
            ):
                _apply_template(project, overlay, report.constants, projects_dir)
        removable = [s for s in slots if s.exists and s.level != tpl.LEVEL_COMMON]
        if removable and c2.button(
            f"🗑 Retirar la plantilla de {removable[0].level}",
            disabled=bool(active),
            key="wz-tpl-remove",
            help="Se mueve a templates/.history/; vuelve a usarse la del nivel siguiente.",
        ):
            dv = version if removable[0].level == tpl.LEVEL_VERSION else ""
            backup = tpl.remove_template(project.path, dv)
            st.session_state["wz-tpl-done"] = f"Plantilla retirada (copia en `{backup}`)."
            st.rerun()

    with st.expander("⬆ Subir una plantilla DSP para este proyecto"):
        mode = st.radio(
            "Origen", ["Subir archivo", "Ruta local"], horizontal=True, key="wz-tpl-mode"
        )
        source: Any = None
        if mode == "Subir archivo":
            upload = st.file_uploader("dsp_template.py", type=["py"], key="wz-tpl-up")
            source = upload.getvalue() if upload is not None else None
        else:
            path = st.text_input("Ruta del .py", key="wz-tpl-path").strip().strip('"')
            if path and not Path(path).is_file():
                st.error(f"No existe el fichero `{path}`.")
            elif path:
                source = Path(path).read_bytes()
        if source is None:
            return
        report = _template_report(source)
        _show_template_report(report)
        if not report.ok:
            return
        levels = {"Todo el proyecto": ""}
        if version:
            levels[f"Solo la versión de datos {version}"] = version
        target = st.radio("Instalar para", list(levels), horizontal=True, key="wz-tpl-level")
        force = False
        if not report.parity:
            force = st.checkbox(
                "Instalar igualmente (el modelo no verá las mismas features que el dsp_*.py)",
                key="wz-tpl-force",
            )
        apply = st.checkbox("Aplicar también su DSP a project.yaml", True, key="wz-tpl-apply-cfg")
        if st.button(
            "💾 Instalar plantilla",
            disabled=bool(active) or (not report.parity and not force),
            key="wz-tpl-install",
        ):
            try:
                result = tpl.install_template(source, project.path, levels[target], force)
            except tpl.TemplateError as exc:
                st.error(str(exc))
                return
            msg = f"Plantilla instalada en `{result.path.as_posix()}`."
            if result.backup:
                msg += f" Anterior en `{result.backup.as_posix()}`."
            st.session_state["wz-tpl-done"] = msg
            if apply:
                _apply_template(project, overlay, report.constants, projects_dir)
            st.rerun()


def _step_config(project, version, projects_dir, checks, active) -> None:
    base = wz.base_config(project.product, project.turbine)
    overlay = wz.read_overlay(project.product, project.turbine, projects_dir)
    effective = _effective(project, version)
    path = wz.overlay_path(project.product, project.turbine, projects_dir)
    st.markdown(
        f"Los valores base salen de `configs/*.yaml`; lo que cambies aquí se guarda solo en "
        f"`{path.as_posix()}` (copia de la versión anterior en `{wz.HISTORY_DIR}/`)."
    )
    saved = st.session_state.pop("wz-saved", None)
    if saved:
        st.success(f"project.yaml guardado. Copia anterior: `{saved}`")
    template = next((c for c in checks if c.name == "Plantilla DSP"), None)
    if template and template.status == services.ERROR:
        st.error(f"Plantilla DSP: {template.detail}")
    _template_section(project, version, projects_dir, overlay, active)
    if not effective:
        st.error("La configuración actual no es válida; corrígela en el editor YAML.")
    else:
        _profile_picker(project, overlay, effective, projects_dir)
        values = _config_form(project, effective)
        new = dict(overlay)
        for key, value in values.items():
            new = wz.set_override(new, base, key, value)
        try:
            preview = services.to_builtin(
                wz.validate_overlay(project.product, project.turbine, new)
            )
            diff = wz.overlay_diff(base, preview)
            error = ""
        except ValueError as exc:
            diff, error = [], str(exc)
        st.markdown("##### Diferencias con la configuración base")
        if diff:
            st.dataframe(diff, hide_index=True, use_container_width=True)
        else:
            st.caption("Ninguna: el proyecto usa la configuración base.")
        if error:
            st.error(error)
        changed = wz.normalize_overlay(new, base) != wz.normalize_overlay(overlay, base)
        if changed:
            st.warning("Hay cambios sin guardar. Un cambio de configuración genera un run nuevo.")
        if st.button("💾 Guardar project.yaml", disabled=not changed or bool(error), key="wz-save"):
            _save(project, new, projects_dir)

    with st.expander("Editor YAML avanzado"):
        text = st.text_area(
            "project.yaml (secciones: paths, data, train, export, storage, optuna, dsp)",
            yaml.safe_dump(overlay, sort_keys=False, allow_unicode=True) if overlay else "",
            height=300,
            key=f"wz-yaml-{project.name}",
        )
        if st.button("💾 Guardar YAML", key="wz-save-yaml"):
            try:
                data = yaml.safe_load(text) or {}
            except yaml.YAMLError as exc:
                st.error(f"YAML inválido: {exc}")
            else:
                if isinstance(data, dict):
                    _save(project, data, projects_dir)
                else:
                    st.error("project.yaml debe ser un diccionario de secciones")
        history = sorted((path.parent / wz.HISTORY_DIR).glob("project_*.yaml"), reverse=True)
        if history:
            st.caption(
                f"{len(history)} versiones anteriores en {wz.HISTORY_DIR}/; la más reciente:"
            )
            st.code(history[0].read_text(encoding="utf-8"), language="yaml")


def _step_train(project, version, projects_dir, checks, active) -> None:
    runs = _runs(project, version)
    effective = _effective(project, version)
    optuna = effective.get("optuna", {})
    mode = wz.train_mode(effective) if effective else "search_all"
    st.markdown(f"Modo: **{wz.TRAIN_MODE_LABELS[mode]}**.")
    if mode == "two_phase":
        n1, n2 = phase_trial_counts(
            int(optuna.get("n_trials") or 1),
            optuna.get("phase1_trials"),
            optuna.get("phase2_trials"),
        )
        st.markdown(
            f"Fase 1: **{n1} trials** buscando el DSP con una red mínima "
            f"({optuna.get('phase1_epochs')} épocas). Fase 2: **{n2} trials** buscando la red con "
            f"el mejor DSP (hasta {optuna.get('trial_epochs')} épocas); después, entrenamiento "
            f"final de {effective.get('train', {}).get('epochs')} épocas."
        )
    elif mode != "fixed":
        target = "la red" if mode == "search_net" else "el DSP y la red"
        st.markdown(
            f"Búsqueda Optuna de {target}: **{optuna.get('n_trials')} trials** de hasta "
            f"{optuna.get('trial_epochs')} épocas; después, entrenamiento final de "
            f"{effective.get('train', {}).get('epochs')} épocas."
        )
    else:
        st.markdown(
            f"Se entrena una vez con `train.model_params` y `dsp.mfe_blocks` "
            f"({effective.get('train', {}).get('epochs')} épocas)."
        )
    st.caption("Cambia el modo en el paso Configuración.")
    st.caption("Si ya existe un run con la misma configuración y datos, se reutiliza.")
    profile = effective.get("dsp", {}).get("profile", "default")
    if profile != wz.EXPORT_PROFILE:
        st.warning(
            f"Perfil DSP **{profile}**: el modelo entrenado no se podrá exportar con la plantilla "
            f"(exige **{wz.EXPORT_PROFILE}**)."
        )
        st.button("Ir a Configuración", on_click=_goto, args=("config",), key="wz-to-config")
    params = {"product": project.product, "turbine": project.turbine, "data_version": version}
    problems = services.blockers("train", checks, len(runs))
    execute.launch_button("train", params, projects_dir, problems, active, "wz-train", "▶ Entrenar")
    if not active:
        st.markdown("#### Optuna")
        try:
            common.render_optuna(wz.optuna_progress(project.product, project.turbine, version))
        except Exception as exc:  # noqa: BLE001 - el panel es informativo
            st.caption(f"No se pudo leer el estudio Optuna: {exc}")
    if runs:
        st.markdown(f"#### Último run: `{runs[0].name}`")
        common.render_run(runs[0], "wz-train-run")


def _step_evaluate(project, version, projects_dir, checks, active) -> None:
    runs = _runs(project, version)
    st.markdown("Evalúa el run con `data/test` y genera el informe (Markdown, HTML y PDF).")
    params: Dict[str, Any] = {
        "product": project.product,
        "turbine": project.turbine,
        "data_version": version,
        "run_dir": execute._run_choice(runs, "wz-eval-run"),
    }
    c1, c2 = st.columns(2)
    params["min_accuracy"] = c1.number_input("Accuracy mínima", 0.0, 1.0, 0.0, 0.01, key="wz-acc")
    params["min_f1"] = c2.number_input("F1 macro mínimo", 0.0, 1.0, 0.0, 0.01, key="wz-f1")
    st.caption("Con umbrales a 0 el veredicto es informativo.")
    problems = services.blockers("evaluate", checks, len(runs))
    execute.launch_button(
        "evaluate", params, projects_dir, problems, active, "wz-evaluate", "▶ Evaluar"
    )
    if not runs:
        return
    metrics = wz.test_metrics(runs[0])
    verdict = metrics.get("verdict") or {}
    if metrics:
        segment = metrics.get("segment") or {}
        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy (test)", f"{segment.get('accuracy', 0):.2%}")
        c2.metric("F1 macro (test)", f"{segment.get('f1_macro', 0):.3f}")
        c3.metric("Veredicto", "apto" if verdict.get("passed") else "no apto")
        for fail in verdict.get("failed", []):
            st.error(fail)
    common.render_run(runs[0], "wz-eval-run-view")


def _step_export(project, version, projects_dir, checks, active) -> None:
    runs = _runs(project, version)
    suggested = services.next_version(project.export_versions)
    st.markdown(
        f"Genera `dsp_{project.name}-<v>.py`, `nn_{project.name}-<v>.tflite` y el manifiesto "
        "del dataset en `export/<v>/` (norma ClaudIA)."
    )
    if runs:
        verdict = wz.test_metrics(runs[0]).get("verdict") or {}
        if not verdict:
            st.warning("El último run no está evaluado; conviene evaluarlo antes de exportar.")
        elif not verdict.get("passed"):
            st.warning("El último run no supera los umbrales de evaluación.")
    params: Dict[str, Any] = {
        "product": project.product,
        "turbine": project.turbine,
        "data_version": version,
        "run_dir": execute._run_choice(runs, "wz-export-run"),
        "version": st.text_input("Versión del modelo", suggested, key="wz-export-v"),
    }
    allowed = execute.overwrite_ok("export", params, project, prefix="wz-")
    problems = services.blockers("export", checks, len(runs))
    chosen_run = Path(params["run_dir"]) if params["run_dir"] else (runs[0].path if runs else None)
    profile = wz.run_dsp_profile(chosen_run) if chosen_run else None
    if profile not in (None, wz.EXPORT_PROFILE):
        st.error(
            f"El run se entrenó con el perfil DSP **{profile}** y la plantilla exige "
            f"**{wz.EXPORT_PROFILE}**: el `dsp_*.py` no reproduciría las features del modelo. "
            "Cambia el perfil en Configuración y reentrena."
        )
        st.button("Ir a Configuración", on_click=_goto, args=("config",), key="wz-export-config")
        if st.checkbox(
            "Exportar igualmente (allow_aux_mismatch, solo para pruebas)", key="wz-export-force"
        ):
            params["allow_aux_mismatch"] = True
        else:
            problems = problems + [f"el run usa el perfil DSP {profile}"]
    execute.launch_button(
        "export", params, projects_dir, problems, active, "wz-export", "▶ Exportar", allowed
    )
    if runs:
        done = wz.exports_of_run(project.path, runs[0])
        if done:
            st.success(f"El último run ya está exportado como {', '.join(done)}.")
            rows = services.list_dir(project.path, f"export/{done[-1]}")
            st.dataframe(
                pd.DataFrame(rows).drop(columns=["is_dir"]),
                hide_index=True,
                use_container_width=True,
            )


def _step_dashboard(project, version, projects_dir, checks, active) -> None:
    from ds_project.reporting.dashboard import DEFAULT_LIMITS, write_dashboard

    runs = _runs(project, version)
    st.markdown(
        "Genera un único HTML interactivo y offline para negocio (combinaciones factibles, "
        "rendimiento y parámetros). Los límites son solo valores iniciales; se cambian en el HTML."
    )
    if not runs:
        st.info("Entrena un run antes de generar el dashboard.")
        return
    run = next(
        (
            r
            for r in runs
            if r.name == st.selectbox("Run", [r.name for r in runs], key="wz-dash-run")
        ),
        runs[0],
    )
    c1, c2, c3, c4 = st.columns(4)
    limits = {
        "max_window_s": c1.number_input(
            "Ventana máx. (s)", 0.1, 30.0, float(DEFAULT_LIMITS["max_window_s"]), 0.1, key="wz-d1"
        ),
        "max_size_kb": c2.number_input(
            "Tamaño máx. (KB)",
            1.0,
            100000.0,
            float(DEFAULT_LIMITS["max_size_kb"]),
            32.0,
            key="wz-d2",
        ),
        "max_mops": c3.number_input(
            "Cómputo máx. (Mops/s)",
            0.1,
            100000.0,
            float(DEFAULT_LIMITS["max_mops"]),
            10.0,
            key="wz-d3",
        ),
        "max_false_alarm": c4.number_input(
            "Falsas alarmas máx.",
            0.0,
            1.0,
            float(DEFAULT_LIMITS["max_false_alarm"]),
            0.01,
            key="wz-d4",
        ),
    }
    if wz.test_metrics(run) == {}:
        st.caption("Run sin evaluación de test: el dashboard usará solo la validación interna.")
    if st.button("▶ Generar dashboard", disabled=bool(active), key="wz-dash-gen"):
        try:
            with st.spinner("Generando dashboard..."):
                write_dashboard(run.path, limits=limits)
            st.rerun()
        except Exception as exc:  # noqa: BLE001 - se muestra al usuario
            st.error(f"No se pudo generar el dashboard: {exc}")
    files = wz.dashboards_of_run(run.path)
    if not files:
        return
    path = files[0]
    st.success(f"{path.name} ({path.stat().st_size / 2**20:.1f} MB) en `{path}`")
    st.download_button(
        "Descargar dashboard", path.read_bytes(), path.name, "text/html", key="wz-dash-dl"
    )
    if st.checkbox("Previsualizar aquí", key="wz-dash-preview"):
        components.html(path.read_text(encoding="utf-8"), height=900, scrolling=True)


RENDERERS = {
    "project": _step_project,
    "data": _step_data,
    "config": _step_config,
    "train": _step_train,
    "evaluate": _step_evaluate,
    "export": _step_export,
    "dashboard": _step_dashboard,
}


def render(
    project: Optional[services.ProjectInfo],
    version: str,
    projects_dir: str,
    checks: List[services.Check],
) -> None:
    """Asistente; ``project`` es ``None`` si aún no hay proyectos."""
    exts = _effective(project, version).get("data", {}).get("extensions") if project else None
    steps = wz.wizard_steps(project, version, checks, projects_dir, exts)
    by_key = {s.key: s for s in steps}
    keys = [s.key for s in steps]
    if st.session_state.get(STEP_KEY) not in keys:
        st.session_state[STEP_KEY] = wz.current_step(steps)

    done = sum(s.status == wz.DONE for s in steps)
    st.progress(done / len(steps), text=f"{done} de {len(steps)} pasos completados")
    # Streamlit recrea el radio cuando cambian las etiquetas (iconos de estado), así que el
    # paso vive en STEP_KEY y la clave del widget depende de él para no volver al índice 0.
    current = st.session_state[STEP_KEY]
    radio_key = f"{RADIO_PREFIX}{current}"
    st.radio(
        "Paso",
        keys,
        index=keys.index(current),
        format_func=lambda k: f"{STEP_ICONS[by_key[k].status]} {by_key[k].title}",
        horizontal=True,
        key=radio_key,
        on_change=lambda: _goto(st.session_state[radio_key]),
        label_visibility="collapsed",
    )
    step = by_key[st.session_state[STEP_KEY]]
    st.subheader(f"{STEP_ICONS[step.status]} {step.title}")
    if step.detail:
        st.caption(step.detail)

    active = execute.render_monitor(jobs.jobs_dir(projects_dir), prefix="wz-")
    if step.status == wz.LOCKED or (project is None and step.key != "project"):
        pending = wz.current_step(steps)
        st.info(f"Completa antes «{by_key[pending].title}».")
        st.button("Ir a ese paso", on_click=_goto, args=(pending,), key="wz-goto")
    else:
        RENDERERS[step.key](project, version, projects_dir, checks, active)

    st.divider()
    index = keys.index(step.key)
    left, _, right = st.columns([1, 4, 1])
    if index > 0:
        left.button("← Anterior", on_click=_goto, args=(keys[index - 1],), key="wz-prev")
    if index < len(keys) - 1:
        right.button("Siguiente →", on_click=_goto, args=(keys[index + 1],), key="wz-next")
