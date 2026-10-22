"""Piezas de interfaz compartidas por las pestañas y el asistente."""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from ds_project import services, wizard
from ds_project.data import importer

ICONS = {services.OK: "✅", services.WARN: "⚠️", services.ERROR: "❌"}
RUN_FIGURES = ("confusion_matrix.png", "training_history.png", "optuna_history.png")


def render_counts(counts: Dict[str, int], title: str) -> None:
    st.markdown(f"#### {title}")
    if not counts:
        st.info("Sin datos.")
        return
    df = pd.DataFrame({"clase": list(counts), "audios": list(counts.values())})
    left, right = st.columns([1, 2])
    left.dataframe(df, hide_index=True, use_container_width=True)
    right.bar_chart(df.set_index("clase"))


def _zip_preview(source: Any, extensions: Optional[List[str]]) -> Optional[importer.ImportPlan]:
    try:
        with zipfile.ZipFile(source) as zf:
            plan = importer.plan_zip(zf, extensions)
    except (importer.ZipImportError, zipfile.BadZipFile, OSError) as exc:
        st.error(f"Zip no válido: {exc}")
        return None
    if hasattr(source, "seek"):
        source.seek(0)
    counts = plan.counts()
    labels = sorted({c for split in counts.values() for c in split})
    st.dataframe(
        pd.DataFrame(
            {
                "clase": labels,
                "data/raw": [counts.get("raw", {}).get(c, 0) for c in labels],
                "data/test": [counts.get("test", {}).get(c, 0) for c in labels],
            }
        ),
        hide_index=True,
    )
    st.caption(
        f"Raíz: `{plan.root or '(raíz del zip)'}` · a data/: "
        + (", ".join(f"`{d}`" for _, d in plan.extras) or "nada")
        + (
            f" · {len(plan.skipped)} ficheros fuera de la estructura (se omiten)"
            if plan.skipped
            else ""
        )
    )
    return plan


def render_zip_import(
    data_dir: Path, extensions: Optional[List[str]], active: Any, prefix: str
) -> None:
    """Sube o indica un .zip y lo importa a ``data_dir`` (training->raw, testing->test)."""
    done = st.session_state.pop(f"{prefix}-zip-done", None)
    if done:
        st.success("Importación completada.")
        st.code(done, language="text")
    with st.expander("📦 Importar datos desde un .zip", expanded=False):
        st.caption(
            "Estructura: `<raíz>/training/<clase>/*.wav` → `data/raw`, "
            "`<raíz>/testing/<clase>/*.wav` → `data/test`, CSV de la raíz (manifests) → `data/`."
        )
        mode = st.radio(
            "Origen", ["Subir archivo", "Ruta local"], horizontal=True, key=f"{prefix}-zip-mode"
        )
        if mode == "Subir archivo":
            source: Any = st.file_uploader("Zip de datos", type=["zip"], key=f"{prefix}-zip-up")
        else:
            path = st.text_input("Ruta del .zip", key=f"{prefix}-zip-path").strip().strip('"')
            source = Path(path) if path else None
            if source is not None and not source.is_file():
                st.error(f"No existe el fichero `{source}`.")
                source = None
        if source is None:
            return
        plan = _zip_preview(source, extensions)
        if plan is None:
            return
        nested = False
        if plan.nested:
            nested = st.checkbox(
                "Extraer también los zip anidados: "
                + ", ".join(f"`{Path(n).name}`" for n in plan.nested)
                + " (si no, se copian a data/ sin extraer)",
                key=f"{prefix}-zip-nested",
            )
        existing = importer.existing_audio(data_dir, extensions)
        replace = False
        if existing:
            st.warning(
                f"Esta versión ya tiene {existing} audios. Lo recomendable es crear una versión de "
                "datos nueva; sustituir cambia los resultados de los runs ya entrenados."
            )
            replace = st.checkbox(
                "Sustituir raw/, test/ y los manifests actuales", key=f"{prefix}-zip-replace"
            )
        if active:
            st.info(f"Hay un proceso en curso ({active['task']}): espera a que termine.")
        disabled = bool(active) or (existing > 0 and not replace)
        if st.button("📥 Importar", type="primary", disabled=disabled, key=f"{prefix}-zip-go"):
            with st.spinner("Extrayendo..."):
                try:
                    result = importer.import_zip(
                        source,
                        data_dir,
                        replace=replace,
                        extract_nested=nested,
                        extensions=extensions,
                    )
                except (importer.ZipImportError, zipfile.BadZipFile, OSError) as exc:
                    st.error(f"No se pudo importar: {exc}")
                    return
            st.session_state[f"{prefix}-zip-done"] = importer.format_result(result)
            st.rerun()


def render_run(run: services.RunInfo, key: str = "run") -> None:
    c1, c2 = st.columns(2)
    c1.metric("Accuracy (validación)", "—" if run.accuracy is None else f"{run.accuracy:.2%}")
    c2.metric("F1 macro (validación)", "—" if run.f1_macro is None else f"{run.f1_macro:.3f}")
    if run.missing:
        st.warning(f"Faltan artefactos: {', '.join(run.missing)}")
    if run.search:
        phases = " → ".join(
            f"{p['phase']}: {p['n_trials']} trials × {p['trial_epochs']} épocas"
            for p in run.search.get("phases", [])
        )
        st.caption(f"Búsqueda Optuna: {run.search.get('mode_label', '')} ({phases})")
        with st.expander("Configuración de la búsqueda"):
            st.json(run.search)
    figures = [run.path / f for f in RUN_FIGURES if (run.path / f).is_file()]
    for fig, col in zip(figures, st.columns(max(len(figures), 1))):
        col.image(str(fig), caption=fig.name)
    evaluation = run.path / "evaluation"
    if not (evaluation / "report.md").is_file():
        st.info("Este run no tiene informe de evaluación (invoke evaluate).")
        return
    st.markdown("#### Informe de evaluación")
    for ext, mime in (("pdf", "application/pdf"), ("html", "text/html"), ("md", "text/markdown")):
        f = evaluation / f"report.{ext}"
        if f.is_file():
            st.download_button(
                f"Descargar {f.name}", f.read_bytes(), f.name, mime, key=f"{key}-dl-{ext}"
            )
    html = evaluation / "report.html"
    if html.is_file():
        components.html(html.read_text(encoding="utf-8"), height=800, scrolling=True)


def render_optuna(progress: Optional[Dict[str, Any]]) -> None:
    """Resumen del estudio Optuna del ámbito (trials, mejor valor y evolución)."""
    if not progress:
        st.caption("Sin estudio Optuna todavía para este ámbito.")
        return
    target = progress.get("target") or 0
    finished = progress["finished"]
    labels = {"dsp": " · fase 1 (DSP)", "net": " · fase 2 (red)"}
    st.markdown(f"**Optuna** · `{progress['study_name']}`{labels.get(progress.get('phase'), '')}")
    if target:
        st.progress(min(finished / target, 1.0), text=f"{finished} / {target} trials")
    states = ", ".join(f"{k.lower()}: {v}" for k, v in sorted(progress["states"].items()))
    c1, c2 = st.columns(2)
    best = progress.get("best_value")
    c1.metric(
        f"Mejor {progress['metric']} ({progress['direction']})",
        "—" if best is None else f"{best:.4f}",
        help=f"trial {progress.get('best_trial')}",
    )
    c2.caption(states)
    if progress["values"]:
        st.line_chart(pd.DataFrame(progress["values"]).set_index("trial"))
    if progress.get("best_params"):
        with st.expander("Mejores hiperparámetros"):
            st.json(progress["best_params"])


def optuna_for_job(job: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    params = job.get("params") or {}
    if job.get("task") not in ("train", "all") or not params.get("product"):
        return None
    try:
        return wizard.optuna_progress(
            params["product"], params.get("turbine", "all"), params.get("data_version", "")
        )
    except Exception:  # noqa: BLE001 - el panel es informativo
        return None
