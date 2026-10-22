# Changelog

## [Sin publicar]

### Añadido
- `invoke dashboard` (extra opcional `dashboard`, plotly): genera un único HTML autocontenido y offline con explorador de combinaciones (filtros cruzados, frente de Pareto), límites de factibilidad ajustables (ventana, tamaño, cómputo, falsas alarmas), rendimiento de test, punto de operación por umbral, parámetros e importancias. Los trials de Optuna guardan `n_params`/`epochs_run`; tamaño y cómputo son estimaciones analíticas del perfil `pinbolt`. El asistente de la UI incorpora el paso «Dashboard» (tras «Exportar») para generarlo, descargarlo y previsualizarlo.
- Búsqueda Optuna en dos fases (`optuna.strategy: two_phase`, cuarto modo del asistente): fase 1 busca el DSP con una red mínima (`phase1_epochs`, `phase1_model_params`), fase 2 busca la red con el mejor DSP; reparto con `phase1_trials`/`phase2_trials` (por defecto `n_trials` en cada fase), estudios `__dsp`/`__net` e informe con ambas fases. `joint` sigue siendo el valor por defecto.
- `search_config.json` en cada run con Optuna (modo `joint`/`two_phase`/`net_only`, trials y épocas por fase), `search_mode` en `optuna_summary.json` y columna «búsqueda» en la tabla de runs de la UI.
- Plantilla DSP por proyecto (`ds_project.dsp.templates`, `invoke dsp-template`, sección «Plantilla DSP» del paso «Configuración»): subida por archivo o ruta a nivel de proyecto o de versión de datos, comprobación en subproceso (constantes y paridad con el DSP `pinbolt` de entrenamiento; sin paridad exige `--force`), copia de la anterior en `templates/.history/` y `--apply-config` para fijar su DSP en `project.yaml`.
- `optuna.tune_dsp` (por defecto `true`): con `false` Optuna solo busca la red y usa `dsp.mfe_blocks`. El asistente ofrece tres modos: buscar DSP y red, DSP fijo + buscar red, y sin Optuna con `train.model_params` editables.
- `invoke export`: `--window-step-sec` por defecto toma `WINDOW_STEP_SEC` de la plantilla (antes, 0.3 fijo).
- Importador de datos desde `.zip` (`ds_project.data.importer`, `invoke import-data`, panel en el paso «Datos» del asistente y en la pestaña «Datos»): `training/<clase>` → `data/raw`, `testing/<clase>` → `data/test`, CSV de la raíz → `data/`; zips anidados opcionales (`--extract-nested`); no sobrescribe sin `--replace` y restaura los datos previos si falla. La UI admite subidas de hasta 4 GB.
- UI: recarga los módulos de `ds_project` modificados con la interfaz abierta (antes, editar p. ej. `wizard.py` sin reiniciar daba `AttributeError`).
- Asistente guiado en la UI (pestaña «Asistente», lógica en `ds_project.wizard`): proyecto → datos → configuración → entrenar → evaluar → exportar, con estado por paso y bloqueo de los posteriores. Editor de `project.yaml` (formulario + YAML) que valida y guarda solo las diferencias con `configs/*.yaml`, con copia previa en `.history/`; selector de perfil DSP (`pinbolt` necesario para exportar con la plantilla); progreso de Optuna en vivo también en el monitor de «Ejecutar». `load_settings(overlay=...)` valida un `project.yaml` sin escribirlo.
- Interfaz web local (`invoke ui`, Streamlit, extra opcional `ui`): estado/validación del proyecto, datos por clase, runs con métricas e informe de evaluación embebido, exports y explorador de archivos. La lógica vive en `ds_project.services` (sin dependencia de la UI).
- Pestaña «Ejecutar» de la UI: lanza `init-project`, `train`, `evaluate`, `export` y `all` en segundo plano (`ds_project.jobs`) con validación previa, comando equivalente, log en vivo, botón de detener, historial en `projects/.jobs/` y un único trabajo activo. Streamlit pasa a `>=1.37,<1.40`.
- `invoke evaluate`: informe de evaluación sobre `data/test` en Markdown, HTML y PDF (`<run>/evaluation/`), con métricas por segmento y fichero, trials de Optuna (`optuna_trials.json`) y validación de los manifests de origen (`ds_project.reporting`, `ds_project.data.source_manifest`). Umbrales opcionales `--min-accuracy` / `--min-f1`; `invoke all` lo ejecuta tras `train` y luego exporta.
- Multiproyecto bajo `projects/<producto>/<turbina>/` (norma de nombres ClaudIA): `--product` / `--turbine` / `--data-version` (o `DS_PRODUCT` / `DS_TURBINE` / `DS_DATA_VERSION`) en `invoke train|export|all|init-project`, por defecto `test` / `all` / `v1`. Datos (`raw`, `interim`, `processed`, `test`), y modelos (con `optuna.db`) cuelgan de `<datos>/`; el export en `export/<vN>/`. Overlay `project.yaml` y `templates/` por proyecto (`DS_PROJECTS_DIR` cambia la raíz). Se elimina `--name`: el paquete se llama `dsp_<producto>_<turbina>-vN.py` / `nn_<producto>_<turbina>-vN.tflite` y la versión debe ser `v<número>`.
- Manifiesto del dataset (`dataset_manifest.json`, sha1 por fichero y `dataset_hash`): entra en el hash del run, se guarda en el run y se copia al export. Los datos no se versionan en Git.
- Optuna: el estudio se llama `<study>__<producto>_<turbina>__<versión>__<hash>` y guarda `product`, `turbine`, `data_version`, `dataset_hash` y `run_hash` como user attrs.
- Plantillas por proyecto/datos: `projects/<p>/<t>/templates/<datos>/dsp_template.py` → `projects/<p>/<t>/templates/dsp_template.py` → `templates/dsp_template.py`.
- `invoke.yaml` con `run.in_stream: false` (evita `OSError 22` en Windows).
- Reproducibilidad: `train.deterministic` (por defecto `true`) activa las operaciones deterministas de TensorFlow; el sampler TPE de cada worker de Optuna usa `optuna.seed + worker_idx` (antes sin semilla) y el modelo final se resemilla con `data.seed` tras la búsqueda.
- Perfil DSP `pinbolt` (`profile: pinbolt`, `configs/dsp_pinbolt.yaml`): MFE aplanado sin escalar + 25 features auxiliares calculadas sobre ese vector, idéntico a `templates/dsp_template.py` (`ds_project.dsp.pinbolt`); `Reshape` previo a la CNN (`mfe_reshape`), `zero_padding`, `n_filters_final` y scaler `none`. Un test verifica la paridad con el DSP generado.
- `invoke export`: genera el paquete PinBolt (`dsp_<producto>_<turbina>-vN.py` + `nn_<producto>_<turbina>-vN.tflite`) rellenando `templates/dsp_template.py` con los resultados de un run (`ds_project.dsp.module_export`).
- Modelo de dos ramas: `input_mfe` (CNN sobre los bloques MFE) e `input_feats` (Dense sobre features auxiliares agregadas), unidas con `Concatenate` antes de la clasificación. Se configura con `aux_features` y `aux_stats` en `configs/dsp.yaml`; vacío mantiene el modelo de una entrada.
- Scaler independiente para la rama auxiliar (`feature_scaler.joblib`, `aux_scaler` en `dsp_params.json` y `DSP_AUX_SCALER_*` en `dsp_params.h`).
- Diagrama de la pipeline, la búsqueda de parámetros y las entradas del modelo en `docs/architecture.md`.
- Entrenamiento, Optuna, evaluación, exportación TFLite (incl. `int8`) e inferencia con dos entradas.
- Features DSP nuevas en `features.audio`: `spectral_rolloff95`, `spectral_entropy`, `spectral_skewness`, `spectral_kurtosis`, `clipping`, `peak_to_rms`, `crest_factor`, `energy_variance`, `harmonic`, `modulation`, `cepstral` y `tkeo` (ver `docs/api.md`).
- Parámetros DSP `n_ceps` y `clipping_threshold`; `dsp_params.json` los incluye, así que hay que regenerarlo al reexportar.
- Conjunto de test hold-out en `<datos>/data/test/` con la misma estructura por clase que `data/raw/`.
- Módulo `ds_project.data.test_set` y tests de fuga entre splits.
- Clase `Tendon_rupture` con muestras sintéticas (`synthetic001`–`100`) y una real (`ES76_008`).
- Datos de nuevas máquinas: `ES1Y_012`, `ES78_005`, `ES76_007`, `ES76_008`.
- Configuración `configs/smoke.yaml` y `scripts/smoke_check.py`.
- Scripts `check_env.py`, `lint_all.py`, `cleanup_data_dirs.py`.
- Tests de higiene del repositorio, de `.gitignore` y de rutas absolutas.
- Documentación: `data.md`, `smoke_run.md`, `full_run.md`, `windows.md`.

### Corregido
- `invoke train` ejecuta `python -m ds_project.pipeline` con `configs/logging.yaml`: los mensajes INFO del pipeline (run, modo de búsqueda) ya aparecen en la consola y en el log de los trabajos de la UI.

### Pendiente
- Fusionar `.gitignore (añadir al final del archivo existente)` en `.gitignore` y eliminarlo.
- Retirar `.env` del control de versiones si está rastreado.
