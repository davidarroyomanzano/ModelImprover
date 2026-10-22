# Ejecución completa

## Preparación

Antes de ejecutar la pipeline:

~~~bash
poetry install
poetry run python scripts/check_env.py
~~~

Verificar que los datos están disponibles en
`projects/<producto>/<turbina>/<datos>/data/raw/<clase>/` (por defecto `projects/test/all/v1/`) y
revisar la configuración activa.

## Proyectos y versiones de datos

Todo cuelga de `projects/<producto>/<turbina>/`. Producto y turbina siguen la norma de nombres
ClaudIA (minúsculas, sin `_`): `pinbolt`/`g5x`, `bathear`/`all`... Por defecto `test` / `all` /
`v1` (`data.product`, `data.turbine`, `data.version` en `configs/config.yaml`).

~~~bash
poetry run invoke init-project --product pinbolt --turbine g5x --data-version v3 --export-version v150
poetry run invoke all    --product pinbolt --turbine g5x --data-version v3
poetry run invoke train  --product pinbolt --turbine g5x --data-version v3
poetry run invoke evaluate --product pinbolt --turbine g5x --data-version v3 --min-f1 0.9
poetry run invoke export --product pinbolt --turbine g5x --data-version v3 --version v150
~~~

También con `DS_PRODUCT`, `DS_TURBINE`, `DS_DATA_VERSION` y `DS_PROJECTS_DIR` (raíz de proyectos).

~~~text
projects/<producto>/<turbina>/
  project.yaml                      (overlay: paths, data, train, export, storage, optuna, dsp)
  templates/[<datos>/]dsp_template.py   (→ templates/dsp_template.py global)
  <datos>/data/{raw,interim,processed,test}/
  <datos>/models/<fecha>_<hash>/    + optuna.db (SQLite del estudio)
  export/<vN>/                      dsp_<producto>_<turbina>-vN.py, nn_..-vN.tflite, manifiesto
~~~

- Los datos no se versionan en Git: cada run guarda `dataset_manifest.json` (sha1 por fichero y
  `dataset_hash`), que entra en el hash del run. Añadir muestras crea un run y un estudio nuevos.
- Solo `project.yaml` y `templates/` de cada proyecto se versionan; datos, modelos y
  export están en `.gitignore`.
- El estudio Optuna se llama `<study>__<producto>_<turbina>__<datos>__<hash>` y guarda producto,
  turbina, versión, `dataset_hash` y `run_hash` como atributos.
- `--data-version` identifica los datos (`v3`); `--version` del export es la versión del modelo
  publicado (`v150`, formato `v<número>`).

## Calidad del código

Validaciones recomendadas:

~~~bash
poetry run invoke lint
poetry run invoke test
~~~

El proyecto utiliza Ruff, pytest y pre-commit para garantizar la calidad y consistencia del código.

## Ejecución rápida

Para validar que el proyecto funciona correctamente:

~~~bash
poetry run invoke smoke
~~~

También puede ejecutarse:

~~~bash
poetry run python scripts/check_env.py
~~~

## Pipeline completa

Ejecutar la pipeline completa:

~~~bash
poetry run invoke train
~~~

o bien:

~~~bash
poetry run python -m ds_project.pipeline
~~~

Configuraciones principales:

- `configs/config.yaml`
- `configs/smoke.yaml`
- `configs/optuna.yaml`
- `configs/dsp.yaml`

## Pasos

~~~bash
poetry run python scripts/check_env.py
poetry run invoke --list
poetry run python -m ds_project.pipeline
~~~

## Configuración

- Ajustar `configs/optuna.yaml` (número de trials, `n_jobs`, almacenamiento del estudio).
- Ajustar `configs/dsp.yaml` antes del tuning.
- Revisar `configs/config.yaml`.
- Revisar `configs/smoke.yaml`.
- La GPU se usa automáticamente si está disponible.

## Verificación de artefactos

Comprobar que la ejecución ha generado:

- `model.tflite`
- `model.keras`
- `scaler.joblib`
- `dsp_params.json`
- `metrics.json`
- `feature_scaler.joblib` (solo si `aux_features` no está vacío)

Puede verificarse mediante:

~~~bash
poetry run python scripts/smoke_check.py artifacts
~~~

## Informe de evaluación (`invoke evaluate`)

Entre `train` y `export`. Reevalúa el último run del ámbito con `<datos>/data/test/` (nunca usado
para entrenar) y escribe en `<run>/evaluation/`: `report.md`, `report.html` (autocontenido) y
`report.pdf`, más `metrics_test.json`, `predictions.csv`, `source_manifest_check.json` y
`figures/`. Contenido: resumen y veredicto, datos (por clase y validación contra los
`*manifest*.csv` de origen de `<datos>/data/`), modelo y bloques MFE, métricas por segmento y por
fichero (con IC 95 % por bootstrap de ficheros, ROC-AUC/AP, calibración y errores más
confiados), trials de Optuna (`optuna_trials.json` en el run; en runs antiguos se carga de
`optuna.db`) y reproducibilidad.

~~~bash
poetry run invoke evaluate --product pinbolt --turbine g5x --data-version v3 --min-accuracy 0.95 --min-f1 0.9
~~~

Con `--min-accuracy` / `--min-f1` la tarea termina con código 1 si no se alcanzan (sin ellos el
veredicto es informativo). Si `data/test` está vacío, la evaluación se omite con un aviso.
`invoke all` = lint + test + train + evaluate + export (`--version`, por defecto `v150`).

## Exportar el paquete PinBolt (DSP + TFLite)

Entrena con el perfil `pinbolt`, que calcula el MFE aplanado y las 25 features auxiliares
exactamente como la plantilla (`configs/dsp_pinbolt.yaml`; `configs/dsp.yaml` no cambia):

~~~bash
DS_DSP_CONFIG=configs/dsp_pinbolt.yaml poetry run invoke train --product pinbolt --turbine g5x
poetry run invoke export --product pinbolt --turbine g5x --version v150 --window-step-sec 0.3
~~~

Toma el run más reciente de `<proyecto>/<datos>/models/` (o `--run-dir`) y genera en
`projects/<producto>/<turbina>/export/<versión>/` `dsp_<producto>_<turbina>-<versión>.py`,
`nn_<producto>_<turbina>-<versión>.tflite` y `dataset_manifest_<producto>_<turbina>-<versión>.json`,
según la norma de nombres ClaudIA (la versión debe ser `v<número>`). La plantilla se resuelve por
proyecto y versión de datos (ver arriba) y no se modifica: se sustituyen sus constantes.

| Constante | Origen |
|---|---|
| `SAMPLE_RATE`, `PREEMPHASIS_COEFF` | `dsp_params.json` (`sample_rate`, `pre_emphasis`) |
| `WINDOW_SIZE_SEC` | `clip_duration` |
| `ZERO_PADDING`, `N_FILTERS_FINAL` | `zero_padding`, `n_filters_final` (0 = filtros del primer bloque) |
| `WINDOW_STEP_SEC` | opción `--window-step-sec` (0.3); no sale del entrenamiento |
| `CLASS_NAMES` | `labels` |
| `MFE_BLOCKS` | `mfe_blocks` (tras la búsqueda Optuna) |
| `AUX_FEATURE_DIM`, `AUX_MEAN`, `AUX_SCALE` | `aux_scaler` (`offset`/`divisor`) |

En el perfil `pinbolt` el MFE no se escala, la red aplica `Reshape((N_FRAMES, N_FILTERS_FINAL))`
sobre el vector concatenado y las 25 features se calculan sobre ese vector (como el DSP
desplegado). La cuantización ×256 de la plantilla se replica tal cual.

Otras opciones: `--output-dir`, `--dsp-version`, `--template`. El comando falla si el run no usa
`profile: pinbolt` o su vector auxiliar no tiene la dimensión de la plantilla (25), porque el DSP
generado no reproduciría el entrenamiento; `--allow-aux-mismatch` lo omite. Los runs entrenados
con `configs/dsp.yaml` no son exportables: hay que reentrenar con el perfil.

## Reproducibilidad

Las semillas salen de `data.seed` (splits, entrenamiento final) y `optuna.seed` (sampler TPE y
`optuna.seed + trial` para cada trial). Con `train.deterministic: true` TensorFlow usa operaciones
deterministas (algo más lento; en GPU falla si una op no tiene versión determinista, en ese caso
pon `false`). Con `optuna.n_workers > 1` el orden en que los workers toman trials depende del
tiempo de ejecución, así que la búsqueda solo es reproducible con `n_workers: 1`.

## Salidas

| Ruta | Contenido |
|---|---|
| `projects/<producto>/<turbina>/<datos>/models/` | Runs: modelo TFLite, cabecera C, parámetros DSP, escalador, manifiesto; `optuna.db` |
| `projects/<producto>/<turbina>/<datos>/models/<run>/evaluation/` | Informe de evaluación (md, html, pdf), métricas, predicciones y figuras |
| `projects/<producto>/<turbina>/export/<vN>/` | Paquete DSP + TFLite versionado |

## Evaluación

Las métricas finales se calculan sobre `<datos>/data/test/` del proyecto.

Por el desbalance, reportar F1 macro y recall por clase, especialmente en `Hydraulic_charge_stop` y `Tendon_rupture` (esta última mayoritariamente sintética y con mayor riesgo de degradación en producción).
