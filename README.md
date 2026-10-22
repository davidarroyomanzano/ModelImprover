# Clasificador de audio: DSP, Optuna y TFLite

Pipeline de clasificación de audio para detectar eventos mecánicos en aerogeneradores.

1. Extrae features con un DSP parametrizado (`configs/dsp.yaml`).
2. Busca hiperparámetros con Optuna, en paralelo y con GPU opcional (`configs/optuna.yaml`).
3. Exporta artefactos para firmware: modelo TFLite, cabecera C y parámetros DSP.

## Clases

| Etiqueta | Tipo de muestra | Origen |
|---|---|---|
| `Good` | Fondo (funcionamiento normal) | Real |
| `Stopped` | Fondo (máquina parada) | Real |
| `Hydraulic_charge_start` | Evento | Real |
| `Hydraulic_charge_stop` | Evento | Real (pocas muestras) |
| `Impacts_hub` | Evento | Real + sintético |
| `Impacts_pitch` | Evento | Real + sintético |
| `Impacts_yaw` | Evento | Real |
| `Tendon_rupture` | Evento | Mayoritariamente sintético |

La etiqueta se obtiene del nombre de la carpeta: `data/<split>/<etiqueta>/*.wav`. Ver [Datos](docs/data.md).

> El dataset está muy desbalanceado: `Good` y `Stopped` concentran la mayoría de muestras, mientras que `Hydraulic_charge_stop` tiene unas pocas decenas. Ver [Datos](docs/data.md#balance-de-clases).

## Flujo general

~~~mermaid
flowchart LR
    A[projects/&lt;p&gt;/&lt;t&gt;/&lt;datos&gt;/data/raw] --> B[ingest]
    T[.../data/test] --> B
    B --> C[process<br/>data/interim, data/processed]
    C --> D[features<br/>DSP + scaling]
    D --> E[Optuna tuning<br/>configs/optuna.yaml]
    E --> F[train + evaluate]
    F --> G[export<br/>TFLite + .h + DSP params]
    G --> H[&lt;datos&gt;/models/]
~~~

## Inicio rápido

~~~bash
# 1. Instalar dependencias (entorno virtual local, ver poetry.toml)
poetry install

# 2. Configurar variables de entorno
cp .env.example .env

# 3. Verificar el entorno
poetry run python scripts/check_env.py

# 4. Ejecución mínima de extremo a extremo
poetry run python scripts/smoke_check.py

# 5. Ejecutar los tests
poetry run pytest
~~~

Las tareas habituales (entrenamiento, lint, limpieza) están definidas en `tasks.py`:

~~~bash
poetry run invoke --list
~~~

### Importar datos desde un .zip

Los datos llegan como un zip con `training/` y `testing/` (en la raíz o en una carpeta, p. ej.
`N155_OUT/`). El importador los coloca en la versión de datos del proyecto:

| En el zip | Destino |
|---|---|
| `<raíz>/training/<clase>/*.wav` | `data/raw/<clase>/` |
| `<raíz>/testing/<clase>/*.wav` | `data/test/<clase>/` |
| `<raíz>/*.csv` (manifests) | `data/` |
| `.zip` anidados en `training/`/`testing/` | `data/` sin extraer (o extraídos con `--extract-nested`) |

~~~bash
poetry run invoke import-data --zip-path N155_OUT.zip --product pinbolt --turbine n155 --data-version v2
~~~

Crea el ámbito si falta y no sobrescribe una versión que ya tenga audios salvo con `--replace`
(lo recomendable es una versión de datos nueva). Extrae primero a una carpeta temporal: si algo
falla, los datos previos se mantienen. También desde la UI: paso **Datos** del asistente o
pestaña **Datos** (subiendo el zip o indicando su ruta local).

### Plantilla DSP por proyecto

El export rellena la primera `dsp_template.py` que encuentra, en este orden:
`projects/<p>/<t>/templates/<versión>/` → `projects/<p>/<t>/templates/` → `templates/` (común).
Para subir la de un proyecto:

~~~bash
# Solo comprobar (no instala)
poetry run invoke dsp-template --path dsp_template.py --check-only
# Instalar para todo el proyecto y fijar su DSP en project.yaml
poetry run invoke dsp-template --path dsp_template.py --product pinbolt --turbine n155 --apply-config
# Solo para una versión de datos / retirar la plantilla
poetry run invoke dsp-template --path dsp_template.py --product pinbolt --turbine n155 --data-version v2
poetry run invoke dsp-template --product pinbolt --turbine n155 --remove
~~~

Antes de instalarla se ejecuta en un subproceso y se comprueba que define las constantes que
rellena el export y que, sobre unos clips de prueba, **reproduce el DSP de entrenamiento** (perfil
`pinbolt`) con sus propios bloques MFE. Si la plantilla falla, se rechaza; si funciona pero su
matemática no coincide, solo se instala con `--force` (el modelo vería features distintas en el
dispositivo). La anterior se guarda en `templates/.history/`.

`--apply-config` copia a `project.yaml` sus bloques MFE, `CLIP_DURATION`, `PRE_EMPHASIS`,
`ZERO_PADDING` y `N_FILTERS_FINAL`, y pone `optuna.tune_dsp: false`. El salto entre inferencias
del export es el `WINDOW_STEP_SEC` de la plantilla, salvo que se pase `--window-step-sec`.

Modos de entrenamiento (`optuna.enabled`, `optuna.tune_dsp` y `optuna.strategy`):

| Modo | Qué hace |
|---|---|
| Buscar DSP y red (por defecto) | Optuna prueba `dsp_search_space` y la red a la vez (`strategy: joint`) |
| Optuna en dos fases | `strategy: two_phase`: 1) busca el DSP con una red mínima y pocas épocas; 2) fija el mejor DSP (cacheado) y busca la red |
| DSP fijo, buscar la red | usa `dsp.mfe_blocks` tal cual (p. ej. los de la plantilla) |
| Sin Optuna | DSP fijo y red fija con `train.model_params` |

En dos fases, `optuna.n_trials` es el número de trials de **cada** fase; se puede indicar otro por fase con
`optuna.phase1_trials` (DSP) y `optuna.phase2_trials` (red). La fase 1 usa `optuna.phase1_epochs` (15) y la red mínima de
`models.tuning.PHASE1_MODEL_PARAMS`, ajustable con `optuna.phase1_model_params`. Los estudios se llaman
`<estudio>__dsp` y `<estudio>__net`, y el informe los muestra por separado. Al empezar la búsqueda, cada run
guarda `search_config.json` (modo, estudios, trials y épocas por fase, qué se busca) y lo escribe en el log
(`Búsqueda Optuna: ...`); la UI lo muestra en la columna «búsqueda» y en el detalle del run. Compensa cuando calcular el DSP
domina el coste del trial (audios largos o muchos ficheros); la red simple puede preferir un DSP distinto
al de la red final.

En la UI todo esto está en el paso **Configuración** del asistente: sección «Plantilla DSP»
(niveles, informe de la activa, subida por archivo o ruta local) y selector del modo, con un
editor de los parámetros de la red en el modo «Sin Optuna».

### Interfaz web (opcional)

~~~bash
poetry install -E ui      # una vez
poetry run invoke ui      # http://localhost:8501
~~~

Muestra el estado y las comprobaciones de cada proyecto/versión de datos, el recuento por clase,
los runs con su informe de evaluación, los exports y un explorador de archivos.

### Dashboard para negocio (opcional)

~~~bash
poetry install -E dashboard
poetry run invoke dashboard --product pinbolt --turbine g5x --data-version v0
~~~

Genera `dashboard_<run>.html` dentro del run: un único fichero (~5 MB, sin servidor ni internet)
que se puede enviar por correo. Los límites (`--max-window`, `--max-size-kb`, `--max-mops`,
`--max-false-alarm`) son solo valores iniciales; se cambian con sliders en el propio HTML.
Tamaño y cómputo son estimaciones; las falsas alarmas solo se evalúan para el modelo final.
También se genera, descarga y previsualiza desde el paso «Dashboard» del asistente de la UI
(requiere `poetry install -E dashboard -E ui`).

La pestaña **Asistente** (la primera) guía el flujo completo en seis pasos con su estado
(✅ hecho, 🔵 pendiente, ⚠️ revisar, 🔒 bloqueado hasta completar el anterior):

1. **Proyecto**: elige o crea `projects/<producto>/<turbina>/<versión>/`.
2. **Datos**: importación desde `.zip`, rutas de `data/raw` y `data/test`, recuento por clase y
   avisos (desequilibrio, clases pequeñas, proporción de test, clases distintas).
3. **Configuración**: formulario con los parámetros habituales (datos, entrenamiento, Optuna,
   cuantización, DSP y bloques MFE) y editor YAML avanzado. Solo se guardan en `project.yaml` los
   valores que difieren de `configs/*.yaml`, tras validarlos; la versión anterior se copia en
   `projects/<p>/<t>/.history/`. El selector **Perfil DSP** aplica `configs/dsp_pinbolt.yaml` a la
   sección `dsp`; es necesario para exportar con la plantilla (`dsp_*.py`).
4. **Entrenar**: lanza `train` y muestra el progreso de Optuna en vivo (trials, mejor valor,
   curva y mejores hiperparámetros). Avisa si el perfil DSP no permitirá exportar.
5. **Evaluar**: umbrales opcionales, veredicto e informe.
6. **Exportar**: versión `v<n>` sugerida y ficheros generados. Se bloquea si el run no usa el
   perfil `pinbolt` (salvo `allow_aux_mismatch`, solo para pruebas).

La pestaña **Ejecutar** lanza `init-project`, `train`, `evaluate`, `export` y `all` con formularios
que validan los parámetros (norma de nombres, versión `v<n>`, umbrales) y muestran el comando
`poetry run invoke ...` equivalente. El botón se bloquea si las comprobaciones del proyecto lo
impiden (p. ej. sin datos, sin plantilla DSP o sin runs para evaluar/exportar) y pide confirmación
antes de sobrescribir una versión ya exportada. Solo se ejecuta un trabajo a la vez (GPU y
`optuna.db` compartidos); su log se sigue en vivo y se puede detener. Cada trabajo se guarda en
`projects/.jobs/<id>/` (`job.json` y `output.log`). Si se cierra la UI, el trabajo en curso deja
de seguirse (aparece como «sin seguimiento»).

## Estructura del repositorio

| Ruta | Contenido |
|---|---|
| `configs/` | YAML: general (`config.yaml`), `dsp.yaml`, `optuna.yaml`, `logging.yaml`, `smoke.yaml` |
| `projects/<producto>/<turbina>/` | Un proyecto: `project.yaml`, `templates/`, `export/<vN>/` y, por versión de datos, `<datos>/data/{raw,test,interim,processed}`, `<datos>/models/` (runs, con su informe en `evaluation/`, y `optuna.db`). Solo `project.yaml` y `templates/` se versionan |
| `data/external/` | Datos externos comunes (no versionados) |
| `src/ds_project/` | Paquete: `data`, `dsp`, `features`, `models`, `utils`, `visualization`, `pipeline.py` |
| `scripts/` | `check_env.py`, `smoke_check.py`, `lint_all.py`, `cleanup_data_dirs.py` |
| `templates/` | Plantilla DSP común (fallback de `projects/.../templates/`) |
| `notebooks/` | Exploración (ver `notebooks/README.md`) |
| `tests/` | Suite de pytest |
| `docs/` | Documentación detallada |

## Documentación

- [Instalación y configuración](docs/setup.md)
- [Datos y convenciones de nombres](docs/data.md)
- [Arquitectura](docs/architecture.md)
- [API del paquete](docs/api.md)
- [Ejecución smoke](docs/smoke_run.md)
- [Ejecución completa](docs/full_run.md)
- [Notas para Windows](docs/windows.md)
- [Guía de contribución](docs/contributing.md)
- [Changelog](docs/changelog.md)

## Avisos de higiene del repositorio

- `.env` contiene configuración local y **no debe versionarse**; usar `.env.example` como plantilla.
- El archivo `.gitignore (añadir al final del archivo existente)` es un residuo: su contenido debe fusionarse en `.gitignore` y el archivo eliminarse (lo verifican `tests/test_root_filenames.py` y `tests/test_repo_hygiene.py`).
- `.ruff_cache/` es caché local y debe estar ignorada.
