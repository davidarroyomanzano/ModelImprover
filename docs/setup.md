# Instalación y configuración

## Requisitos

- Python según `pyproject.toml`.
- [Poetry](https://python-poetry.org/). `poetry.toml` crea el entorno virtual dentro del proyecto (`.venv/`).
- GPU NVIDIA opcional (CUDA) para acelerar el tuning. Sin GPU el pipeline funciona en CPU.

## Instalación

~~~bash
poetry install
cp .env.example .env
poetry run python scripts/check_env.py
~~~

`check_env.py` comprueba versión de Python, dependencias, disponibilidad de GPU (`ds_project.utils.gpu`) y la estructura de `data/`.

## Variables de entorno

Definidas en `.env` (plantilla en `.env.example`). Se usan para rutas, nivel de log y selección de dispositivo. Nunca se versiona `.env`.

## Ficheros de configuración

| Fichero | Propósito |
|---|---|
| `configs/config.yaml` | Rutas, semilla, splits, clases |
| `configs/dsp.yaml` | Parámetros del DSP (frecuencia de muestreo, ventanas, bandas, etc.) |
| `configs/optuna.yaml` | Espacio de búsqueda, número de trials, paralelismo |
| `configs/logging.yaml` | Configuración de `logging` |
| `configs/smoke.yaml` | Configuración reducida para la ejecución smoke |

La carga se centraliza en `ds_project.config`.

## Herramientas de desarrollo

~~~bash
poetry run pre-commit install
poetry run python scripts/lint_all.py
poetry run pytest
~~~
