"""Tareas de invoke. Se ejecutan siempre con el intérprete del .venv local."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from invoke import task

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

if sys.platform.startswith("win"):
    _VENV_PY = ROOT / ".venv" / "Scripts" / "python.exe"
else:
    _VENV_PY = ROOT / ".venv" / "bin" / "python"

VENV_PY = _VENV_PY if _VENV_PY.exists() else Path(sys.executable)
PY = f'"{VENV_PY}"'
MAX_UPLOAD_MB = 4096


def _assert_local_package() -> None:
    """Aborta si ds_project se importa desde un proyecto distinto a este repo."""
    code = "import ds_project, pathlib; print(pathlib.Path(ds_project.__file__).resolve())"
    result = subprocess.run(
        [str(VENV_PY), "-c", code],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )
    if result.returncode != 0:
        sys.exit(
            "ERROR: no se pudo importar 'ds_project' con "
            f"{VENV_PY}.\nEjecuta: poetry install\n{result.stderr}"
        )
    pkg_path = Path(result.stdout.strip())
    try:
        pkg_path.relative_to(SRC.resolve())
    except ValueError:
        sys.exit(
            "ERROR: 'ds_project' se está importando desde otro proyecto:\n"
            f"  {pkg_path}\n"
            f"Se esperaba dentro de: {SRC}\n"
            "Solución: 'conda deactivate' y luego 'poetry install' en este repo, "
            "o 'pip uninstall -y ds-project' en el entorno activo."
        )


@task
def setup(c):
    """Instala dependencias y los hooks de pre-commit."""
    c.run("poetry install", pty=False)
    c.run(f"{PY} -m pre_commit install", pty=False)


@task
def check(c):
    """Verifica que el paquete importado es el de este repo."""
    _assert_local_package()
    print(f"OK: ds_project se importa desde {SRC} con {VENV_PY}")


@task
def format(c):  # noqa: A001
    """Aplica todos los fixers (ruff + fixers de pre-commit)."""
    c.run(f"{PY} -m ruff check --fix .", pty=False, warn=True)
    c.run(f"{PY} -m ruff format .", pty=False)
    c.run(
        f"{PY} -m pre_commit run --hook-stage manual --all-files",
        pty=False,
        warn=True,
    )


@task
def lint(c):
    """Comprueba el estilo sin modificar archivos."""
    c.run(f"{PY} -m ruff check --no-fix .", pty=False)
    c.run(f"{PY} -m ruff format --check .", pty=False)


@task
def test(c):
    """Ejecuta la suite de tests."""
    _assert_local_package()
    _ensure_scope(c)
    c.run(f"{PY} -m pytest", pty=False)


@task
def smoke(c):
    """Ejecución mínima de extremo a extremo."""
    _assert_local_package()
    c.run(f'{PY} "{ROOT / "scripts" / "smoke_check.py"}"', pty=False)


def _scope_env(product: str, turbine: str, data_version: str) -> dict:
    env = {}
    if product:
        env["DS_PRODUCT"] = product
    if turbine:
        env["DS_TURBINE"] = turbine
    if data_version:
        env["DS_DATA_VERSION"] = data_version
    return env


@task(
    help={
        "product": "Producto (pinbolt, bathear, bladehiss...)",
        "turbine": "Turbina o ámbito (g5x, n155, all...)",
        "data_version": "Versión de los datos (p. ej. v3)",
    }
)
def train(c, product="", turbine="", data_version=""):
    """Ejecuta el pipeline completo en projects/<producto>/<turbina>/<datos>/."""
    _assert_local_package()
    _ensure_scope(c, product, turbine, data_version)
    c.run(
        f"{PY} -m ds_project.pipeline",
        pty=False,
        env=_scope_env(product, turbine, data_version),
    )


@task(
    help={
        "product": "Producto (pinbolt, bathear, bladehiss...)",
        "turbine": "Turbina o ámbito (g5x, n155, all...)",
        "data_version": "Versión de datos: solo considera runs entrenados con ella",
        "run_dir": "Run a exportar (por defecto, el último de <proyecto>/<datos>/models)",
        "output_dir": "Carpeta de salida (por defecto, <proyecto>/export/<version>/)",
        "version": "Versión del modelo, p. ej. v150 (dsp_<producto>_<turbina>-v150.py)",
        "window_step_sec": "Salto entre inferencias en s (0 = WINDOW_STEP_SEC de la plantilla)",
        "dsp_version": "DSP_VERSION (por defecto, el de la plantilla)",
        "template": "Plantilla DSP (por defecto, <proyecto>/templates/<datos>/ -> común)",
        "allow_aux_mismatch": "Permite que el vector auxiliar no coincida con la plantilla",
    }
)
def export(  # noqa: PLR0913
    c,
    product="",
    turbine="",
    data_version="",
    run_dir="",
    output_dir="",
    version="v150",
    window_step_sec=0.0,
    dsp_version="",
    template="",
    allow_aux_mismatch=False,
):
    """Genera dsp_<producto>_<turbina>-<version>.py y nn_...-<version>.tflite desde un run."""
    _assert_local_package()
    code = (
        "import sys;"
        "from ds_project.dsp.module_export import export_for_project;"
        "a=sys.argv[1:];"
        "r=export_for_project(a[0],a[1],a[2],a[3],a[4],a[5],float(a[6]) or None,"
        "a[7] or None,a[8]=='1',a[9],a[10]);"
        "[print(k+':',v) for k,v in r.items()]"
    )
    args = [
        product,
        turbine,
        version,
        run_dir,
        output_dir,
        template and str(ROOT / template),
        str(window_step_sec),
        dsp_version,
        "1" if allow_aux_mismatch else "0",
        data_version,
        str(ROOT / "templates"),
    ]
    quoted = " ".join(f'"{a}"' for a in args)
    c.run(f'{PY} -c "{code}" {quoted}', pty=False)


@task(
    help={
        "product": "Producto (pinbolt, bathear, bladehiss...)",
        "turbine": "Turbina o ámbito (g5x, n155, all...)",
        "data_version": "Versión de los datos (p. ej. v3)",
        "run_dir": "Run a evaluar (por defecto, el último del ámbito)",
        "min_accuracy": "Accuracy mínima por segmento; si no se alcanza, sale con código 1",
        "min_f1": "F1 macro mínimo por segmento; si no se alcanza, sale con código 1",
    }
)
def evaluate(  # noqa: PLR0913
    c, product="", turbine="", data_version="", run_dir="", min_accuracy=0.0, min_f1=0.0
):
    """Evalúa un run con data/test y genera el informe (md, html, pdf) en <run>/evaluation/."""
    _assert_local_package()
    code = (
        "import sys;"
        "from ds_project.reporting.build import evaluate_for_project;"
        "a=sys.argv[1:];"
        "r=evaluate_for_project(a[0],a[1],a[2],a[3],float(a[4]),float(a[5]));"
        "[print(k+':',r[k]) for k in ('run_dir','markdown','html','pdf','accuracy','f1_macro',"
        "'skipped') if k in r];"
        "v=r['verdict'];"
        "print('Veredicto:', 'informativo' if v['informative'] else "
        "('APTO' if v['passed'] else v['failed']));"
        "sys.exit(0 if v['passed'] else 1)"
    )
    args = [product, turbine, data_version, run_dir, str(min_accuracy), str(min_f1)]
    quoted = " ".join(f'"{a}"' for a in args)
    c.run(f'{PY} -c "{code}" {quoted}', pty=False, env=_scope_env(product, turbine, data_version))


@task(
    help={
        "product": "Producto (pinbolt, bathear, bladehiss...)",
        "turbine": "Turbina o ámbito (g5x, n155, all...)",
        "data_version": "Versión de los datos (p. ej. v3)",
        "run_dir": "Run del dashboard (por defecto, el último del ámbito)",
        "output": "Fichero HTML de salida (por defecto <run>/dashboard_<run>.html)",
        "max_window": "Ventana máxima por defecto (s)",
        "max_size_kb": "Tamaño máximo por defecto del modelo (KB)",
        "max_mops": "Cómputo máximo por defecto (Mops/s estimados)",
        "max_false_alarm": "Falsas alarmas máximas por defecto (0-1)",
    }
)
def dashboard(  # noqa: PLR0913
    c,
    product="",
    turbine="",
    data_version="",
    run_dir="",
    output="",
    max_window=1.0,
    max_size_kb=1024.0,
    max_mops=150.0,
    max_false_alarm=0.05,
):
    """Genera un cuadro de mandos HTML autocontenido (un solo fichero) para negocio."""
    _assert_local_package()
    code = (
        "import sys;"
        "from ds_project.reporting.dashboard import dashboard_for_project;"
        "a=sys.argv[1:];"
        "r=dashboard_for_project(a[0],a[1],a[2],a[3],a[4],"
        "{'max_window_s':float(a[5]),'max_size_kb':float(a[6]),"
        "'max_mops':float(a[7]),'max_false_alarm':float(a[8])});"
        "[print(k+':',v) for k,v in r.items()]"
    )
    args = [
        product,
        turbine,
        data_version,
        run_dir,
        output,
        str(max_window),
        str(max_size_kb),
        str(max_mops),
        str(max_false_alarm),
    ]
    quoted = " ".join(f'"{a}"' for a in args)
    c.run(f'{PY} -c "{code}" {quoted}', pty=False, env=_scope_env(product, turbine, data_version))


def _ensure_scope(c, product="", turbine="", data_version=""):
    """Crea (si falta) el ámbito del proyecto activo; es idempotente."""
    code = (
        "import sys;"
        "from ds_project.layout import init_project;"
        "a=sys.argv[1:];"
        "[print('creado:', p) for p in init_project(a[0],a[1],a[2],a[3])]"
    )
    quoted = " ".join(f'"{a}"' for a in (product, turbine, data_version, str(ROOT)))
    c.run(f'{PY} -c "{code}" {quoted}', pty=False)


@task(
    help={
        "product": "Producto (por defecto, el de configs/config.yaml)",
        "turbine": "Turbina o ámbito (g5x, n155, all...)",
        "data_version": "Versión de datos (p. ej. v3)",
        "export_version": "Versión del modelo a preparar en <proyecto>/export/ (opcional)",
    }
)
def init_project(c, product="", turbine="", data_version="", export_version=""):
    """Crea projects/<producto>/<turbina>/ con datos, modelos, export y templates."""
    _assert_local_package()
    code = (
        "import sys;"
        "from ds_project.layout import init_project;"
        "a=sys.argv[1:];"
        "[print(p) for p in init_project(a[0],a[1],a[2],a[3],a[4])]"
    )
    quoted = " ".join(f'"{a}"' for a in (product, turbine, data_version, str(ROOT), export_version))
    c.run(f'{PY} -c "{code}" {quoted}', pty=False)


@task(
    help={
        "zip_path": "Zip con <raíz>/training/<clase>/*.wav, testing/<clase>/*.wav y manifests",
        "product": "Producto (por defecto, el de configs/config.yaml)",
        "turbine": "Turbina o ámbito (g5x, n155, all...)",
        "data_version": "Versión de datos destino (p. ej. v2)",
        "replace": "Sustituye los audios y manifests existentes de esa versión",
        "extract_nested": "Extrae también los .zip anidados (p. ej. datos sintéticos)",
    }
)
def import_data(  # noqa: PLR0913
    c, zip_path, product="", turbine="", data_version="", replace=False, extract_nested=False
):
    """Importa un .zip de datos: training -> data/raw, testing -> data/test, CSV -> data/."""
    _assert_local_package()
    _ensure_scope(c, product, turbine, data_version)
    args = [f'"{zip_path}"']
    scope = {"product": product, "turbine": turbine, "data-version": data_version}
    args += [f'--{k} "{v}"' for k, v in scope.items() if v]
    if replace:
        args.append("--replace")
    if extract_nested:
        args.append("--extract-nested")
    c.run(f"{PY} -m ds_project.data.importer {' '.join(args)}", pty=False)


@task(
    help={
        "path": "dsp_template.py a comprobar o instalar (vacío con --remove)",
        "product": "Producto (por defecto, el de configs/config.yaml)",
        "turbine": "Turbina o ámbito (g5x, n155, all...)",
        "data_version": "Instala solo para esta versión de datos (vacío = todo el proyecto)",
        "check_only": "Solo comprueba la plantilla, sin instalarla",
        "force": "Instala aunque no reproduzca el DSP de entrenamiento",
        "apply_config": "Fija en project.yaml el DSP de la plantilla (sin búsqueda del DSP)",
        "remove": "Retira la plantilla del nivel indicado (queda en .history/)",
    }
)
def dsp_template(  # noqa: PLR0913
    c,
    path="",
    product="",
    turbine="",
    data_version="",
    check_only=False,
    force=False,
    apply_config=False,
    remove=False,
):
    """Comprueba e instala la plantilla DSP de un proyecto (projects/<p>/<t>/templates/)."""
    _assert_local_package()
    if check_only:
        c.run(f'{PY} -m ds_project.dsp.templates check "{path}"', pty=False)
        return
    _ensure_scope(c, product, turbine, data_version)
    args = ["remove"] if remove else ["install", f'"{path}"']
    scope = {"product": product, "turbine": turbine, "data-version": data_version}
    args += [f'--{k} "{v}"' for k, v in scope.items() if v]
    if force and not remove:
        args.append("--force")
    if apply_config and not remove:
        args.append("--apply-config")
    c.run(f"{PY} -m ds_project.dsp.templates {' '.join(args)}", pty=False)


@task(help={"port": "Puerto local de la interfaz"})
def ui(c, port=8501):
    """Abre la interfaz web local (requiere poetry install -E ui)."""
    _assert_local_package()
    app = SRC / "ds_project" / "ui" / "app.py"
    print(f"Interfaz en http://localhost:{port} (Ctrl+C para cerrar)")
    c.run(
        f'{PY} -m streamlit run "{app}" --server.address localhost --server.port {port}'
        " --server.headless true --browser.gatherUsageStats false --client.toolbarMode minimal"
        f" --server.maxUploadSize {MAX_UPLOAD_MB}",
        pty=False,
        env={"DS_REPO_ROOT": str(ROOT)},
    )


@task
def clean(c):
    """Elimina caches locales."""
    for name in (".ruff_cache", ".pytest_cache"):
        shutil.rmtree(ROOT / name, ignore_errors=True)
    for cache in ROOT.rglob("__pycache__"):
        shutil.rmtree(cache, ignore_errors=True)


@task(
    pre=[lint, test],
    name="all",
    help={
        "product": "Producto (pinbolt, bathear, bladehiss...)",
        "turbine": "Turbina o ámbito (g5x, n155, all...)",
        "data_version": "Versión de los datos (p. ej. v3)",
        "version": "Versión del modelo a exportar (p. ej. v150)",
    },
)
def all_(c, product="", turbine="", data_version="", version="v150"):
    """lint + test + train + evaluate + export (inicializa el proyecto si no existe)."""
    train(c, product=product, turbine=turbine, data_version=data_version)
    evaluate(c, product=product, turbine=turbine, data_version=data_version)
    export(c, product=product, turbine=turbine, data_version=data_version, version=version)
    print("Pipeline completo terminado.")
