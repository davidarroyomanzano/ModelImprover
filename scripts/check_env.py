"""Diagnóstico y reparación de la instalación editable de ds_project.

Detecta si el paquete ``ds_project`` importado proviene de otra copia del
proyecto (por ejemplo, tras mover el repositorio de carpeta) y, con
``--fix``, lo reinstala en modo editable desde la raíz actual.

Uso:
    python scripts/check_env.py          # solo diagnóstico
    python scripts/check_env.py --fix    # reinstala si está desalineado
"""

from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SRC = ROOT / "src"


def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def imported_package_path() -> Path | None:
    """Devuelve la ruta del paquete ds_project importado, o None si no existe."""
    # Quitamos la raíz del repo del path para ver lo que realmente está instalado.
    sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != ROOT]
    try:
        module = importlib.import_module("ds_project")
    except ImportError:
        return None
    return Path(module.__file__).resolve()


def interpreter_is_stale() -> bool:
    """True si el intérprete está fuera del repo y en una ruta 'Proyecto_Data_Science'."""
    exe = Path(sys.executable).resolve()
    return "Proyecto_Data_Science" in str(exe)


def reinstall() -> int:
    print("Desinstalando ds_project...")
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", "ds_project", "ds-project"],
        check=False,
    )
    print(f"Reinstalando en modo editable desde {ROOT} ...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", str(ROOT), "--no-deps"],
        check=False,
    )
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fix", action="store_true", help="Reinstala si está desalineado")
    args = parser.parse_args()

    print(f"Raíz del proyecto : {ROOT}")
    print(f"Intérprete        : {sys.executable}")

    if interpreter_is_stale():
        print(
            "ERROR: el intérprete vive en la ruta antigua. Recrea el entorno:\n"
            "  poetry env remove --all && poetry install"
        )
        return 2

    pkg = imported_package_path()
    print(f"ds_project        : {pkg}")

    if pkg is not None and _is_relative_to(pkg, EXPECTED_SRC):
        print("OK: ds_project se importa desde este repositorio.")
        return 0

    print("AVISO: ds_project no se importa desde este repositorio.")
    if not args.fix:
        print("Ejecuta: python scripts/check_env.py --fix")
        return 1

    code = reinstall()
    if code != 0:
        print("La reinstalación falló.")
        return code
    print("Reinstalación completada. Vuelve a ejecutar el script para verificar.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
