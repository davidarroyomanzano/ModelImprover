"""Formatea y valida todo el repositorio (no solo los ficheros staged)."""

from __future__ import annotations

import subprocess
import sys

COMMANDS: list[list[str]] = [
    ["black", "."],
    ["ruff", "check", ".", "--fix"],
    ["ruff", "check", "."],
]


def main() -> int:
    """Ejecuta black y ruff sobre todo el proyecto.

    Returns:
        Código de salida: 0 si todo pasa y 1 si queda algún error.
    """
    status = 0
    for cmd in COMMANDS:
        print(f"$ {' '.join(cmd)}")
        result = subprocess.run([sys.executable, "-m", *cmd], check=False)
        if result.returncode != 0:
            status = 1
    return status


if __name__ == "__main__":
    raise SystemExit(main())
