"""Elimina subcarpetas de ``data/`` que no se referencian en ningún proceso.

Uso:
    poetry run python scripts/cleanup_data_dirs.py           # solo informa
    poetry run python scripts/cleanup_data_dirs.py --apply   # elimina
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
SCAN_TARGETS = ("src", "configs", "tests", "tasks.py", "pyproject.toml")
SCAN_SUFFIXES = {".py", ".yaml", ".yml", ".toml", ".json"}
# Carpetas que el requerimiento declara en uso explícitamente.
PROTECTED = {"raw", "test"}


def _iter_sources(root: Path):
    for target in SCAN_TARGETS:
        path = root / target
        if path.is_file():
            yield path
        elif path.is_dir():
            yield from (p for p in path.rglob("*") if p.is_file() and p.suffix in SCAN_SUFFIXES)


def find_unused_dirs(root: Path = PROJECT_ROOT) -> list[Path]:
    """Devuelve las subcarpetas de data/ sin referencias en el código o la configuración."""
    data_dir = root / "data"
    candidates = [d for d in sorted(data_dir.iterdir()) if d.is_dir() and d.name not in PROTECTED]
    corpus = "\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in _iter_sources(root))
    unused = []
    for d in candidates:
        patterns = (f"data/{d.name}", f'"{d.name}"', f"'{d.name}'", f": {d.name}")
        if not any(pat in corpus for pat in patterns):
            unused.append(d)
    return unused


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Eliminar realmente")
    args = parser.parse_args(argv)

    unused = find_unused_dirs()
    if not unused:
        print("No hay carpetas sin usar en data/.")
        return 0
    for d in unused:
        if args.apply:
            shutil.rmtree(d)
            print(f"Eliminada: {d.relative_to(PROJECT_ROOT)}")
        else:
            print(f"Sin uso (se eliminaría con --apply): {d.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
