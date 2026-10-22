"""Conjunto de evaluación externa (``<ámbito>/data/test``).

Las métricas, informes y validaciones externas deben calcularse con este
conjunto, nunca con particiones de ``data/raw``.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

AUDIO_EXTENSIONS: frozenset[str] = frozenset({".wav"})

# Identificador de la grabación de origen: <SITE>.<fuente>.<timestamp de 10 dígitos>
_SOURCE_ID_RE = re.compile(r"^[^.]+\.(?P<site>[^.]+)\.[^.]+\.(?P<ts>\d{10})\.")


@dataclass(frozen=True)
class LabeledFile:
    """Fichero de audio con su etiqueta (nombre de la carpeta padre)."""

    path: Path
    label: str


@dataclass
class TestSetReport:
    """Resultado de validar el conjunto de test frente al de entrenamiento."""

    n_files: int
    class_counts: dict[str, int]
    missing_in_test: list[str] = field(default_factory=list)
    unknown_in_test: list[str] = field(default_factory=list)
    leaked_sources: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Es válido si no hay etiquetas desconocidas ni fugas de datos."""
        return not self.unknown_in_test and not self.leaked_sources


def discover_labeled_files(root: Path) -> list[LabeledFile]:
    """Devuelve los audios de ``root/<etiqueta>/*`` ordenados por ruta."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"No existe el directorio de datos: {root}")
    files = [
        LabeledFile(path=p, label=p.parent.name)
        for p in sorted(root.glob("*/*"))
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    ]
    if not files:
        raise ValueError(f"No se encontraron audios en {root}")
    return files


def source_id(path: Path) -> str | None:
    """Extrae el identificador de la grabación de origen, si existe."""
    match = _SOURCE_ID_RE.match(Path(path).name)
    if match is None:
        return None
    return f"{match['site']}:{match['ts']}"


def _default_dirs() -> tuple[Path, Path]:
    from ds_project.config import load_settings

    paths = load_settings().resolved_paths()
    return Path(paths.raw_dir), Path(paths.test_dir)


def load_test_set(test_dir: Path | None = None) -> tuple[list[Path], list[str]]:
    """Carga rutas y etiquetas del conjunto de evaluación externa (por defecto, el del ámbito)."""
    files = discover_labeled_files(test_dir if test_dir is not None else _default_dirs()[1])
    return [f.path for f in files], [f.label for f in files]


def validate_test_set(
    train_dir: Path | None = None,
    test_dir: Path | None = None,
) -> TestSetReport:
    """Compara etiquetas y detecta grabaciones de origen compartidas."""
    default_train, default_test = _default_dirs()
    train = discover_labeled_files(train_dir if train_dir is not None else default_train)
    test = discover_labeled_files(test_dir if test_dir is not None else default_test)

    train_labels = {f.label for f in train}
    test_labels = {f.label for f in test}

    train_sources = {s for f in train if (s := source_id(f.path))}
    test_sources = {s for f in test if (s := source_id(f.path))}

    return TestSetReport(
        n_files=len(test),
        class_counts=dict(sorted(Counter(f.label for f in test).items())),
        missing_in_test=sorted(train_labels - test_labels),
        unknown_in_test=sorted(test_labels - train_labels),
        leaked_sources=sorted(train_sources & test_sources),
    )
