"""Descubrimiento de audios (estructura raw_dir/<etiqueta>/*.wav) y sincronización remota."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

from ds_project.utils.storage import StorageBackend

logger = logging.getLogger(__name__)
PathLike = Union[str, Path]
DEFAULT_EXTENSIONS = (".wav", ".flac", ".ogg")


def normalize_audio_filenames(
    raw_dir: PathLike,
    extensions: Optional[Sequence[str]] = None,
) -> int:
    """Normaliza los nombres de los ficheros de audio dentro de raw_dir.

    Reemplaza espacios y paréntesis por nombres seguros y devuelve el número
    de archivos renombrados.
    """
    raw_dir = Path(raw_dir)
    exts = {e.lower() for e in (extensions or DEFAULT_EXTENSIONS)}

    renamed = 0

    for p in raw_dir.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in exts:
            continue

        safe_name = sanitize_filename(p.name)

        if safe_name != p.name:
            safe_path = p.with_name(safe_name)

            if not safe_path.exists():
                logger.warning(
                    "Renombrando fichero: '%s' -> '%s'",
                    p.name,
                    safe_name,
                )
                p.rename(safe_path)
                renamed += 1

    return renamed


def list_audio_files(
    raw_dir: PathLike,
    extensions: Optional[Sequence[str]] = None,
) -> Dict[str, List[Path]]:
    """Devuelve {etiqueta: [ficheros]}. La etiqueta es la subcarpeta de primer nivel."""

    normalize_audio_filenames(raw_dir)

    raw_dir = Path(raw_dir)
    exts = {e.lower() for e in (extensions or DEFAULT_EXTENSIONS)}
    result: Dict[str, List[Path]] = {}

    if not raw_dir.exists():
        return result

    for label_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        files = sorted(p for p in label_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts)

        if files:
            result[label_dir.name] = files

    return result


def sync_raw_data(
    storage: StorageBackend,
    remote_prefix: str,
    raw_dir: PathLike,
    extensions: Optional[Sequence[str]] = None,
) -> int:
    """Descarga los audios remotos que falten en raw_dir. Devuelve cuántos se descargaron."""
    raw_dir = Path(raw_dir)
    exts = {e.lower() for e in (extensions or DEFAULT_EXTENSIONS)}
    downloaded = 0
    for key in storage.list_files(remote_prefix):
        rel = key[len(remote_prefix) :].lstrip("/") if remote_prefix else key
        if Path(rel).suffix.lower() not in exts:
            continue
        rel_path = Path(rel)
        safe_parts = [sanitize_filename(part) for part in rel_path.parts]
        local = raw_dir.joinpath(*safe_parts)

        if local.exists():
            continue
        storage.download_file(key, local)
        downloaded += 1
    logger.info("Sincronizados %d ficheros desde '%s'", downloaded, remote_prefix)
    return downloaded


def sanitize_filename(name: str) -> str:
    return name.replace(" ", "_").replace("(", "").replace(")", "")
