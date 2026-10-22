"""Identidad del conjunto de datos: manifiesto (ruta, clase, tamaño, sha1) y su hash."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

PathLike = Union[str, Path]


def _sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def build_manifest(
    files: Dict[str, List[Path]],
    root: PathLike,
    cache_path: Optional[PathLike] = None,
) -> Dict[str, Any]:
    """Manifiesto ordenado de ``{etiqueta: [ficheros]}``.

    El sha1 de cada fichero se cachea en ``cache_path`` por (ruta, tamaño, mtime): al ampliar
    el dataset solo se leen los ficheros nuevos o modificados.
    """
    root = Path(root)
    cache: Dict[str, Any] = {}
    if cache_path and Path(cache_path).exists():
        cache = json.loads(Path(cache_path).read_text(encoding="utf-8"))

    entries: List[Dict[str, Any]] = []
    new_cache: Dict[str, Any] = {}
    for label in sorted(files):
        for path in sorted(files[label]):
            rel = _relative(path, root)
            try:
                st = path.stat()
            except OSError:
                entries.append({"path": rel, "label": label, "size": -1, "sha1": ""})
                continue
            hit = cache.get(rel)
            if hit and hit["size"] == st.st_size and hit["mtime_ns"] == st.st_mtime_ns:
                sha1 = hit["sha1"]
            else:
                sha1 = _sha1_file(path)
            new_cache[rel] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "sha1": sha1}
            entries.append({"path": rel, "label": label, "size": st.st_size, "sha1": sha1})

    if cache_path and new_cache and new_cache != cache:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        Path(cache_path).write_text(json.dumps(new_cache), encoding="utf-8")

    payload = json.dumps(entries, sort_keys=True).encode()
    per_class = {label: len(files[label]) for label in sorted(files)}
    return {
        "dataset_hash": hashlib.sha1(payload).hexdigest()[:12],
        "n_files": len(entries),
        "per_class": per_class,
        "files": entries,
    }


def diff_manifests(old: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, List[str]]:
    """Ficheros añadidos, eliminados y modificados entre dos manifiestos."""
    a = {e["path"]: e["sha1"] for e in old["files"]}
    b = {e["path"]: e["sha1"] for e in new["files"]}
    return {
        "added": sorted(set(b) - set(a)),
        "removed": sorted(set(a) - set(b)),
        "changed": sorted(p for p in set(a) & set(b) if a[p] != b[p]),
    }
