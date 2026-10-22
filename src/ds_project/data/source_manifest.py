"""Coherencia entre los manifests de origen (``*manifest*.csv``) y los WAV en disco.

Cada fila describe un WAV extraído (``output_file``), su grabación de origen y su rol
(``background``, ``target``, ``hard_negative``). Se cruza por nombre de fichero.
"""

from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

PathLike = Union[str, Path]
_EXAMPLES = 5


def find_source_manifests(data_dir: PathLike) -> List[Path]:
    """CSV ``*manifest*.csv`` en la raíz de ``data_dir`` (no recursivo)."""
    return sorted(Path(data_dir).glob("*manifest*.csv"))


def _basename(value: str) -> str:
    return re.split(r"[\\/]", value.strip())[-1]


def _source_class(source_file: str) -> str:
    parts = re.split(r"[\\/]", source_file.strip())
    return parts[-2] if len(parts) >= 2 else ""


def load_source_rows(manifests: List[Path]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for path in manifests:
        with path.open(newline="", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                row["_manifest"] = path.name
                rows.append(row)
    return rows


def _wavs(root: Path) -> Dict[str, str]:
    """``{nombre: etiqueta}`` de ``root/<etiqueta>/*.wav``."""
    if not root.is_dir():
        return {}
    return {p.name: p.parent.name for p in root.glob("*/*.wav")}


def _wav_frames(path: Path) -> Optional[Dict[str, int]]:
    import wave

    try:
        with wave.open(str(path), "rb") as wf:
            return {"sr": int(wf.getframerate()), "frames": int(wf.getnframes())}
    except (wave.Error, EOFError, OSError):
        pass
    try:
        import soundfile as sf

        info = sf.info(str(path))
    except Exception:  # noqa: BLE001 - fichero ilegible: se informa como incoherente
        return None
    return {"sr": int(info.samplerate), "frames": int(info.frames)}


def validate_source_manifests(
    data_dir: PathLike,
    raw_dir: PathLike,
    test_dir: PathLike,
    sample_rate: Optional[int] = None,
    check_headers: bool = True,
) -> Dict[str, Any]:
    """Valida los manifests de origen frente a ``raw`` y ``test``.

    Comprobaciones: filas sin WAV, WAV sin fila, carpeta distinta a la clase de origen,
    duplicados, grabaciones de origen compartidas entre ``raw`` y ``test``, frecuencia y
    duración de cada WAV frente al manifest. Devuelve un diccionario serializable.
    """
    manifests = find_source_manifests(data_dir)
    result: Dict[str, Any] = {"manifests": [m.name for m in manifests], "available": bool(manifests)}
    if not manifests:
        result.update(ok=True, warnings=["No hay manifests de origen (*manifest*.csv)."])
        return result

    rows = load_source_rows(manifests)
    raw, test = _wavs(Path(raw_dir)), _wavs(Path(test_dir))
    on_disk = {**raw, **test}
    split_of = {**dict.fromkeys(raw, "raw"), **dict.fromkeys(test, "test")}

    names = [_basename(r.get("output_file", "")) for r in rows]
    dup = [n for n, c in Counter(names).items() if c > 1]
    missing = [(n, _source_class(r.get("source_file", ""))) for n, r in zip(names, rows) if n not in on_disk]
    unlisted = sorted(set(on_disk) - set(names))

    mismatch: List[Dict[str, str]] = []
    roles: Dict[str, Counter] = defaultdict(Counter)
    sources: Dict[str, set] = defaultdict(set)
    sr_bad = frames_bad = jitter_fb = 0
    unreadable = 0
    for name, row in zip(names, rows):
        if name not in on_disk:
            continue
        split = split_of[name]
        folder = on_disk[name]
        expected = _source_class(row.get("source_file", ""))
        if expected and expected != folder:
            mismatch.append({"file": name, "folder": folder, "source_class": expected})
        roles[split][row.get("class_role", "?")] += 1
        sources[split].add(row.get("source_stem", ""))
        if row.get("edge_action") == "jitter_fallback_nominal":
            jitter_fb += 1
        if sample_rate and row.get("sample_rate_hz") and int(row["sample_rate_hz"]) != sample_rate:
            sr_bad += 1
        if check_headers:
            base = Path(raw_dir if split == "raw" else test_dir) / folder / name
            info = _wav_frames(base)
            if info is None:
                unreadable += 1
                continue
            try:
                exp_sr = int(row["sample_rate_hz"])
                exp_frames = round(float(row["extract_seconds"]) * exp_sr)
            except (KeyError, ValueError):
                continue
            if info["sr"] != exp_sr:
                sr_bad += 1
            if abs(info["frames"] - exp_frames) > 1:
                frames_bad += 1

    shared = sorted((sources["raw"] & sources["test"]) - {""})
    by_class_missing = Counter(c for _, c in missing)
    by_class_unlisted = Counter(on_disk[n] for n in unlisted)
    prefix_unlisted = Counter(n.split(".")[0] for n in unlisted)

    warnings: List[str] = []
    if missing:
        warnings.append(f"{len(missing)} filas del manifest no tienen WAV en raw/test.")
    if unlisted:
        warnings.append(f"{len(unlisted)} WAV en disco no figuran en ningún manifest.")
    if mismatch:
        warnings.append(f"{len(mismatch)} WAV están en una carpeta distinta a su clase de origen.")
    if dup:
        warnings.append(f"{len(dup)} ficheros repetidos en los manifests.")
    if shared:
        warnings.append(f"{len(shared)} grabaciones de origen aparecen en raw y en test (fuga).")
    if sr_bad:
        warnings.append(f"{sr_bad} WAV con frecuencia de muestreo distinta a la esperada.")
    if frames_bad:
        warnings.append(f"{frames_bad} WAV con duración distinta a extract_seconds.")
    if unreadable:
        warnings.append(f"{unreadable} WAV ilegibles.")

    result.update(
        n_rows=len(rows),
        n_wavs=len(on_disk),
        n_matched=len(rows) - len(missing),
        missing_on_disk={
            "count": len(missing),
            "by_class": dict(sorted(by_class_missing.items())),
            "examples": [n for n, _ in missing[:_EXAMPLES]],
        },
        unlisted_on_disk={
            "count": len(unlisted),
            "by_class": dict(sorted(by_class_unlisted.items())),
            "by_prefix": dict(prefix_unlisted.most_common(6)),
            "examples": unlisted[:_EXAMPLES],
        },
        class_mismatch={"count": len(mismatch), "examples": mismatch[:_EXAMPLES]},
        duplicates=len(dup),
        shared_sources={"count": len(shared), "examples": shared[:_EXAMPLES]},
        sample_rate_mismatch=sr_bad,
        duration_mismatch=frames_bad,
        unreadable=unreadable,
        jitter_fallback=jitter_fb,
        roles_by_split={k: dict(sorted(v.items())) for k, v in roles.items()},
        n_sources={k: len(v - {""}) for k, v in sources.items()},
        warnings=warnings,
        ok=not (mismatch or dup or shared or sr_bad or frames_bad or unreadable),
    )
    return result
