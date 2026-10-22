"""Importa un ``.zip`` de datos al ámbito ``<proyecto>/<versión>/data``.

Estructura esperada (con o sin carpeta raíz, p. ej. ``N155_OUT/``)::

    <raíz>/training/<clase>/*.wav   -> data/raw/<clase>/
    <raíz>/testing/<clase>/*.wav    -> data/test/<clase>/
    <raíz>/*.csv (manifests, etc.)  -> data/

Otros ficheros dentro de ``training``/``testing`` (p. ej. un ``.zip`` sintético) se copian a
``data/`` sin extraer, salvo que se pida ``extract_nested``: entonces su contenido se importa
con las mismas reglas y sus ficheros sueltos van a ``data/<nombre del zip>/``.
"""

from __future__ import annotations

import argparse
import io
import shutil
import uuid
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import IO, Dict, List, Optional, Sequence, Tuple, Union

from ds_project.data.ingest import DEFAULT_EXTENSIONS, sanitize_filename

PathLike = Union[str, Path]
ZipSource = Union[PathLike, IO[bytes]]
SPLITS = {"training": "raw", "train": "raw", "testing": "test", "test": "test"}
IGNORED = ("__MACOSX",)


class ZipImportError(ValueError):
    """El zip no tiene la estructura esperada o el destino no admite la importación."""


@dataclass
class ImportPlan:
    """Qué se copiará y adónde (rutas relativas a ``data/``)."""

    root: str = ""
    audio: List[Tuple[str, str]] = field(default_factory=list)
    extras: List[Tuple[str, str]] = field(default_factory=list)
    nested: List[str] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)

    def counts(self) -> Dict[str, Dict[str, int]]:
        """``{"raw": {clase: n}, "test": {...}}``."""
        out: Dict[str, Counter] = defaultdict(Counter)
        for _, dest in self.audio:
            split, label = PurePosixPath(dest).parts[:2]
            out[split][label] += 1
        return {s: dict(sorted(c.items())) for s, c in sorted(out.items())}


@dataclass
class ImportResult:
    counts: Dict[str, Dict[str, int]]
    extras: List[str]
    skipped: List[str]
    replaced: bool


def _safe(part: str) -> str:
    name = sanitize_filename(part.strip())
    if name in ("", ".", "..") or any(c in name for c in '<>:"|?*'):
        raise ZipImportError(f"Nombre no válido en el zip: {part!r}")
    return name


def _members(zf: zipfile.ZipFile) -> List[Tuple[str, Tuple[str, ...]]]:
    out = []
    for info in zf.infolist():
        if info.is_dir():
            continue
        parts = tuple(p for p in info.filename.replace("\\", "/").split("/") if p)
        if not parts or parts[0] in IGNORED or parts[-1].startswith("."):
            continue
        out.append((info.filename, parts))
    return out


def _find_root(members: List[Tuple[str, Tuple[str, ...]]]) -> Tuple[str, ...]:
    roots = set()
    for _, parts in members:
        idx = next((i for i, p in enumerate(parts[:2]) if p.lower() in SPLITS), None)
        if idx is not None and idx < len(parts) - 1:
            roots.add(parts[:idx])
    if not roots:
        raise ZipImportError(
            "El zip no contiene carpetas training/ ni testing/ (en la raíz o en una subcarpeta)."
        )
    if len(roots) > 1:
        found = ", ".join(sorted("/".join(r) or "(raíz)" for r in roots))
        raise ZipImportError(f"El zip tiene varias raíces con training/testing: {found}.")
    return roots.pop()


def plan_zip(
    zf: zipfile.ZipFile, extensions: Optional[Sequence[str]] = None, prefix: str = ""
) -> ImportPlan:
    """Clasifica los miembros del zip sin escribir nada."""
    exts = {e.lower() for e in (extensions or DEFAULT_EXTENSIONS)}
    members = _members(zf)
    root = _find_root(members)
    plan = ImportPlan(root="/".join(root))
    for name, parts in members:
        if parts[: len(root)] != root:
            plan.skipped.append(name)
            continue
        rel = parts[len(root) :]
        suffix = PurePosixPath(rel[-1]).suffix.lower()
        if len(rel) == 1:
            plan.extras.append((name, prefix + _safe(rel[0])))
        elif rel[0].lower() not in SPLITS:
            plan.skipped.append(name)
        elif len(rel) == 3 and suffix in exts:
            split = SPLITS[rel[0].lower()]
            plan.audio.append((name, f"{split}/{_safe(rel[1])}/{_safe(rel[2])}"))
        elif len(rel) == 2 and suffix == ".zip":
            plan.nested.append(name)
        else:
            plan.skipped.append(name)
    if not plan.audio and not plan.nested:
        raise ZipImportError("El zip no contiene audios en training/<clase>/ ni testing/<clase>/.")
    return plan


def _check_duplicates(dests: List[str]) -> None:
    dup = [d for d, n in Counter(d.lower() for d in dests).items() if n > 1]
    if dup:
        raise ZipImportError(f"{len(dup)} ficheros colisionan en el destino, p. ej. {dup[0]}.")


def existing_audio(data_dir: PathLike, extensions: Optional[Sequence[str]] = None) -> int:
    exts = {e.lower() for e in (extensions or DEFAULT_EXTENSIONS)}
    data_dir = Path(data_dir)
    return sum(
        1
        for split in ("raw", "test")
        for p in (data_dir / split).glob("*/*")
        if p.is_file() and p.suffix.lower() in exts
    )


def _extract(zf: zipfile.ZipFile, pairs: List[Tuple[str, str]], staging: Path) -> None:
    for member, dest in pairs:
        target = staging / dest
        target.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(member) as src, target.open("wb") as dst:
            shutil.copyfileobj(src, dst)


def _set_aside(data_dir: Path, backup: Path) -> None:
    """Mueve ``raw/``, ``test/`` y los manifests de ``data/`` a ``backup``."""
    backup.mkdir(parents=True)
    for path in [data_dir / "raw", data_dir / "test", *data_dir.glob("*manifest*.csv")]:
        if path.exists():
            path.replace(backup / path.name)


def _restore(data_dir: Path, backup: Path) -> None:
    for split in ("raw", "test"):
        if (backup / split).exists():
            shutil.rmtree(data_dir / split, ignore_errors=True)
    for path in backup.iterdir():
        path.replace(data_dir / path.name)


def _move(staging: Path, data_dir: Path) -> None:
    for src in sorted(p for p in staging.rglob("*") if p.is_file()):
        dest = data_dir / src.relative_to(staging)
        dest.parent.mkdir(parents=True, exist_ok=True)
        src.replace(dest)


def import_zip(
    source: ZipSource,
    data_dir: PathLike,
    replace: bool = False,
    extract_nested: bool = False,
    extensions: Optional[Sequence[str]] = None,
) -> ImportResult:
    """Importa ``source`` en ``data_dir``.

    Falla si ``data/raw`` o ``data/test`` ya tienen audios, salvo con ``replace=True``
    (que sustituye ``raw/``, ``test/`` y los ``*manifest*.csv`` de ``data/``). Extrae primero
    a una carpeta temporal dentro de ``data_dir`` y aparta los datos previos hasta terminar:
    si algo falla, se restauran.
    """
    data_dir = Path(data_dir)
    current = existing_audio(data_dir, extensions)
    if current and not replace:
        raise ZipImportError(
            f"{data_dir} ya contiene {current} audios en raw/test. Usa --replace (o «Sustituir» "
            "en la UI) para sustituirlos, o crea una versión de datos nueva."
        )
    with zipfile.ZipFile(source) as zf:
        plan = plan_zip(zf, extensions)
        audio, extras = list(plan.audio), list(plan.extras)
        skipped = list(plan.skipped)
        inner_zips: List[Tuple[str, zipfile.ZipFile]] = []
        for member in plan.nested:
            if not extract_nested:
                extras.append((member, _safe(PurePosixPath(member).name)))
                continue
            inner = zipfile.ZipFile(io.BytesIO(zf.read(member)))
            stem = _safe(PurePosixPath(member).stem)
            sub = plan_zip(inner, extensions, prefix=f"{stem}/")
            inner_zips.append((member, inner))
            audio += [(f"{member}::{m}", d) for m, d in sub.audio]
            extras += [(f"{member}::{m}", d) for m, d in sub.extras]
            skipped += [f"{member}::{m}" for m in sub.skipped + sub.nested]
        _check_duplicates([d for _, d in audio + extras])

        tag = uuid.uuid4().hex[:8]
        staging = data_dir / f".import-{tag}"
        backup = data_dir / f".replaced-{tag}"
        try:
            outer = [(m, d) for m, d in audio + extras if "::" not in m]
            _extract(zf, outer, staging)
            for member, inner in inner_zips:
                pref = f"{member}::"
                pairs = [(m[len(pref) :], d) for m, d in audio + extras if m.startswith(pref)]
                _extract(inner, pairs, staging)
            if replace:
                _set_aside(data_dir, backup)
            try:
                _move(staging, data_dir)
            except BaseException:
                if backup.exists():
                    _restore(data_dir, backup)
                raise
            shutil.rmtree(backup, ignore_errors=True)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
            for _, inner in inner_zips:
                inner.close()

    counts: Dict[str, Counter] = defaultdict(Counter)
    for _, dest in audio:
        split, label = PurePosixPath(dest).parts[:2]
        counts[split][label] += 1
    for split in ("raw", "test"):
        (data_dir / split).mkdir(parents=True, exist_ok=True)
    return ImportResult(
        counts={s: dict(sorted(c.items())) for s, c in sorted(counts.items())},
        extras=sorted(d for _, d in extras),
        skipped=skipped,
        replaced=bool(current) and replace,
    )


def format_result(result: ImportResult) -> str:
    lines = []
    for split, counts in result.counts.items():
        lines.append(f"data/{split}: {sum(counts.values())} audios")
        lines += [f"  {label}: {n}" for label, n in counts.items()]
    if result.extras:
        lines.append("Otros ficheros en data/: " + ", ".join(result.extras))
    if result.skipped:
        lines.append(f"Omitidos (fuera de la estructura): {len(result.skipped)}")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> None:
    from ds_project.config import load_settings

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("zip_path")
    ap.add_argument("--product", default="")
    ap.add_argument("--turbine", default="")
    ap.add_argument("--data-version", default="")
    ap.add_argument("--replace", action="store_true")
    ap.add_argument("--extract-nested", action="store_true")
    args = ap.parse_args(argv)
    settings = load_settings(
        product=args.product or None,
        turbine=args.turbine or None,
        data_version=args.data_version or None,
    )
    data_dir = Path(settings.resolved_paths().raw_dir).parent
    try:
        result = import_zip(
            args.zip_path,
            data_dir,
            replace=args.replace,
            extract_nested=args.extract_nested,
            extensions=settings.data.extensions,
        )
    except (ZipImportError, zipfile.BadZipFile, FileNotFoundError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
    print(f"Importado en {data_dir}")
    print(format_result(result))


if __name__ == "__main__":
    main()
