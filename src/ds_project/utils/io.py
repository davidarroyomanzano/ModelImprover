"""Utilidades de E/S: JSON/YAML seguros con NumPy, identificadores de ejecución."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Union

import numpy as np
import yaml

PathLike = Union[str, Path]


def ensure_dir(path: PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def to_builtin(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return to_builtin(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_builtin(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def save_json(data: Any, path: PathLike, indent: int = 2) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(
        json.dumps(to_builtin(data), indent=indent, ensure_ascii=False),
        encoding="utf-8",
    )
    return p


def load_json(path: PathLike) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_yaml(data: Any, path: PathLike) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(to_builtin(data), fh, sort_keys=False, allow_unicode=True)
    return p


def load_yaml(path: PathLike) -> Any:
    with Path(path).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def config_hash(data: Any, length: int = 8) -> str:
    payload = json.dumps(to_builtin(data), sort_keys=True).encode()
    return hashlib.sha1(payload).hexdigest()[:length]


def make_run_id(data: Any) -> str:
    return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{config_hash(data)}"
