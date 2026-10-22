"""Extracción de features por lotes con caché en disco."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

from ds_project.dsp.params import DSPParams
from ds_project.features.audio import extract_auxiliary_features, extract_features

logger = logging.getLogger(__name__)
PathLike = Union[str, Path]


def build_features(signals: Sequence[np.ndarray], params: DSPParams) -> np.ndarray:
    if len(signals) == 0:
        return np.zeros((0, *params.feature_shape), dtype=np.float32)
    return np.stack([extract_features(s, params) for s in signals]).astype(np.float32)


def build_auxiliary_features(signals: Sequence[np.ndarray], params: DSPParams) -> np.ndarray:
    """Construye los vectores de la segunda entrada del modelo."""
    if not params.has_aux:
        return np.empty((len(signals), 0), dtype=np.float32)
    return np.stack([extract_auxiliary_features(s, params) for s in signals]).astype(np.float32)


def build_auxiliary_features_cached(
    signals: np.ndarray,
    params: DSPParams,
    cache_dir: Optional[PathLike] = None,
    name: str = "aux_features",
) -> np.ndarray:
    if cache_dir is None:
        return build_auxiliary_features(signals, params)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{name}_{_cache_key(signals, params)}.npy"
    if path.exists():
        logger.info("Features auxiliares desde caché: %s", path)
        return np.load(path)
    feats = build_auxiliary_features(signals, params)
    np.save(path, feats)
    logger.info("Features auxiliares guardadas en caché: %s %s", path, feats.shape)
    return feats


def _cache_key(signals: np.ndarray, params: DSPParams) -> str:
    h = hashlib.sha1()
    h.update(json.dumps(params.to_dict(), sort_keys=True).encode())
    h.update(np.ascontiguousarray(signals).tobytes())
    return h.hexdigest()[:16]


def build_features_cached(
    signals: np.ndarray,
    params: DSPParams,
    cache_dir: Optional[PathLike] = None,
    name: str = "features",
) -> np.ndarray:
    if cache_dir is None:
        return build_features(signals, params)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{name}_{_cache_key(signals, params)}.npy"
    if path.exists():
        logger.info("Features desde caché: %s", path)
        return np.load(path)
    feats = build_features(signals, params)
    np.save(path, feats)
    logger.info("Features guardadas en caché: %s %s", path, feats.shape)
    return feats
