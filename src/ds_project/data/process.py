"""Carga, remuestreo, segmentación y partición (por fichero, sin fugas) de audios."""

from __future__ import annotations

import logging
from math import gcd
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
from sklearn.model_selection import train_test_split

from ds_project.dsp.params import DSPParams
from ds_project.features.audio import fix_length

logger = logging.getLogger(__name__)
PathLike = Union[str, Path]
SplitItems = List[Tuple[Path, int]]


def resample(signal: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return signal.astype(np.float32)
    from scipy.signal import resample_poly

    g = gcd(int(orig_sr), int(target_sr))
    return resample_poly(signal, target_sr // g, orig_sr // g).astype(np.float32)


def load_audio(path: PathLike, target_sr: int) -> np.ndarray:
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return resample(data, sr, target_sr)


def segment_signal(
    signal: np.ndarray, n_samples: int, overlap: float = 0.0, min_ratio: float = 0.5
) -> List[np.ndarray]:
    signal = np.asarray(signal, dtype=np.float32)
    if signal.shape[0] <= n_samples:
        return [fix_length(signal, n_samples)]
    hop = max(1, int(round(n_samples * (1.0 - overlap))))
    segments: List[np.ndarray] = []
    for start in range(0, signal.shape[0], hop):
        chunk = signal[start : start + n_samples]
        if chunk.shape[0] < n_samples:
            if chunk.shape[0] >= min_ratio * n_samples:
                segments.append(fix_length(chunk, n_samples))
            break
        segments.append(chunk)
    return segments


def _safe_split(items, labels, test_size, seed):
    counts = np.bincount(labels)
    stratify = labels if counts.min() >= 2 else None
    if stratify is None:
        logger.warning("Alguna clase tiene <2 ficheros: partición sin estratificar")
    return train_test_split(
        items, labels, test_size=test_size, random_state=seed, stratify=stratify
    )


def split_files(
    files_by_label: Dict[str, List[Path]], test_size: float, val_size: float, seed: int
) -> Tuple[Dict[str, SplitItems], List[str]]:
    """Partición estratificada a nivel de fichero (evita fugas entre segmentos)."""
    labels = sorted(files_by_label)
    paths: List[Path] = []
    y: List[int] = []
    for idx, lbl in enumerate(labels):
        paths += files_by_label[lbl]
        y += [idx] * len(files_by_label[lbl])
    y_arr = np.asarray(y)

    p_trval, p_test, y_trval, y_test = _safe_split(paths, y_arr, test_size, seed)
    rel_val = val_size / (1.0 - test_size)
    p_train, p_val, y_train, y_val = _safe_split(p_trval, np.asarray(y_trval), rel_val, seed)

    splits = {
        "train": list(zip(p_train, map(int, y_train))),
        "val": list(zip(p_val, map(int, y_val))),
        "test": list(zip(p_test, map(int, y_test))),
    }
    return splits, labels


def load_split(
    items: Sequence[Tuple[Path, int]],
    params: DSPParams,
    overlap: float = 0.0,
    min_ratio: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    signals: List[np.ndarray] = []
    targets: List[int] = []
    for path, label in items:
        audio = load_audio(path, params.sample_rate)
        segs = segment_signal(audio, params.n_samples, overlap, min_ratio)
        signals += segs
        targets += [label] * len(segs)
    if not signals:
        return np.zeros((0, params.n_samples), np.float32), np.zeros((0,), np.int64)
    return np.stack(signals).astype(np.float32), np.asarray(targets, dtype=np.int64)
