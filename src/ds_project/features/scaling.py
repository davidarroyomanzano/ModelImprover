"""Scalers ajustados solo con train; soportan tensores (N, ..., F) escalando el último eje."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union

import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

PathLike = Union[str, Path]


class FeatureScaler:
    def __init__(self, kind: str = "standard") -> None:
        if kind not in ("standard", "minmax", "none"):
            raise ValueError("kind debe ser 'standard', 'minmax' o 'none'")
        self.kind = kind
        self._scaler = {"standard": StandardScaler, "minmax": MinMaxScaler}.get(
            kind, lambda: None
        )()
        self._dim = 0
        self.fitted = False

    @staticmethod
    def _flatten(X: np.ndarray) -> np.ndarray:
        return X.reshape(-1, X.shape[-1])

    def fit(self, X: np.ndarray) -> FeatureScaler:
        X = np.asarray(X, dtype=np.float32)
        self._dim = int(X.shape[-1])
        if self._scaler is not None:
            self._scaler.fit(self._flatten(X))
        self.fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("El scaler no está ajustado")
        X = np.asarray(X, dtype=np.float32)
        if self._scaler is None:
            return X
        return self._scaler.transform(self._flatten(X)).reshape(X.shape).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def to_dict(self) -> Dict[str, Any]:
        """Forma unificada para firmware: y = (x - offset) / divisor."""
        if not self.fitted:
            raise RuntimeError("El scaler no está ajustado")
        if self.kind == "none":
            offset = np.zeros(self._dim)
            divisor = np.ones(self._dim)
        elif self.kind == "standard":
            offset = self._scaler.mean_
            divisor = self._scaler.scale_
        else:
            scale = self._scaler.scale_
            safe = np.where(scale == 0, 1.0, scale)
            offset = -self._scaler.min_ / safe
            divisor = 1.0 / safe
        return {
            "kind": self.kind,
            "offset": [float(v) for v in offset],
            "divisor": [float(v) for v in divisor],
        }

    def save(self, path: PathLike) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, path: PathLike) -> FeatureScaler:
        obj = joblib.load(Path(path))
        if not isinstance(obj, cls):
            raise TypeError(f"{path} no contiene un FeatureScaler")
        return obj
