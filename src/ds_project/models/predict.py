"""Inferencia de extremo a extremo: audio -> DSP -> scaler -> TFLite."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

from ds_project.data.process import load_audio, resample
from ds_project.dsp.params import DSPParams
from ds_project.features.audio import extract_auxiliary_features, extract_features
from ds_project.features.scaling import FeatureScaler
from ds_project.models.evaluate import tflite_predict_proba
from ds_project.utils.io import load_json

PathLike = Union[str, Path]


class AudioClassifier:
    def __init__(self, model_path: PathLike, scaler_path: PathLike, dsp_json_path: PathLike):
        artifact = load_json(dsp_json_path)
        self.model_path = Path(model_path)
        self.params = DSPParams.from_dict(artifact["dsp"])
        self.labels = list(artifact["labels"])
        self.scaler = FeatureScaler.load(scaler_path)
        self.feature_scaler = (
            FeatureScaler.load(Path(scaler_path).with_name("feature_scaler.joblib"))
            if self.params.has_aux
            else None
        )

    @classmethod
    def from_run_dir(cls, run_dir: PathLike) -> AudioClassifier:
        run_dir = Path(run_dir)
        return cls(
            run_dir / "model.tflite",
            run_dir / "scaler.joblib",
            run_dir / "dsp_params.json",
        )

    def predict_signal(self, signal: np.ndarray, sr: Optional[int] = None) -> Dict[str, Any]:
        if sr is not None:
            signal = resample(np.asarray(signal, np.float32), sr, self.params.sample_rate)
        feats = self.scaler.transform(extract_features(signal, self.params)[None, ...])
        aux_feats = None
        if self.feature_scaler is not None:
            aux_feats = self.feature_scaler.transform(
                extract_auxiliary_features(signal, self.params)[None, ...]
            )
        probs = (
            tflite_predict_proba(self.model_path, feats, aux_feats)[0]
            if aux_feats is not None
            else tflite_predict_proba(self.model_path, feats)[0]
        )
        idx = int(np.argmax(probs))
        return {
            "label": self.labels[idx],
            "index": idx,
            "probabilities": {lbl: float(p) for lbl, p in zip(self.labels, probs)},
        }

    def predict_file(self, path: PathLike) -> Dict[str, Any]:
        return self.predict_signal(load_audio(path, self.params.sample_rate))
