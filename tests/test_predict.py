from unittest.mock import patch

import numpy as np
import pytest

from ds_project.dsp.export import export_dsp_json
from ds_project.features.scaling import FeatureScaler
from ds_project.models import predict
from ds_project.models.predict import AudioClassifier


@pytest.fixture
def run_dir(tmp_path, small_params):
    feats = np.random.default_rng(0).normal(size=(4, *small_params.feature_shape))
    FeatureScaler().fit(feats.astype(np.float32)).save(tmp_path / "scaler.joblib")
    export_dsp_json(small_params, ["a", "b"], tmp_path / "dsp_params.json")
    (tmp_path / "model.tflite").write_bytes(b"")
    return tmp_path


def _proba(values):
    return patch.object(
        predict, "tflite_predict_proba", return_value=np.array([values], np.float32)
    )


def test_predict_signal(run_dir, small_params, sine):
    with _proba([0.1, 0.9]) as tp:
        clf = AudioClassifier.from_run_dir(run_dir)
        out = clf.predict_signal(sine(1000.0, 8000, 0.1))
    assert clf.params == small_params and clf.labels == ["a", "b"]
    assert (out["label"], out["index"]) == ("b", 1)
    assert out["probabilities"] == pytest.approx({"a": 0.1, "b": 0.9})
    path, feats = tp.call_args.args
    assert path == run_dir / "model.tflite"
    assert feats.shape == (1, *small_params.feature_shape)


def test_predict_signal_resamples(run_dir):
    with patch.object(predict, "resample", wraps=predict.resample) as rs, _proba([0.7, 0.3]):
        out = AudioClassifier.from_run_dir(run_dir).predict_signal(
            np.zeros(1600, np.float32), sr=16000
        )
    assert rs.call_args.args[1:] == (16000, 8000)
    assert out["label"] == "a"


def test_predict_file(run_dir):
    with patch.object(predict, "load_audio", return_value=np.zeros(800, np.float32)) as la, _proba(
        [0.4, 0.6]
    ):
        out = AudioClassifier.from_run_dir(run_dir).predict_file("x.wav")
    la.assert_called_once_with("x.wav", 8000)
    assert out["label"] == "b"
