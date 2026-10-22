import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ds_project.models import evaluate as ev
from ds_project.models import train as tr
from ds_project.models.architectures import DEFAULT_MODEL_PARAMS


# ---------- train ----------
def test_class_weights():
    w = tr.compute_class_weights(np.array([0, 0, 0, 1]))
    assert w == pytest.approx({0: 4 / 6, 1: 2.0})


def test_set_global_seed(fake_tf, monkeypatch):
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    tr.set_global_seed(7)
    a = np.random.rand(3)
    tr.set_global_seed(7)
    assert np.array_equal(a, np.random.rand(3))
    assert os.environ["PYTHONHASHSEED"] == "7"
    fake_tf.random.set_seed.assert_called_with(7)


def test_set_global_seed_deterministic(fake_tf, monkeypatch):
    monkeypatch.delenv("TF_DETERMINISTIC_OPS", raising=False)
    tr.set_global_seed(7, deterministic=True)
    fake_tf.config.experimental.enable_op_determinism.assert_called_once()
    assert os.environ["TF_DETERMINISTIC_OPS"] == "1"
    fake_tf.config.experimental.enable_op_determinism.reset_mock()
    tr.set_global_seed(7)
    fake_tf.config.experimental.enable_op_determinism.assert_not_called()


@pytest.mark.parametrize("use_cw", [True, False])
def test_train_model_mocked(fake_tf, use_cw):
    model = MagicMock()
    model.fit.return_value = SimpleNamespace(
        history={"loss": [np.float32(1.0), np.float32(0.5)], "val_loss": [0.9, 0.7]}
    )
    X = np.zeros((4, 5, 3), np.float32)
    y = np.array([0, 0, 0, 1])
    extra = object()
    with patch.object(tr, "build_model", return_value=model) as bm:
        m, h = tr.train_model(
            X,
            y,
            X,
            y,
            n_classes=2,
            params={"batch_size": 2},
            epochs=3,
            patience=4,
            use_class_weights=use_cw,
            extra_callbacks=[extra],
        )
    assert m is model
    shape, n_classes, params = bm.call_args.args
    assert (tuple(shape), n_classes) == ((5, 3), 2)
    assert params["batch_size"] == 2 and params["units"] == DEFAULT_MODEL_PARAMS["units"]
    kw = model.fit.call_args.kwargs
    assert (kw["batch_size"], kw["epochs"]) == (2, 3)
    assert len(kw["callbacks"]) == 3 and kw["callbacks"][-1] is extra
    assert (kw["class_weight"] is not None) == use_cw
    fake_tf.keras.callbacks.EarlyStopping.assert_called_once_with(
        monitor="val_loss", patience=4, restore_best_weights=True
    )
    assert fake_tf.keras.callbacks.ReduceLROnPlateau.call_args.kwargs["patience"] == 2
    assert h == {"loss": [1.0, 0.5], "val_loss": [0.9, 0.7]}
    assert all(isinstance(v, float) for vals in h.values() for v in vals)


# ---------- evaluate ----------
def test_compute_metrics():
    m = ev.compute_metrics(np.array([0, 1, 1]), np.array([0, 1, 1]), ["a", "b", "c"])
    assert m["accuracy"] == 1.0
    assert m["confusion_matrix"] == [[1, 0, 0], [0, 2, 0], [0, 0, 0]]
    assert set(m["report"]) >= {"a", "b", "c"}


def test_evaluate_keras():
    model = MagicMock()
    model.predict.return_value = np.array([[0.9, 0.1], [0.2, 0.8]])
    m = ev.evaluate_keras(model, np.zeros((2, 3)), np.array([0, 0]), ["a", "b"])
    assert m["accuracy"] == 0.5
    model.predict.assert_called_once()


def _interp(fake_tf, inp, out, outputs):
    it = fake_tf.lite.Interpreter.return_value
    it.get_input_details.return_value = [inp]
    it.get_output_details.return_value = [out]
    it.get_tensor.side_effect = outputs
    return it


def test_tflite_predict_float(fake_tf):
    f32 = {"dtype": np.float32, "quantization": (0.0, 0)}
    it = _interp(
        fake_tf,
        {"index": 0, **f32},
        {"index": 1, **f32},
        [np.array([[0.2, 0.8]], np.float32), np.array([[0.6, 0.4]], np.float32)],
    )
    probs = ev.tflite_predict_proba("m.tflite", np.zeros((2, 5, 3)))
    assert np.allclose(probs, [[0.2, 0.8], [0.6, 0.4]])
    fake_tf.lite.Interpreter.assert_called_once_with(model_path="m.tflite")
    it.resize_tensor_input.assert_called_once_with(0, [1, 5, 3])
    assert it.invoke.call_count == 2


def test_tflite_predict_int8(fake_tf):
    it = _interp(
        fake_tf,
        {"index": 0, "dtype": np.int8, "quantization": (0.5, 0)},
        {"index": 1, "dtype": np.int8, "quantization": (1 / 256, -128)},
        [np.array([[-128, 127]], np.int8)],
    )
    probs = ev.tflite_predict_proba("m.tflite", np.array([[1.0, 1000.0]], np.float32))
    sent = it.set_tensor.call_args.args[1]
    assert sent.dtype == np.int8 and sent.tolist() == [[2, 127]]
    assert np.allclose(probs, [[0.0, 255 / 256]])


def test_tflite_predict_empty(fake_tf):
    f32 = {"dtype": np.float32, "quantization": (0.0, 0)}
    _interp(fake_tf, {"index": 0, **f32}, {"index": 1, **f32}, [])
    assert ev.tflite_predict_proba("m.tflite", np.zeros((0, 3))).shape == (0,)


def test_evaluate_tflite():
    with patch.object(ev, "tflite_predict_proba", return_value=np.array([[0.1, 0.9]])) as p:
        m = ev.evaluate_tflite("m.tflite", np.zeros((1, 2)), np.array([1]), ["a", "b"])
    assert m["accuracy"] == 1.0
    assert p.call_args.args[0] == "m.tflite"


# ---------- architectures (TF real) ----------
@pytest.mark.slow
def test_build_model_real():
    pytest.importorskip("tensorflow")
    from ds_project.models.architectures import build_model

    seq = build_model((10, 4), 3, {"n_conv_layers": 2, "filters": 4, "units": 8})
    assert seq.output_shape == (None, 3)
    assert seq.get_layer("conv_1").filters == 8
    assert build_model((12,), 2, {}).output_shape == (None, 2)
    dual = build_model((10, 4), 3, {"n_conv_layers": 1, "units": 8}, aux_input_shape=(6,))
    assert len(dual.inputs) == 2 and dual.get_layer("feature_concat")
    with pytest.raises(ValueError):
        build_model((2, 3, 4), 2, {})
    reshaped = build_model((12,), 2, {"n_conv_layers": 1, "units": 8}, mfe_reshape=(4, 3))
    assert reshaped.get_layer("mfe_reshape").target_shape == (4, 3)
    with pytest.raises(ValueError):
        build_model((12,), 2, {}, mfe_reshape=(5, 3))
