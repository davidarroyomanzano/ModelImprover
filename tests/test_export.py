import warnings

import numpy as np
import pytest

from ds_project.models.export import export_tflite

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message="Statistics for quantized inputs were expected*",
    )


def _converter(fake_tf, payload=b"tfl"):
    conv = fake_tf.lite.TFLiteConverter.from_keras_model.return_value
    conv.convert.return_value = payload
    return conv


# ---------- Rápidos (TF simulado) ----------
@pytest.mark.parametrize("quant", ["float32", "float16", "dynamic"])
def test_export_modes(fake_tf, tmp_path, quant):
    conv = _converter(fake_tf)
    model = object()
    out = export_tflite(model, tmp_path / "sub" / "m.tflite", quant)
    assert out.read_bytes() == b"tfl"
    fake_tf.lite.TFLiteConverter.from_keras_model.assert_called_once_with(model)
    if quant != "float32":
        assert conv.optimizations == [fake_tf.lite.Optimize.DEFAULT]
    if quant == "float16":
        assert conv.target_spec.supported_types == [fake_tf.float16]


def test_export_int8_representative(fake_tf, tmp_path):
    conv = _converter(fake_tf)
    X = np.arange(30, dtype=np.float64).reshape(10, 3)
    export_tflite(
        object(),
        tmp_path / "m.tflite",
        "int8",
        representative_data=X,
        n_representative=4,
    )
    samples = list(conv.representative_dataset())
    assert len(samples) == 4
    assert samples[0][0].shape == (1, 3) and samples[0][0].dtype == np.float32
    assert conv.inference_input_type is fake_tf.int8
    assert conv.inference_output_type is fake_tf.int8
    assert conv.target_spec.supported_ops == [fake_tf.lite.OpsSet.TFLITE_BUILTINS_INT8]


def test_export_int8_two_inputs_representative(fake_tf, tmp_path):
    conv = _converter(fake_tf)
    mfe = np.arange(30, dtype=np.float32).reshape(10, 3)
    feats = np.arange(20, dtype=np.float32).reshape(10, 2)
    export_tflite(
        object(),
        tmp_path / "m.tflite",
        "int8",
        representative_data=[mfe, feats],
        n_representative=4,
    )
    samples = list(conv.representative_dataset())
    assert len(samples) == 4
    assert [sample.shape for sample in samples[0]] == [(1, 3), (1, 2)]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"quantization": "int8"},
        {"quantization": "int8", "representative_data": np.zeros((0, 3))},
        {"quantization": "int4"},
    ],
)
def test_export_errors(fake_tf, tmp_path, kwargs):
    _converter(fake_tf)
    with pytest.raises(ValueError):
        export_tflite(object(), tmp_path / "m.tflite", **kwargs)
    assert not (tmp_path / "m.tflite").exists()


# ---------- Integración (TF real) ----------
@pytest.mark.slow
@pytest.mark.parametrize("shape", [(10, 4), (12,)])
def test_float32_parity(tmp_path, shape):
    pytest.importorskip("tensorflow")
    from ds_project.models.architectures import build_model
    from ds_project.models.evaluate import tflite_predict_proba

    X = np.random.default_rng(0).normal(size=(8, *shape)).astype(np.float32)
    model = build_model(shape, 3, {"n_conv_layers": 1, "units": 8, "filters": 4})
    path = export_tflite(model, tmp_path / "m.tflite", "float32")
    assert np.allclose(tflite_predict_proba(path, X), model.predict(X, verbose=0), atol=1e-4)


@pytest.mark.slow
def test_float32_dual_input_parity(tmp_path):
    pytest.importorskip("tensorflow")
    from ds_project.models.architectures import build_model
    from ds_project.models.evaluate import tflite_predict_proba

    rng = np.random.default_rng(2)
    mfe = rng.normal(size=(8, 10, 4)).astype(np.float32)
    feats = rng.normal(size=(8, 6)).astype(np.float32)
    model = build_model(
        (10, 4), 3, {"n_conv_layers": 1, "units": 8, "filters": 4}, aux_input_shape=(6,)
    )
    path = export_tflite(model, tmp_path / "dual.tflite", "float32")
    expected = model.predict([mfe, feats], verbose=0)
    assert np.allclose(tflite_predict_proba(path, mfe, feats), expected, atol=1e-4)


@pytest.mark.slow
def test_int8_runs(tmp_path):
    pytest.importorskip("tensorflow")
    from ds_project.models.architectures import build_model
    from ds_project.models.evaluate import tflite_predict_proba

    X = np.random.default_rng(1).normal(size=(16, 10, 4)).astype(np.float32)
    model = build_model((10, 4), 2, {"n_conv_layers": 1, "units": 8, "filters": 4})
    path = export_tflite(model, tmp_path / "m.tflite", "int8", representative_data=X)
    probs = tflite_predict_proba(path, X)
    assert probs.shape == (16, 2) and np.isfinite(probs).all()
