import joblib
import numpy as np
import pytest

from ds_project.features.scaling import FeatureScaler


def test_scaler_none_is_identity():
    X = np.random.default_rng(0).normal(size=(5, 7)).astype(np.float32)
    s = FeatureScaler("none")
    assert np.array_equal(s.fit_transform(X), X)
    d = s.to_dict()
    assert d["offset"] == [0.0] * 7 and d["divisor"] == [1.0] * 7


@pytest.mark.parametrize("kind", ["standard", "minmax"])
def test_scaler_3d_and_firmware_formula(kind, tmp_path):
    X = np.random.default_rng(0).normal(5.0, 3.0, size=(20, 10, 4)).astype(np.float32)
    s = FeatureScaler(kind)
    Xt = s.fit_transform(X)
    assert Xt.shape == X.shape and Xt.dtype == np.float32
    d = s.to_dict()
    assert d["kind"] == kind and len(d["offset"]) == len(d["divisor"]) == 4
    assert np.allclose((X - np.array(d["offset"])) / np.array(d["divisor"]), Xt, atol=1e-4)
    loaded = FeatureScaler.load(s.save(tmp_path / "sub" / "s.joblib"))
    assert np.allclose(loaded.transform(X), Xt)


def test_standard_stats():
    X = np.random.default_rng(1).normal(2, 4, size=(500, 3)).astype(np.float32)
    Xt = FeatureScaler("standard").fit_transform(X)
    assert np.allclose(Xt.mean(axis=0), 0, atol=1e-4)
    assert np.allclose(Xt.std(axis=0), 1, atol=1e-3)


def test_minmax_constant_column_is_finite():
    X = np.column_stack([np.ones(10), np.arange(10)]).astype(np.float32)
    d = FeatureScaler("minmax").fit(X).to_dict()
    assert np.isfinite(d["offset"]).all() and np.isfinite(d["divisor"]).all()


def test_invalid_kind():
    with pytest.raises(ValueError):
        FeatureScaler("robust")


def test_unfitted_raises():
    with pytest.raises(RuntimeError):
        FeatureScaler().transform(np.zeros((2, 2)))
    with pytest.raises(RuntimeError):
        FeatureScaler().to_dict()


def test_load_wrong_type(tmp_path):
    joblib.dump({"a": 1}, tmp_path / "x.joblib")
    with pytest.raises(TypeError):
        FeatureScaler.load(tmp_path / "x.joblib")


@pytest.mark.xfail(
    strict=True,
    raises=ValueError,
    reason="BUG: sklearn no acepta 0 muestras; run_pipeline falla con X_test vacío",
)
def test_transform_empty():
    s = FeatureScaler().fit(np.ones((4, 2), np.float32))
    assert s.transform(np.zeros((0, 2), np.float32)).shape == (0, 2)
