import dataclasses
from unittest.mock import patch

import numpy as np
import pytest

from ds_project.dsp.params import VALID_FEATURES, VALID_STATS, DSPParams
from ds_project.features import audio
from ds_project.features import build as build_mod
from ds_project.features.build import build_features, build_features_cached


# ---------- Preprocesado ----------
def test_fix_length():
    x = np.arange(5, dtype=np.float32)
    assert audio.fix_length(x, 3).tolist() == [0, 1, 2]
    assert audio.fix_length(x, 7).tolist() == [0, 1, 2, 3, 4, 0, 0]


def test_pre_emphasis():
    x = np.array([1.0, 2.0, 3.0], np.float32)
    assert np.allclose(audio.apply_pre_emphasis(x, 0.5), [1.0, 1.5, 2.0])
    assert np.array_equal(audio.apply_pre_emphasis(x, 0.0), x)


@pytest.mark.parametrize("name", ["hann", "hamming", "rect"])
def test_window(name):
    w = audio.get_window(name, 8)
    assert w.shape == (8,) and w.dtype == np.float32


def test_window_invalid():
    with pytest.raises(ValueError):
        audio.get_window("kaiser", 8)


def test_hann_is_periodic():
    sig = pytest.importorskip("scipy.signal")
    assert np.allclose(audio.get_window("hann", 16), sig.get_window("hann", 16, fftbins=True))


def test_frame_signal():
    assert audio.frame_signal(np.zeros(1000, np.float32), 400, 160).shape == (4, 400)
    assert audio.frame_signal(np.ones(10, np.float32), 16, 8).shape == (1, 16)


# ---------- Mel / MFCC ----------
def test_mel_roundtrip():
    hz = np.array([0.0, 440.0, 8000.0])
    assert np.allclose(audio.mel_to_hz(audio.hz_to_mel(hz)), hz)


def test_mel_filterbank_shape():
    fb = audio.mel_filterbank(16000, 512, 40, 20.0, 8000.0)
    assert fb.shape == (40, 257)
    assert (fb >= 0).all() and (fb.sum(axis=1) > 0).all()


def test_mel_parity_librosa():
    librosa = pytest.importorskip("librosa")
    ref = librosa.filters.mel(
        sr=16000, n_fft=512, n_mels=40, fmin=20.0, fmax=8000.0, htk=True, norm=None
    )
    assert np.allclose(audio.mel_filterbank(16000, 512, 40, 20.0, 8000.0), ref, atol=1e-5)


def test_mfcc_constant_log_mel():
    out = audio.mfcc_from_log_mel(np.ones((3, 20), np.float32), 5)
    assert out.shape == (3, 5)
    assert np.allclose(out[:, 1:], 0.0, atol=1e-5) and (out[:, 0] > 0).all()


# ---------- Features por frame ----------
def test_rms_and_zcr():
    frames = np.array([[3, -3, 3, -3], [1, 1, 1, 1]], np.float32)
    assert np.allclose(audio.rms(frames), [3.0, 1.0])
    assert np.allclose(audio.zero_crossing_rate(frames), [1.0, 0.0])


def test_spectral_single_bin():
    freqs = np.array([0, 100, 200, 300], np.float32)
    mag = np.array([[0, 0, 1, 0]], np.float32)
    assert np.allclose(audio.spectral_centroid(mag, freqs), 200.0)
    assert np.allclose(audio.spectral_bandwidth(mag, freqs), 0.0, atol=1e-3)
    assert np.allclose(audio.spectral_rolloff(mag, freqs, 0.85), 200.0)


def test_spectral_flatness():
    tone = np.zeros((1, 16), np.float32)
    tone[0, 3] = 1.0
    assert np.allclose(audio.spectral_flatness(np.ones((1, 16), np.float32)), 1.0, atol=1e-4)
    assert audio.spectral_flatness(tone)[0] < 1e-3


# ---------- Orquestación ----------
@pytest.mark.parametrize("feat", VALID_FEATURES)
def test_each_feature(small_params, sine, feat):
    p = dataclasses.replace(small_params, features=(feat,))
    out = audio.extract_features(sine(1000.0, 8000, 0.1), p)
    assert out.shape == p.feature_shape and np.isfinite(out).all()


@pytest.mark.parametrize("length", [100, 800, 5000])
def test_any_input_length(small_params, length):
    sig = np.random.default_rng(0).normal(size=length).astype(np.float32)
    assert audio.extract_features(sig, small_params).shape == small_params.feature_shape


def test_aggregate_all_stats(small_params, sine):
    p = dataclasses.replace(
        small_params,
        mode="aggregate",
        features=("log_mel", "spectral_centroid"),
        stats=VALID_STATS,
    )
    out = audio.extract_features(sine(500.0, 8000, 0.1), p)
    assert out.shape == p.feature_shape
    assert len(audio.feature_names(p)) == p.feature_shape[0]


def test_feature_names():
    seq = DSPParams(features=("mfcc", "rms"))
    names = audio.feature_names(seq)
    assert len(names) == 14 and names[0] == "mfcc_0" and names[-1] == "rms"
    agg = DSPParams(mode="aggregate", features=("rms",), stats=("mean", "std"))
    assert audio.feature_names(agg) == ["rms_mean", "rms_std"]


def test_auxiliary_features_are_aggregated(small_params, sine):
    p = dataclasses.replace(
        small_params,
        features=("mfe",),
        aux_features=("rms", "clipping", "cepstral"),
        aux_stats=("mean", "std"),
    )
    out = audio.extract_auxiliary_features(sine(500.0, 8000, 0.1), p)
    assert out.shape == p.aux_feature_shape and np.isfinite(out).all()
    assert len(audio.auxiliary_feature_names(p)) == out.size


def test_centroid_close_to_tone(sine):
    p = DSPParams(pre_emphasis=0.0, features=("spectral_centroid",))
    centroid = audio.extract_features(sine(1000.0), p)[:, 0]
    assert np.allclose(np.median(centroid), 1000.0, rtol=0.05)


def test_silence_rms_zero():
    p = DSPParams(features=("rms",))
    assert np.allclose(audio.extract_features(np.zeros(16000, np.float32), p), 0.0)


# ---------- Lotes y caché ----------
def test_build_empty(small_params):
    assert build_features([], small_params).shape == (0, *small_params.feature_shape)


def test_cache_hit(tmp_path, small_params, sine):
    signals = np.stack([sine(500, 8000, 0.1), sine(2000, 8000, 0.1)])
    with patch.object(build_mod, "build_features", wraps=build_mod.build_features) as spy:
        a = build_features_cached(signals, small_params, tmp_path, "train")
        b = build_features_cached(signals, small_params, tmp_path, "train")
    assert spy.call_count == 1
    assert np.array_equal(a, b)
    assert np.array_equal(a, build_features(signals, small_params))
    assert len(list(tmp_path.glob("train_*.npy"))) == 1


def test_cache_key_depends_on_params(tmp_path, small_params, sine):
    signals = np.stack([sine(500, 8000, 0.1)])
    build_features_cached(signals, small_params, tmp_path, "x")
    build_features_cached(signals, dataclasses.replace(small_params, n_mfcc=5), tmp_path, "x")
    assert len(list(tmp_path.glob("x_*.npy"))) == 2


def test_no_cache_dir(tmp_path, small_params, sine):
    signals = np.stack([sine(500, 8000, 0.1)])
    assert np.array_equal(
        build_features_cached(signals, small_params),
        build_features(signals, small_params),
    )
    assert not list(tmp_path.iterdir())
