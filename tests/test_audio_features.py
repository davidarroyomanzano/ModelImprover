import numpy as np

from ds_project.dsp.params import DSPParams
from ds_project.features import audio
from ds_project.features.build import build_features, build_features_cached


def _sine(freq=1000.0, sr=16000, dur=1.0):
    t = np.arange(int(sr * dur)) / sr
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_frame_signal_count():
    frames = audio.frame_signal(np.zeros(1000, np.float32), 400, 160)
    assert frames.shape == (1 + (1000 - 400) // 160, 400)


def test_sequence_and_aggregate_shapes():
    seq = DSPParams(features=("mfcc", "rms", "zcr"))
    assert audio.extract_features(_sine(), seq).shape == seq.feature_shape
    agg = DSPParams(
        mode="aggregate",
        features=("log_mel", "spectral_centroid"),
        stats=("mean", "max"),
    )
    assert audio.extract_features(_sine(), agg).shape == agg.feature_shape
    assert len(audio.feature_names(agg)) == agg.feature_shape[0]


def test_centroid_close_to_tone():
    p = DSPParams(pre_emphasis=0.0, features=("spectral_centroid",))
    centroid = audio.extract_features(_sine(1000.0), p)[:, 0]
    assert np.allclose(np.median(centroid), 1000.0, rtol=0.05)


def test_build_and_cache(tmp_path):
    p = DSPParams()
    signals = np.stack([_sine(500), _sine(2000)])
    a = build_features_cached(signals, p, tmp_path, "train")
    b = build_features_cached(signals, p, tmp_path, "train")
    assert np.array_equal(a, b)
    assert np.array_equal(a, build_features(signals, p))
