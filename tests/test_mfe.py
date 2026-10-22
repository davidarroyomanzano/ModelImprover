import dataclasses
from unittest.mock import MagicMock

import numpy as np
import pytest

from ds_project.config import OptunaCfg, Settings  # pylint: disable=E0611
from ds_project.dsp.params import DSPParams, MFEBlock  # pylint: disable=E0611,E0401
from ds_project.features import audio  # pylint: disable=E0611
from ds_project.models import tuning  # pylint: disable=E0611


def test_mfe_shape_and_range():
    p = DSPParams(
        sample_rate=8000,
        fmax=4000.0,
        clip_duration=0.2,
        features=("mfe",),
        mfe_blocks=(
            MFEBlock(n_filters=8, high_freq=3000.0),
            MFEBlock(n_filters=16, frame_length=0.05, frame_stride=0.025, high_freq=3500.0),
        ),
    )
    sig = np.random.default_rng(0).normal(scale=0.1, size=1600).astype(np.float32)
    out = audio.extract_features(sig, p)
    assert out.shape == p.feature_shape and out.shape[1] == 16
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_mfe_rejects_mixed_features():
    with pytest.raises(ValueError, match="mfe"):
        DSPParams(features=("mfe", "rms"))


def test_dsp_roundtrip_blocks():
    p = DSPParams(features=("mfe",))
    assert DSPParams.from_dict(p.to_dict()) == p


def test_resolve_dsp_settings():
    space = {"window_size": {}, "mfe_blocks": [{}, {}]}
    s = Settings(optuna=OptunaCfg(dsp_search_space=space), dsp=DSPParams(features=("mfe",)))
    out = tuning.resolve_dsp_settings(
        s,
        {"window_size": 0.5, "window_step": 0.25, "n_filters_2": 8, "low_freq_1": 100},
        local=True,
    )
    assert out.dsp.clip_duration == 0.5 and out.data.segment_overlap == 0.5
    assert out.dsp.mfe_blocks[1].n_filters == 8 and out.dsp.mfe_blocks[0].low_freq == 100
    assert out.data.source == "local"


def test_suggest_dsp_params_caps_step():
    trial = MagicMock()
    trial.suggest_float.side_effect = (
        lambda name, lo, hi, **kw: 0.3 if name == "window_size" else hi
    )
    space = {
        "window_size": {"type": "float", "low": 0.1, "high": 1.0, "step": 0.05},
        "window_step": {"type": "float", "low": 0.1, "high": 1.0, "step": 0.05},
    }
    assert tuning.suggest_dsp_params(trial, space)["window_step"] == 0.3
    assert dataclasses.is_dataclass(MFEBlock)
