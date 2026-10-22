"""Matemática del perfil ``pinbolt``: reproduce templates/dsp_template.py durante el entrenamiento.

El MFE se entrega aplanado, sin ventana y sin escalar; las 25 features auxiliares se calculan
sobre ese vector aplanado (no sobre la forma de onda), igual que el DSP desplegado.
"""

from __future__ import annotations

import math
from typing import List

import numpy as np
from scipy.fftpack import dct
from scipy.signal import correlate
from scipy.stats import entropy, kurtosis, skew

from ds_project.dsp.params import DSPParams, MFEBlock

AUX_FEATURE_NAMES: List[str] = (
    ["spectral_centroid", "spectral_bandwidth", "spectral_rolloff", "spectral_rolloff95"]
    + ["spectral_flatness", "spectral_entropy", "spectral_skewness", "spectral_kurtosis"]
    + ["clipping_ratio", "clipping_events", "clipping_avg_length"]
    + ["rms", "zcr", "peak_to_rms", "crest_factor", "energy_variance"]
    + ["harmonic_hnr", "harmonic_pitch"]
    + ["modulation_energy_delta", "modulation_rms_delta"]
    + ["cepstral_0", "cepstral_1", "cepstral_2"]
    + ["tkeo_mean", "tkeo_var"]
)


# ---------------- MFE ----------------
def preemphasis(x: np.ndarray, coeff: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if len(x) == 0:
        return x
    return np.append(x[0], x[1:] - coeff * x[:-1]).astype(np.float32)


def _frequency_to_mel(freq_hz: float) -> float:
    return 2595.0 * np.log10(1.0 + (float(freq_hz) / 700.0))


def _mel_to_frequency(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _triangle(x: np.ndarray, left: float, middle: float, right: float) -> np.ndarray:
    y = np.zeros_like(x, dtype=np.float32)
    if middle != left:
        left_mask = (x >= left) & (x <= middle)
        y[left_mask] = (x[left_mask] - left) / (middle - left)
    if right != middle:
        right_mask = (x >= middle) & (x <= right)
        y[right_mask] = (right - x[right_mask]) / (right - middle)
    return np.maximum(y, 0.0).astype(np.float32)


def _filterbanks(num_filter: int, coefficients: int, sr: int, low: float, high: float):
    fftpoints = (coefficients - 1) * 2
    mels = np.linspace(_frequency_to_mel(low), _frequency_to_mel(high), num_filter + 2)
    hertz = _mel_to_frequency(mels)
    freq_index = np.floor((fftpoints + 1) * hertz / sr).astype(int)
    filterbank = np.zeros([num_filter, coefficients], dtype=np.float32)
    for i in range(num_filter):
        left, middle, right = (int(freq_index[i + k]) for k in range(3))
        if right <= left:
            continue
        left = max(left, 0)
        right = min(right, coefficients - 1)
        z = np.linspace(left, right, num=right - left + 1)
        filterbank[i, left : right + 1] = _triangle(z, left=left, middle=middle, right=right)
    return filterbank


def _stack_frames(sig: np.ndarray, sr: int, frame_length: float, stride: float, zero_padding: bool):
    length_signal = sig.shape[0]
    frame_sample_length = int(np.round(sr * frame_length))
    frame_stride_samples = float(np.round(sr * stride))
    if zero_padding:
        numframes = int(math.ceil((length_signal - frame_sample_length) / frame_stride_samples))
        len_sig = int(numframes * frame_stride_samples + frame_sample_length)
        signal = np.concatenate((sig, np.zeros((len_sig - length_signal,), dtype=sig.dtype)))
    else:
        x = length_signal - (frame_sample_length - frame_stride_samples)
        numframes = max(int(math.floor(x / frame_stride_samples)), 1)
        len_sig = int((numframes - 1) * frame_stride_samples + frame_sample_length)
        signal = sig[0:len_sig]
    indices = (
        np.tile(np.arange(0, frame_sample_length), (numframes, 1))
        + np.tile(
            np.arange(0, numframes * frame_stride_samples, frame_stride_samples),
            (frame_sample_length, 1),
        ).T
    )
    return signal[np.asarray(indices, dtype=np.int32)]


def _mfe_block(signal: np.ndarray, block: MFEBlock, params: DSPParams) -> np.ndarray:
    sr = params.sample_rate
    frames = _stack_frames(signal, sr, block.frame_length, block.frame_stride, params.zero_padding)
    spec = np.fft.rfft(frames, n=block.fft_length)
    power = ((1.0 / block.fft_length) * (np.absolute(spec) ** 2)).astype(np.float32)
    banks = _filterbanks(block.n_filters, power.shape[1], sr, block.low_freq, block.high_freq)
    feats = np.dot(power, banks.T)
    mfe = np.where(feats == 0, np.finfo(float).eps, feats).astype(np.float32)
    mfe_db = 10.0 * np.log10(np.clip(mfe, 1e-30, None))
    mfe_norm = (mfe_db - block.noise_floor_db) / (abs(block.noise_floor_db) + 12.0)
    mfe_qu = np.uint8(np.round(np.clip(mfe_norm, 0.0, 1.0) * 256.0))
    return (mfe_qu.astype(np.float32) / 256.0).flatten()


def mfe_vector(signal: np.ndarray, params: DSPParams) -> np.ndarray:
    """Vector MFE aplanado de una ventana de ``clip_duration`` s (ya con longitud fija)."""
    x = np.clip(np.asarray(signal, dtype=np.float32), -1.0, 1.0)
    x = preemphasis(x, params.pre_emphasis)
    out = np.concatenate([_mfe_block(x, b, params) for b in params.mfe_blocks]).astype(np.float32)
    nf = params.effective_n_filters_final
    return out[: (out.shape[0] // nf) * nf]


# ---------------- Features auxiliares (sobre el vector MFE aplanado) ----------------
def _spectral(X: np.ndarray, sr: int) -> np.ndarray:
    n_frames, n_bins = X.shape
    freqs = np.linspace(0, sr / 2, n_bins)
    X_safe = np.clip(X, 1e-12, None)
    energy = X_safe / np.sum(X_safe, axis=1, keepdims=True)
    centroid = np.sum(freqs * energy, axis=1)
    bandwidth = np.sqrt(np.sum(((freqs - centroid[:, None]) ** 2) * energy, axis=1))
    cumsum = np.cumsum(X_safe, axis=1)
    total = np.sum(X_safe, axis=1, keepdims=True)
    rolloff = np.zeros(n_frames)
    rolloff95 = np.zeros(n_frames)
    for i in range(n_frames):
        rolloff[i] = freqs[min(np.searchsorted(cumsum[i], float(0.85 * total[i, 0])), n_bins - 1)]
        rolloff95[i] = freqs[min(np.searchsorted(cumsum[i], float(0.95 * total[i, 0])), n_bins - 1)]
    flatness = np.exp(np.mean(np.log(X_safe), axis=1)) / np.mean(X_safe, axis=1)
    spec_entropy = entropy(energy.T)
    varying = np.std(X_safe, axis=1) > np.finfo(np.float32).eps
    spec_skew = np.zeros(n_frames, dtype=np.float32)
    spec_kurt = np.zeros(n_frames, dtype=np.float32)
    if np.any(varying):
        spec_skew[varying] = skew(X_safe[varying], axis=1).astype(np.float32)
        spec_kurt[varying] = kurtosis(X_safe[varying], axis=1).astype(np.float32)
    return np.stack(
        [centroid, bandwidth, rolloff, rolloff95, flatness, spec_entropy, spec_skew, spec_kurt],
        axis=1,
    )


def _clipping(X: np.ndarray, threshold: float = 0.999) -> np.ndarray:
    n_frames, n_samples = X.shape
    feats = np.zeros((n_frames, 3), dtype=np.float32)
    for i in range(n_frames):
        clipped = np.abs(X[i]) >= threshold
        n_clipped = np.sum(clipped)
        events = np.sum(np.diff(clipped.astype(int)) == 1)
        avg_len = 0.0
        if n_clipped:
            lengths, count = [], 0
            for c in clipped:
                if c:
                    count += 1
                elif count > 0:
                    lengths.append(count)
                    count = 0
            if count > 0:
                lengths.append(count)
            avg_len = np.mean(lengths) if lengths else 0.0
        feats[i] = [n_clipped / n_samples, events, avg_len]
    return feats


def _temporal(X: np.ndarray) -> np.ndarray:
    feats = np.zeros((X.shape[0], 5), dtype=np.float32)
    for i in range(X.shape[0]):
        audio = X[i]
        rms = np.sqrt(np.mean(audio**2))
        zcr = np.mean(np.abs(np.diff(np.sign(audio)))) / 2
        peak_rms_ratio = np.max(np.abs(audio)) / (rms + 1e-12)
        energy_var = np.var([np.sum(s**2) for s in np.array_split(audio, 4)])
        feats[i] = [rms, zcr, peak_rms_ratio, peak_rms_ratio, energy_var]
    return feats


def _harmonic(X: np.ndarray, sr: int) -> np.ndarray:
    feats = np.zeros((X.shape[0], 2), dtype=np.float32)
    for i in range(X.shape[0]):
        audio = X[i] - np.mean(X[i])
        if np.all(audio == 0):
            continue
        corr = correlate(audio, audio, mode="full")
        corr = corr[corr.size // 2 :]
        max_val = np.max(corr)
        if max_val <= 0:
            continue
        corr = corr / max_val
        starts = np.where(np.diff(corr) > 0)[0]
        start = starts[0] if starts.size > 0 else 1
        peak = np.argmax(corr[start:]) + start if start < len(corr) else 0
        pitch_freq = sr / peak if peak > 0 else 0
        val_peak = max(float(corr[peak]), 0.0) if peak < len(corr) else 0.0
        ratio = max(val_peak / (1.0 - val_peak + 1e-12), 1e-12)
        feats[i] = [10.0 * np.log10(ratio), pitch_freq]
    return feats


def _modulation(X: np.ndarray) -> np.ndarray:
    feats = np.zeros((X.shape[0], 2), dtype=np.float32)
    energies = np.sum(X**2, axis=1)
    rms_vals = np.sqrt(np.mean(X**2, axis=1))
    feats[:, 0] = np.diff(energies, prepend=energies[0])
    feats[:, 1] = np.diff(rms_vals, prepend=rms_vals[0])
    return feats


def _cepstral(X: np.ndarray, n_ceps: int = 3) -> np.ndarray:
    feats = np.zeros((X.shape[0], n_ceps), dtype=np.float32)
    for i in range(X.shape[0]):
        log_spectrum = np.log(np.abs(np.fft.rfft(X[i])) ** 2 + 1e-12)
        feats[i] = dct(log_spectrum, type=2, norm="ortho")[:n_ceps]
    return feats


def _nonlinear_energy(X: np.ndarray) -> np.ndarray:
    feats = np.zeros((X.shape[0], 2), dtype=np.float32)
    for i in range(X.shape[0]):
        x = X[i]
        if len(x) < 3:
            continue
        tkeo = x[1:-1] ** 2 - x[:-2] * x[2:]
        feats[i] = [np.mean(tkeo), np.var(tkeo)]
    return feats


def aux_vector(mfe: np.ndarray, sample_rate: int) -> np.ndarray:
    """25 features auxiliares (sin escalar) calculadas sobre el vector MFE aplanado."""
    X = np.asarray(mfe, dtype=np.float32).reshape(1, -1)
    feats = [
        _spectral(X, sample_rate),
        _clipping(X),
        _temporal(X),
        _harmonic(X, sample_rate),
        _modulation(X),
        _cepstral(X, n_ceps=3),
        _nonlinear_energy(X),
    ]
    aux = np.concatenate(feats, axis=1).astype(np.float32)
    return np.nan_to_num(aux, nan=0.0, posinf=0.0, neginf=0.0).reshape(-1)
