"""Funciones puras de extracción de features de audio (NumPy/SciPy).

Implementación determinista y portable a firmware:
- Banco de filtros mel HTK sin normalización (equivalente a librosa htk=True, norm=None).
- MFCC = DCT-II ortonormal sobre log natural del espectro mel de potencia.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List

import numpy as np
from scipy.fft import dct
from scipy.stats import kurtosis, skew

from ds_project.dsp import pinbolt
from ds_project.dsp.params import MULTI_FEATURE_SUFFIXES, DSPParams, MFEBlock

EPS = 1e-10


# ---------------- Preprocesado ----------------
def fix_length(signal: np.ndarray, n_samples: int) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float32)
    if signal.shape[0] >= n_samples:
        return signal[:n_samples]
    return np.pad(signal, (0, n_samples - signal.shape[0]))


def apply_pre_emphasis(signal: np.ndarray, coef: float) -> np.ndarray:
    if coef <= 0.0:
        return signal.astype(np.float32)
    out = np.empty_like(signal, dtype=np.float32)
    out[0] = signal[0]
    out[1:] = signal[1:] - coef * signal[:-1]
    return out


def get_window(name: str, n: int) -> np.ndarray:
    """Ventanas periódicas (como scipy.signal.get_window(..., fftbins=True))."""
    if name == "hann":
        return np.hanning(n + 1)[:-1].astype(np.float32)
    if name == "hamming":
        return np.hamming(n + 1)[:-1].astype(np.float32)
    if name == "rect":
        return np.ones(n, dtype=np.float32)
    raise ValueError(f"Ventana no soportada: {name}")


def frame_signal(signal: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    if signal.shape[0] < frame_length:
        signal = np.pad(signal, (0, frame_length - signal.shape[0]))
    n_frames = 1 + (signal.shape[0] - frame_length) // hop_length
    idx = np.arange(frame_length)[None, :] + hop_length * np.arange(n_frames)[:, None]
    return signal[idx]


def power_spectrum(frames: np.ndarray, window: np.ndarray, n_fft: int) -> np.ndarray:
    spec = np.fft.rfft(frames * window[None, :], n=n_fft, axis=-1)
    return (np.abs(spec) ** 2).astype(np.float32)


# ---------------- Mel / MFCC ----------------
def hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + np.asarray(hz, dtype=np.float64) / 700.0)


def mel_to_hz(mel):
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    """Matriz (n_mels, n_fft//2+1) de filtros triangulares HTK."""
    fft_freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    mel_pts = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_pts = mel_to_hz(mel_pts)
    fdiff = np.diff(hz_pts)
    ramps = hz_pts[:, None] - fft_freqs[None, :]
    weights = np.zeros((n_mels, fft_freqs.shape[0]), dtype=np.float64)
    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0.0, np.minimum(lower, upper))
    return weights.astype(np.float32)


def log_mel_spectrogram(power: np.ndarray, fbank: np.ndarray, log_offset: float) -> np.ndarray:
    mel = power @ fbank.T
    return np.log(mel + log_offset).astype(np.float32)


def mfcc_from_log_mel(log_mel: np.ndarray, n_mfcc: int) -> np.ndarray:
    return dct(log_mel, type=2, norm="ortho", axis=-1)[:, :n_mfcc].astype(np.float32)  # pylint: disable=E1126


# ---------------- Features temporales / espectrales ----------------
def rms(frames: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=-1)).astype(np.float32)


def zero_crossing_rate(frames: np.ndarray) -> np.ndarray:
    signs = np.signbit(frames)
    return np.mean(signs[:, 1:] != signs[:, :-1], axis=-1).astype(np.float32)


def spectral_centroid(mag: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    return (np.sum(mag * freqs[None, :], axis=-1) / (np.sum(mag, axis=-1) + EPS)).astype(np.float32)


def spectral_bandwidth(mag: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    centroid = spectral_centroid(mag, freqs)
    dev = (freqs[None, :] - centroid[:, None]) ** 2
    return np.sqrt(np.sum(mag * dev, axis=-1) / (np.sum(mag, axis=-1) + EPS)).astype(np.float32)


def spectral_rolloff(mag: np.ndarray, freqs: np.ndarray, percent: float) -> np.ndarray:
    cumsum = np.cumsum(mag, axis=-1)
    threshold = percent * cumsum[:, -1:]
    idx = np.argmax(cumsum >= threshold, axis=-1)
    return freqs[idx].astype(np.float32)


def spectral_flatness(power: np.ndarray) -> np.ndarray:
    geo = np.exp(np.mean(np.log(power + EPS), axis=-1))
    arith = np.mean(power, axis=-1) + EPS
    return (geo / arith).astype(np.float32)


def spectral_entropy(mag: np.ndarray) -> np.ndarray:
    p = np.clip(mag, EPS, None)
    p = p / np.sum(p, axis=-1, keepdims=True)
    return (-np.sum(p * np.log(p), axis=-1)).astype(np.float32)


def spectral_skewness(mag: np.ndarray) -> np.ndarray:
    return np.nan_to_num(skew(np.clip(mag, EPS, None), axis=-1)).astype(np.float32)


def spectral_kurtosis(mag: np.ndarray) -> np.ndarray:
    return np.nan_to_num(kurtosis(np.clip(mag, EPS, None), axis=-1)).astype(np.float32)


# ---------------- Clipping ----------------
def clipping_features(frames: np.ndarray, threshold: float = 0.999) -> np.ndarray:
    """Devuelve (n_frames, 3): ratio, nº de episodios y longitud media de episodio."""
    clipped = np.abs(frames) >= threshold
    padded = np.pad(clipped, ((0, 0), (1, 1))).astype(np.int8)
    edges = np.diff(padded, axis=-1)
    n_clipped = clipped.sum(axis=-1)
    events = (edges == 1).sum(axis=-1)
    ratio = n_clipped / frames.shape[-1]
    avg_len = np.where(events > 0, n_clipped / np.maximum(events, 1), 0.0)
    return np.stack([ratio, events, avg_len], axis=-1).astype(np.float32)


# ---------------- Temporales ----------------
def peak_to_rms(frames: np.ndarray) -> np.ndarray:
    return (np.max(np.abs(frames), axis=-1) / (rms(frames) + EPS)).astype(np.float32)


def energy_variance(frames: np.ndarray, n_sub: int = 4) -> np.ndarray:
    """Varianza de la energía de n_sub subventanas de cada frame."""
    out = np.empty(frames.shape[0], dtype=np.float32)
    for i, f in enumerate(frames):
        out[i] = np.var([np.sum(s.astype(np.float64) ** 2) for s in np.array_split(f, n_sub)])
    return out


# ---------------- Armónicas ----------------
def harmonic_features(frames: np.ndarray, sample_rate: int) -> np.ndarray:
    """Devuelve (n_frames, 2): HNR (dB) y pitch (Hz) por autocorrelación."""
    out = np.zeros((frames.shape[0], 2), dtype=np.float32)
    n = frames.shape[-1]
    for i, f in enumerate(frames):
        x = f.astype(np.float64) - np.mean(f)
        if not np.any(x):
            continue
        spec = np.fft.rfft(x, n=2 * n)
        corr = np.fft.irfft(np.abs(spec) ** 2)[:n]
        corr = corr / (corr[0] + EPS)
        rising = np.where(np.diff(corr) > 0)[0]
        start = rising[0] if rising.size else 1
        peak = int(np.argmax(corr[start:]) + start)
        r = float(np.clip(corr[peak], 0.0, 1.0 - 1e-12))
        out[i, 0] = 10.0 * np.log10((r + 1e-12) / (1.0 - r + 1e-12))
        out[i, 1] = sample_rate / peak if peak > 0 else 0.0
    return out


# ---------------- Modulación ----------------
def modulation_features(frames: np.ndarray) -> np.ndarray:
    """Devuelve (n_frames, 2): delta de energía y delta de RMS entre frames consecutivos."""
    f = frames.astype(np.float64)
    energy = np.sum(f**2, axis=-1)
    rms_vals = np.sqrt(np.mean(f**2, axis=-1))
    return np.stack(
        [np.diff(energy, prepend=energy[:1]), np.diff(rms_vals, prepend=rms_vals[:1])], axis=-1
    ).astype(np.float32)


# ---------------- Cepstrales ----------------
def cepstral_features(power: np.ndarray, n_ceps: int) -> np.ndarray:
    return dct(np.log(power + EPS), type=2, norm="ortho", axis=-1)[:, :n_ceps].astype(  # pylint: disable=E1126
        np.float32
    )


# ---------------- Energía no lineal (TKEO) ----------------
def tkeo_features(frames: np.ndarray) -> np.ndarray:
    """Devuelve (n_frames, 2): media y varianza del operador de Teager-Kaiser."""
    f = frames.astype(np.float64)
    if f.shape[-1] < 3:
        return np.zeros((f.shape[0], 2), dtype=np.float32)
    tkeo = f[:, 1:-1] ** 2 - f[:, :-2] * f[:, 2:]
    return np.stack([tkeo.mean(axis=-1), tkeo.var(axis=-1)], axis=-1).astype(np.float32)


# ---------------- MFE (banco mel en dB, normalizado y cuantizado) ----------------
def mfe_block(x: np.ndarray, block: MFEBlock, params: DSPParams) -> np.ndarray:
    """Devuelve (n_frames, n_filters) en [0, 1], cuantizado a 8 bits."""
    sr = params.sample_rate
    frame_len = block.frame_samples(sr)
    frames = frame_signal(x, frame_len, block.hop_samples(sr))
    power = power_spectrum(frames, get_window(params.window, frame_len), block.fft_length)
    power = power / block.fft_length
    fb = mel_filterbank(sr, block.fft_length, block.n_filters, block.low_freq, block.high_freq)
    mel_db = 10.0 * np.log10(np.clip(power @ fb.T, 1e-30, None))
    norm = (mel_db - block.noise_floor_db) / (abs(block.noise_floor_db) + 12.0)
    return (np.round(np.clip(norm, 0.0, 1.0) * 255.0) / 255.0).astype(np.float32)


def mfe_features(x: np.ndarray, params: DSPParams) -> np.ndarray:
    """Apila los bloques en el eje temporal; el eje de filtros se rellena con ceros al máximo."""
    width = params.feature_dim("mfe")
    out = [
        np.pad(mfe_block(x, b, params), ((0, 0), (0, width - b.n_filters)))
        for b in params.mfe_blocks
    ]
    return np.concatenate(out, axis=0)


# ---------------- Orquestación por señal ----------------
def extract_frame_features(signal: np.ndarray, params: DSPParams) -> np.ndarray:
    """Devuelve una matriz (n_frames, frame_feature_dim) en el orden de params.features."""
    raw = fix_length(signal, params.n_samples)
    x = apply_pre_emphasis(raw, params.pre_emphasis)
    if params.features == ("mfe",):
        return mfe_features(x, params)
    frames = frame_signal(x, params.frame_length, params.hop_length)
    raw_frames = frame_signal(raw, params.frame_length, params.hop_length)
    window = get_window(params.window, params.frame_length)
    power = power_spectrum(frames, window, params.n_fft)
    mag = np.sqrt(power)
    freqs = np.fft.rfftfreq(params.n_fft, d=1.0 / params.sample_rate).astype(np.float32)

    cache: Dict[str, np.ndarray] = {}

    def _log_mel() -> np.ndarray:
        if "log_mel" not in cache:
            fb = mel_filterbank(
                params.sample_rate,
                params.n_fft,
                params.n_mels,
                params.fmin,
                params.fmax,
            )
            cache["log_mel"] = log_mel_spectrogram(power, fb, params.log_offset)
        return cache["log_mel"]

    blocks: List[np.ndarray] = []
    for name in params.features:
        if name == "mfcc":
            blocks.append(mfcc_from_log_mel(_log_mel(), params.n_mfcc))
        elif name == "log_mel":
            blocks.append(_log_mel())
        elif name == "rms":
            blocks.append(rms(frames)[:, None])
        elif name == "zcr":
            blocks.append(zero_crossing_rate(frames)[:, None])
        elif name == "spectral_centroid":
            blocks.append(spectral_centroid(mag, freqs)[:, None])
        elif name == "spectral_bandwidth":
            blocks.append(spectral_bandwidth(mag, freqs)[:, None])
        elif name == "spectral_rolloff":
            blocks.append(spectral_rolloff(mag, freqs, params.rolloff_percent)[:, None])
        elif name == "spectral_flatness":
            blocks.append(spectral_flatness(power)[:, None])
        elif name == "spectral_rolloff95":
            blocks.append(spectral_rolloff(mag, freqs, 0.95)[:, None])
        elif name == "spectral_entropy":
            blocks.append(spectral_entropy(mag)[:, None])
        elif name == "spectral_skewness":
            blocks.append(spectral_skewness(mag)[:, None])
        elif name == "spectral_kurtosis":
            blocks.append(spectral_kurtosis(mag)[:, None])
        elif name == "clipping":
            blocks.append(clipping_features(raw_frames, params.clipping_threshold))
        elif name == "peak_to_rms":
            blocks.append(peak_to_rms(frames)[:, None])
        elif name == "crest_factor":
            # Mismo valor que peak_to_rms (se mantiene como feature separada)
            blocks.append(peak_to_rms(frames)[:, None])
        elif name == "energy_variance":
            blocks.append(energy_variance(frames)[:, None])
        elif name == "harmonic":
            blocks.append(harmonic_features(frames, params.sample_rate))
        elif name == "modulation":
            blocks.append(modulation_features(frames))
        elif name == "cepstral":
            blocks.append(cepstral_features(power, params.n_ceps))
        elif name == "tkeo":
            blocks.append(tkeo_features(frames))
        else:  # pragma: no cover - validado en DSPParams
            raise ValueError(f"Feature desconocida: {name}")
    return np.concatenate(blocks, axis=-1).astype(np.float32)


_STAT_FUNCS = {
    "mean": np.mean,
    "std": np.std,
    "min": np.min,
    "max": np.max,
    "median": np.median,
}


def aggregate_features(frame_feats: np.ndarray, stats) -> np.ndarray:
    return np.concatenate([_STAT_FUNCS[s](frame_feats, axis=0) for s in stats]).astype(np.float32)


def extract_features(signal: np.ndarray, params: DSPParams) -> np.ndarray:
    if params.is_pinbolt:
        return pinbolt.mfe_vector(fix_length(signal, params.n_samples), params)
    frame_feats = extract_frame_features(signal, params)
    if params.mode == "sequence":
        return frame_feats
    return aggregate_features(frame_feats, params.stats)


def extract_auxiliary_features(signal: np.ndarray, params: DSPParams) -> np.ndarray:
    """Extrae el vector agregado que alimenta la rama auxiliar del modelo."""
    if params.is_pinbolt:
        return pinbolt.aux_vector(extract_features(signal, params), params.sample_rate)
    if not params.aux_features:
        return np.empty((0,), dtype=np.float32)
    aux_params = dataclasses.replace(
        params,
        mode="aggregate",
        features=params.aux_features,
        stats=params.aux_stats,
        aux_features=(),
    )
    return extract_features(signal, aux_params)


def auxiliary_feature_names(params: DSPParams) -> List[str]:
    """Nombres, en orden, del vector que alimenta la rama auxiliar."""
    if params.is_pinbolt:
        return list(pinbolt.AUX_FEATURE_NAMES)
    if not params.aux_features:
        return []
    aux_params = dataclasses.replace(
        params,
        mode="aggregate",
        features=params.aux_features,
        stats=params.aux_stats,
        aux_features=(),
    )
    return feature_names(aux_params)


def feature_names(params: DSPParams) -> List[str]:
    if params.is_pinbolt:
        return [f"mfe_{i}" for i in range(params.mfe_flat_length)]
    names: List[str] = []
    for f in params.features:
        dim = params.feature_dim(f)
        if f in MULTI_FEATURE_SUFFIXES:
            names += [f"{f}_{s}" for s in MULTI_FEATURE_SUFFIXES[f]]
        else:
            names += [f"{f}_{i}" for i in range(dim)] if dim > 1 else [f]
    if params.mode == "aggregate":
        return [f"{n}_{s}" for s in params.stats for n in names]
    return names
