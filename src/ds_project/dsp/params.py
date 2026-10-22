"""Dataclass inmutable con todos los parámetros del DSP."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import yaml

VALID_FEATURES = (
    "mfcc",
    "log_mel",
    "rms",
    "zcr",
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_rolloff",
    "spectral_flatness",
    "mfe",
    "spectral_rolloff95",
    "spectral_entropy",
    "spectral_skewness",
    "spectral_kurtosis",
    "clipping",
    "peak_to_rms",
    "crest_factor",
    "energy_variance",
    "harmonic",
    "modulation",
    "cepstral",
    "tkeo",
)
# Features que aportan varias columnas: nombre -> sufijos (None = depende de params)
MULTI_FEATURE_SUFFIXES = {
    "clipping": ("ratio", "events", "avg_length"),
    "harmonic": ("hnr", "pitch"),
    "modulation": ("energy_delta", "rms_delta"),
    "tkeo": ("mean", "var"),
}
VALID_STATS = ("mean", "std", "min", "max", "median")
VALID_WINDOWS = ("hann", "hamming", "rect")
VALID_MODES = ("sequence", "aggregate")
VALID_PROFILES = ("default", "pinbolt")
# Vector auxiliar del perfil pinbolt: spectral(8) + clipping(3) + temporal(5) + harmonic(2)
# + modulation(2) + cepstral(3) + nonlinear energy(2)
PINBOLT_AUX_DIM = 25


@dataclass(frozen=True)
class MFEBlock:
    """Un banco MFE (log-mel cuantizado). Tiempos en segundos, frecuencias en Hz."""

    frame_length: float = 0.03
    frame_stride: float = 0.015
    n_filters: int = 32
    fft_length: int = 512
    low_freq: float = 50.0
    high_freq: float = 1500.0
    noise_floor_db: float = -50.0

    def __post_init__(self) -> None:
        if self.frame_length <= 0 or self.frame_stride <= 0:
            raise ValueError("frame_length y frame_stride deben ser > 0")
        if self.n_filters < 1 or self.fft_length < 2:
            raise ValueError("n_filters >= 1 y fft_length >= 2")
        if not 0 <= self.low_freq < self.high_freq:
            raise ValueError("Se requiere 0 <= low_freq < high_freq")
        if self.noise_floor_db >= 0:
            raise ValueError("noise_floor_db debe ser negativo")

    def frame_samples(self, sample_rate: int) -> int:
        return max(1, int(round(self.frame_length * sample_rate)))

    def hop_samples(self, sample_rate: int) -> int:
        return max(1, int(round(self.frame_stride * sample_rate)))

    def n_frames(self, n_samples: int, sample_rate: int) -> int:
        fl = self.frame_samples(sample_rate)
        return 1 + (max(n_samples, fl) - fl) // self.hop_samples(sample_rate)

    def pinbolt_n_frames(self, n_samples: int, sample_rate: int, zero_padding: bool) -> int:
        """Nº de frames con el apilado de la plantilla DSP de PinBolt."""
        length = int(round(sample_rate * self.frame_length))
        stride = float(round(sample_rate * self.frame_stride))
        if zero_padding:
            return max(int(math.ceil((n_samples - length) / stride)), 1)
        return max(int(math.floor((n_samples - (length - stride)) / stride)), 1)


_DEFAULT_MFE_BLOCKS = (
    MFEBlock(low_freq=50.0, high_freq=1500.0),
    MFEBlock(low_freq=1500.0, high_freq=3500.0),
)


@dataclass(frozen=True)
class DSPParams:
    version: str = "1.0"
    sample_rate: int = 16000
    clip_duration: float = 1.0
    frame_length: int = 512
    hop_length: int = 256
    n_fft: int = 512
    window: str = "hann"
    pre_emphasis: float = 0.97
    n_mels: int = 40
    n_mfcc: int = 13
    fmin: float = 20.0
    fmax: float = 8000.0
    log_offset: float = 1e-6
    rolloff_percent: float = 0.85
    n_ceps: int = 3
    clipping_threshold: float = 0.999
    mode: str = "sequence"
    features: Tuple[str, ...] = ("mfcc",)
    stats: Tuple[str, ...] = ("mean", "std")
    aux_features: Tuple[str, ...] = ()
    aux_stats: Tuple[str, ...] = ("mean", "std")
    mfe_blocks: Tuple[MFEBlock, ...] = field(default=_DEFAULT_MFE_BLOCKS)
    # profile=pinbolt reproduce la plantilla DSP de PinBolt: vector MFE aplanado sin escalar y
    # 25 features auxiliares calculadas sobre ese vector (aux_features debe quedar vacío).
    profile: str = "default"
    zero_padding: bool = False
    n_filters_final: int = 0  # 0 = nº de filtros del primer bloque (solo perfil pinbolt)

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate debe ser > 0")
        if self.n_fft < self.frame_length:
            raise ValueError("n_fft debe ser >= frame_length")
        if not 0 < self.hop_length <= self.frame_length:
            raise ValueError("hop_length debe estar en (0, frame_length]")
        if self.window not in VALID_WINDOWS:
            raise ValueError(f"window debe ser uno de {VALID_WINDOWS}")
        if self.mode not in VALID_MODES:
            raise ValueError(f"mode debe ser uno de {VALID_MODES}")
        if not self.features:
            raise ValueError("Debe haber al menos una feature")
        bad = [f for f in self.features if f not in VALID_FEATURES]
        if bad:
            raise ValueError(f"Features no válidas: {bad}")
        bad_stats = [s for s in self.stats if s not in VALID_STATS]
        if bad_stats:
            raise ValueError(f"Estadísticos no válidos: {bad_stats}")
        bad_aux = [f for f in self.aux_features if f not in VALID_FEATURES or f == "mfe"]
        if bad_aux:
            raise ValueError(f"Features auxiliares no válidas: {bad_aux}")
        bad_aux_stats = [s for s in self.aux_stats if s not in VALID_STATS]
        if bad_aux_stats:
            raise ValueError(f"Estadísticos auxiliares no válidos: {bad_aux_stats}")
        if self.aux_features and not self.aux_stats:
            raise ValueError("aux_features requiere al menos un aux_stat")
        if self.mode == "aggregate" and not self.stats:
            raise ValueError("mode=aggregate requiere al menos un estadístico")
        if not 0 <= self.fmin < self.fmax <= self.sample_rate / 2:
            raise ValueError("Se requiere 0 <= fmin < fmax <= sample_rate/2")
        if self.n_mfcc > self.n_mels:
            raise ValueError("n_mfcc debe ser <= n_mels")
        if not 0.0 <= self.pre_emphasis < 1.0:
            raise ValueError("pre_emphasis debe estar en [0, 1)")
        if not 0.0 < self.rolloff_percent < 1.0:
            raise ValueError("rolloff_percent debe estar en (0, 1)")
        if not 1 <= self.n_ceps <= self.n_fft // 2 + 1:
            raise ValueError("n_ceps debe estar en [1, n_fft//2 + 1]")
        if not 0.0 < self.clipping_threshold <= 1.0:
            raise ValueError("clipping_threshold debe estar en (0, 1]")
        if "mfe" in self.features:
            if self.features != ("mfe",):
                raise ValueError("'mfe' no se puede combinar con otras features")
            if not self.mfe_blocks:
                raise ValueError("features=mfe requiere al menos un mfe_block")
            if any(b.high_freq > self.sample_rate / 2 for b in self.mfe_blocks):
                raise ValueError("mfe_block.high_freq debe ser <= sample_rate/2")
        if self.profile not in VALID_PROFILES:
            raise ValueError(f"profile debe ser uno de {VALID_PROFILES}")
        if self.profile == "pinbolt":
            if self.features != ("mfe",):
                raise ValueError("profile=pinbolt requiere features=[mfe]")
            if self.aux_features:
                raise ValueError("profile=pinbolt calcula sus features auxiliares: aux_features=[]")
            if self.n_filters_final < 0:
                raise ValueError("n_filters_final debe ser >= 0")
            if self.mfe_flat_length < 1:
                raise ValueError("n_filters_final excede la longitud del vector MFE")

    # ---------------- Derivados ----------------
    @property
    def is_pinbolt(self) -> bool:
        return self.profile == "pinbolt"

    @property
    def has_aux(self) -> bool:
        return self.is_pinbolt or bool(self.aux_features)

    def scaler_kinds(self, kind: str) -> Tuple[str, str]:
        """Tipo de scaler de (MFE, auxiliares). En pinbolt: MFE sin escalar y aux estándar."""
        return ("none", "standard") if self.is_pinbolt else (kind, kind)

    @property
    def effective_n_filters_final(self) -> int:
        return self.n_filters_final or self.mfe_blocks[0].n_filters

    @property
    def mfe_flat_length(self) -> int:
        """Longitud del vector MFE aplanado (recortado a múltiplo de n_filters_final)."""
        total = sum(
            b.pinbolt_n_frames(self.n_samples, self.sample_rate, self.zero_padding) * b.n_filters
            for b in self.mfe_blocks
        )
        nf = self.effective_n_filters_final
        return (total // nf) * nf

    @property
    def mfe_reshape(self) -> Optional[Tuple[int, int]]:
        if not self.is_pinbolt:
            return None
        nf = self.effective_n_filters_final
        return (self.mfe_flat_length // nf, nf)

    @property
    def n_samples(self) -> int:
        return int(round(self.sample_rate * self.clip_duration))

    @property
    def n_frames(self) -> int:
        if self.is_pinbolt:
            return sum(
                b.pinbolt_n_frames(self.n_samples, self.sample_rate, self.zero_padding)
                for b in self.mfe_blocks
            )
        if "mfe" in self.features:
            return sum(b.n_frames(self.n_samples, self.sample_rate) for b in self.mfe_blocks)
        n = max(self.n_samples, self.frame_length)
        return 1 + (n - self.frame_length) // self.hop_length

    @property
    def n_bins(self) -> int:
        return self.n_fft // 2 + 1

    def feature_dim(self, name: str) -> int:
        if name == "mfcc":
            return self.n_mfcc
        if name == "log_mel":
            return self.n_mels
        if name == "mfe":
            return max(b.n_filters for b in self.mfe_blocks)
        if name == "cepstral":
            return self.n_ceps
        if name in MULTI_FEATURE_SUFFIXES:
            return len(MULTI_FEATURE_SUFFIXES[name])
        return 1

    @property
    def frame_feature_dim(self) -> int:
        return sum(self.feature_dim(f) for f in self.features)

    @property
    def feature_shape(self) -> Tuple[int, ...]:
        if self.is_pinbolt:
            return (self.mfe_flat_length,)
        if self.mode == "sequence":
            return (self.n_frames, self.frame_feature_dim)
        return (self.frame_feature_dim * len(self.stats),)

    @property
    def aux_feature_dim(self) -> int:
        if self.is_pinbolt:
            return PINBOLT_AUX_DIM
        return sum(self.feature_dim(f) for f in self.aux_features)

    @property
    def aux_feature_shape(self) -> Tuple[int, ...]:
        if self.is_pinbolt:
            return (PINBOLT_AUX_DIM,)
        return (self.aux_feature_dim * len(self.aux_stats),)

    # ---------------- Serialización ----------------
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["features"] = list(self.features)
        d["stats"] = list(self.stats)
        d["aux_features"] = list(self.aux_features)
        d["aux_stats"] = list(self.aux_stats)
        d["mfe_blocks"] = [asdict(b) for b in self.mfe_blocks]
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DSPParams:
        data = dict(data or {})
        known = {f.name: f for f in fields(cls)}
        unknown = set(data) - set(known)
        if unknown:
            raise ValueError(f"Claves DSP desconocidas: {sorted(unknown)}")
        defaults = cls()
        kwargs: Dict[str, Any] = {}
        for key, value in data.items():
            default = getattr(defaults, key)
            if key == "mfe_blocks":
                kwargs[key] = tuple(
                    b if isinstance(b, MFEBlock) else MFEBlock(**b) for b in (value or [])
                )
            elif isinstance(default, tuple):
                kwargs[key] = tuple(value)
            elif isinstance(default, bool):
                kwargs[key] = bool(value)
            elif isinstance(default, float):
                kwargs[key] = float(value)
            elif isinstance(default, int):
                kwargs[key] = int(value)
            else:
                kwargs[key] = str(value)
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> DSPParams:
        with Path(path).open("r", encoding="utf-8") as fh:
            return cls.from_dict(yaml.safe_load(fh) or {})
