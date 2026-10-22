"""Carga y validación de configuración (config.yaml + optuna.yaml + dsp.yaml)."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

from ds_project.dsp.params import DSPParams

PathLike = Union[str, Path]

VALID_SOURCES = ("local", "s3", "onedrive")
VALID_BACKENDS = ("none", "local", "s3", "onedrive")
VALID_QUANT = ("float32", "float16", "dynamic", "int8")
VALID_SCALERS = ("standard", "minmax")
_LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_PRODUCT_RE = re.compile(r"^[a-z][a-z0-9]*$")
_TURBINE_RE = re.compile(r"^[a-z0-9]+$")
_CONFIG_SECTIONS = ("paths", "data", "train", "export", "storage")


def _check_scope(product: str, turbine: str, version: str = "") -> None:
    """Producto y turbina siguen la norma de nombres (minúsculas, sin ``_`` ni espacios)."""
    if product and not _PRODUCT_RE.match(product):
        raise ValueError("data.product: minúsculas y dígitos, sin '_' ni espacios (p. ej. pinbolt)")
    if turbine and not _TURBINE_RE.match(turbine):
        raise ValueError("data.turbine: minúsculas y dígitos, sin '_' ni espacios (p. ej. g5x)")
    if turbine and not product:
        raise ValueError("data.turbine requiere data.product")
    if version and not _LABEL_RE.match(version):
        raise ValueError("data.version solo admite letras, números, '_', '-' y '.'")


def _from_dict(cls, data: Optional[Dict[str, Any]]):
    data = data or {}
    names = {f.name for f in fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise ValueError(f"Claves desconocidas en {cls.__name__}: {sorted(unknown)}")
    return cls(**data)


@dataclass
class PathsCfg:
    """Rutas relativas al ámbito ``<projects_dir>/<producto>/<turbina>/<datos>`` si hay producto."""

    raw_dir: str = "data/raw"
    interim_dir: str = "data/interim"
    processed_dir: str = "data/processed"
    test_dir: str = "data/test"
    models_dir: str = "models"
    projects_dir: str = "projects"


@dataclass
class DataCfg:
    source: str = "local"
    remote_prefix: str = "raw"
    extensions: List[str] = field(default_factory=lambda: [".wav"])
    test_size: float = 0.15
    val_size: float = 0.15
    segment_overlap: float = 0.0
    min_segment_ratio: float = 0.5
    seed: int = 42
    product: str = ""  # p. ej. pinbolt; "" = sin proyecto (rutas tal cual, sin ámbito)
    turbine: str = ""  # p. ej. g5x, n155, all; por defecto "all" si hay producto
    version: str = ""  # versión de los datos, p. ej. v3 (no es la versión del modelo exportado)

    def __post_init__(self) -> None:
        if self.product and not self.turbine:
            self.turbine = "all"
        _check_scope(self.product, self.turbine, self.version)
        if self.source not in VALID_SOURCES:
            raise ValueError(f"data.source debe ser uno de {VALID_SOURCES}")
        if not 0.0 <= self.segment_overlap < 1.0:
            raise ValueError("data.segment_overlap debe estar en [0, 1)")
        if self.test_size + self.val_size >= 1.0:
            raise ValueError("test_size + val_size debe ser < 1")


@dataclass
class TrainCfg:
    epochs: int = 100
    patience: int = 10
    use_class_weights: bool = True
    scaler: str = "standard"
    deterministic: bool = True  # operaciones deterministas de TensorFlow (reproducibilidad)
    model_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.scaler not in VALID_SCALERS:
            raise ValueError(f"train.scaler debe ser uno de {VALID_SCALERS}")


@dataclass
class ExportCfg:
    quantization: str = "float16"
    representative_samples: int = 200
    export_header: bool = True

    def __post_init__(self) -> None:
        if self.quantization not in VALID_QUANT:
            raise ValueError(f"export.quantization debe ser uno de {VALID_QUANT}")


@dataclass
class StorageCfg:
    backend: str = "none"
    local_root: str = ""
    s3_bucket: str = ""
    s3_prefix: str = ""
    onedrive_root: str = ""
    upload_artifacts: bool = False

    def __post_init__(self) -> None:
        if self.backend not in VALID_BACKENDS:
            raise ValueError(f"storage.backend debe ser uno de {VALID_BACKENDS}")


def phase_trial_counts(
    n_trials: int, dsp: Optional[int] = None, net: Optional[int] = None
) -> Tuple[int, int]:
    """Trials de (fase DSP, fase red): ``n_trials`` en cada una salvo que se indique otro."""
    return int(dsp or n_trials), int(net or n_trials)


@dataclass
class OptunaCfg:
    enabled: bool = True
    study_name: str = "audio_tuning"
    storage: str = "sqlite:///models/optuna.db"
    direction: str = "maximize"
    metric: str = "val_accuracy"
    n_trials: int = 50
    n_workers: int = 1
    timeout: Optional[float] = None
    trial_epochs: int = 30
    trial_patience: int = 5
    n_startup_trials: int = 5
    n_warmup_steps: int = 5
    seed: int = 42
    search_space: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    dsp_search_space: Dict[str, Any] = field(default_factory=dict)
    tune_dsp: bool = True  # False: DSP fijo (sección dsp) y Optuna solo busca la red
    # "joint": DSP y red en el mismo estudio. "two_phase": primero el DSP con una red fija
    # pequeña (``phase1_*``) y después la red con ese DSP (sus features se calculan una vez).
    strategy: str = "joint"
    phase1_trials: Optional[int] = None  # trials de la fase del DSP; por defecto n_trials
    phase2_trials: Optional[int] = None  # trials de la fase de la red; por defecto n_trials
    phase1_epochs: int = 15
    phase1_model_params: Dict[str, Any] = field(default_factory=dict)  # sobre una red mínima

    def __post_init__(self) -> None:
        if self.direction not in ("maximize", "minimize"):
            raise ValueError("optuna.direction debe ser 'maximize' o 'minimize'")
        if self.n_workers < 1:
            raise ValueError("optuna.n_workers debe ser >= 1")
        if self.strategy not in ("joint", "two_phase"):
            raise ValueError("optuna.strategy debe ser 'joint' o 'two_phase'")
        if self.phase1_epochs < 1:
            raise ValueError("optuna.phase1_epochs debe ser >= 1")
        for name in ("phase1_trials", "phase2_trials"):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ValueError(f"optuna.{name} debe ser >= 1")

    @property
    def active_dsp_space(self) -> Dict[str, Any]:
        """Espacio de búsqueda del DSP que se usa realmente (vacío si ``tune_dsp`` es False)."""
        return self.dsp_search_space if self.tune_dsp else {}

    @property
    def two_phase(self) -> bool:
        """Hay dos fases solo si se pidió y existe un DSP que buscar."""
        return self.strategy == "two_phase" and bool(self.active_dsp_space)

    def phase_trials(self) -> Tuple[int, int]:
        """Trials de (fase DSP, fase red)."""
        return phase_trial_counts(self.n_trials, self.phase1_trials, self.phase2_trials)


@dataclass
class Settings:
    paths: PathsCfg = field(default_factory=PathsCfg)
    data: DataCfg = field(default_factory=DataCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    export: ExportCfg = field(default_factory=ExportCfg)
    storage: StorageCfg = field(default_factory=StorageCfg)
    optuna: OptunaCfg = field(default_factory=OptunaCfg)
    dsp: DSPParams = field(default_factory=DSPParams)

    @property
    def project_name(self) -> str:
        """``<producto>_<turbina>`` (identidad del modelo en la norma de nombres) o ``""``."""
        return f"{self.data.product}_{self.data.turbine}" if self.data.product else ""

    def project_dir(self) -> Optional[Path]:
        """``<projects_dir>/<producto>/<turbina>``; ``None`` si no hay producto."""
        if not self.data.product:
            return None
        return Path(self.paths.projects_dir) / self.data.product / self.data.turbine

    def scope_dir(self) -> Optional[Path]:
        """Ámbito de datos y modelos: ``<project_dir>/<versión de datos>``."""
        root = self.project_dir()
        if root is None:
            return None
        return root / self.data.version if self.data.version else root

    def resolved_paths(self) -> PathsCfg:
        """Rutas de trabajo: con producto, cada carpeta cuelga del ámbito del proyecto."""
        scope = self.scope_dir()
        if scope is None:
            return self.paths
        scoped = {
            f.name: str(scope / getattr(self.paths, f.name))
            for f in fields(PathsCfg)
            if f.name != "projects_dir"
        }
        return PathsCfg(projects_dir=self.paths.projects_dir, **scoped)


def _deep_merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _read_yaml(path: Optional[PathLike]) -> Dict[str, Any]:
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_settings(
    config_path: Optional[PathLike] = None,
    optuna_path: Optional[PathLike] = None,
    dsp_path: Optional[PathLike] = None,
    product: Optional[str] = None,
    turbine: Optional[str] = None,
    data_version: Optional[str] = None,
    overlay: Optional[Dict[str, Any]] = None,
) -> Settings:
    """Carga la configuración. Los ficheros inexistentes usan valores por defecto.

    Precedencia: valores base < ``projects/<producto>/<turbina>/project.yaml`` < argumentos /
    variables ``DS_PRODUCT``, ``DS_TURBINE`` y ``DS_DATA_VERSION``. La carpeta raíz de proyectos
    puede cambiarse con ``DS_PROJECTS_DIR``. ``overlay`` sustituye al ``project.yaml`` en disco
    (sirve para validar cambios antes de guardarlos).
    """
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:  # pragma: no cover
        pass

    config_path = config_path or os.getenv("DS_CONFIG", "configs/config.yaml")
    optuna_path = optuna_path or os.getenv("DS_OPTUNA_CONFIG", "configs/optuna.yaml")
    dsp_path = dsp_path or os.getenv("DS_DSP_CONFIG", "configs/dsp.yaml")

    cfg = _read_yaml(config_path)
    unknown = set(cfg) - set(_CONFIG_SECTIONS)
    if unknown:
        raise ValueError(f"Secciones desconocidas en config.yaml: {sorted(unknown)}")
    optuna_cfg = _read_yaml(optuna_path)
    dsp_cfg = _read_yaml(dsp_path)

    data_cfg = dict(cfg.get("data") or {})
    product = product or os.getenv("DS_PRODUCT") or data_cfg.get("product", "")
    turbine = turbine or os.getenv("DS_TURBINE") or data_cfg.get("turbine", "")
    data_version = data_version or os.getenv("DS_DATA_VERSION")
    if product and not turbine:
        turbine = "all"
    _check_scope(product, turbine)
    projects_dir = os.getenv("DS_PROJECTS_DIR")
    if projects_dir:
        cfg["paths"] = {**(cfg.get("paths") or {}), "projects_dir": projects_dir}
    if product:
        root = (cfg.get("paths") or {}).get("projects_dir", PathsCfg.projects_dir)
        if overlay is None:
            overlay = _read_yaml(Path(root) / product / turbine / "project.yaml")
        if not isinstance(overlay, dict):
            raise ValueError("project.yaml debe ser un diccionario de secciones")
        unknown = set(overlay) - set(_CONFIG_SECTIONS) - {"optuna", "dsp"}
        if unknown:
            raise ValueError(
                f"Secciones desconocidas en el proyecto {product}_{turbine}: {sorted(unknown)}"
            )
        for section in _CONFIG_SECTIONS:
            if section in overlay:
                cfg[section] = _deep_merge(cfg.get(section) or {}, overlay[section] or {})
        optuna_cfg = _deep_merge(optuna_cfg, overlay.get("optuna") or {})
        dsp_cfg = _deep_merge(dsp_cfg, overlay.get("dsp") or {})
        data_cfg = {**(cfg.get("data") or {}), "product": product, "turbine": turbine}
    if data_version:
        data_cfg["version"] = data_version
    cfg["data"] = data_cfg

    return Settings(
        paths=_from_dict(PathsCfg, cfg.get("paths")),
        data=_from_dict(DataCfg, cfg.get("data")),
        train=_from_dict(TrainCfg, cfg.get("train")),
        export=_from_dict(ExportCfg, cfg.get("export")),
        storage=_from_dict(StorageCfg, cfg.get("storage")),
        optuna=_from_dict(OptunaCfg, optuna_cfg),
        dsp=DSPParams.from_dict(dsp_cfg),
    )
