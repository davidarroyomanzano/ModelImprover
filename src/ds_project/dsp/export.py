"""Exportación de parámetros DSP + scaler + etiquetas a JSON y cabecera C."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from ds_project.dsp.params import VALID_FEATURES, DSPParams
from ds_project.utils.io import ensure_dir, save_json

PathLike = Union[str, Path]


def build_dsp_artifact(
    params: DSPParams,
    labels: Sequence[str],
    scaler_dict: Optional[Dict[str, Any]] = None,
    aux_scaler_dict: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    from ds_project.features.audio import auxiliary_feature_names, feature_names

    artifact: Dict[str, Any] = {
        "dsp": params.to_dict(),
        "derived": {
            "n_samples": params.n_samples,
            "n_frames": params.n_frames,
            "n_bins": params.n_bins,
            "frame_feature_dim": params.frame_feature_dim,
            "feature_shape": list(params.feature_shape),
            "feature_names": feature_names(params),
            "aux_feature_shape": list(params.aux_feature_shape),
            "aux_feature_names": auxiliary_feature_names(params),
        },
        "labels": list(labels),
        "scaler": scaler_dict,
        "aux_scaler": aux_scaler_dict,
    }
    if extra:
        artifact["extra"] = extra
    return artifact


def export_dsp_json(
    params: DSPParams,
    labels: Sequence[str],
    path: PathLike,
    scaler_dict: Optional[Dict[str, Any]] = None,
    aux_scaler_dict: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    return save_json(build_dsp_artifact(params, labels, scaler_dict, aux_scaler_dict, extra), path)


def _c_float(v: float) -> str:
    return f"{float(v)!r}f"


def _c_array(name: str, values: List[float]) -> List[str]:
    body = ", ".join(_c_float(v) for v in values)
    return [f"static const float {name}[{len(values)}] = {{{body}}};"]


def export_dsp_header(
    params: DSPParams,
    labels: Sequence[str],
    path: PathLike,
    scaler_dict: Optional[Dict[str, Any]] = None,
    aux_scaler_dict: Optional[Dict[str, Any]] = None,
    guard: str = "DSP_PARAMS_H",
) -> Path:
    """Genera una cabecera C con los parámetros DSP, el scaler y las etiquetas."""
    lines: List[str] = [
        "/* Auto-generado por ds_project.dsp.export. NO EDITAR. */",
        f"#ifndef {guard}",
        f"#define {guard}",
        "",
        f'#define DSP_VERSION "{params.version}"',
        f"#define DSP_SAMPLE_RATE {params.sample_rate}",
        f"#define DSP_CLIP_DURATION {_c_float(params.clip_duration)}",
        f"#define DSP_N_SAMPLES {params.n_samples}",
        f"#define DSP_FRAME_LENGTH {params.frame_length}",
        f"#define DSP_HOP_LENGTH {params.hop_length}",
        f"#define DSP_N_FFT {params.n_fft}",
        f"#define DSP_N_BINS {params.n_bins}",
        f"#define DSP_N_FRAMES {params.n_frames}",
        f'#define DSP_WINDOW "{params.window}"',
        f"#define DSP_WINDOW_{params.window.upper()} 1",
        f"#define DSP_PRE_EMPHASIS {_c_float(params.pre_emphasis)}",
        f"#define DSP_N_MELS {params.n_mels}",
        f"#define DSP_N_MFCC {params.n_mfcc}",
        f"#define DSP_FMIN {_c_float(params.fmin)}",
        f"#define DSP_FMAX {_c_float(params.fmax)}",
        f"#define DSP_LOG_OFFSET {_c_float(params.log_offset)}",
        f"#define DSP_ROLLOFF_PERCENT {_c_float(params.rolloff_percent)}",
        f"#define DSP_MODE_{params.mode.upper()} 1",
        f"#define DSP_FRAME_FEATURE_DIM {params.frame_feature_dim}",
        f"#define DSP_FEATURE_SIZE {_prod(params.feature_shape)}",
        "",
    ]
    for feat in VALID_FEATURES:
        lines.append(f"#define DSP_USE_{feat.upper()} {1 if feat in params.features else 0}")
    for stat in ("mean", "std", "min", "max", "median"):
        enabled = params.mode == "aggregate" and stat in params.stats
        lines.append(f"#define DSP_STAT_{stat.upper()} {1 if enabled else 0}")
    lines.append("")

    if "mfe" in params.features:
        lines.append(f"#define DSP_N_MFE_BLOCKS {len(params.mfe_blocks)}")
        for i, b in enumerate(params.mfe_blocks, start=1):
            pre = f"DSP_MFE{i}"
            lines += [
                f"#define {pre}_FRAME_LENGTH {b.frame_samples(params.sample_rate)}",
                f"#define {pre}_HOP_LENGTH {b.hop_samples(params.sample_rate)}",
                f"#define {pre}_N_FRAMES {b.n_frames(params.n_samples, params.sample_rate)}",
                f"#define {pre}_N_FILTERS {b.n_filters}",
                f"#define {pre}_FFT_LENGTH {b.fft_length}",
                f"#define {pre}_LOW_FREQ {_c_float(b.low_freq)}",
                f"#define {pre}_HIGH_FREQ {_c_float(b.high_freq)}",
                f"#define {pre}_NOISE_FLOOR_DB {_c_float(b.noise_floor_db)}",
            ]
        lines.append("")

    lines.append(f"#define DSP_N_CLASSES {len(labels)}")
    label_body = ", ".join(f'"{lbl}"' for lbl in labels)
    lines.append(f"static const char *DSP_LABELS[{len(labels)}] = {{{label_body}}};")
    lines.append("")

    if scaler_dict:
        lines.append(f'/* Scaler ({scaler_dict["kind"]}): y = (x - offset) / divisor */')
        lines.append(f"#define DSP_SCALER_DIM {len(scaler_dict['offset'])}")
        lines += _c_array("DSP_SCALER_OFFSET", scaler_dict["offset"])
        lines += _c_array("DSP_SCALER_DIVISOR", scaler_dict["divisor"])
        lines.append("")
    if aux_scaler_dict:
        lines.append(
            f'/* Scaler auxiliar ({aux_scaler_dict["kind"]}): y = (x - offset) / divisor */'
        )
        lines.append(f"#define DSP_AUX_SCALER_DIM {len(aux_scaler_dict['offset'])}")
        lines += _c_array("DSP_AUX_SCALER_OFFSET", aux_scaler_dict["offset"])
        lines += _c_array("DSP_AUX_SCALER_DIVISOR", aux_scaler_dict["divisor"])
        lines.append("")

    lines.append(f"#endif /* {guard} */")
    out = Path(path)
    ensure_dir(out.parent)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def _prod(shape) -> int:
    result = 1
    for s in shape:
        result *= int(s)
    return result
