"""
ClaudIA / PinBolt - STANDARD TFLITE DSP CONTRACT TEMPLATE

DSP implementation version: 3.0
DSP contract template revision: 1.0
Template date: 2026-09-09
Compatibility target: PinBolt upload3.py v43.3 (2026-09-08)
Validation note: v43.3 changes only FOLDER ground-truth filename parsing;
the TFLite DSP contract is unchanged from v43.2.

Purpose
-------
This file is the reference structure for future ``dsp_<model>.py`` modules.
The objective is NOT to force every future model to use this MFE/AUX
mathematics. The objective is to standardize the interface and architectural
boundary that allow different model-specific DSPs to run on the same generic
PinBolt Runner.

Normative architectural rule
----------------------------
The DSP owns ONLY model-specific mathematical preprocessing and the capture
cadence required to reproduce the trained pipeline. Parameters that change the
mathematics seen by the model MUST remain coupled to the DSP implementation.
Typical examples are:

- SAMPLE_RATE
- inference WINDOW_SIZE / WINDOW_STEP
- MFE / MFCC / spectrogram definitions
- pre-emphasis or other signal transforms
- auxiliary-feature definitions
- fitted normalization/scaler parameters
- class labels when no authoritative external metadata is available
- explicit tensor mapping required by the deployed TFLite model

The Runner owns runtime policy and orchestration. A DSP MUST NOT import or
reimplement Runner policy such as:

- output decimation
- smoothing / aggregation
- decision thresholds / target-event policy
- Ring-Buffer PRE/POST context
- scheduler behavior
- file I/O, WAV export, SFTP or publication
- Modbus, dashboard or plugin orchestration

In particular, current Runner versions explicitly reject
``output_decimation`` inside ``get_capture_params()``. Configure
``TFLITE_OUTPUT_DECIMATION`` in Runner configuration instead.

PinBolt DSP public contract
---------------------------
For this standard template, future DSP modules SHOULD keep the following public
entry points. The first four groups are the compatibility surface consumed by
the Runner; helper functions may vary internally by model:

1. ``get_labels()``
   Return a non-empty Python ``list[str]`` in exactly the output-class order
   used by the deployed model.

2. ``get_capture_params()``
   Return ONLY the model/capture cadence required by preprocessing:
   ``sample_rate``, ``window_size`` and ``inference_hop_size``.

3. ``validate_config()`` and ``validate_config_multi()``
   Fail fast when model input dtype/shape/layout is incompatible with this DSP.
   Do not silently pad/truncate unknown model tensor layouts.

4. ``preprocess()`` and ``preprocess_multi()``
   Return tensor(s) ready for the TFLite interpreter. For multi-input models,
   a dict keyed by TFLite input index is the preferred unambiguous form.

5. ``get_dsp_info()`` and ``extract_model_inputs()``
   Recommended diagnostic/internal-contract helpers. They improve inspection,
   testing and ModelGuard integration but are not Runner policy controls.

Reliability rules for the data scientist
----------------------------------------
- The DSP and the TFLite model form one versioned mathematical unit.
- Changing sample rate, window/hop, feature extraction, scaling or tensor
  mapping is a model-interface change and requires explicit validation and,
  where appropriate, retraining.
- Runtime dependencies that are part of the trained preprocessing should be
  minimized. Small fitted numeric assets (for example StandardScaler arrays
  stored in an .npz during training) should be embedded as validated numeric
  constants in the DSP for deployment rather than loaded from a separate file.
- Importing the DSP must have no logging/printing, network, file-write or other
  runtime side effects. Diagnostics must be explicit function calls.
- Invalid or ambiguous configurations must fail early with a clear exception;
  producing a syntactically valid but semantically wrong tensor is worse than
  refusing to start.

Reference implementation note
-----------------------------
This template implements the current MFE + 25 auxiliary-feature pipeline. The
auxiliary vector is intentionally extracted from the flattened MFE vector, not
from the raw waveform, because that reproduces the trained pipeline. Future
models may use different mathematics, but MUST preserve the public contract and
architecture boundary above.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.stats import skew, kurtosis, entropy
from scipy.signal import correlate
from scipy.fftpack import dct

# ============================================================================
# Model-specific preprocessing constants
# ============================================================================

# DSP_VERSION identifies the mathematical preprocessing implementation paired
# with the model. Do not bump it for comments-only edits; bump it when the
# deployed preprocessing contract/math changes and re-validation is performed.
DSP_VERSION = "3.0"
DSP_TEMPLATE_REVISION = "1.0"
DSP_TYPE = "MFE_PLUS_AUX_FEATURES"
EXPECTED_INPUT_DTYPE = np.float32

# These values are intrinsic to the trained preprocessing pipeline.
# The runner must obtain them through get_capture_params(); they are not global
# runtime tuning parameters.
SAMPLE_RATE = 16_000
WINDOW_SIZE_SEC = 0.9
WINDOW_STEP_SEC = 0.3
ZERO_PADDING = False
PREEMPHASIS_COEFF = 0.98

WINDOW_SIZE_SAMPLES = int(round(WINDOW_SIZE_SEC * SAMPLE_RATE))
WINDOW_STEP_SAMPLES = int(round(WINDOW_STEP_SEC * SAMPLE_RATE))

if SAMPLE_RATE <= 0:
    raise RuntimeError("SAMPLE_RATE must be > 0")
if WINDOW_SIZE_SEC <= 0 or WINDOW_STEP_SEC <= 0:
    raise RuntimeError("DSP window and hop must be > 0")
if WINDOW_STEP_SEC > WINDOW_SIZE_SEC:
    raise RuntimeError("DSP window step cannot exceed DSP window size")
if WINDOW_SIZE_SAMPLES <= 0 or WINDOW_STEP_SAMPLES <= 0:
    raise RuntimeError("DSP window/hop produced an invalid sample count")

CLASS_NAMES = [
    "Generator_fan",
    "Good",
    "Hydraulic_charge",
    "Impacts_hub_ext",
    "Impacts_hub_int",
    "Impacts_yaw",
    "Stopped",
    "Yaw_lub_mf",
    "Yaw_lub_tracks",
]

# Two MFE branches reproduced from the training/deployment preprocessing.
MFE_BLOCKS: List[Dict[str, Any]] = [
    {
        "FRAME_LENGTH": 0.025,
        "FRAME_STRIDE": 0.400,
        "N_FILTERS": 8,
        "FFT_LENGTH": 2048,
        "LOW_FREQ": 550,
        "HIGH_FREQ": 3350,
        "NOISE_FLOOR_DB": -30,
        "WIN_SIZE": 101,  # Edge Impulse export metadata; not used directly.
    },
    {
        "FRAME_LENGTH": 0.140,
        "FRAME_STRIDE": 0.100,
        "N_FILTERS": 128,
        "FFT_LENGTH": 2048,
        "LOW_FREQ": 50,
        "HIGH_FREQ": 6400,
        "NOISE_FLOOR_DB": -44,
        "WIN_SIZE": 101,  # Edge Impulse export metadata; not used directly.
    },
]

# Optional grouping/trimming parameter. Keep None unless the trained model
# explicitly requires MFE length grouping by a final filter count.
N_FILTERS_FINAL = None

# Auxiliary feature vector = spectral(8) + clipping(3) + temporal(5)
# + harmonic(2) + modulation(2) + cepstral(3) + nonlinear energy(2) = 25.
AUX_FEATURE_DIM = 25

# Fitted StandardScaler parameters embedded from the deployment scaler.
# These constants are inseparable from the trained model preprocessing.
#
# DEPLOYMENT STANDARD FOR .npz ASSETS:
# During training/development the scaler may be stored in an .npz file. For
# PinBolt deployment, extract the required numeric arrays OFFLINE and paste them
# here as typed NumPy constants. Do NOT call np.load(<external .npz>) at runtime.
# After embedding, verify array names/shapes and numerical parity against the
# original artifact with fixed test vectors before releasing DSP + model.
AUX_MEAN = np.asarray([
    4.0831130e+03, 2.2625916e+03, 6.7912490e+03, 7.6079673e+03,
    8.0407190e-01, 6.7320971e+00, -3.4952682e-01, 6.3518291e+00,
    0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 3.4681270e-01,
    9.0465443e-03, 2.0223541e+00, 2.0223541e+00, 5.3288369e+00,
    3.5749583e+00, 1.4510826e+02, 5.9165945e-03, 1.3358312e-05,
   -4.0248337e+01, 1.5480359e+01, 4.7738810e+00, 2.5970773e-03,
    1.0174820e-03,
], dtype=np.float32)

AUX_SCALE = np.asarray([
    3.2456281e+02, 2.0324632e+02, 4.2303244e+02, 3.6058661e+02,
    3.0839151e-01, 6.9596285e-01, 2.3838899e+00, 5.6630367e+01,
    1.0000000e+00, 1.0000000e+00, 1.0000000e+00, 1.4743635e-01,
    2.2481607e-02, 2.4901166e+00, 2.4901166e+00, 2.8497803e+01,
    2.4762967e+00, 2.8253690e+02, 6.2825770e+00, 9.8982500e-03,
    1.5446986e+02, 5.2411799e+00, 2.3984811e+00, 1.3688602e-03,
    6.3391909e-04,
], dtype=np.float32)

def _validate_static_configuration() -> None:
    """Validate deployment constants at import time without side effects.

    This function deliberately performs only deterministic checks. Importing a
    DSP may raise on an invalid package, but it must never print, write files,
    access the network or mutate Runner state. Failing during load is preferable
    to running a model with a mismatched preprocessing definition.
    """
    if not isinstance(DSP_VERSION, str) or not DSP_VERSION.strip():
        raise RuntimeError("DSP_VERSION must be a non-empty string")
    if not isinstance(DSP_TYPE, str) or not DSP_TYPE.strip():
        raise RuntimeError("DSP_TYPE must be a non-empty string")
    if not isinstance(ZERO_PADDING, (bool, np.bool_)):
        raise RuntimeError("ZERO_PADDING must be boolean")
    if not np.isfinite(float(PREEMPHASIS_COEFF)):
        raise RuntimeError("PREEMPHASIS_COEFF must be finite")

    if not isinstance(CLASS_NAMES, list) or not CLASS_NAMES:
        raise RuntimeError("CLASS_NAMES must be a non-empty Python list")
    if any(not isinstance(label, str) or not label.strip() for label in CLASS_NAMES):
        raise RuntimeError("Every CLASS_NAMES entry must be a non-empty string")
    if len(set(CLASS_NAMES)) != len(CLASS_NAMES):
        raise RuntimeError("CLASS_NAMES contains duplicate labels")

    if not isinstance(MFE_BLOCKS, list) or not MFE_BLOCKS:
        raise RuntimeError("MFE_BLOCKS must be a non-empty list")
    nyquist = SAMPLE_RATE / 2.0
    required_keys = {"FRAME_LENGTH", "FRAME_STRIDE", "N_FILTERS", "FFT_LENGTH"}
    for idx, block in enumerate(MFE_BLOCKS, start=1):
        if not isinstance(block, dict):
            raise RuntimeError(f"MFE block {idx} must be a dict")
        missing = sorted(required_keys.difference(block))
        if missing:
            raise RuntimeError(f"MFE block {idx} missing keys: {missing}")
        if float(block["FRAME_LENGTH"]) <= 0 or float(block["FRAME_STRIDE"]) <= 0:
            raise RuntimeError(f"MFE block {idx} frame length/stride must be > 0")
        if int(block["N_FILTERS"]) <= 0 or int(block["FFT_LENGTH"]) <= 0:
            raise RuntimeError(f"MFE block {idx} filters/FFT length must be > 0")
        low = block.get("LOW_FREQ")
        high = block.get("HIGH_FREQ")
        low_eff = 300.0 if low is None else float(low)
        high_eff = nyquist if high is None else float(high)
        if not (0.0 <= low_eff < high_eff <= nyquist):
            raise RuntimeError(
                f"MFE block {idx} frequency range must satisfy "
                f"0 <= LOW_FREQ < HIGH_FREQ <= Nyquist ({nyquist:g} Hz)"
            )
        noise_floor = float(block.get("NOISE_FLOOR_DB", -32))
        if not np.isfinite(noise_floor):
            raise RuntimeError(f"MFE block {idx} NOISE_FLOOR_DB must be finite")

    if N_FILTERS_FINAL is not None and int(N_FILTERS_FINAL) <= 0:
        raise RuntimeError("N_FILTERS_FINAL must be None or a positive integer")

    if AUX_MEAN.shape != (AUX_FEATURE_DIM,) or AUX_SCALE.shape != (AUX_FEATURE_DIM,):
        raise RuntimeError(
            f"Embedded auxiliary scaler must contain exactly {AUX_FEATURE_DIM} mean/scale values"
        )
    if not np.all(np.isfinite(AUX_MEAN)) or not np.all(np.isfinite(AUX_SCALE)):
        raise RuntimeError("Embedded auxiliary scaler contains non-finite values")
    if np.any(AUX_SCALE <= 0):
        raise RuntimeError("Embedded auxiliary scaler contains non-positive scale values")


_validate_static_configuration()


def print_runtime_configuration() -> None:
    """Developer diagnostic. Never called automatically at module import."""
    print("\n" + "=" * 80)
    print("TFLITE DSP CONFIGURATION")
    print("=" * 80)
    print(f"DSP_VERSION            = {DSP_VERSION}")
    print(f"DSP_TEMPLATE_REVISION  = {DSP_TEMPLATE_REVISION}")
    print(f"SAMPLE_RATE            = {SAMPLE_RATE}")
    print(f"WINDOW_SIZE_SEC        = {WINDOW_SIZE_SEC}")
    print(f"WINDOW_STEP_SEC        = {WINDOW_STEP_SEC}")
    print(f"WINDOW_SIZE_SAMPLES    = {WINDOW_SIZE_SAMPLES}")
    print(f"WINDOW_STEP_SAMPLES    = {WINDOW_STEP_SAMPLES}")
    print(f"N_FILTERS_FINAL        = {N_FILTERS_FINAL}")
    print(f"AUX_FEATURE_DIM        = {AUX_FEATURE_DIM}")
    print("AUX_SCALER_EMBEDDED    = True")
    print("OUTPUT_DECIMATION      = runner-owned / not defined in DSP")
    print("\nMFE_BLOCKS:")
    for i, block in enumerate(MFE_BLOCKS, start=1):
        print(f"\nBlock {i}")
        for key, value in block.items():
            print(f"   {key:20s}: {value}")
    print("\nCLASSES:")
    for label in CLASS_NAMES:
        print(f"   {label}")
    print("=" * 80)

# ============================================================================
# DSP module contract
# ============================================================================

def get_labels() -> List[str]:
    """Return class labels in EXACT model-output order.

    The Runner uses this list to interpret model scores. Reordering labels
    without a matching model change silently corrupts semantics, so always
    version and validate labels together with the TFLite model.
    """
    return list(CLASS_NAMES)


def get_capture_params() -> Dict[str, int]:
    """Return ONLY capture cadence intrinsic to the trained preprocessing.

    Contract with upload3.py v43.3:
    - sample_rate: required audio sampling frequency in Hz.
    - window_size: number of samples consumed by one model inference.
    - inference_hop_size: sample advance between consecutive inferences.

    Do NOT add output_decimation, aggregation, thresholds, Ring-Buffer settings
    or other runtime policy here. Current Runner versions explicitly reject
    output_decimation in this dictionary.
    """
    return {
        "sample_rate": SAMPLE_RATE,
        "window_size": WINDOW_SIZE_SAMPLES,
        "inference_hop_size": WINDOW_STEP_SAMPLES,
    }


def get_dsp_info() -> Dict[str, Any]:
    """Return descriptive DSP metadata for diagnostics/introspection.

    This function is informational: it must not become a back door for Runner
    policy. Keep runtime decisions in Runner configuration.
    """
    return {
        "dsp_version": DSP_VERSION,
        "dsp_type": DSP_TYPE,
        "sample_rate": SAMPLE_RATE,
        "window_size_sec": WINDOW_SIZE_SEC,
        "window_step_sec": WINDOW_STEP_SEC,
        "window_size_samples": WINDOW_SIZE_SAMPLES,
        "window_step_samples": WINDOW_STEP_SAMPLES,
        "mfe_blocks": MFE_BLOCKS,
        "aux_feature_dim": AUX_FEATURE_DIM,
        "n_filters_final": N_FILTERS_FINAL,
        "labels": list(CLASS_NAMES),
        "aux_scaler_embedded": True,
        "preemphasis_coeff": PREEMPHASIS_COEFF,
        "output_decimation_owned_by_runner": True,
    }


def validate_config(input_shape: Any, input_dtype: Any) -> None:
    """Validate a single-input TFLite tensor definition.

    Keep this entry point even when the new model is multi-input: it preserves
    legacy/single-input compatibility and provides a simple fail-fast check.
    Exact feature-length compatibility is checked again by preprocess().
    """
    shape = _shape_tuple(input_shape)
    dtype = np.dtype(input_dtype)
    if dtype != np.dtype(EXPECTED_INPUT_DTYPE):
        raise RuntimeError(f"Invalid input dtype: expected {EXPECTED_INPUT_DTYPE}, got {dtype}")
    if len(shape) != 2 or int(shape[0]) != 1 or int(shape[1]) <= 0:
        raise RuntimeError(f"Expected model input shape [1, N>0], got {shape}")


def validate_config_multi(input_details: List[Dict[str, Any]]) -> None:
    """Validate TFLite metadata for one or more input tensors.

    This validator checks metadata integrity before feature extraction. The
    actual mapping of each tensor to MFE/AUX/combined features remains explicit
    in preprocess_multi(); unknown or ambiguous layouts must fail, never be
    repaired by silent padding/truncation.
    """
    if not isinstance(input_details, (list, tuple)) or not input_details:
        raise RuntimeError("TFLite model has no valid input_details sequence")

    seen_indices = set()
    for position, d in enumerate(input_details):
        if not isinstance(d, dict):
            raise RuntimeError(
                f"Invalid TFLite input metadata at position {position}: expected dict"
            )
        for key in ("index", "shape", "dtype"):
            if key not in d:
                raise RuntimeError(
                    f"Incomplete TFLite input metadata at position {position}: missing {key!r}"
                )

        try:
            tensor_index = int(d["index"])
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError(
                f"Invalid TFLite input index at position {position}: {d.get('index')!r}"
            ) from exc
        if tensor_index in seen_indices:
            raise RuntimeError(f"Duplicate TFLite input index: {tensor_index}")
        seen_indices.add(tensor_index)

        shape = _shape_tuple(d["shape"])
        dtype = np.dtype(d["dtype"])
        if dtype != np.dtype(EXPECTED_INPUT_DTYPE):
            raise RuntimeError(
                f"Invalid input dtype for {d.get('name')}: expected float32, got {dtype}"
            )
        if len(shape) != 2 or int(shape[0]) != 1 or int(shape[1]) <= 0:
            raise RuntimeError(
                f"Expected input {d.get('name')} shape [1, N>0], got {shape}"
            )


def preprocess(audio_window: np.ndarray, input_shape: Any, input_dtype: Any) -> np.ndarray:
    """Return one tensor ready for TFLite set_tensor(), with strict dimensions.

    IMPORTANT FOR FUTURE MODELS: adapt this mapping to the model actually
    trained. Never make an incompatible model "fit" by generic padding or
    truncation; that can keep the program running while invalidating inference.
    """
    shape = _shape_tuple(input_shape)
    dtype = np.dtype(input_dtype)
    validate_config(shape, dtype)

    mfe, aux = extract_model_inputs(audio_window, SAMPLE_RATE)
    expected_n = int(shape[-1])
    if expected_n == mfe.shape[0]:
        x = mfe
    elif expected_n == aux.shape[0]:
        x = aux
    elif expected_n == mfe.shape[0] + aux.shape[0]:
        x = np.concatenate([mfe, aux]).astype(np.float32)
    else:
        raise RuntimeError(
            "Unsupported single-input tensor length: "
            f"model expects {expected_n}, while DSP produces "
            f"mfe={mfe.shape[0]}, aux={aux.shape[0]}, combined={mfe.shape[0] + aux.shape[0]}"
        )
    return x.reshape(shape).astype(dtype, copy=False)


def preprocess_multi(audio_window: np.ndarray, input_details: List[Dict[str, Any]]) -> Dict[int, np.ndarray]:
    """Return tensors keyed by TFLite input index for multi-input models.

    A dict {input_index: ndarray} is the preferred Runner contract because it
    is explicit and independent of interpreter input ordering. This reference
    implementation maps MFE/AUX by semantic name first and exact length second.
    A future DSP with different or repeated inputs should implement its mapping
    explicitly; do not broaden heuristics to guess unknown tensor semantics.
    """
    validate_config_multi(input_details)
    mfe, aux = extract_model_inputs(audio_window, SAMPLE_RATE)
    combined = np.concatenate([mfe, aux]).astype(np.float32)

    tensors: Dict[int, np.ndarray] = {}
    assigned_mfe = False
    assigned_aux = False

    for detail in input_details:
        name = str(detail.get("name", "")).lower()
        shape = _shape_tuple(detail.get("shape"))
        dtype = np.dtype(detail.get("dtype"))
        expected_n = int(shape[-1])

        if "mfe" in name:
            if expected_n != mfe.shape[0]:
                raise RuntimeError(
                    f"MFE input {detail.get('name')!r} expects {expected_n} values, "
                    f"but DSP produces {mfe.shape[0]}"
                )
            x = mfe
            assigned_mfe = True
        elif "feat" in name or "aux" in name:
            if expected_n != aux.shape[0]:
                raise RuntimeError(
                    f"Aux input {detail.get('name')!r} expects {expected_n} values, "
                    f"but DSP produces {aux.shape[0]}"
                )
            x = aux
            assigned_aux = True
        elif expected_n == mfe.shape[0] and not assigned_mfe:
            x = mfe
            assigned_mfe = True
        elif expected_n == aux.shape[0] and not assigned_aux:
            x = aux
            assigned_aux = True
        elif expected_n == combined.shape[0] and len(input_details) == 1:
            x = combined
        else:
            raise RuntimeError(
                f"Cannot map TFLite input {detail.get('name')!r} with shape {shape}. "
                f"DSP produces mfe={mfe.shape[0]} and aux={aux.shape[0]} values."
            )

        tensors[int(detail["index"])] = x.reshape(shape).astype(dtype, copy=False)

    return tensors

# ============================================================================
# Feature extraction
# ============================================================================


def extract_model_inputs(audio: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
    """Build the exact model inputs from one Runner-provided audio window.

    Keep one canonical function that reproduces the training preprocessing.
    MIC and FOLDER must call the same mathematics through this DSP contract;
    mode-specific policy belongs to the Runner, not to separate DSP branches.

    IMPORTANT: auxiliary features are intentionally extracted from the MFE
    vector. This mirrors the current trained pipeline. Do not change this to
    raw audio unless the model is retrained or the original training pipeline
    proves otherwise.
    """
    x = _validate_audio(audio, sr)
    mfe = extract_mfe_feature_vector(x, sr)
    aux = extract_aux_feature_vector(mfe, sr)
    return mfe.astype(np.float32), aux.astype(np.float32)


def _validate_audio(audio: np.ndarray, sr: int) -> np.ndarray:
    if int(sr) != SAMPLE_RATE:
        raise RuntimeError(f"Invalid sample rate: expected {SAMPLE_RATE}, got {sr}")
    x = np.asarray(audio)
    if x.size == 0:
        raise RuntimeError("DSP received an empty audio window")
    if x.ndim == 2:
        if x.shape[1] == 1:
            x = x[:, 0]
        elif x.shape[0] == 1:
            x = x[0]
        else:
            raise RuntimeError(f"DSP expects mono audio, got shape {x.shape}")
    elif x.ndim != 1:
        raise RuntimeError(f"DSP expects a 1-D mono audio window, got shape {x.shape}")
    x = x.astype(np.float32, copy=False)
    if not np.all(np.isfinite(x)):
        raise RuntimeError("DSP received non-finite audio samples")
    # Runner paths may provide int16-like samples. Training preprocessing used
    # segment / 2**15, while already-normalized float audio is kept unchanged.
    if np.max(np.abs(x)) > 2.0:
        x = x / float(2 ** 15)
    if len(x) < WINDOW_SIZE_SAMPLES:
        x = np.pad(x, (0, WINDOW_SIZE_SAMPLES - len(x)))
    elif len(x) > WINDOW_SIZE_SAMPLES:
        x = x[:WINDOW_SIZE_SAMPLES]
    return np.clip(x, -1.0, 1.0).astype(np.float32, copy=False)


def extract_mfe_feature_vector(signal: np.ndarray, sr: int) -> np.ndarray:
    # Training preprocessing applies speechpy.processing.preemphasis(signal, cof=0.98)
    signal = _preemphasis(signal, coeff=PREEMPHASIS_COEFF)
    blocks = []
    for block_cfg in MFE_BLOCKS:
        mfe = _sp_mfe_evol_numpy(
            signal,
            sampling_frequency=sr,
            frame_length=float(block_cfg["FRAME_LENGTH"]),
            frame_stride=float(block_cfg["FRAME_STRIDE"]),
            num_filters=int(block_cfg["N_FILTERS"]),
            fft_length=int(block_cfg["FFT_LENGTH"]),
            low_frequency=block_cfg.get("LOW_FREQ"),
            high_frequency=block_cfg.get("HIGH_FREQ"),
            zero_padding=ZERO_PADDING,
        )
        noise_floor_db = float(block_cfg.get("NOISE_FLOOR_DB", -32))
        mfe_db = 10.0 * np.log10(np.clip(mfe, 1e-30, None))
        mfe_norm = (mfe_db - noise_floor_db) / (abs(noise_floor_db) + 12.0)
        mfe_clip = np.clip(mfe_norm, 0.0, 1.0)
        mfe_qu = np.uint8(np.round(mfe_clip * 256.0))
        mfe_fl = mfe_qu.astype(np.float32) / 256.0
        blocks.append(mfe_fl.flatten())
    out = np.concatenate(blocks).astype(np.float32)
    if N_FILTERS_FINAL:
        expected_len = (out.shape[0] // N_FILTERS_FINAL) * N_FILTERS_FINAL
        out = out[:expected_len]
    return out


def extract_aux_feature_vector(signal: np.ndarray, sr: int) -> np.ndarray:
    X = signal.reshape(1, -1).astype(np.float32)
    feats = [
        compute_spectral_features(X, sr=sr),
        compute_clipping_features(X),
        compute_temporal_features(X),
        compute_harmonic_features(X, sr=sr),
        compute_modulation_features(X),
        compute_cepstral_features(X, n_ceps=3),
        compute_nonlinear_energy_features(X),
    ]
    aux = np.concatenate(feats, axis=1).astype(np.float32)
    aux = np.nan_to_num(aux, nan=0.0, posinf=0.0, neginf=0.0)
    aux = _apply_aux_scaler(aux)
    return aux.reshape(-1).astype(np.float32)


def _apply_aux_scaler(aux_2d: np.ndarray) -> np.ndarray:
    """Apply the fitted StandardScaler embedded in this DSP."""
    if aux_2d.ndim != 2 or aux_2d.shape[1] != AUX_FEATURE_DIM:
        raise RuntimeError(
            f"Auxiliary feature dimension mismatch: expected {AUX_FEATURE_DIM}, got {aux_2d.shape}"
        )
    mean = AUX_MEAN.reshape(1, -1)
    scale = AUX_SCALE.reshape(1, -1)
    scaled = (aux_2d.astype(np.float32, copy=False) - mean) / scale
    return np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

# ============================================================================
# MFE helpers, speechpy-compatible enough for deployment dimensions
# ============================================================================

def _preemphasis(x: np.ndarray, coeff: float = 0.98) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if len(x) == 0:
        return x
    return np.append(x[0], x[1:] - coeff * x[:-1]).astype(np.float32)


def _sp_mfe_evol_numpy(
    signal: np.ndarray,
    sampling_frequency: int,
    frame_length: float,
    frame_stride: float,
    num_filters: int,
    fft_length: int,
    low_frequency: Any,
    high_frequency: Any,
    zero_padding: bool,
) -> np.ndarray:
    frames = _own_stack_frames(signal, sampling_frequency, frame_length, frame_stride, zero_padding)
    power_spectrum = _power_spectrum(frames, fft_length)
    coefficients = power_spectrum.shape[1]
    high_frequency = high_frequency or sampling_frequency / 2
    filter_banks = _sp_filterbanks_evol(num_filters, coefficients, sampling_frequency, low_frequency, high_frequency)
    features = np.dot(power_spectrum, filter_banks.T)
    return np.where(features == 0, np.finfo(float).eps, features).astype(np.float32)


def _power_spectrum(frames: np.ndarray, fft_length: int) -> np.ndarray:
    spec = np.fft.rfft(frames, n=fft_length)
    mag = np.absolute(spec)
    return ((1.0 / fft_length) * (mag ** 2)).astype(np.float32)


def _sp_filterbanks_evol(num_filter: int, coefficients: int, sampling_freq: int, low_freq: Any, high_freq: Any) -> np.ndarray:
    high_freq = high_freq or sampling_freq / 2
    if low_freq is None:
        low_freq = 300
    fftpoints = (coefficients - 1) * 2
    mels = np.linspace(_frequency_to_mel(low_freq), _frequency_to_mel(high_freq), num_filter + 2)
    hertz = _mel_to_frequency(mels)
    freq_index = np.floor((fftpoints + 1) * hertz / sampling_freq).astype(int)
    filterbank = np.zeros([num_filter, coefficients], dtype=np.float32)
    for i in range(0, num_filter):
        left = int(freq_index[i])
        middle = int(freq_index[i + 1])
        right = int(freq_index[i + 2])
        if right <= left:
            continue
        left = max(left, 0)
        right = min(right, coefficients - 1)
        z = np.linspace(left, right, num=right - left + 1)
        filterbank[i, left:right + 1] = _triangle(z, left=left, middle=middle, right=right)
    return filterbank


def _own_stack_frames(sig: np.ndarray, sampling_frequency: int, frame_length: float, frame_stride: float, zero_padding: bool) -> np.ndarray:
    # Do not use assert for deployment validation: Python -O can disable it.
    if sig.ndim != 1:
        raise RuntimeError(f"Signal dimension should be (N,), got {sig.shape}")
    length_signal = sig.shape[0]
    frame_sample_length = int(np.round(sampling_frequency * frame_length))
    frame_stride_samples = float(np.round(sampling_frequency * frame_stride))
    if zero_padding:
        numframes = int(math.ceil((length_signal - frame_sample_length) / frame_stride_samples))
        len_sig = int(numframes * frame_stride_samples + frame_sample_length)
        signal = np.concatenate((sig, np.zeros((len_sig - length_signal,), dtype=sig.dtype)))
    else:
        x = (length_signal - (frame_sample_length - frame_stride_samples))
        numframes = int(math.floor(x / frame_stride_samples))
        numframes = max(numframes, 1)
        len_sig = int((numframes - 1) * frame_stride_samples + frame_sample_length)
        signal = sig[0:len_sig]
    indices = np.tile(np.arange(0, frame_sample_length), (numframes, 1)) + np.tile(
        np.arange(0, numframes * frame_stride_samples, frame_stride_samples),
        (frame_sample_length, 1),
    ).T
    return signal[np.asarray(indices, dtype=np.int32)]


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

# ============================================================================
# Auxiliary feature functions
# ============================================================================

def compute_spectral_features(X, sr, n_mels=None, freq_min=0, freq_max=None):
    n_frames, n_bins = X.shape
    if freq_max is None:
        freq_max = sr / 2
    freqs = np.linspace(freq_min, freq_max, n_bins)
    X_safe = np.clip(X, 1e-12, None)
    energy = X_safe / np.sum(X_safe, axis=1, keepdims=True)
    centroid = np.sum(freqs * energy, axis=1)
    bandwidth = np.sqrt(np.sum(((freqs - centroid[:, None]) ** 2) * energy, axis=1))
    rolloff_threshold = 0.85 * np.sum(X_safe, axis=1, keepdims=True)
    rolloff = np.zeros(n_frames)
    cumsum = np.cumsum(X_safe, axis=1)
    for i in range(n_frames):
        rolloff[i] = freqs[min(np.searchsorted(cumsum[i], float(rolloff_threshold[i, 0])), len(freqs)-1)]
    flatness = np.exp(np.mean(np.log(X_safe), axis=1)) / np.mean(X_safe, axis=1)
    spec_entropy = entropy(energy.T)

    # scipy.stats emits precision-loss warnings for constant/silent vectors and
    # returns NaN. The deployed pipeline later normalizes NaN to zero anyway,
    # so handle that case explicitly to keep silence deterministic and quiet.
    row_std = np.std(X_safe, axis=1)
    varying = row_std > np.finfo(np.float32).eps
    spec_skew = np.zeros(n_frames, dtype=np.float32)
    spec_kurt = np.zeros(n_frames, dtype=np.float32)
    if np.any(varying):
        spec_skew[varying] = skew(X_safe[varying], axis=1).astype(np.float32)
        spec_kurt[varying] = kurtosis(X_safe[varying], axis=1).astype(np.float32)
    rolloff95_threshold = 0.95 * np.sum(X_safe, axis=1, keepdims=True)
    rolloff95 = np.zeros(n_frames)
    for i in range(n_frames):
        rolloff95[i] = freqs[min(np.searchsorted(cumsum[i], float(rolloff95_threshold[i, 0])), len(freqs)-1)]
    return np.stack([centroid, bandwidth, rolloff, rolloff95, flatness, spec_entropy, spec_skew, spec_kurt], axis=1)


def compute_clipping_features(X, threshold=0.999):
    n_frames, n_samples = X.shape
    feats = np.zeros((n_frames, 3), dtype=np.float32)
    for i in range(n_frames):
        audio = X[i]
        clipped = np.abs(audio) >= threshold
        n_clipped = np.sum(clipped)
        ratio = n_clipped / n_samples
        events = np.sum(np.diff(clipped.astype(int)) == 1)
        if n_clipped == 0:
            avg_len = 0.0
        else:
            lengths, count = [], 0
            for c in clipped:
                if c:
                    count += 1
                elif count > 0:
                    lengths.append(count); count = 0
            if count > 0:
                lengths.append(count)
            avg_len = np.mean(lengths) if lengths else 0.0
        feats[i] = [ratio, events, avg_len]
    return feats


def compute_temporal_features(X):
    n_frames, _ = X.shape
    feats = np.zeros((n_frames, 5), dtype=np.float32)
    for i in range(n_frames):
        audio = X[i]
        rms = np.sqrt(np.mean(audio**2))
        zcr = np.mean(np.abs(np.diff(np.sign(audio)))) / 2
        peak = np.max(np.abs(audio))
        peak_rms_ratio = peak / (rms + 1e-12)
        crest_factor = peak_rms_ratio
        splits = np.array_split(audio, 4)
        energies = [np.sum(s**2) for s in splits]
        energy_var = np.var(energies)
        feats[i] = [rms, zcr, peak_rms_ratio, crest_factor, energy_var]
    return feats


def compute_harmonic_features(X, sr):
    n_frames, _ = X.shape
    feats = np.zeros((n_frames, 2), dtype=np.float32)
    for i in range(n_frames):
        audio = X[i] - np.mean(X[i])
        if np.all(audio == 0):
            continue
        corr = correlate(audio, audio, mode="full")
        corr = corr[corr.size // 2:]
        max_val = np.max(corr)
        if max_val > 0:
            corr = corr / max_val
        else:
            continue
        d = np.diff(corr)
        starts = np.where(d > 0)[0]
        start = starts[0] if starts.size > 0 else 1
        peak = np.argmax(corr[start:]) + start if start < len(corr) else 0
        pitch_freq = sr / peak if peak > 0 else 0
        val_peak = max(float(corr[peak]), 0.0) if peak < len(corr) else 0.0
        ratio = val_peak / (1.0 - val_peak + 1e-12)
        ratio = max(ratio, 1e-12)
        hnr = 10.0 * np.log10(ratio)
        feats[i] = [hnr, pitch_freq]
    return feats


def compute_modulation_features(X):
    n_frames, _ = X.shape
    feats = np.zeros((n_frames, 2), dtype=np.float32)
    energies = np.sum(X**2, axis=1)
    rms_vals = np.sqrt(np.mean(X**2, axis=1))
    feats[:, 0] = np.diff(energies, prepend=energies[0])
    feats[:, 1] = np.diff(rms_vals, prepend=rms_vals[0])
    return feats


def compute_cepstral_features(X, n_ceps=3):
    n_frames, _ = X.shape
    feats = np.zeros((n_frames, n_ceps), dtype=np.float32)
    for i in range(n_frames):
        spectrum = np.abs(np.fft.rfft(X[i]))**2
        log_spectrum = np.log(spectrum + 1e-12)
        feats[i] = dct(log_spectrum, type=2, norm='ortho')[:n_ceps]
    return feats


def compute_nonlinear_energy_features(X):
    n_frames, n_samples = X.shape
    feats = np.zeros((n_frames, 2), dtype=np.float32)
    for i in range(n_frames):
        x = X[i]
        if len(x) < 3:
            continue
        tkeo = x[1:-1]**2 - x[:-2] * x[2:]
        feats[i] = [np.mean(tkeo), np.var(tkeo)]
    return feats

# ============================================================================
# Small utilities
# ============================================================================

def _shape_tuple(shape: Any) -> Tuple[int, ...]:
    return tuple(int(x) for x in np.asarray(shape).tolist())