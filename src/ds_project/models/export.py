"""Conversión Keras -> TFLite (float32, float16, dynamic, int8)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

logger = logging.getLogger(__name__)
PathLike = Union[str, Path]


def export_tflite(
    model,
    output_path: PathLike,
    quantization: str = "float32",
    representative_data: Optional[Union[np.ndarray, Sequence[np.ndarray]]] = None,
    n_representative: int = 200,
) -> Path:
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if quantization == "float16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif quantization == "dynamic":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    elif quantization == "int8":
        if representative_data is None or len(representative_data) == 0:
            raise ValueError("int8 requiere representative_data")
        if isinstance(representative_data, np.ndarray):
            rep = [np.asarray(representative_data[:n_representative], dtype=np.float32)]
        else:
            rep = [np.asarray(x[:n_representative], dtype=np.float32) for x in representative_data]
        if not rep or any(len(x) == 0 for x in rep):
            raise ValueError("int8 requiere representative_data")
        if len({len(x) for x in rep}) != 1:
            raise ValueError("Las entradas representativas deben tener el mismo número de muestras")

        def _gen():
            for i in range(len(rep[0])):
                yield [x[i : i + 1] for x in rep]

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = _gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    elif quantization != "float32":
        raise ValueError(f"Cuantización no soportada: {quantization}")

    tflite_bytes = converter.convert()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(tflite_bytes)
    logger.info("TFLite (%s) exportado: %s (%d bytes)", quantization, out, len(tflite_bytes))
    return out
