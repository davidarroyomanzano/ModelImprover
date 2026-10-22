"""Métricas para modelos Keras y TFLite."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

PathLike = Union[str, Path]


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, labels: Sequence[str]
) -> Dict[str, Any]:
    idx = list(range(len(labels)))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=idx, average="macro", zero_division=0)),
        "precision_macro": float(
            precision_score(y_true, y_pred, labels=idx, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, labels=idx, average="macro", zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=idx).tolist(),
        "report": classification_report(
            y_true,
            y_pred,
            labels=idx,
            target_names=list(labels),
            output_dict=True,
            zero_division=0,
        ),
    }


def evaluate_keras(
    model,
    X: np.ndarray,
    y: np.ndarray,
    labels: Sequence[str],
    X_feats: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    probs = model.predict([X, X_feats] if X_feats is not None else X, verbose=0)
    return compute_metrics(y, np.argmax(probs, axis=-1), labels)


def tflite_predict_proba(
    model_path: PathLike, X: np.ndarray, X_feats: Optional[np.ndarray] = None
) -> np.ndarray:
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    sources = [X] if X_feats is None else [X, X_feats]
    inputs = interpreter.get_input_details()
    if len(inputs) != len(sources):
        raise ValueError(f"El modelo espera {len(inputs)} entradas, se recibieron {len(sources)}")
    source_by_name = {"input_mfe": X}
    if X_feats is not None:
        source_by_name["input_feats"] = X_feats
    ordered_sources = []
    for index, inp in enumerate(inputs):
        matched = [name for name in source_by_name if name in inp.get("name", "")]
        ordered_sources.append(source_by_name[matched[0]] if matched else sources[index])
    if len({len(source) for source in ordered_sources}) != 1:
        raise ValueError("Las entradas deben tener el mismo número de muestras")
    for inp, source in zip(inputs, ordered_sources):
        interpreter.resize_tensor_input(inp["index"], [1, *source.shape[1:]])
    interpreter.allocate_tensors()
    inputs = interpreter.get_input_details()
    out = interpreter.get_output_details()[0]
    out_scale, out_zero = out["quantization"]

    results = []
    for values in zip(*[np.asarray(source, dtype=np.float32) for source in ordered_sources]):
        for inp, value in zip(inputs, values):
            value = value[None, ...]
            in_scale, in_zero = inp["quantization"]
            if np.issubdtype(inp["dtype"], np.integer):
                info = np.iinfo(inp["dtype"])
                value = np.clip(np.round(value / in_scale + in_zero), info.min, info.max).astype(
                    inp["dtype"]
                )
            interpreter.set_tensor(inp["index"], value)
        interpreter.invoke()
        y = interpreter.get_tensor(out["index"])[0]
        if np.issubdtype(out["dtype"], np.integer):
            y = (y.astype(np.float32) - out_zero) * out_scale
        results.append(y.astype(np.float32))
    return np.stack(results) if results else np.zeros((0,), np.float32)


def evaluate_tflite(
    model_path: PathLike,
    X: np.ndarray,
    y: np.ndarray,
    labels: Sequence[str],
    X_feats: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    probs = tflite_predict_proba(model_path, X, X_feats)
    return compute_metrics(y, np.argmax(probs, axis=-1), labels)
