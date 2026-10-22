"""Entrenamiento con semillas, early stopping y pesos de clase."""

from __future__ import annotations

import os
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ds_project.models.architectures import DEFAULT_MODEL_PARAMS, build_model


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    """Fija las semillas de Python, NumPy y TensorFlow.

    Con ``deterministic`` activa además las operaciones deterministas de TensorFlow
    (algo más lento; en GPU hace fallar las ops sin implementación determinista).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
        if deterministic:
            os.environ["TF_DETERMINISTIC_OPS"] = "1"
            tf.config.experimental.enable_op_determinism()
    except ImportError:  # pragma: no cover
        pass


def compute_class_weights(y: np.ndarray) -> Dict[int, float]:
    from sklearn.utils.class_weight import compute_class_weight

    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int,
    params: Optional[Dict[str, Any]] = None,
    epochs: int = 100,
    patience: int = 10,
    use_class_weights: bool = True,
    extra_callbacks: Optional[List[Any]] = None,
    verbose: int = 0,
    X_train_feats: Optional[np.ndarray] = None,
    X_val_feats: Optional[np.ndarray] = None,
    mfe_reshape: Optional[Tuple[int, int]] = None,
) -> Tuple[Any, Dict[str, List[float]]]:
    import tensorflow as tf

    p = {**DEFAULT_MODEL_PARAMS, **(params or {})}
    if (X_train_feats is None) != (X_val_feats is None):
        raise ValueError("X_train_feats y X_val_feats deben proporcionarse juntos")
    model = build_model(
        X_train.shape[1:],
        n_classes,
        p,
        aux_input_shape=X_train_feats.shape[1:] if X_train_feats is not None else None,
        mfe_reshape=mfe_reshape,
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=max(1, patience // 2), min_lr=1e-6
        ),
    ] + list(extra_callbacks or [])

    history = model.fit(
        [X_train, X_train_feats] if X_train_feats is not None else X_train,
        y_train,
        validation_data=(
            ([X_val, X_val_feats], y_val) if X_val_feats is not None else (X_val, y_val)
        ),
        epochs=epochs,
        batch_size=int(p["batch_size"]),
        class_weight=compute_class_weights(y_train) if use_class_weights else None,
        callbacks=callbacks,
        verbose=verbose,
    )
    return model, {k: [float(v) for v in vals] for k, vals in history.history.items()}
