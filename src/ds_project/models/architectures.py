"""Arquitecturas Keras compatibles con TFLite (Conv1D para secuencias, MLP para vectores)."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

DEFAULT_MODEL_PARAMS: Dict[str, Any] = {
    "learning_rate": 1e-3,
    "batch_size": 32,
    "n_conv_layers": 2,
    "filters": 32,
    "kernel_size": 3,
    "n_dense_layers": 1,
    "units": 64,
    "dropout": 0.3,
}


def build_model(
    input_shape: Tuple[int, ...],
    n_classes: int,
    params: Dict[str, Any],
    aux_input_shape: Optional[Tuple[int, ...]] = None,
    mfe_reshape: Optional[Tuple[int, int]] = None,
):
    import tensorflow as tf

    p = {**DEFAULT_MODEL_PARAMS, **(params or {})}
    inputs = tf.keras.Input(shape=tuple(input_shape), name="input_mfe")
    x = inputs

    if mfe_reshape is not None:
        if len(input_shape) != 1 or mfe_reshape[0] * mfe_reshape[1] != input_shape[0]:
            raise ValueError(f"mfe_reshape {mfe_reshape} no encaja con input_shape {input_shape}")
        x = tf.keras.layers.Reshape(tuple(mfe_reshape), name="mfe_reshape")(x)

    if len(input_shape) == 2 or mfe_reshape is not None:
        for i in range(int(p["n_conv_layers"])):
            x = tf.keras.layers.Conv1D(
                int(p["filters"]) * (2**i),
                int(p["kernel_size"]),
                padding="same",
                activation="relu",
                name=f"conv_{i}",
            )(x)
            x = tf.keras.layers.BatchNormalization(name=f"bn_{i}")(x)
            if x.shape[1] is not None and x.shape[1] >= 2:
                x = tf.keras.layers.MaxPooling1D(2, name=f"pool_{i}")(x)
        x = tf.keras.layers.Flatten(name="mfe_flatten")(x)
    elif len(input_shape) != 1:
        raise ValueError(f"input_shape no soportado: {input_shape}")

    model_inputs = inputs
    if aux_input_shape is not None:
        aux_inputs = tf.keras.Input(shape=tuple(aux_input_shape), name="input_feats")
        aux = aux_inputs
        for i in range(int(p["n_dense_layers"])):
            aux = tf.keras.layers.Dense(int(p["units"]), activation="relu", name=f"aux_dense_{i}")(
                aux
            )
            aux = tf.keras.layers.Dropout(float(p["dropout"]), name=f"aux_dropout_{i}")(aux)
        x = tf.keras.layers.Concatenate(name="feature_concat")([x, aux])
        model_inputs = [inputs, aux_inputs]

    for i in range(int(p["n_dense_layers"])):
        x = tf.keras.layers.Dense(int(p["units"]), activation="relu", name=f"dense_{i}")(x)
        x = tf.keras.layers.Dropout(float(p["dropout"]), name=f"dropout_{i}")(x)

    outputs = tf.keras.layers.Dense(n_classes, activation="softmax", name="probs")(x)
    model = tf.keras.Model(model_inputs, outputs, name="audio_classifier")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=float(p["learning_rate"])),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
