"""Detección y asignación de GPU por worker (sin importar TF en el proceso padre)."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess

logger = logging.getLogger(__name__)


def detect_gpu_count() -> int:
    env = os.getenv("DS_NUM_GPUS")
    if env is not None:
        return max(0, int(env))
    if shutil.which("nvidia-smi") is None:
        return 0
    try:
        out = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return 0
    return sum(1 for line in out.splitlines() if line.strip().startswith("GPU"))


def assign_gpu(worker_idx: int, n_gpus: int) -> str:
    """Fija CUDA_VISIBLE_DEVICES. Llamar antes de importar TensorFlow."""
    value = str(worker_idx % n_gpus) if n_gpus > 0 else "-1"
    os.environ["CUDA_VISIBLE_DEVICES"] = value
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    logger.info("Worker %d -> CUDA_VISIBLE_DEVICES=%s", worker_idx, value)
    return value


def configure_memory_growth() -> None:
    try:
        import tensorflow as tf
    except ImportError:  # pragma: no cover
        return
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as exc:  # TF ya inicializado
            logger.warning("No se pudo activar memory growth: %s", exc)
