import os
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from ds_project.utils import gpu


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.delenv("DS_NUM_GPUS", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("TF_CPP_MIN_LOG_LEVEL", "0")


@pytest.mark.parametrize("value,expected", [("3", 3), ("0", 0), ("-2", 0)])
def test_env_override(monkeypatch, value, expected):
    monkeypatch.setenv("DS_NUM_GPUS", value)
    assert gpu.detect_gpu_count() == expected


def test_no_nvidia_smi():
    with patch.object(gpu.shutil, "which", return_value=None):
        assert gpu.detect_gpu_count() == 0


def test_parse_nvidia_smi():
    out = MagicMock(stdout="GPU 0: A100 (UUID: x)\nGPU 1: A100 (UUID: y)\n\n")
    with patch.object(gpu.shutil, "which", return_value="/usr/bin/nvidia-smi"), patch.object(
        gpu.subprocess, "run", return_value=out
    ) as run:
        assert gpu.detect_gpu_count() == 2
    assert run.call_args.args[0] == ["nvidia-smi", "-L"]


@pytest.mark.parametrize("exc", [OSError("x"), subprocess.TimeoutExpired("nvidia-smi", 10)])
def test_nvidia_smi_errors(exc):
    with patch.object(gpu.shutil, "which", return_value="/usr/bin/nvidia-smi"), patch.object(
        gpu.subprocess, "run", side_effect=exc
    ):
        assert gpu.detect_gpu_count() == 0


@pytest.mark.parametrize("idx,n,expected", [(0, 2, "0"), (3, 2, "1"), (1, 0, "-1")])
def test_assign_gpu(idx, n, expected):
    assert gpu.assign_gpu(idx, n) == expected
    assert os.environ["CUDA_VISIBLE_DEVICES"] == expected


def test_memory_growth_tolerates_runtime_error(fake_tf):
    fake_tf.config.list_physical_devices.return_value = ["g0", "g1"]
    fake_tf.config.experimental.set_memory_growth.side_effect = [
        None,
        RuntimeError("init"),
    ]
    gpu.configure_memory_growth()
    assert fake_tf.config.experimental.set_memory_growth.call_count == 2
