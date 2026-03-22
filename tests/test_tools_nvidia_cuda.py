from __future__ import annotations

from unittest import mock

from tensorflow.python.client import device_lib as tf_device_lib

import tools.tools_nvidia_cuda as nvc


def test_run_cmd_success(monkeypatch):
    class P:
        returncode = 0
        stdout = "Driver Version: 1.2.3    CUDA Version: 12.0"
        stderr = ""

    monkeypatch.setattr(nvc.subprocess, "run", lambda *a, **k: P())
    code, out, err = nvc._run_cmd(["nvidia-smi"])
    assert code == 0
    assert "CUDA Version" in out


def test_run_cmd_not_found(monkeypatch):
    monkeypatch.setattr(nvc.subprocess, "run", mock.Mock(side_effect=FileNotFoundError()))
    code, out, err = nvc._run_cmd(["missing-cmd"])
    assert code == 127
    assert "not found" in err.lower()


def test_check_nvidia_driver_failure(monkeypatch, capsys):
    monkeypatch.setattr(nvc, "_run_cmd", lambda *_a, **_k: (1, "", "err"))
    assert nvc.check_nvidia_driver_and_cuda() is None


def test_check_nvidia_driver_parse_ok(monkeypatch, capsys):
    out = "| Driver Version: 550.0    CUDA Version: 12.4 |"
    monkeypatch.setattr(nvc, "_run_cmd", lambda *_a, **_k: (0, out, ""))
    info = nvc.check_nvidia_driver_and_cuda()
    assert info["driver_version"] == "550.0"
    assert info["cuda_runtime_version"] == "12.4"


def test_check_nvidia_driver_no_parse(monkeypatch):
    monkeypatch.setattr(nvc, "_run_cmd", lambda *_a, **_k: (0, "no version line", ""))
    nvc.check_nvidia_driver_and_cuda()


def test_check_nvcc_missing(monkeypatch):
    monkeypatch.setattr(nvc, "_run_cmd", lambda *_a, **_k: (1, "", ""))
    assert nvc.check_nvcc() is None


def test_check_nvcc_ok(monkeypatch):
    out = "Cuda compilation tools, release 12.4, V12.4.99"
    monkeypatch.setattr(nvc, "_run_cmd", lambda *_a, **_k: (0, out, ""))
    info = nvc.check_nvcc()
    assert info["cuda_toolkit_version"] == "12.4"


def test_check_nvcc_no_release_line(monkeypatch):
    monkeypatch.setattr(nvc, "_run_cmd", lambda *_a, **_k: (0, "hello", ""))
    nvc.check_nvcc()


def test_check_tf_cuda_build_info_ok(monkeypatch):
    monkeypatch.setattr(
        nvc.tf.sysconfig,
        "get_build_info",
        lambda: {"cuda_version": "12", "cudnn_version": "8"},
    )
    nvc.check_tf_cuda_build_info()


def test_check_tf_cuda_build_info_exc(monkeypatch):
    monkeypatch.setattr(nvc.tf.sysconfig, "get_build_info", mock.Mock(side_effect=RuntimeError("x")))
    assert nvc.check_tf_cuda_build_info() is None


def test_check_tf_gpu_usable_no_gpu(monkeypatch):
    cpu = mock.Mock(device_type="CPU")
    monkeypatch.setattr(tf_device_lib, "list_local_devices", lambda: [cpu])
    assert nvc.check_tf_gpu_usable() == "cpu"


def test_check_tf_gpu_usable_gpu_ok(monkeypatch):
    gpu = mock.Mock(device_type="GPU")
    monkeypatch.setattr(tf_device_lib, "list_local_devices", lambda: [gpu])
    xmock = mock.Mock()
    ymock = mock.Mock()
    ymock.numpy = lambda: None
    monkeypatch.setattr(nvc.tf.random, "uniform", lambda shape: xmock)
    monkeypatch.setattr(nvc.tf, "matmul", lambda a, b: ymock)
    assert nvc.check_tf_gpu_usable() == "gpu"


def test_check_tf_gpu_usable_gpu_fails(monkeypatch):
    gpu = mock.Mock(device_type="GPU")
    monkeypatch.setattr(tf_device_lib, "list_local_devices", lambda: [gpu])

    def boom(*_a, **_k):
        raise RuntimeError("gpu bad")

    monkeypatch.setattr(nvc.tf.random, "uniform", boom)
    assert nvc.check_tf_gpu_usable() == "cpu"
