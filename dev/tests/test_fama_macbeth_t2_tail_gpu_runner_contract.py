"""Hosted contract coverage for the focused Fama-MacBeth extreme t(2) GPU gate."""

from __future__ import annotations

import numpy as np
import pytest

from dev.benchmarks import validate_fama_macbeth_t2_tail_gpu as t2_gpu_gate
from statgpu.inference._reference_distribution import two_sided_reference_inference


def test_extreme_t2_tail_numpy_preserves_representable_subnormal_probability():
    statistic_value = t2_gpu_gate._EXTREME_STATISTIC
    pvalues, critical = two_sided_reference_inference(
        np.asarray([statistic_value], dtype=np.float64),
        distribution="t",
        alpha=0.05,
        backend="numpy",
        xp=np,
        df=2,
    )

    observed = float(np.asarray(pvalues)[0])
    expected = t2_gpu_gate._expected_tail(statistic_value)
    assert observed > 0.0
    assert np.isfinite(observed)
    np.testing.assert_allclose(observed, expected, rtol=2e-15, atol=0.0)
    assert np.isfinite(float(np.asarray(critical)))


def test_extreme_t2_gpu_runner_contract_requires_both_cuda_backends():
    assert t2_gpu_gate.SCHEMA_VERSION == 2
    assert t2_gpu_gate._validate_acceptance_backends(["cupy", "torch"]) == [
        "cupy",
        "torch",
    ]
    with pytest.raises(ValueError, match="requires exactly both GPU backends"):
        t2_gpu_gate._validate_acceptance_backends(["cupy"])

    expected = t2_gpu_gate._expected_tail(t2_gpu_gate._EXTREME_STATISTIC)
    assert expected > 0.0
    assert expected == pytest.approx(1.0e-308, rel=2e-15, abs=0.0)


def test_extreme_t2_gpu_runner_rejects_cross_device_cupy_trace(monkeypatch):
    class Device:
        def __init__(self, device_id):
            self.id = device_id

    class FakeCuPyArray:
        def __init__(self, device_id):
            self.device = Device(device_id)

    monkeypatch.setattr(
        t2_gpu_gate,
        "_is_cupy_array",
        lambda value: isinstance(value, FakeCuPyArray),
    )

    same = FakeCuPyArray(0)
    assert (
        t2_gpu_gate._assert_cuda_native_and_same_device(same, same, same, "cupy")
        == "cuda:0"
    )
    with pytest.raises(AssertionError, match="crossed CUDA devices"):
        t2_gpu_gate._assert_cuda_native_and_same_device(
            FakeCuPyArray(0), FakeCuPyArray(1), FakeCuPyArray(0), "cupy"
        )


def test_extreme_t2_gpu_runner_rejects_cross_device_torch_trace(monkeypatch):
    class FakeTorchTensor:
        def __init__(self, device):
            self.device = device

    monkeypatch.setattr(
        t2_gpu_gate,
        "_is_torch_array",
        lambda value: isinstance(value, FakeTorchTensor),
    )

    same = FakeTorchTensor("cuda:0")
    assert (
        t2_gpu_gate._assert_cuda_native_and_same_device(same, same, same, "torch")
        == "cuda:0"
    )
    with pytest.raises(AssertionError, match="crossed CUDA devices"):
        t2_gpu_gate._assert_cuda_native_and_same_device(
            FakeTorchTensor("cuda:0"),
            FakeTorchTensor("cuda:1"),
            FakeTorchTensor("cuda:0"),
            "torch",
        )


def test_extreme_t2_gpu_runner_rejects_non_native_outputs(monkeypatch):
    class FakeTorchTensor:
        def __init__(self, device):
            self.device = device

    monkeypatch.setattr(
        t2_gpu_gate,
        "_is_torch_array",
        lambda value: isinstance(value, FakeTorchTensor),
    )

    with pytest.raises(AssertionError, match="left the requested CUDA backend"):
        t2_gpu_gate._assert_cuda_native_and_same_device(
            FakeTorchTensor("cuda:0"),
            np.asarray([1.0]),
            FakeTorchTensor("cuda:0"),
            "torch",
        )
