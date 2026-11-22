"""Characterize Gaussian conversion dispatch without launching GPU workloads."""

from __future__ import annotations

import sys
import types
import importlib

import numpy as np
import pytest

from statgpu.linear_model import _gaussian_inference as gi


@pytest.fixture
def torch_dispatch(monkeypatch):
    """Record device/dtype choices; this is not numerical Torch evidence."""
    state = {"available": False, "probes": 0, "calls": []}
    float64 = types.SimpleNamespace(is_floating_point=True, name="float64")
    float32 = types.SimpleNamespace(is_floating_point=True, name="float32")
    int64 = types.SimpleNamespace(is_floating_point=False, name="int64")

    class Tensor:
        def __init__(self, device, dtype):
            self.device = device
            self.dtype = dtype

        def to(self, *, device, dtype):
            state["calls"].append(("to", self, device, dtype))
            return Tensor(device, dtype)

    def as_tensor(value, *, device, dtype):
        state["calls"].append(("as_tensor", value, device, dtype))
        return Tensor(device, dtype)

    def is_available():
        state["probes"] += 1
        return state["available"]

    fake = types.SimpleNamespace(
        Tensor=Tensor,
        float64=float64,
        float32=float32,
        int64=int64,
        as_tensor=as_tensor,
        cuda=types.SimpleNamespace(is_available=is_available),
    )
    monkeypatch.setitem(sys.modules, "torch", fake)
    return fake, state


@pytest.mark.parametrize("available, expected", [(False, "cpu"), (True, "cuda")])
@pytest.mark.parametrize("like", [None, [9.0]])
def test_host_only_inference_without_device_uses_availability(
    torch_dispatch, available, expected, like
):
    torch, state = torch_dispatch
    state["available"] = available
    value = [1.0, 2.0]

    result = gi._as_backend_array(value, "torch", like=like)

    assert result.device == expected
    assert result.dtype is torch.float64
    assert state["probes"] == 1
    assert state["calls"] == [("as_tensor", value, expected, torch.float64)]


@pytest.mark.parametrize("device", ["cpu", "cuda", "cuda:3"])
def test_explicit_device_does_not_probe_or_substitute(torch_dispatch, device):
    torch, state = torch_dispatch
    result = gi._as_backend_array([1.0], "torch", device=device)

    assert result.device == device
    assert result.dtype is torch.float64
    assert state["probes"] == 0


@pytest.mark.parametrize("device", [None, "cpu", "cuda:5"])
@pytest.mark.parametrize("native_dtype", ["float32", "int64"])
def test_native_value_owns_device_and_floating_precision(
    torch_dispatch, device, native_dtype
):
    torch, state = torch_dispatch
    value = torch.Tensor("cuda:2", getattr(torch, native_dtype))

    result = gi._as_backend_array(value, "torch", device=device)

    dtype = torch.float32 if native_dtype == "float32" else torch.float64
    assert result.device == "cuda:2"
    assert result.dtype is dtype
    assert state["probes"] == 0
    assert state["calls"] == [("to", value, "cuda:2", dtype)]


@pytest.mark.parametrize("native_value", [False, True])
@pytest.mark.parametrize("native_dtype", ["float32", "int64"])
def test_native_reference_takes_precedence_over_value_and_explicit_device(
    torch_dispatch, native_value, native_dtype
):
    torch, state = torch_dispatch
    value = torch.Tensor("cuda:2", torch.float64) if native_value else [1.0]
    like = torch.Tensor("cpu", getattr(torch, native_dtype))

    result = gi._as_backend_array(value, "torch", like=like, device="cuda:5")

    dtype = torch.float32 if native_dtype == "float32" else torch.float64
    assert result.device == "cpu"
    assert result.dtype is dtype
    assert state["probes"] == 0
    operation = "to" if native_value else "as_tensor"
    assert state["calls"] == [(operation, value, "cpu", dtype)]


def test_explicit_unavailable_device_error_is_not_retried_on_cpu(torch_dispatch):
    torch, state = torch_dispatch

    def unavailable(value, *, device, dtype):
        state["calls"].append(device)
        raise RuntimeError("requested CUDA device unavailable")

    torch.as_tensor = unavailable
    with pytest.raises(RuntimeError, match="requested CUDA device unavailable"):
        gi._as_backend_array([1.0], "torch", device="cuda:3")
    assert state["calls"] == ["cuda:3"]
    assert state["probes"] == 0


def test_numpy_conversion_keeps_float64_without_loading_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    result = gi._as_backend_array(np.asarray([1.0], dtype=np.float32), "numpy")
    assert result.dtype == np.float64
    np.testing.assert_array_equal(result, [1.0])


def test_historical_installer_import_and_calls_never_rebind_helper():
    original = gi._as_backend_array
    compat = importlib.import_module(
        "statgpu.linear_model._gaussian_inference_device_contract"
    )
    for _ in range(2):
        compat._install_host_torch_device_contract()
        importlib.reload(compat)
        assert gi._as_backend_array is original
    assert original.__module__ == "statgpu.linear_model._gaussian_inference"
    assert not hasattr(original, "_statgpu_host_torch_device_contract")


def test_gaussian_consumers_share_the_canonical_conversion_function():
    from statgpu.linear_model.penalized import _post_selection_ols
    from statgpu.linear_model import _gaussian_residual_bootstrap_backend_contract

    assert _post_selection_ols._as_backend_array is gi._as_backend_array
    assert (
        _gaussian_residual_bootstrap_backend_contract._as_backend_array
        is gi._as_backend_array
    )
