"""Regression coverage for accelerator provenance during public finite validation."""

from __future__ import annotations

import sys
import types

import pytest

from statgpu.backends._validation import check_finite


def test_cupy_finite_reduction_failure_records_backend_provenance(monkeypatch):
    class FakeCuPyArray:
        __module__ = "cupy"

    def fail_isfinite(value):
        raise RuntimeError("synthetic CuPy finite-check allocation failure")

    monkeypatch.setitem(
        sys.modules,
        "cupy",
        types.SimpleNamespace(isfinite=fail_isfinite),
    )

    with pytest.raises(RuntimeError, match="synthetic CuPy") as exc_info:
        check_finite(FakeCuPyArray(), name="X")

    assert getattr(exc_info.value, "_statgpu_finite_backend", None) == "cupy"


def test_cuda_torch_finite_reduction_failure_records_backend_provenance(monkeypatch):
    strided = object()

    class FakeTorchTensor:
        __module__ = "torch"
        layout = strided
        device = types.SimpleNamespace(type="cuda")

    def fail_isfinite(value):
        raise RuntimeError("synthetic Torch finite-check allocation failure")

    monkeypatch.setitem(
        sys.modules,
        "torch",
        types.SimpleNamespace(strided=strided, isfinite=fail_isfinite),
    )

    with pytest.raises(RuntimeError, match="synthetic Torch") as exc_info:
        check_finite(FakeTorchTensor(), name="X")

    assert getattr(exc_info.value, "_statgpu_finite_backend", None) == "torch"
