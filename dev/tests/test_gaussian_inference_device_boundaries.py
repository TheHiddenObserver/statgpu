"""Regression tests for Gaussian inference backend/device boundaries."""

from __future__ import annotations

import sys
import types

import numpy as np

import statgpu.linear_model._gaussian_inference as gi


class _HostTransferOnlyArray:
    """GPU-like array that forbids implicit NumPy coercion."""

    def __init__(self, value):
        self._value = np.asarray(value)

    def __array__(self, *args, **kwargs):
        raise TypeError("implicit NumPy conversion is forbidden")

    def get(self):
        return self._value.copy()


def test_numpy_boundary_uses_explicit_backend_to_host_conversion():
    source = _HostTransferOnlyArray([1.0, 2.0, 3.0])

    converted = gi._as_backend_array(source, "numpy")

    np.testing.assert_array_equal(converted, np.asarray([1.0, 2.0, 3.0]))
    assert converted.dtype == np.float64


def test_cupy_reference_inference_runs_on_statistic_device(monkeypatch):
    state = {"current": 0, "entered": [], "reference_calls": 0}

    class FakeArray:
        def __init__(self, device):
            self.device = types.SimpleNamespace(id=int(device))

    class FakeDevice:
        def __init__(self, device):
            self.device = int(device)
            self.previous = None

        def __enter__(self):
            self.previous = state["current"]
            state["current"] = self.device
            state["entered"].append(self.device)
            return self

        def __exit__(self, exc_type, exc, tb):
            state["current"] = self.previous

    fake_cupy = types.SimpleNamespace(
        abs=lambda value: value,
        cuda=types.SimpleNamespace(Device=FakeDevice),
    )
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)

    statistic = FakeArray(1)
    expected = (object(), object())

    def fake_reference(
        statistic_abs,
        *,
        distribution,
        alpha,
        backend,
        xp,
        df=None,
        device=None,
    ):
        state["reference_calls"] += 1
        assert statistic_abs is statistic
        assert distribution == "normal"
        assert alpha == 0.05
        assert backend == "cupy"
        assert xp is fake_cupy
        assert device is None
        assert state["current"] == 1
        return expected

    monkeypatch.setattr(gi, "two_sided_reference_inference", fake_reference)

    result = gi._reference_inference(
        statistic,
        distribution="normal",
        alpha=0.05,
        backend="cupy",
        device="cuda:1",
    )

    assert result == expected
    assert state["entered"] == [1]
    assert state["reference_calls"] == 1
    assert state["current"] == 0
