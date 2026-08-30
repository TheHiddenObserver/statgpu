"""Regression for LinearRegression dispatcher-failure cleanup ownership."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest


def test_linear_cupy_dispatch_failure_cleans_and_invalidates_once(monkeypatch):
    import statgpu.linear_model.wrappers._linear as linear_module
    from statgpu.linear_model import LinearRegression

    X = np.arange(18.0, dtype=np.float64).reshape(6, 3)
    y = np.linspace(-1.0, 1.0, 6)
    model = LinearRegression(
        fit_intercept=False,
        device="cpu",
        compute_inference=True,
        gpu_memory_cleanup=True,
    ).fit(X, y)
    assert model._fitted is True
    assert model.coef_ is not None

    events = []

    class FakeArray:
        def __init__(self, shape, device=1):
            self.shape = tuple(shape)
            self.ndim = len(self.shape)
            self.device = types.SimpleNamespace(id=int(device))

    class FakeDevice:
        def __init__(self, device):
            self.device = int(device)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    fake_cupy = types.SimpleNamespace(cuda=types.SimpleNamespace(Device=FakeDevice))
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)
    monkeypatch.setattr(
        model, "_get_backend", lambda backend="auto": types.SimpleNamespace(name="cupy")
    )
    monkeypatch.setattr(
        model,
        "_to_array",
        lambda value, backend=None: FakeArray(np.asarray(value).shape),
    )
    monkeypatch.setattr(
        linear_module,
        "_cupy_asarray_on_device",
        lambda value, target_device, dtype=None: value,
    )
    monkeypatch.setattr(
        model, "_cleanup_cuda_memory", lambda: events.append("cleanup")
    )

    def fail_after_dispatch(*args, **kwargs):
        events.append("fit")
        raise RuntimeError("synthetic dispatcher failure")

    monkeypatch.setattr(model, "_fit_gpu", fail_after_dispatch)

    with pytest.raises(RuntimeError, match="synthetic dispatcher failure"):
        model.fit(X, y)

    assert events == ["fit", "cleanup"]
    assert model._fitted is False
    assert model.coef_ is None
    assert model.intercept_ is None
    assert model._params is None
    assert model._inference_result is None
    assert model._selected_backend_name is None
    with pytest.raises(RuntimeError):
        model.predict(X)
