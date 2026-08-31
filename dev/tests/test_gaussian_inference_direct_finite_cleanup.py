"""Regression for direct-device cleanup after outer PGLM finite validation."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest


def test_pglm_outer_finite_cleanup_bypasses_stale_device_delegate(monkeypatch):
    from statgpu.linear_model import PenalizedGeneralizedLinearModel

    X = np.arange(12.0, dtype=np.float64).reshape(6, 2)
    y = np.linspace(-1.0, 1.0, 6)
    model = PenalizedGeneralizedLinearModel(
        loss="squared_error",
        penalty="l1",
        alpha=0.05,
        fit_intercept=False,
        device="cpu",
        compute_inference=False,
        gpu_memory_cleanup=True,
    ).fit(X, y)

    # Reproduce the review failure: the previous fit says cuda:0, while the
    # rejected refit's finite-check temporaries belong to a CuPy X on cuda:1.
    model._selected_backend_name = "cupy"
    model._selected_backend_device = "cuda:0"
    state = {"current": 0, "entered": [], "cleaned": []}

    class FakeCuPyArray:
        __module__ = "cupy"
        device = types.SimpleNamespace(id=1)

    class FakeFiniteReduction:
        def all(self):
            return self

        def item(self):
            return False

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
            return False

    monkeypatch.setitem(
        sys.modules,
        "cupy",
        types.SimpleNamespace(
            isfinite=lambda value: FakeFiniteReduction(),
            cuda=types.SimpleNamespace(Device=FakeDevice),
        ),
    )
    monkeypatch.setattr(
        model,
        "_cleanup_cuda_memory",
        lambda: state["cleaned"].append(state["current"]),
    )

    with pytest.raises(ValueError, match="X must contain only finite values"):
        model.fit(FakeCuPyArray(), y)

    # A tracked inner cleanup delegate would re-enter stale cuda:0 here. The
    # direct cleanup captured outside that wrapper stack must run only on cuda:1.
    assert state["entered"] == [1]
    assert state["cleaned"] == [1]
    assert state["current"] == 0
    assert model._fitted is False
    assert model.coef_ is None
    assert model.intercept_ is None
    assert model._params is None
    assert model._selected_backend_name is None
    assert model._selected_backend_device is None
