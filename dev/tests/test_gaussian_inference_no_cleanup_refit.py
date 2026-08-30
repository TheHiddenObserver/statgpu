"""Regression for estimation-only failed refits when cleanup is disabled."""

from __future__ import annotations

import types

import numpy as np
import pytest


def test_no_inference_failed_refit_invalidates_state_without_cleanup(monkeypatch):
    from statgpu.linear_model import PenalizedGeneralizedLinearModel

    X = np.arange(18.0, dtype=np.float64).reshape(6, 3)
    y = np.linspace(-1.0, 1.0, 6)
    model = PenalizedGeneralizedLinearModel(
        loss="squared_error",
        penalty="l2",
        alpha=0.2,
        fit_intercept=False,
        device="cpu",
        solver="exact",
        compute_inference=False,
        gpu_memory_cleanup=False,
    )
    model.fit(X, y)
    assert model._fitted is True
    assert model.coef_ is not None

    class FakeArray:
        def __init__(self, shape, device=1):
            self.shape = tuple(shape)
            self.ndim = len(self.shape)
            self.device = types.SimpleNamespace(id=int(device))

    monkeypatch.setattr(
        model, "_get_backend", lambda backend="auto": types.SimpleNamespace(name="cupy")
    )
    monkeypatch.setattr(
        model, "_auto_backend_override", lambda backend_name, X: backend_name
    )
    monkeypatch.setattr(
        model,
        "_select_solver",
        lambda loss, backend_name=None, X=None: "exact",
    )

    calls = {"count": 0}

    def staged_to_array(value, device=None, backend=None):
        calls["count"] += 1
        if calls["count"] == 1:
            return FakeArray(np.asarray(value).shape)
        raise RuntimeError("synthetic estimation-only conversion failure")

    monkeypatch.setattr(model, "_to_array", staged_to_array)
    monkeypatch.setattr(
        model,
        "_fit_gpu",
        lambda *args, **kwargs: pytest.fail("backend dispatch must not start"),
    )

    with pytest.raises(RuntimeError, match="synthetic estimation-only conversion failure"):
        model.fit(X, y)

    assert model._fitted is False
    assert model.coef_ is None
    assert model.intercept_ is None
    assert model._params is None
    assert model._selected_backend_name is None
    with pytest.raises(RuntimeError, match="Model has not been fitted yet"):
        model.predict(X)
