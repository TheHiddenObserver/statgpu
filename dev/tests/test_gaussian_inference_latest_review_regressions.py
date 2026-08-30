"""Regressions for the latest Gaussian inference review/fix cycle."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

import statgpu.linear_model._gaussian_inference as gi


def _weighted_problem():
    X = np.asarray(
        [
            [-2.0, 0.5],
            [-1.0, 1.5],
            [0.0, -0.5],
            [1.0, 2.0],
            [2.0, 0.25],
            [3.0, 1.0],
        ],
        dtype=np.float64,
    )
    y = np.asarray([-1.2, 0.1, 0.4, 2.2, 1.6, 3.8], dtype=np.float64)
    weights = np.asarray([1.0, 4.0, 2.0, 9.0, 3.0, 5.0], dtype=np.float64)
    return X, y, weights


def test_public_ridge_exact_weighted_diagnostics_use_raw_outcomes():
    from statgpu.linear_model import Ridge

    X, y, weights = _weighted_problem()
    model = Ridge(
        alpha=0.15,
        fit_intercept=True,
        device="cpu",
        solver="exact",
        compute_inference=True,
    ).fit(X, y, sample_weight=weights)

    pred = np.asarray(model.predict(X), dtype=float)
    raw_resid = y - pred
    y_mean = np.average(y, weights=weights)
    ss_tot = float(np.sum(weights * (y - y_mean) ** 2))
    ss_res = float(np.sum(weights * raw_resid ** 2))
    expected_r2 = 1.0 - ss_res / ss_tot
    expected_f = ((ss_tot - ss_res) / X.shape[1]) / (
        ss_res / model._df_resid
    )

    np.testing.assert_allclose(model._y, y, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(model._raw_resid, raw_resid, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(model._sample_weight_fit, weights, rtol=0.0, atol=0.0)
    assert model.rsquared == pytest.approx(expected_r2, rel=1e-10, abs=1e-10)
    assert model.fvalue == pytest.approx(expected_f, rel=1e-10, abs=1e-10)


def test_explicit_torch_host_only_gaussian_helpers_use_cpu_without_cuda(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    X = np.asarray(
        [[-1.0, 0.5], [0.0, 1.0], [1.0, 1.5], [2.0, 2.0]],
        dtype=np.float64,
    )
    coef = np.asarray([0.6, -0.25], dtype=np.float64)
    y = X @ coef + np.asarray([0.1, -0.1, 0.05, -0.05])

    state = gi.build_gaussian_fit_state(
        X,
        y,
        coef,
        0.0,
        False,
        backend="torch",
    )
    assert state.backend == "torch"
    assert state.device == "cpu"
    assert isinstance(state.X_design, torch.Tensor)
    assert state.X_design.device.type == "cpu"

    X_design = np.column_stack([np.ones(X.shape[0]), X[:, 0]])
    params = np.asarray([0.5, 0.2], dtype=np.float64)
    resid = np.asarray([0.1, -0.1, 0.05, -0.05], dtype=np.float64)
    result = gi.compute_gaussian_inference(
        X_design,
        params,
        resid,
        np.asarray(0.02, dtype=np.float64),
        df_resid=2,
        cov_type="nonrobust",
        backend="torch",
    )
    assert result is not None
    assert result.metadata["numerical_backend"] == "torch"
    assert result.metadata["numerical_device"] == "cpu"


def test_no_inference_gpu_fit_does_not_repeat_backend_cleanup(monkeypatch):
    import statgpu.linear_model.penalized._fit_mixin as fit_mixin_module
    from statgpu.linear_model import PenalizedGeneralizedLinearModel

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
        fit_mixin_module,
        "_cupy_asarray_on_device",
        lambda value, target_device, dtype=None: value,
    )

    model = PenalizedGeneralizedLinearModel(
        loss="squared_error",
        penalty="l2",
        alpha=0.2,
        fit_intercept=False,
        device="cpu",
        solver="exact",
        compute_inference=False,
        gpu_memory_cleanup=True,
    )
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
    monkeypatch.setattr(
        model,
        "_to_array",
        lambda value, device=None, backend=None: FakeArray(np.asarray(value).shape),
    )
    monkeypatch.setattr(
        model, "_cleanup_cuda_memory", lambda: events.append("cleanup")
    )

    def fake_fit_gpu(X, y, sample_weight=None):
        events.append("fit")
        model._cleanup_cuda_memory()

    monkeypatch.setattr(model, "_fit_gpu", fake_fit_gpu)

    X = np.arange(18.0, dtype=np.float64).reshape(6, 3)
    y = np.linspace(-1.0, 1.0, 6)
    model.fit(X, y)

    assert events == ["fit", "cleanup"]
    assert model._fitted is True
