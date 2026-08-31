"""Regressions for the final Gaussian inference review-fix contracts."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest


class _FakeArray:
    def __init__(self, shape, device=1):
        self.shape = tuple(shape)
        self.ndim = len(self.shape)
        self.device = types.SimpleNamespace(id=int(device))


class _FakeDevice:
    events = []
    fail_enter = False

    def __init__(self, device):
        self.device = int(device)

    def __enter__(self):
        type(self).events.append(f"enter:{self.device}")
        if type(self).fail_enter:
            raise RuntimeError("synthetic device context failure")
        return self

    def __exit__(self, exc_type, exc, tb):
        type(self).events.append(f"exit:{self.device}")
        return False


def _install_fake_cupy(monkeypatch, *, current_device=1, finite=True, fail_enter=False):
    _FakeDevice.events = []
    _FakeDevice.fail_enter = bool(fail_enter)

    class _Reduction:
        def all(self):
            return self

        def item(self):
            return bool(finite)

    fake_cupy = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            Device=_FakeDevice,
            runtime=types.SimpleNamespace(getDevice=lambda: int(current_device)),
        ),
        isfinite=lambda value: _Reduction(),
    )
    monkeypatch.setitem(sys.modules, "cupy", fake_cupy)
    return fake_cupy


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


def test_linear_failed_refit_cleans_current_attempt_device(monkeypatch):
    from statgpu.linear_model import LinearRegression

    X = np.arange(18.0, dtype=np.float64).reshape(6, 3)
    y = np.linspace(-1.0, 1.0, 6)
    model = LinearRegression(
        fit_intercept=False,
        device="cpu",
        compute_inference=True,
        gpu_memory_cleanup=True,
    ).fit(X, y)
    model._selected_backend_name = "cupy"
    model._selected_backend_device = "cuda:0"

    events = []
    _install_fake_cupy(monkeypatch, current_device=1)
    monkeypatch.setattr(
        model,
        "_get_backend",
        lambda backend="auto": types.SimpleNamespace(name="cupy"),
    )

    calls = {"count": 0}

    def staged_to_array(value, backend=None):
        calls["count"] += 1
        if calls["count"] == 1:
            return _FakeArray(np.asarray(value).shape, device=1)
        raise RuntimeError("synthetic response conversion failure")

    monkeypatch.setattr(model, "_to_array", staged_to_array)
    monkeypatch.setattr(
        model,
        "_cleanup_cuda_memory",
        lambda: events.append("cleanup"),
    )
    monkeypatch.setattr(
        model,
        "_fit_gpu",
        lambda *args, **kwargs: pytest.fail("backend dispatch must not start"),
    )

    with pytest.raises(RuntimeError, match="synthetic response conversion failure"):
        model.fit(X, y)

    assert _FakeDevice.events == ["enter:1", "exit:1"]
    assert events == ["cleanup"]
    assert model._fitted is False
    assert model.coef_ is None
    assert model.intercept_ is None
    assert model._selected_backend_device is None


@pytest.mark.parametrize("compute_inference", [False, True])
def test_pglm_weight_conversion_failure_uses_current_attempt_device(
    monkeypatch,
    compute_inference,
):
    from statgpu.linear_model import PenalizedGeneralizedLinearModel

    X = np.arange(18.0, dtype=np.float64).reshape(6, 3)
    y = np.linspace(-1.0, 1.0, 6)
    weights = np.linspace(1.0, 2.0, 6)
    model = PenalizedGeneralizedLinearModel(
        loss="squared_error",
        penalty="l2",
        alpha=0.2,
        fit_intercept=False,
        device="cpu",
        solver="exact",
        compute_inference=compute_inference,
        gpu_memory_cleanup=True,
    ).fit(X, y)
    model._selected_backend_name = "cupy"
    model._selected_backend_device = "cuda:0"

    events = []
    _install_fake_cupy(monkeypatch, current_device=1)
    monkeypatch.setattr(
        model,
        "_get_backend",
        lambda backend="auto": types.SimpleNamespace(name="cupy"),
    )
    monkeypatch.setattr(
        model,
        "_auto_backend_override",
        lambda backend_name, X: backend_name,
    )
    monkeypatch.setattr(
        model,
        "_select_solver",
        lambda loss, backend_name=None, X=None: "exact",
    )

    def fail_first_conversion(value, device=None, backend=None):
        raise RuntimeError("synthetic sample-weight conversion failure")

    monkeypatch.setattr(model, "_to_array", fail_first_conversion)
    monkeypatch.setattr(
        model,
        "_cleanup_cuda_memory",
        lambda: events.append("cleanup"),
    )
    monkeypatch.setattr(
        model,
        "_fit_gpu",
        lambda *args, **kwargs: pytest.fail("backend dispatch must not start"),
    )

    with pytest.raises(RuntimeError, match="synthetic sample-weight conversion failure"):
        model.fit(X, y, sample_weight=weights)

    assert _FakeDevice.events == ["enter:1", "exit:1"]
    assert events == ["cleanup"]
    assert model._fitted is False
    assert model.coef_ is None
    assert model.intercept_ is None
    assert model._params is None
    assert model._selected_backend_device is None


def test_pglm_public_validation_context_failure_preserves_original_error(monkeypatch):
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

    events = []
    _install_fake_cupy(
        monkeypatch,
        current_device=1,
        finite=False,
        fail_enter=True,
    )
    monkeypatch.setattr(
        model,
        "_cleanup_cuda_memory",
        lambda: events.append("cleanup"),
    )

    class NonFiniteCuPyArray:
        __module__ = "cupy"

        def __init__(self):
            self.device = types.SimpleNamespace(id=1)

    with pytest.raises(
        ValueError,
        match="X must contain only finite values",
    ):
        model.fit(NonFiniteCuPyArray(), y)

    assert _FakeDevice.events == ["enter:1"]
    assert events == ["cleanup"]
    assert model._fitted is False
    assert model.coef_ is None
    assert "n_features_in_" not in model.__dict__


def test_torch_exact_weighted_precompute_preserves_raw_reporting_response(
    monkeypatch,
):
    torch = pytest.importorskip("torch")
    from statgpu.linear_model.penalized._penalized_linear import (
        PenalizedLinearRegression,
    )

    X, y, weights = _weighted_problem()
    model = PenalizedLinearRegression(
        penalty="l2",
        alpha=0.15,
        fit_intercept=True,
        device="cpu",
        solver="exact",
        compute_inference=True,
        gpu_memory_cleanup=False,
    )
    monkeypatch.setattr(
        model,
        "_get_backend",
        lambda backend="auto": types.SimpleNamespace(name="torch"),
    )
    monkeypatch.setattr(
        model,
        "_auto_backend_override",
        lambda backend_name, X: backend_name,
    )

    model.fit(X, y, sample_weight=weights)

    assert model._selected_backend_name == "torch"
    np.testing.assert_allclose(model._y, y, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        model._sample_weight_fit,
        weights,
        rtol=0.0,
        atol=0.0,
    )

    pred = np.asarray(model.predict(X), dtype=float)
    raw_resid = y - pred
    weighted_resid = raw_resid * np.sqrt(weights)
    np.testing.assert_allclose(
        model._raw_resid,
        raw_resid,
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        model._resid,
        weighted_resid,
        rtol=1e-10,
        atol=1e-10,
    )

    y_mean = np.average(y, weights=weights)
    ss_tot = float(np.sum(weights * (y - y_mean) ** 2))
    ss_res = float(np.sum(weights * raw_resid ** 2))
    expected_r2 = 1.0 - ss_res / ss_tot
    expected_f = ((ss_tot - ss_res) / X.shape[1]) / (
        ss_res / model._df_resid
    )
    assert model.rsquared == pytest.approx(expected_r2, rel=1e-10, abs=1e-10)
    assert model.fvalue == pytest.approx(expected_f, rel=1e-10, abs=1e-10)

    # Confirm the numerical path really executed on Torch CPU.
    assert torch.device("cpu").type == "cpu"
