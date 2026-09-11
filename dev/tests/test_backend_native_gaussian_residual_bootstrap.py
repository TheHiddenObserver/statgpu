"""Contract tests for Issue #145 backend-native Gaussian residual bootstrap."""

from __future__ import annotations

import numpy as np
import pytest

from statgpu.linear_model import PenalizedLinearRegression
from statgpu.linear_model import (
    _gaussian_residual_bootstrap_backend_contract as _bootstrap,
)


def _data(seed=145, n=56, p=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta = np.array([0.8, -0.45, 0.25])[:p]
    y = 0.35 + X @ beta + rng.normal(scale=0.25, size=n)
    return X, y


def _cpu_model(*, penalty="l1", alpha=0.06, l1_ratio=0.5, seed=17, B=8):
    model = PenalizedLinearRegression(
        penalty=penalty,
        alpha=alpha,
        l1_ratio=l1_ratio,
        fit_intercept=True,
        device="cpu",
        solver="fista",
        max_iter=1200,
        tol=1e-8,
        compute_inference=True,
        inference_method="bootstrap",
        cov_type="nonrobust",
    )
    model.n_bootstrap = B
    model.bootstrap_random_state = seed
    return model


def test_resample_schedule_is_backend_neutral_and_reproducible():
    left = _bootstrap._draw_resample_indices(11, 7, 1234)
    right = _bootstrap._draw_resample_indices(11, 7, 1234)
    other = _bootstrap._draw_resample_indices(11, 7, 1235)

    assert left.dtype == np.int64
    assert left.shape == (7, 11)
    np.testing.assert_array_equal(left, right)
    assert not np.array_equal(left, other)


def test_cpu_bootstrap_preserves_contract_and_is_reproducible():
    X, y = _data()
    first = _cpu_model(seed=77, B=10).fit(X, y)
    second = _cpu_model(seed=77, B=10).fit(X, y)

    assert first.inference_method_ == "residual_bootstrap"
    assert first.inference_target_ == "penalized_coefficient_distribution"
    assert first._inference_result.method == "residual_bootstrap"
    metadata = first._inference_result.metadata
    assert metadata["resampling_scope"] == "unweighted_gaussian_residual"
    assert metadata["resampling_schedule"] == "numpy_generator_control_plane"
    assert metadata["refit_penalty"] == "l1"
    assert metadata["numerical_backend"] == "numpy"
    assert metadata["numerical_device"] == "cpu"
    assert metadata["reporting_backend"] == "numpy"
    assert metadata["reporting_boundary"] == "post_numerical_inference"
    assert metadata["n_bootstrap"] == 10
    assert metadata["random_state"] == 77

    np.testing.assert_allclose(first._bse, second._bse, rtol=0, atol=0)
    np.testing.assert_allclose(first._pvalues, second._pvalues, rtol=0, atol=0)
    np.testing.assert_allclose(first._conf_int, second._conf_int, rtol=0, atol=0)


def test_cpu_elasticnet_bootstrap_preserves_penalty_family():
    X, y = _data(seed=146)
    model = _cpu_model(
        penalty="elasticnet", alpha=0.05, l1_ratio=0.35, seed=5, B=6
    ).fit(X, y)

    assert model._inference_result.metadata["refit_penalty"] == "elasticnet"
    assert np.all(np.isfinite(model._bse))
    assert np.all(np.isfinite(model._conf_int))


def test_weighted_bootstrap_remains_fail_closed():
    X, y = _data(seed=147)
    model = _cpu_model(B=4)
    weights = np.linspace(0.5, 1.5, X.shape[0])

    with pytest.raises(NotImplementedError, match="Weighted Gaussian residual-bootstrap"):
        model.fit(X, y, sample_weight=weights)


def test_too_few_draws_remains_fail_closed():
    X, y = _data(seed=148)
    model = _cpu_model(B=1)

    with pytest.raises(ValueError, match="n_bootstrap must be an integer >= 2"):
        model.fit(X, y)
    assert not getattr(model, "_fitted", False)


def test_torch_backend_consumes_native_bootstrap_responses(monkeypatch):
    torch = pytest.importorskip("torch")
    X_np, y_np = _data(seed=149, n=24, p=2)
    X = torch.as_tensor(X_np, dtype=torch.float64)
    y = torch.as_tensor(y_np, dtype=torch.float64)

    owner = PenalizedLinearRegression(
        penalty="l1",
        alpha=0.04,
        fit_intercept=True,
        device="cpu",
        solver="fista",
        compute_inference=False,
    )
    owner._selected_backend_name = "torch"
    owner._selected_backend_device = "cpu"
    owner._effective_intercept = True
    owner.coef_ = np.array([0.4, -0.2], dtype=np.float64)
    owner.intercept_ = 0.1
    owner._params = np.array([0.1, 0.4, -0.2], dtype=np.float64)
    owner.n_bootstrap = 4
    owner.bootstrap_random_state = 22
    owner.cov_type = "nonrobust"

    seen = []

    class _Child:
        def fit(self, X_value, y_value):
            assert isinstance(X_value, torch.Tensor)
            assert isinstance(y_value, torch.Tensor)
            assert X_value.device.type == "cpu"
            assert y_value.device.type == "cpu"
            seen.append(y_value.detach().clone())
            self._selected_backend_name = "torch"
            self._selected_backend_device = "cpu"
            # Child-public reporting is the established post-fit NumPy boundary.
            y_mean = float(torch.mean(y_value))
            self._params = np.array([y_mean, 0.4, -0.2], dtype=np.float64)
            return self

    monkeypatch.setattr(_bootstrap, "_make_child_refit", lambda *_args, **_kwargs: _Child())
    _bootstrap._backend_native_gaussian_residual_bootstrap(owner, X, y)

    assert len(seen) == 4
    assert owner._inference_result.metadata["numerical_backend"] == "torch"
    assert owner._inference_result.metadata["numerical_device"] == "cpu"
    assert np.all(np.isfinite(owner._bse))


def test_backend_child_provenance_mismatch_fails_closed():
    class _Child:
        _selected_backend_name = "numpy"
        _selected_backend_device = "cpu"

    with pytest.raises(RuntimeError, match="changed execution provenance"):
        _bootstrap._assert_child_provenance(
            _Child(), backend="cupy", device="cuda:0"
        )
