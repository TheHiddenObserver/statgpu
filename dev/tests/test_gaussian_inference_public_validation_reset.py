"""Regression coverage for public finite-validation Gaussian refit rollback."""

from __future__ import annotations

import numpy as np
import pytest


def test_public_finite_validation_failed_refit_clears_gaussian_l2_state():
    """NaN rejection outside the inner fit transaction must fail closed."""
    from statgpu.linear_model import PenalizedLinearRegression

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

    model = PenalizedLinearRegression(
        penalty="l2",
        alpha=0.15,
        fit_intercept=True,
        device="cpu",
        solver="exact",
        compute_inference=True,
    )
    model.fit(X, y)
    assert model._fitted is True
    assert model.coef_ is not None
    assert model._inference_result is not None

    X_bad = X.copy()
    X_bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        model.fit(X_bad, y)

    assert model._fitted is False
    assert model.coef_ is None
    assert model.intercept_ is None
    assert model._params is None
    assert model._inference_result is None
    assert model._selected_backend_name is None
    assert model.__sklearn_is_fitted__() is False
    with pytest.raises(RuntimeError, match="Model has not been fitted yet"):
        model.predict(X)


def test_public_ridge_internal_validation_failed_refit_clears_state():
    """Optimized Ridge validation failures must not expose the prior fit."""
    from statgpu.linear_model import Ridge

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

    model = Ridge(
        alpha=0.15,
        fit_intercept=True,
        device="cpu",
        solver="exact",
        compute_inference=True,
    ).fit(X, y)
    assert model._fitted is True
    assert model.coef_ is not None
    assert model._inference_result is not None

    with pytest.raises(ValueError, match="sample_weight must have length n_samples"):
        model.fit(X, y, sample_weight=np.ones(X.shape[0] - 1))

    assert model._fitted is False
    assert model.coef_ is None
    assert model.intercept_ is None
    assert model._params is None
    assert model._inference_result is None
    assert model._selected_backend_name is None
    assert model.__sklearn_is_fitted__() is False
    with pytest.raises(RuntimeError, match="Model has not been fitted yet"):
        model.predict(X)
