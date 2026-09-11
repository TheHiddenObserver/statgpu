"""Boundary tests for weighted inverse-link Gamma smooth solvers."""

from __future__ import annotations

import numpy as np
import pytest

from statgpu.linear_model import GammaRegression


def _data(seed=15601, n=80, p=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(scale=0.2, size=(n, p)).astype(np.float64)
    eta = np.clip(1.0 + X @ np.array([0.08, -0.05, 0.04]), 0.7, 1.3)
    mu = 1.0 / eta
    y = (mu * rng.lognormal(0.0, 0.035, size=n)).astype(np.float64)
    weights = np.linspace(0.55, 1.65, n, dtype=np.float64)
    return X, y, weights


@pytest.mark.parametrize("solver", ["newton", "lbfgs"])
def test_weighted_inverse_gamma_no_intercept_fails_closed_precisely(solver):
    X, y, weights = _data()
    model = GammaRegression(
        link="inverse_power",
        fit_intercept=False,
        solver=solver,
        device="cpu",
        max_iter=300,
        tol=1e-8,
    )

    with pytest.raises(ValueError, match="requires fit_intercept=True"):
        model.fit(X, y, sample_weight=weights)


def test_unweighted_inverse_gamma_no_intercept_keeps_historical_boundary():
    # Issue #150 narrows only the newly opened weighted row.  It does not turn
    # the historical unweighted no-intercept path into a new API migration.
    X, y, _ = _data(seed=15602)
    model = GammaRegression(
        link="inverse_power",
        fit_intercept=False,
        solver="lbfgs",
        device="cpu",
        max_iter=5,
        tol=1e-8,
    )
    # The historical path may converge or emit its existing numerical warning,
    # but it must not be rejected by the new weighted-only capability guard.
    try:
        model.fit(X, y)
    except ValueError as exc:
        assert "requires fit_intercept=True" not in str(exc)
