"""Additional review-closure regressions for PR #151 weight classification."""

from __future__ import annotations

import numpy as np
import pytest

from statgpu.glm_core._logistic import LogisticLoss
from statgpu.linear_model import GeneralizedLinearModel
from statgpu.solvers._lbfgs import _prepare_lbfgs_sample_weight
from statgpu.solvers._newton import _prepare_newton_sample_weight


def test_weight_preparation_keeps_fractional_weights_for_integral_design():
    X = np.ones((4, 2), dtype=np.int64)
    weights = np.array([0.5, 1.5, 0.75, 2.25], dtype=np.float64)

    newton_weight = _prepare_newton_sample_weight(weights, 4, "numpy", X)
    lbfgs_weight = _prepare_lbfgs_sample_weight(
        weights, 4, "numpy", X, LogisticLoss()
    )

    for prepared in (newton_weight, lbfgs_weight):
        assert prepared is not None
        assert np.issubdtype(np.asarray(prepared).dtype, np.floating)
        np.testing.assert_allclose(prepared, weights, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("solver", ["newton", "lbfgs"])
def test_no_intercept_integer_design_matches_float_design_with_fractional_weights(solver):
    rng = np.random.default_rng(151901)
    X_int = rng.integers(-2, 3, size=(140, 3), dtype=np.int64)
    beta = np.array([0.34, -0.22, 0.13], dtype=np.float64)
    prob = 1.0 / (1.0 + np.exp(-(X_int @ beta)))
    y = rng.binomial(1, prob).astype(np.float64)
    y[0], y[1] = 0.0, 1.0
    weights = np.linspace(0.35, 1.75, X_int.shape[0], dtype=np.float64)

    common = dict(
        family="binomial",
        solver=solver,
        fit_intercept=False,
        device="cpu",
        max_iter=800,
        tol=1e-10,
    )
    integer_fit = GeneralizedLinearModel(**common).fit(
        X_int, y, sample_weight=weights
    )
    float_fit = GeneralizedLinearModel(**common).fit(
        X_int.astype(np.float64), y, sample_weight=weights
    )

    assert integer_fit.intercept_ == float_fit.intercept_ == 0.0
    np.testing.assert_allclose(
        integer_fit.coef_, float_fit.coef_, rtol=0.0, atol=2e-12
    )
