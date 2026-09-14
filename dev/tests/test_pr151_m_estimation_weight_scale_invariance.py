"""PR151 regressions for analytic-weight M-estimation inference identity."""

from __future__ import annotations

import numpy as np
import pytest

from statgpu.linear_model import GeneralizedLinearModel
from statgpu.linear_model.penalized import PenalizedGeneralizedLinearModel
from statgpu.solvers._smooth_domain import _prepare_analytic_sample_weight


def _logistic_data(seed=151921, n=160, p=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(scale=0.65, size=(n, p)).astype(np.float64)
    beta = np.array([0.46, -0.28, 0.17], dtype=np.float64)[:p]
    eta = -0.13 + X @ beta
    prob = 1.0 / (1.0 + np.exp(-eta))
    y = rng.binomial(1, prob).astype(np.float64)
    y[0], y[1] = 0.0, 1.0
    weights = np.linspace(0.35, 1.85, n, dtype=np.float64)
    return X, y, weights


def _assert_same_parameter_inference(base, scaled, *, rtol=2e-9, atol=2e-11):
    np.testing.assert_allclose(base.coef_, scaled.coef_, rtol=rtol, atol=atol)
    assert float(base.intercept_) == pytest.approx(
        float(scaled.intercept_), rel=rtol, abs=atol
    )
    np.testing.assert_allclose(base._bse, scaled._bse, rtol=rtol, atol=atol)
    np.testing.assert_allclose(
        base._pvalues, scaled._pvalues, rtol=rtol, atol=atol
    )
    np.testing.assert_allclose(
        base._conf_int, scaled._conf_int, rtol=rtol, atol=atol
    )


@pytest.mark.parametrize("solver", ["newton", "lbfgs"])
@pytest.mark.parametrize("cov_type", ["nonrobust", "hc0", "hc1"])
def test_ordinary_glm_analytic_weight_inference_is_global_scale_invariant(
    solver, cov_type
):
    X, y, weights = _logistic_data()
    kwargs = dict(
        family="binomial",
        fit_intercept=True,
        solver=solver,
        device="cpu",
        max_iter=700,
        tol=1.0e-10,
        compute_inference=True,
        cov_type=cov_type,
    )

    base = GeneralizedLinearModel(**kwargs).fit(X, y, sample_weight=weights)
    scaled = GeneralizedLinearModel(**kwargs).fit(
        X, y, sample_weight=7.25 * weights
    )

    assert base._selected_solver == scaled._selected_solver == solver
    _assert_same_parameter_inference(base, scaled)


@pytest.mark.parametrize("solver", ["newton", "lbfgs"])
@pytest.mark.parametrize("cov_type", ["nonrobust", "hc0", "hc1"])
def test_penalized_glm_analytic_weight_inference_is_global_scale_invariant(
    solver, cov_type
):
    X, y, weights = _logistic_data(seed=151922)
    kwargs = dict(
        loss="logistic",
        penalty="l2",
        alpha=0.025,
        fit_intercept=True,
        solver=solver,
        device="cpu",
        max_iter=700,
        tol=1.0e-10,
        compute_inference=True,
        inference_method="auto",
        cov_type=cov_type,
    )

    base = PenalizedGeneralizedLinearModel(**kwargs).fit(
        X, y, sample_weight=weights
    )
    scaled = PenalizedGeneralizedLinearModel(**kwargs).fit(
        X, y, sample_weight=7.25 * weights
    )

    assert base._selected_solver == scaled._selected_solver == solver
    assert base.inference_resolved_method_ == "m_estimation"
    assert scaled.inference_resolved_method_ == "m_estimation"
    _assert_same_parameter_inference(base, scaled)


@pytest.mark.parametrize("solver", ["newton", "lbfgs"])
def test_ordinary_smooth_fit_retains_exact_prepared_weight_provenance(solver):
    X, y, weights = _logistic_data(seed=151923, n=104, p=3)
    X = X.astype(np.float32)
    expected = _prepare_analytic_sample_weight(
        weights,
        X.shape[0],
        "numpy",
        X,
    )
    assert expected is not None
    assert expected.dtype == np.float32

    model = GeneralizedLinearModel(
        family="binomial",
        fit_intercept=True,
        solver=solver,
        device="cpu",
        max_iter=700,
        tol=1.0e-8,
        compute_inference=True,
        cov_type="nonrobust",
    ).fit(X, y, sample_weight=weights)

    assert model._statgpu_smooth_effective_unweighted is False
    assert model._statgpu_smooth_prepared_weight.dtype == np.float32
    np.testing.assert_array_equal(model._statgpu_smooth_prepared_weight, expected)
    # Public fitted diagnostics retain the historical raw-weight state; only
    # M-estimation inference consumes the mean-one prepared representation.
    np.testing.assert_allclose(model._sample_weight_inf, weights, rtol=0.0, atol=0.0)


def test_ordinary_nonrobust_inference_survives_float32_raw_sum_overflow():
    X, y, _ = _logistic_data(seed=151924, n=96, p=3)
    X = X.astype(np.float32)
    raw = np.linspace(0.75, 1.05, X.shape[0], dtype=np.float32)
    huge = raw * np.float32(3.0e38)
    base = huge / np.float32(3.0e38)

    assert np.all(np.isfinite(huge))
    with np.errstate(over="ignore"):
        assert not np.isfinite(np.sum(huge, dtype=np.float32))

    kwargs = dict(
        family="binomial",
        fit_intercept=True,
        solver="newton",
        device="cpu",
        max_iter=700,
        tol=1.0e-8,
        compute_inference=True,
        cov_type="nonrobust",
    )
    reference = GeneralizedLinearModel(**kwargs).fit(X, y, sample_weight=base)
    overflow = GeneralizedLinearModel(**kwargs).fit(X, y, sample_weight=huge)

    _assert_same_parameter_inference(
        reference, overflow, rtol=5e-6, atol=5e-7
    )
