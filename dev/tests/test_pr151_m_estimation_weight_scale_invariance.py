"""PR151 regressions for analytic-weight M-estimation inference identity."""

from __future__ import annotations

import numpy as np
import pytest

from statgpu.glm_core import get_glm_loss
from statgpu.inference._sandwich import m_estimation_inference
from statgpu.linear_model import GeneralizedLinearModel
from statgpu.linear_model.penalized import PenalizedGeneralizedLinearModel


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


def test_m_estimation_constant_analytic_weights_equal_unweighted_nonrobust():
    X, y, _ = _logistic_data(seed=151923, n=96, p=2)
    design = np.column_stack([X, np.ones(X.shape[0], dtype=np.float64)])
    coef = np.array([0.18, -0.11, 0.07], dtype=np.float64)
    loss = get_glm_loss("logistic")

    unweighted = m_estimation_inference(
        loss,
        design,
        y,
        coef,
        cov_type="nonrobust",
    )
    constant = m_estimation_inference(
        loss,
        design,
        y,
        coef,
        cov_type="nonrobust",
        sample_weight=np.full(X.shape[0], 3.5, dtype=np.float64),
    )

    for key in ("bse", "statistic", "pvalues", "conf_int", "cov_params"):
        np.testing.assert_allclose(
            constant[key], unweighted[key], rtol=2e-12, atol=2e-13
        )
    assert constant["dispersion"] == pytest.approx(
        unweighted["dispersion"], rel=0.0, abs=0.0
    )
