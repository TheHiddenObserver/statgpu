"""PR151 regression closure for penalized smooth-GLM inference consumers."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from statgpu.linear_model import PenalizedGLM_CV
from statgpu.linear_model.penalized import PenalizedGeneralizedLinearModel


def _logistic_data(seed=151921, n=180, p=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(scale=0.7, size=(n, p)).astype(np.float64)
    beta = np.array([0.46, -0.29, 0.17], dtype=np.float64)[:p]
    eta = -0.16 + X @ beta
    prob = 1.0 / (1.0 + np.exp(-eta))
    y = rng.binomial(1, prob).astype(np.float64)
    y[0], y[1] = 0.0, 1.0
    return X, y


def _almost_uniform_weights(n):
    weights = np.full(n, 3.5, dtype=np.float64)
    weights[-1] += 1.0e-8
    return weights


def _fit_penalized(solver, X, y, sample_weight):
    return PenalizedGeneralizedLinearModel(
        loss="logistic",
        penalty="l2",
        alpha=0.025,
        fit_intercept=True,
        solver=solver,
        device="cpu",
        max_iter=800,
        tol=1.0e-10,
        compute_inference=True,
        inference_method="auto",
        cov_type="hc0",
    ).fit(X, y, sample_weight=sample_weight)


@pytest.mark.parametrize("solver", ["newton", "lbfgs"])
def test_effectively_uniform_penalized_smooth_inference_matches_unweighted(solver):
    X, y = _logistic_data()
    weights = _almost_uniform_weights(X.shape[0])

    base = _fit_penalized(solver, X, y, None)
    almost_uniform = _fit_penalized(solver, X, y, weights)

    assert base._selected_solver == almost_uniform._selected_solver == solver
    assert base.inference_resolved_method_ == "m_estimation"
    assert almost_uniform.inference_resolved_method_ == "m_estimation"
    np.testing.assert_allclose(
        almost_uniform.coef_, base.coef_, rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(
        almost_uniform.intercept_, base.intercept_, rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(
        almost_uniform._bse, base._bse, rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(
        almost_uniform._pvalues, base._pvalues, rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(
        almost_uniform._conf_int, base._conf_int, rtol=0.0, atol=0.0
    )


def test_effectively_uniform_cv_final_refit_inference_matches_unweighted():
    X, y = _logistic_data(seed=151922, n=120)
    weights = _almost_uniform_weights(X.shape[0])
    common = dict(
        loss="logistic",
        penalty="l2",
        alpha_grid=np.array([0.03], dtype=np.float64),
        cv=2,
        random_state=151,
        solver="newton",
        device="cpu",
        max_iter=600,
        tol=1.0e-9,
        compute_inference=True,
        inference_method="auto",
        cov_type="hc0",
    )

    base = PenalizedGLM_CV(**common).fit(X, y)
    almost_uniform = PenalizedGLM_CV(**common).fit(
        X, y, sample_weight=weights
    )

    assert base.alpha_ == almost_uniform.alpha_ == pytest.approx(0.03)
    assert base.inference_method_ == almost_uniform.inference_method_ == "m_estimation"
    assert base._inference_result is base.estimator_._inference_result
    assert almost_uniform._inference_result is almost_uniform.estimator_._inference_result
    np.testing.assert_allclose(
        almost_uniform.estimator_.coef_, base.estimator_.coef_, rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(
        almost_uniform.estimator_.intercept_,
        base.estimator_.intercept_,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        almost_uniform._bse, base._bse, rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(
        almost_uniform._pvalues, base._pvalues, rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(
        almost_uniform._conf_int, base._conf_int, rtol=0.0, atol=0.0
    )


def test_penalized_inference_weight_installer_is_idempotent():
    from statgpu.linear_model import _glm_weighted_explicit_solver_contract as contract

    before = PenalizedGeneralizedLinearModel._compute_post_fit_gaussian_inference
    signature = inspect.signature(before)

    contract.install_glm_weighted_explicit_solver_contract()

    assert PenalizedGeneralizedLinearModel._compute_post_fit_gaussian_inference is before
    assert (
        inspect.signature(PenalizedGeneralizedLinearModel._compute_post_fit_gaussian_inference)
        == signature
    )
    assert hasattr(
        PenalizedGeneralizedLinearModel._compute_post_fit_gaussian_inference,
        "__wrapped__",
    )
