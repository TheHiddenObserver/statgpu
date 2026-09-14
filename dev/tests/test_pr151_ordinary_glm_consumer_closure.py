"""Review-closure regressions for PR151 ordinary-GLM post-fit consumers."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from statgpu.linear_model import GeneralizedLinearModel


def _logistic_data(seed=151911, n=160, p=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(scale=0.7, size=(n, p)).astype(np.float64)
    beta = np.array([0.48, -0.31, 0.19], dtype=np.float64)[:p]
    eta = -0.12 + X @ beta
    prob = 1.0 / (1.0 + np.exp(-eta))
    y = rng.binomial(1, prob).astype(np.float64)
    y[0], y[1] = 0.0, 1.0
    return X, y


@pytest.mark.parametrize("solver", ["newton", "lbfgs"])
def test_effectively_uniform_smooth_fit_uses_same_unweighted_inference_state(solver):
    X, y = _logistic_data()
    weights = np.full(X.shape[0], 3.5, dtype=np.float64)
    weights[-1] += 1.0e-8

    kwargs = dict(
        family="binomial",
        solver=solver,
        device="cpu",
        compute_inference=True,
        cov_type="hc0",
        max_iter=600,
        tol=1.0e-10,
    )
    base = GeneralizedLinearModel(**kwargs).fit(X, y)
    almost_uniform = GeneralizedLinearModel(**kwargs).fit(
        X, y, sample_weight=weights
    )

    # The smooth solver's historical effective-uniform rule canonicalizes this
    # request to the unweighted numerical objective. Post-fit diagnostics and
    # M-estimation must consume that same canonical objective.
    assert almost_uniform._sample_weight_inf is None
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
    assert almost_uniform.loglikelihood == base.loglikelihood
    assert almost_uniform.aic == base.aic
    assert almost_uniform.bic == base.bic


def test_weight_contract_installer_is_idempotent_for_inference_wrapper():
    from statgpu.linear_model import _glm_weighted_explicit_solver_contract as contract

    before = GeneralizedLinearModel._compute_inference
    signature = inspect.signature(before)

    contract.install_glm_weighted_explicit_solver_contract()

    assert GeneralizedLinearModel._compute_inference is before
    assert inspect.signature(GeneralizedLinearModel._compute_inference) == signature
    assert hasattr(GeneralizedLinearModel._compute_inference, "__wrapped__")
