"""Inverse-Gaussian IRLS must not stall on valid small means."""

import numpy as np
import pytest

from statgpu.linear_model import GeneralizedLinearModel


def _low_mu_data(n=2000, p=5, seed=123):
    sm = pytest.importorskip("statsmodels.api")
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    eta = -2.8 + X @ np.linspace(-0.2, 0.25, p)
    mu = np.exp(eta)
    y = mu * rng.gamma(4.0, 0.25, size=n)
    reference = sm.GLM(
        y,
        sm.add_constant(X),
        family=sm.families.InverseGaussian(link=sm.families.links.Log()),
    ).fit(maxiter=1000, tol=1e-10)
    return X, y, reference


def test_irls_converges_for_means_below_the_old_clip_floor():
    X, y, reference = _low_mu_data()
    model = GeneralizedLinearModel(
        family="inverse_gaussian",
        C=0.0,
        max_iter=1000,
        tol=1e-9,
        compute_inference=False,
        device="cpu",
    ).fit(X, y)

    assert model.n_iter_ < 1000
    np.testing.assert_allclose(model.coef_, reference.params[1:], rtol=0.0, atol=1e-8)
    np.testing.assert_allclose(
        model.intercept_, reference.params[0], rtol=0.0, atol=1e-8
    )


def test_irls_inference_matches_statsmodels_on_small_means():
    X, y, reference = _low_mu_data(seed=124)
    model = GeneralizedLinearModel(
        family="inverse_gaussian",
        C=0.0,
        max_iter=1000,
        tol=1e-9,
        compute_inference=True,
        cov_type="nonrobust",
        device="cpu",
    ).fit(X, y)

    assert model.n_iter_ < 1000
    np.testing.assert_allclose(model._bse, reference.bse, rtol=0.0, atol=1e-8)
    np.testing.assert_allclose(
        model._pvalues, reference.pvalues, rtol=0.0, atol=1e-8
    )
