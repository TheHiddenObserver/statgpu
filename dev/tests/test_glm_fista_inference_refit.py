"""FISTA inference must report an unpenalized IRLS-refit stationary solution."""

import numpy as np
import pytest

from statgpu.linear_model import GeneralizedLinearModel


def _gamma_data(seed=20261031, n=240, p=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta = np.array([-0.25, 0.15, 0.35])[:p]
    eta = 0.2 + X @ beta
    y = rng.gamma(shape=4.0, scale=np.exp(eta) / 4.0)
    return X, y


@pytest.mark.parametrize("family", ["gamma", "inverse_gaussian"])
def test_explicit_fista_inference_matches_irls_refit(family):
    X, y = _gamma_data()
    common = dict(
        family=family,
        C=0.0,
        compute_inference=True,
        cov_type="nonrobust",
        max_iter=1000,
        tol=1e-9,
        device="cpu",
    )
    reference = GeneralizedLinearModel(solver="irls", **common).fit(X, y)
    fista = GeneralizedLinearModel(solver="fista", **common).fit(X, y)

    np.testing.assert_allclose(fista.coef_, reference.coef_, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        fista.intercept_, reference.intercept_, rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(fista._bse, reference._bse, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        fista._pvalues, reference._pvalues, rtol=0.0, atol=1e-12
    )
    assert fista._fit_metadata["solver_used"] == "fista+irls-inference"


def test_fista_inference_does_not_apply_the_irls_c_penalty():
    X, y = _gamma_data(seed=20261032)
    common = dict(
        family="gamma",
        solver="fista",
        C=1.0,
        max_iter=1000,
        tol=1e-9,
        device="cpu",
    )
    fit_only = GeneralizedLinearModel(compute_inference=False, **common).fit(X, y)
    inference = GeneralizedLinearModel(compute_inference=True, **common).fit(X, y)
    unpenalized_irls = GeneralizedLinearModel(
        family="gamma",
        solver="irls",
        C=0.0,
        compute_inference=True,
        device="cpu",
        max_iter=1000,
        tol=1e-9,
    ).fit(X, y)

    np.testing.assert_allclose(
        inference.coef_, fit_only.coef_, rtol=0.0, atol=1e-3
    )
    np.testing.assert_allclose(
        inference.coef_, unpenalized_irls.coef_, rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        inference._pvalues, unpenalized_irls._pvalues, rtol=0.0, atol=1e-12
    )


def test_fit_only_fista_remains_available():
    X, y = _gamma_data(seed=20261033, n=120, p=2)
    model = GeneralizedLinearModel(
        family="gamma",
        solver="fista",
        C=0.0,
        compute_inference=False,
        max_iter=1000,
        tol=1e-9,
        device="cpu",
    ).fit(X, y)
    assert np.isfinite(model.coef_).all()
    assert not hasattr(model, "pvalues_")
