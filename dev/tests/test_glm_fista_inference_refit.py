"""FISTA inference must report an IRLS-refit stationary solution."""

import numpy as np
import pytest

from statgpu.linear_model import GeneralizedLinearModel


@pytest.mark.parametrize("family", ["gamma", "inverse_gaussian"])
def test_explicit_fista_inference_matches_irls_refit(family):
    rng = np.random.default_rng(20261031)
    n, p = 240, 3
    X = rng.normal(size=(n, p))
    eta = 0.2 + X @ np.array([-0.25, 0.15, 0.35])
    y = rng.gamma(shape=4.0, scale=np.exp(eta) / 4.0)

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


def test_fit_only_fista_remains_available():
    rng = np.random.default_rng(20261031)
    X = rng.normal(size=(120, 2))
    y = np.exp(0.2 + X @ np.array([0.1, -0.2])) + 0.1
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

