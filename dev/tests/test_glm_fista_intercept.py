import numpy as np
import pytest

from statgpu.linear_model import (
    GammaRegression,
    InverseGaussianRegression,
    TweedieRegression,
    NegativeBinomialRegression,
)


@pytest.mark.parametrize(
    "model_cls, sample_fn, kwargs, atol",
    [
        (
            GammaRegression,
            lambda rng, mu: rng.gamma(shape=2.0, scale=np.clip(mu, 1e-6, None) / 2.0),
            {},
            2e-2,
        ),
        (
            InverseGaussianRegression,
            lambda rng, mu: rng.wald(mean=np.clip(mu, 1e-6, None), scale=1.0),
            {},
            5e-2,
        ),
        (
            TweedieRegression,
            lambda rng, mu: rng.gamma(shape=2.0, scale=np.clip(mu, 1e-6, None) / 2.0),
            {"power": 1.5},
            5e-2,
        ),
    ],
)
def test_non_gaussian_glm_fista_intercept_matches_irls(model_cls, sample_fn, kwargs, atol):
    """FISTA with intercept should optimize same objective as IRLS for non-Gaussian GLM."""
    rng = np.random.default_rng(123)
    n, p = 320, 6
    X = rng.normal(size=(n, p))
    beta = np.array([0.7, -0.5, 0.2, -0.1, 0.3, 0.15])
    intercept = 0.4
    eta = X @ beta + intercept
    mu = np.exp(eta)
    y = sample_fn(rng, mu)

    model_fista = model_cls(
        device="cpu",
        fit_intercept=True,
        solver="fista",
        max_iter=600,
        tol=1e-8,
        **kwargs,
    )
    model_irls = model_cls(
        device="cpu",
        fit_intercept=True,
        solver="irls",
        max_iter=600,
        tol=1e-8,
        **kwargs,
    )

    model_fista.fit(X, y)
    model_irls.fit(X, y)

    assert np.isfinite(model_fista.intercept_)
    assert np.isfinite(model_irls.intercept_)
    assert abs(model_fista.intercept_ - model_irls.intercept_) < atol
    assert np.linalg.norm(model_fista.coef_ - model_irls.coef_) < 10 * atol


def test_negative_binomial_fista_intercept_is_finite_and_nonzero():
    """Regression guard: non-Gaussian FISTA path should learn a real intercept (no y-centering path)."""
    rng = np.random.default_rng(321)
    n, p = 260, 5
    X = rng.normal(size=(n, p))
    beta = np.array([0.4, -0.35, 0.25, 0.2, -0.15])
    intercept = 0.6
    mu = np.exp(X @ beta + intercept)
    y = rng.negative_binomial(n=2.0, p=2.0 / (2.0 + np.clip(mu, 1e-6, None)))

    model = NegativeBinomialRegression(
        alpha=0.5,
        device="cpu",
        fit_intercept=True,
        solver="fista",
        max_iter=600,
        tol=1e-8,
    )
    model.fit(X, y)

    assert np.isfinite(model.intercept_)
    assert abs(model.intercept_) > 1e-6


def test_gamma_inverse_power_rejects_fista_solver():
    rng = np.random.default_rng(99)
    X = rng.normal(size=(120, 4))
    y = np.clip(rng.gamma(shape=2.0, scale=1.0, size=120), 1e-6, None)

    model = GammaRegression(
        device="cpu",
        fit_intercept=True,
        solver="fista",
        link="inverse_power",
        max_iter=80,
        tol=1e-6,
    )

    with pytest.raises(ValueError, match="supports GammaRegression only with link='log'"):
        model.fit(X, y)
