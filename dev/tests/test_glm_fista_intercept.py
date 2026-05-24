import numpy as np
import pytest

from statgpu.linear_model import (
    GammaRegression,
    InverseGaussianRegression,
    TweedieRegression,
    NegativeBinomialRegression,
)
from statgpu.backends import _to_numpy


def _gamma_inverse_power_data(seed=2024, n=180, p=5):
    rng = np.random.default_rng(seed)
    X = rng.normal(scale=0.18, size=(n, p))
    beta = np.array([0.12, -0.08, 0.06, -0.04, 0.03])
    eta = np.clip(0.7 + X @ beta, 0.25, None)
    mu = 1.0 / eta
    y = rng.gamma(shape=4.0, scale=mu / 4.0)
    return X, y


def _gamma_inverse_power_objective(X, y, model):
    eta = np.clip(X @ model.coef_ + model.intercept_, 1e-4, 1e3)
    return float(np.mean(y * eta - np.log(eta)))


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
        max_iter=1000,
        tol=1e-8,
        C=1e12,
        **kwargs,
    )
    model_irls = model_cls(
        device="cpu",
        fit_intercept=True,
        solver="irls",
        max_iter=1000,
        tol=1e-8,
        C=1e12,
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


def test_gamma_rejects_unknown_link():
    with pytest.raises(ValueError, match="GammaRegression link"):
        GammaRegression(link="inverse", solver="irls").fit(
            np.ones((4, 2)), np.ones(4)
        )


@pytest.mark.parametrize("device", ["cpu", "cuda", "torch"])
def test_gamma_inverse_power_fista_matches_cpu_across_backends(device):
    if device == "cuda":
        cp = pytest.importorskip("cupy")
        try:
            cp.cuda.runtime.getDeviceCount()
        except Exception as exc:
            pytest.skip(f"CuPy CUDA unavailable: {exc}")
    if device == "torch":
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("Torch CUDA unavailable")

    X, y = _gamma_inverse_power_data()
    ref = GammaRegression(
        link="inverse_power",
        device="cpu",
        fit_intercept=True,
        solver="fista",
        max_iter=2500,
        tol=1e-7,
    )
    ref.fit(X, y)

    model = GammaRegression(
        link="inverse_power",
        device=device,
        fit_intercept=True,
        solver="fista",
        max_iter=2500,
        tol=1e-7,
    )
    model.fit(X, y)

    pred = np.asarray(_to_numpy(model.predict(X)), dtype=float)
    ref_pred = np.asarray(_to_numpy(ref.predict(X)), dtype=float)
    assert np.all(np.isfinite(pred))
    assert np.all(pred > 0)
    assert abs(_gamma_inverse_power_objective(X, y, model)
               - _gamma_inverse_power_objective(X, y, ref)) < 2e-5
    assert np.max(np.abs(pred - ref_pred)) < 2e-3


def test_gamma_inverse_power_fista_no_intercept_moves_from_degenerate_zero_start():
    """inverse_power FISTA without intercept should not stall at near-zero coef init."""
    X, y = _gamma_inverse_power_data(seed=77, n=240, p=5)
    model = GammaRegression(
        link="inverse_power",
        device="cpu",
        fit_intercept=False,
        solver="fista",
        max_iter=2500,
        tol=1e-8,
    )
    model.fit(X, y)

    pred = np.asarray(model.predict(X), dtype=float)
    assert np.all(np.isfinite(pred))
    assert np.median(pred) < 1e3
    assert np.linalg.norm(model.coef_) > 1e-3


def test_gamma_inverse_power_fista_torch_float32_input_dtype_consistent():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("Torch CUDA unavailable")

    X, y = _gamma_inverse_power_data(seed=9, n=120, p=5)
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    model = GammaRegression(
        link="inverse_power",
        device="torch",
        fit_intercept=True,
        solver="fista",
        max_iter=600,
        tol=1e-7,
    )
    model.fit(X, y)

    pred = np.asarray(_to_numpy(model.predict(X)), dtype=float)
    assert np.all(np.isfinite(pred))
    assert np.all(pred > 0)


def test_gamma_inverse_power_fista_torch_integer_X_runs():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("Torch CUDA unavailable")

    X, y = _gamma_inverse_power_data(seed=11, n=100, p=4)
    X_int = np.rint(X * 3.0).astype(np.int64)
    y = y.astype(np.float64)

    model = GammaRegression(
        link="inverse_power",
        device="torch",
        fit_intercept=True,
        solver="fista",
        max_iter=600,
        tol=1e-7,
    )
    model.fit(X_int, y)

    pred = np.asarray(_to_numpy(model.predict(X_int)), dtype=float)
    assert np.all(np.isfinite(pred))
    assert np.all(pred > 0)
