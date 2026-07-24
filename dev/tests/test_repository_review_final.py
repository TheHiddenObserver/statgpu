import numpy as np
import pytest

from statgpu import Ridge
from statgpu.linear_model.penalized._penalized_linear import PenalizedLinearRegression
from statgpu.survival import CoxPH


def test_ridge_exact_matches_internal_average_loss_objective():
    rng = np.random.default_rng(987)
    n, p = 600, 12
    alpha = 0.17
    X = rng.normal(size=(n, p))
    y = X @ rng.normal(size=p) + 1.7 + rng.normal(scale=0.2, size=n)

    ours = Ridge(
        alpha=alpha,
        fit_intercept=True,
        device="cpu",
        compute_inference=False,
    ).fit(X, y)

    X_mean = X.mean(axis=0)
    y_mean = y.mean()
    X_centered = X - X_mean
    y_centered = y - y_mean
    expected_coef = np.linalg.solve(
        X_centered.T @ X_centered + n * alpha * np.eye(p),
        X_centered.T @ y_centered,
    )
    expected_intercept = y_mean - X_mean @ expected_coef

    np.testing.assert_allclose(ours.coef_, expected_coef, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(ours.intercept_, expected_intercept, rtol=1e-11, atol=1e-11)


def test_ridge_wrapper_matches_penalized_linear_regression():
    rng = np.random.default_rng(988)
    X = rng.normal(size=(350, 9))
    y = X @ rng.normal(size=9) - 0.4 + rng.normal(scale=0.3, size=350)
    alpha = 0.23

    wrapper = Ridge(
        alpha=alpha,
        fit_intercept=True,
        device="cpu",
        compute_inference=False,
    ).fit(X, y)
    framework = PenalizedLinearRegression(
        penalty="l2",
        alpha=alpha,
        fit_intercept=True,
        solver="exact",
        device="cpu",
        compute_inference=False,
    ).fit(X, y)

    np.testing.assert_allclose(wrapper.coef_, framework.coef_, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(wrapper.intercept_, framework.intercept_, rtol=1e-11, atol=1e-11)


def test_ridge_sklearn_mapping_is_explicit_not_same_alpha():
    pytest.importorskip("sklearn")
    from sklearn.linear_model import Ridge as SklearnRidge

    rng = np.random.default_rng(989)
    n, p = 280, 7
    alpha = 0.31
    X = rng.normal(size=(n, p))
    y = X @ rng.normal(size=p) + 0.8 + rng.normal(scale=0.25, size=n)

    ours = Ridge(
        alpha=alpha,
        fit_intercept=True,
        device="cpu",
        compute_inference=False,
    ).fit(X, y)
    reference = SklearnRidge(alpha=n * alpha, fit_intercept=True).fit(X, y)

    np.testing.assert_allclose(ours.coef_, reference.coef_, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(ours.intercept_, reference.intercept_, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("ties", ["breslow", "efron"])
def test_cox_information_orientation_matches_statsmodels(ties):
    statsmodels = pytest.importorskip("statsmodels.duration.api")

    rng = np.random.default_rng(654)
    n, p = 700, 5
    X = rng.normal(size=(n, p))
    beta = rng.normal(scale=0.25, size=p)
    u = np.clip(rng.random(n), 1e-12, 1 - 1e-12)
    true_time = -np.log(u) / (0.04 * np.exp(X @ beta))
    censor = rng.exponential(scale=np.median(true_time), size=n)
    event = (true_time <= censor).astype(int)
    time = np.minimum(true_time, censor)

    ours = CoxPH(ties=ties, device="cpu", max_iter=80, tol=1e-8).fit(
        X, time, event
    )
    reference = statsmodels.PHReg(time, X, status=event, ties=ties).fit()
    assert np.all(ours._bse > 0)
    np.testing.assert_allclose(ours.coef_, reference.params, rtol=2e-2, atol=2e-3)
    np.testing.assert_allclose(ours._bse, reference.bse, rtol=2e-1, atol=2e-3)
