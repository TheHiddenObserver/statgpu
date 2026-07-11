
import numpy as np
import pytest

from statgpu import Ridge
from statgpu.survival import CoxPH


sklearn = pytest.importorskip("sklearn")
statsmodels = pytest.importorskip("statsmodels.duration.api")
from sklearn.linear_model import Ridge as SklearnRidge


def test_ridge_exact_matches_sklearn_alpha_convention():
    rng = np.random.default_rng(987)
    X = rng.normal(size=(600, 12))
    y = X @ rng.normal(size=12) + 1.7 + rng.normal(scale=0.2, size=600)
    ours = Ridge(alpha=1.0, fit_intercept=True, device="cpu").fit(X, y)
    reference = SklearnRidge(alpha=1.0, fit_intercept=True).fit(X, y)
    np.testing.assert_allclose(ours.coef_, reference.coef_, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(ours.intercept_, reference.intercept_, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("ties", ["breslow", "efron"])
def test_cox_information_orientation_matches_statsmodels(ties):
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
