"""Consistency tests against statsmodels and sklearn."""

import numpy as np
import pytest

from statgpu.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from statgpu.survival import CoxPH
from statgpu._config import set_device, cuda_available

try:
    import statsmodels.api as sm
    import statsmodels.duration.api as smd

    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False

try:
    from sklearn.linear_model import Ridge as SklearnRidge
    from sklearn.linear_model import Lasso as SklearnLasso

    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False

try:
    import cupy as cp
    HAS_CUPY = True
except Exception:
    HAS_CUPY = False


@pytest.mark.skipif(not HAS_STATSMODELS, reason="statsmodels not available")
class TestStatsmodelsConsistency:
    """Compare estimation and inference outputs with statsmodels."""

    @pytest.mark.parametrize(
        "n_samples,n_features,seed",
        [
            (2000, 12, 42),
            (8000, 24, 52),
            (20000, 48, 62),
        ],
        ids=["small", "medium", "large"],
    )
    def test_linear_estimation_and_inference_match_statsmodels(self, n_samples, n_features, seed):
        """OLS coefficients and inference should match statsmodels closely."""
        set_device("cpu")
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(n_samples, n_features))
        beta = rng.normal(loc=0.0, scale=1.0, size=n_features)
        y = X @ beta + 3.5 + rng.normal(scale=0.4, size=n_samples)

        sg = LinearRegression(device="cpu")
        sg.fit(X, y)

        X_sm = sm.add_constant(X)
        sm_res = sm.OLS(y, X_sm).fit()

        # Estimation
        assert np.allclose(sg.intercept_, sm_res.params[0], rtol=1e-6, atol=1e-6)
        assert np.allclose(sg.coef_, sm_res.params[1:], rtol=1e-6, atol=1e-6)

        # Inference
        assert np.allclose(sg._bse, sm_res.bse, rtol=1e-4, atol=1e-6)
        assert np.allclose(sg._pvalues, sm_res.pvalues, rtol=1e-3, atol=1e-6)
        assert np.allclose(sg.aic, sm_res.aic, rtol=1e-6, atol=1e-6)
        assert np.allclose(sg.bic, sm_res.bic, rtol=1e-6, atol=1e-6)

    @pytest.mark.parametrize(
        "cov_type,sm_cov_type,n_samples,n_features,seed",
        [
            ("hc0", "HC0", 4000, 12, 72),
            ("hc1", "HC1", 4000, 12, 73),
            ("hc0", "HC0", 12000, 24, 82),
            ("hc1", "HC1", 12000, 24, 83),
        ],
        ids=["hc0-small", "hc1-small", "hc0-medium", "hc1-medium"],
    )
    def test_linear_robust_covariance_matches_statsmodels(
        self, cov_type, sm_cov_type, n_samples, n_features, seed
    ):
        """HC0/HC1 robust SE and inference should align with statsmodels OLS."""
        set_device("cpu")
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(n_samples, n_features))
        beta = rng.normal(size=n_features)

        # Heteroskedastic noise to make robust covariance meaningful.
        noise_scale = 0.2 + 0.8 * np.abs(X[:, 0])
        y = X @ beta + 2.0 + rng.normal(scale=noise_scale, size=n_samples)

        sg = LinearRegression(device="cpu", cov_type=cov_type)
        sg.fit(X, y)

        X_sm = sm.add_constant(X)
        sm_res = sm.OLS(y, X_sm).fit(cov_type=sm_cov_type)

        # Estimation should still match.
        assert np.allclose(sg.intercept_, sm_res.params[0], rtol=1e-6, atol=1e-6)
        assert np.allclose(sg.coef_, sm_res.params[1:], rtol=1e-6, atol=1e-6)

        # Robust inference: statsmodels robust path typically uses z-based p-values.
        assert np.allclose(sg._bse, sm_res.bse, rtol=2e-3, atol=1e-6)
        assert np.allclose(sg._pvalues, sm_res.pvalues, rtol=5e-2, atol=1e-5)
        assert np.allclose(sg._conf_int, sm_res.conf_int(), rtol=2e-2, atol=1e-4)

    @pytest.mark.skipif(
        (not HAS_CUPY) or (not cuda_available()),
        reason="CuPy/CUDA not available",
    )
    @pytest.mark.parametrize(
        "cov_type,sm_cov_type,seed",
        [
            ("hc0", "HC0", 172),
            ("hc1", "HC1", 173),
        ],
        ids=["gpu-hc0", "gpu-hc1"],
    )
    def test_linear_robust_covariance_gpu_matches_statsmodels(self, cov_type, sm_cov_type, seed):
        """GPU HC0/HC1 inference should align with statsmodels robust OLS."""
        set_device("cuda")
        rng = np.random.default_rng(seed)
        n_samples, n_features = 4000, 12
        X = rng.normal(size=(n_samples, n_features))
        beta = rng.normal(size=n_features)
        noise_scale = 0.2 + 0.8 * np.abs(X[:, 0])
        y = X @ beta + 1.5 + rng.normal(scale=noise_scale, size=n_samples)

        sg = LinearRegression(device="cuda", cov_type=cov_type)
        sg.fit(cp.asarray(X), cp.asarray(y))

        X_sm = sm.add_constant(X)
        sm_res = sm.OLS(y, X_sm).fit(cov_type=sm_cov_type)

        assert np.allclose(sg.intercept_, sm_res.params[0], rtol=1e-6, atol=1e-6)
        assert np.allclose(sg.coef_, sm_res.params[1:], rtol=1e-6, atol=1e-6)
        assert np.allclose(sg._bse, sm_res.bse, rtol=2e-3, atol=1e-6)
        assert np.allclose(sg._pvalues, sm_res.pvalues, rtol=5e-2, atol=1e-5)
        assert np.allclose(sg._conf_int, sm_res.conf_int(), rtol=2e-2, atol=1e-4)

    @pytest.mark.parametrize(
        "n_samples,n_features,seed",
        [
            (3000, 8, 7),
            (8000, 16, 17),
            (15000, 24, 27),
        ],
        ids=["small", "medium", "large"],
    )
    def test_logistic_estimation_and_inference_match_statsmodels(self, n_samples, n_features, seed):
        """Logit coefficients and inference should match statsmodels closely."""
        set_device("cpu")
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(n_samples, n_features))
        beta = rng.normal(loc=0.0, scale=0.8, size=n_features)
        logits = X @ beta + 0.15
        p = 1.0 / (1.0 + np.exp(-logits))
        y = (rng.random(n_samples) < p).astype(int)

        # Use very weak regularization to approximate unpenalized MLE.
        sg = LogisticRegression(device="cpu", C=1e10, max_iter=200)
        sg.fit(X, y)

        X_sm = sm.add_constant(X)
        sm_res = sm.Logit(y, X_sm).fit(disp=0, maxiter=200)

        sg_params = np.concatenate(([sg.intercept_], sg.coef_))
        assert np.allclose(sg_params, sm_res.params, rtol=5e-2, atol=5e-2)

        # Inference (SE / p-values) should be close for the same likelihood model.
        assert np.allclose(sg._bse, sm_res.bse, rtol=1e-1, atol=1e-2)
        assert np.allclose(sg._pvalues, sm_res.pvalues, rtol=2e-1, atol=1e-2)

    @pytest.mark.parametrize(
        "cov_type,sm_cov_type,n_samples,n_features,seed",
        [
            ("hc0", "HC0", 5000, 10, 107),
            ("hc1", "HC1", 5000, 10, 108),
        ],
        ids=["logit-hc0-cpu", "logit-hc1-cpu"],
    )
    def test_logistic_robust_covariance_matches_statsmodels(
        self, cov_type, sm_cov_type, n_samples, n_features, seed
    ):
        """Logit HC0/HC1 robust inference should align with statsmodels."""
        set_device("cpu")
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(n_samples, n_features))
        beta = rng.normal(loc=0.0, scale=0.7, size=n_features)
        logits = X @ beta + 0.2
        p = 1.0 / (1.0 + np.exp(-logits))
        y = (rng.random(n_samples) < p).astype(int)

        sg = LogisticRegression(device="cpu", C=1e10, max_iter=200, cov_type=cov_type)
        sg.fit(X, y)

        X_sm = sm.add_constant(X)
        sm_res = sm.Logit(y, X_sm).fit(disp=0, maxiter=200, cov_type=sm_cov_type)

        sg_params = np.concatenate(([sg.intercept_], sg.coef_))
        assert np.allclose(sg_params, sm_res.params, rtol=5e-2, atol=5e-2)
        assert np.allclose(sg._bse, sm_res.bse, rtol=2e-1, atol=2e-2)
        assert np.allclose(sg._pvalues, sm_res.pvalues, rtol=3e-1, atol=2e-2)

    @pytest.mark.skipif(
        (not HAS_CUPY) or (not cuda_available()),
        reason="CuPy/CUDA not available",
    )
    @pytest.mark.parametrize(
        "cov_type,sm_cov_type,seed",
        [
            ("hc0", "HC0", 207),
            ("hc1", "HC1", 208),
        ],
        ids=["logit-hc0-gpu", "logit-hc1-gpu"],
    )
    def test_logistic_robust_covariance_gpu_matches_statsmodels(self, cov_type, sm_cov_type, seed):
        """GPU Logit HC0/HC1 robust inference should align with statsmodels."""
        set_device("cuda")
        rng = np.random.default_rng(seed)
        n_samples, n_features = 5000, 10
        X = rng.normal(size=(n_samples, n_features))
        beta = rng.normal(loc=0.0, scale=0.7, size=n_features)
        logits = X @ beta + 0.2
        p = 1.0 / (1.0 + np.exp(-logits))
        y = (rng.random(n_samples) < p).astype(int)

        sg = LogisticRegression(device="cuda", C=1e10, max_iter=200, cov_type=cov_type)
        sg.fit(cp.asarray(X), cp.asarray(y))

        X_sm = sm.add_constant(X)
        sm_res = sm.Logit(y, X_sm).fit(disp=0, maxiter=200, cov_type=sm_cov_type)

        sg_params = np.concatenate(([sg.intercept_], sg.coef_))
        assert np.allclose(sg_params, sm_res.params, rtol=5e-2, atol=5e-2)
        assert np.allclose(sg._bse, sm_res.bse, rtol=2e-1, atol=2e-2)
        assert np.allclose(sg._pvalues, sm_res.pvalues, rtol=3e-1, atol=2e-2)

    @pytest.mark.parametrize(
        "ties,n_samples,n_features,seed",
        [
            ("breslow", 1200, 10, 310),
            ("efron", 1200, 10, 311),
        ],
        ids=["cox-breslow", "cox-efron"],
    )
    def test_cox_estimation_matches_statsmodels(self, ties, n_samples, n_features, seed):
        """CoxPH coefficients should align with statsmodels PHReg under same ties."""
        set_device("cpu")
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(n_samples, n_features))
        beta = rng.normal(scale=0.35, size=n_features)
        lin = X @ beta
        u = np.clip(rng.random(n_samples), 1e-12, 1 - 1e-12)
        base = 0.03
        t_true = -np.log(u) / (base * np.exp(np.clip(lin, -20, 20)))
        censor = rng.exponential(scale=np.median(t_true), size=n_samples)
        event = (t_true <= censor).astype(int)
        time_obs = np.minimum(t_true, censor)

        sg = CoxPH(device="cpu", ties=ties, max_iter=80, tol=1e-8, compute_inference=True)
        sg.fit(X, time_obs, event)

        sm_res = smd.PHReg(time_obs, X, status=event, ties=ties).fit()

        assert np.allclose(sg.coef_, sm_res.params, rtol=2e-2, atol=2e-3)
        assert np.allclose(sg._bse, sm_res.bse, rtol=2e-1, atol=2e-3)


@pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not available")
class TestSklearnPenaltyConsistency:
    """Compare ridge/lasso estimators with sklearn."""

    @pytest.mark.parametrize(
        "n_samples,n_features,seed",
        [
            (5000, 24, 123),
            (15000, 48, 133),
            (30000, 80, 143),
        ],
        ids=["small", "medium", "large"],
    )
    def test_ridge_estimator_matches_sklearn(self, n_samples, n_features, seed):
        """Ridge coefficients/intercept should align with sklearn."""
        set_device("cpu")
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(n_samples, n_features))
        beta = rng.normal(size=n_features)
        y = X @ beta + 1.3 + rng.normal(scale=0.2, size=n_samples)

        alpha = 1.0
        sg = Ridge(alpha=alpha, fit_intercept=True, device="cpu")
        sg.fit(X, y)

        sk = SklearnRidge(alpha=alpha, fit_intercept=True)
        sk.fit(X, y)

        assert np.allclose(sg.intercept_, sk.intercept_, rtol=1e-6, atol=1e-6)
        assert np.allclose(sg.coef_, sk.coef_, rtol=1e-6, atol=1e-6)

    @pytest.mark.parametrize(
        "n_samples,n_features,seed",
        [
            (3000, 24, 321),
            (8000, 48, 331),
            (15000, 72, 341),
        ],
        ids=["small", "medium", "large"],
    )
    def test_lasso_estimator_matches_sklearn(self, n_samples, n_features, seed):
        """Lasso coefficients/intercept should be close to sklearn."""
        set_device("cpu")
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(n_samples, n_features))
        beta = np.zeros(n_features)
        active = min(6, n_features)
        beta[:active] = np.array([2.2, -1.7, 0.8, 1.1, -0.9, 0.5])[:active]
        y = X @ beta + 0.7 + rng.normal(scale=0.3, size=n_samples)

        alpha = 0.05
        sg = Lasso(
            alpha=alpha,
            fit_intercept=True,
            max_iter=5000,
            tol=1e-6,
            device="cpu",
        )
        sg.fit(X, y)

        sk = SklearnLasso(
            alpha=alpha,
            fit_intercept=True,
            max_iter=5000,
            tol=1e-6,
        )
        sk.fit(X, y)

        # Coordinate-descent implementations may differ slightly in path/stopping;
        # require close but not identical coefficients.
        assert np.allclose(sg.intercept_, sk.intercept_, rtol=2e-2, atol=2e-2)
        assert np.allclose(sg.coef_, sk.coef_, rtol=5e-2, atol=1e-2)
