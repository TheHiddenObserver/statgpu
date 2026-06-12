"""
Tests for PR #54: dispatch table for _compute_cv_scores.

Covers:
1. Loss formula registry (_LOSS_RESIDUAL_FNS, _LOSS_VALLOSS_FNS)
2. Numerical constants consistency
3. _weighted_mean zero-weight guard
4. _evaluate_loss_numpy with loss-specific params (NB alpha, Tweedie power)
5. Weighted R² score on PenalizedGeneralizedLinearModel
6. PenalizedGLM_CV.score() delegation
7. cv_splits parameter
8. _build_cv_cache helper
9. _cv_fold_general helper (via full CV fit)
10. _lasso_cv cache_key_eff fix
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose


# ---------------------------------------------------------------------------
# 1. Loss formula registry — residual and val_loss functions
# ---------------------------------------------------------------------------

class TestLossRegistry:
    """Verify all 6 losses are registered and produce correct values."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from statgpu.linear_model._penalized_cv import (
            _LOSS_RESIDUAL_FNS, _LOSS_VALLOSS_FNS,
        )
        self.res_fns = _LOSS_RESIDUAL_FNS
        self.val_fns = _LOSS_VALLOSS_FNS

    def test_all_losses_registered(self):
        expected = {"logistic", "poisson", "gamma", "inverse_gaussian",
                    "negative_binomial", "tweedie"}
        assert set(self.res_fns.keys()) == expected
        assert set(self.val_fns.keys()) == expected

    def test_logistic_residual(self):
        """Gradient of logistic loss = sigmoid(eta) - y."""
        eta = np.array([0.0, 1.0, -1.0])
        y = np.array([0.0, 1.0, 0.0])
        res = self.res_fns["logistic"](eta, y)
        expected = 1.0 / (1.0 + np.exp(-eta)) - y
        assert_allclose(res, expected, atol=1e-12)

    def test_logistic_val_loss(self):
        """Logistic loss = -y*eta + softplus(eta)."""
        eta = np.array([0.0, 1.0, -1.0])
        y = np.array([0.0, 1.0, 0.0])
        val = self.val_fns["logistic"](eta, y)
        log1pexp = np.log1p(np.exp(-np.abs(eta))) + np.maximum(eta, 0.0)
        expected = -y * eta + log1pexp
        assert_allclose(val, expected, atol=1e-12)

    def test_poisson_residual(self):
        """Gradient of Poisson loss = mu - y."""
        eta = np.array([0.0, 1.0, -0.5])
        y = np.array([1.0, 2.0, 0.5])
        res = self.res_fns["poisson"](eta, y)
        mu = np.exp(eta)
        expected = mu - y
        assert_allclose(res, expected, atol=1e-10)

    def test_poisson_val_loss(self):
        """Poisson loss = mu - y*log(mu)."""
        eta = np.array([0.0, 1.0, -0.5])
        y = np.array([1.0, 2.0, 0.5])
        val = self.val_fns["poisson"](eta, y)
        mu = np.exp(eta)
        expected = mu - y * np.log(mu)
        assert_allclose(val, expected, atol=1e-10)

    def test_gamma_residual(self):
        """Gradient of gamma loss = 1 - y/mu."""
        eta = np.array([0.0, 0.5, -0.5])
        y = np.array([1.0, 2.0, 0.5])
        res = self.res_fns["gamma"](eta, y)
        mu = np.exp(eta)
        expected = 1.0 - y / mu
        assert_allclose(res, expected, atol=1e-10)

    def test_gamma_val_loss(self):
        """Gamma loss = y/mu + log(mu)."""
        eta = np.array([0.0, 0.5, -0.5])
        y = np.array([1.0, 2.0, 0.5])
        val = self.val_fns["gamma"](eta, y)
        mu = np.exp(eta)
        expected = y / mu + np.log(mu)
        assert_allclose(val, expected, atol=1e-10)

    def test_inverse_gaussian_residual(self):
        """Gradient of inv-gauss loss = (mu - y) / mu^2."""
        eta = np.array([0.0, 0.5, -0.5])
        y = np.array([1.0, 2.0, 0.5])
        res = self.res_fns["inverse_gaussian"](eta, y)
        mu = np.exp(eta)
        expected = (mu - y) / (mu * mu)
        assert_allclose(res, expected, atol=1e-10)

    def test_inverse_gaussian_val_loss(self):
        """Inv-gauss loss = y/(2*mu^2) - 1/mu."""
        eta = np.array([0.0, 0.5, -0.5])
        y = np.array([1.0, 2.0, 0.5])
        val = self.val_fns["inverse_gaussian"](eta, y)
        mu = np.exp(eta)
        expected = y / (2.0 * mu * mu) - 1.0 / mu
        assert_allclose(val, expected, atol=1e-10)

    def test_nb_residual_default_alpha(self):
        """NB gradient with alpha=1: (mu - y) / (1 + mu)."""
        eta = np.array([0.0, 1.0, -1.0])
        y = np.array([1.0, 3.0, 0.5])
        res = self.res_fns["negative_binomial"](eta, y, alpha=1.0)
        mu = np.exp(eta)
        expected = (mu - y) / (1.0 + mu)
        assert_allclose(res, expected, atol=1e-10)

    def test_nb_residual_custom_alpha(self):
        """NB gradient with custom alpha."""
        eta = np.array([0.0, 1.0, -1.0])
        y = np.array([1.0, 3.0, 0.5])
        alpha = 2.0
        res = self.res_fns["negative_binomial"](eta, y, alpha=alpha)
        mu = np.exp(eta)
        expected = (mu - y) / (1.0 + alpha * mu)
        assert_allclose(res, expected, atol=1e-10)

    def test_nb_val_loss_default_alpha(self):
        """NB val loss with alpha=1: -y*log(mu/(1+mu)) + log(1+mu)."""
        eta = np.array([0.0, 1.0, -1.0])
        y = np.array([1.0, 3.0, 0.5])
        val = self.val_fns["negative_binomial"](eta, y, alpha=1.0)
        mu = np.exp(eta)
        one_plus = 1.0 + mu
        expected = -y * np.log(mu / one_plus) + np.log(one_plus)
        assert_allclose(val, expected, atol=1e-10)

    def test_tweedie_residual_default_power(self):
        """Tweedie gradient with power=1.5."""
        eta = np.array([0.0, 0.5, -0.5])
        y = np.array([1.0, 2.0, 0.5])
        power = 1.5
        res = self.res_fns["tweedie"](eta, y, power=power)
        mu = np.exp(eta)
        expected = mu ** (1 - power) * (mu - y)
        assert_allclose(res, expected, atol=1e-10)

    def test_tweedie_val_loss_default_power(self):
        """Tweedie val loss with power=1.5."""
        eta = np.array([0.0, 0.5, -0.5])
        y = np.array([1.0, 2.0, 0.5])
        power = 1.5
        val = self.val_fns["tweedie"](eta, y, power=power)
        mu = np.exp(eta)
        expected = -y * mu ** (1 - power) / (1 - power) + mu ** (2 - power) / (2 - power)
        assert_allclose(val, expected, atol=1e-10)

    def test_tweedie_boundary_power_1(self):
        """Tweedie with power=1 (Poisson limit) should use log form."""
        eta = np.array([0.0, 0.5])
        y = np.array([1.0, 2.0])
        val = self.val_fns["tweedie"](eta, y, power=1.0)
        # For power=1: term1 = -y*log(mu), term2 = mu
        mu = np.exp(eta)
        expected = -y * np.log(mu) + mu
        assert_allclose(val, expected, atol=1e-10)

    def test_tweedie_boundary_power_2(self):
        """Tweedie with power=2 (Gamma limit) should use log form."""
        eta = np.array([0.0, 0.5])
        y = np.array([1.0, 2.0])
        val = self.val_fns["tweedie"](eta, y, power=2.0)
        mu = np.exp(eta)
        # For power=2: term1 = y/mu (via exp(-1*log(mu))/(-1)), term2 = log(mu)
        expected = y / mu + np.log(mu)
        assert_allclose(val, expected, atol=1e-10)


# ---------------------------------------------------------------------------
# 2. Numerical constants
# ---------------------------------------------------------------------------

class TestNumericalConstants:
    """Verify constants are consistent with loss class defaults."""

    def test_nb_alpha_default(self):
        from statgpu.linear_model._penalized_cv import _NB_ALPHA_DEFAULT
        from statgpu.glm_core._negative_binomial import NegativeBinomialLoss
        loss = NegativeBinomialLoss()
        assert _NB_ALPHA_DEFAULT == loss.alpha

    def test_tweedie_power_default(self):
        from statgpu.linear_model._penalized_cv import _TWEEDIE_POWER_DEFAULT
        from statgpu.glm_core._tweedie import TweedieLoss
        loss = TweedieLoss()
        assert _TWEEDIE_POWER_DEFAULT == loss.power

    def test_eta_clip_values(self):
        from statgpu.linear_model._penalized_cv import (
            _ETA_CLIP_STANDARD, _ETA_CLIP_TWEEDIE, _ETA_CLIP_LOGISTIC,
        )
        assert _ETA_CLIP_STANDARD == 30.0
        assert _ETA_CLIP_TWEEDIE == 50.0
        assert _ETA_CLIP_LOGISTIC == 500.0

    def test_mu_clip_values(self):
        from statgpu.linear_model._penalized_cv import (
            _MU_LO, _MU_LO_TWEEDIE, _MU_HI_TWEEDIE,
        )
        assert _MU_LO == 1e-10
        assert _MU_LO_TWEEDIE == 1e-3
        assert _MU_HI_TWEEDIE == 1e4


# ---------------------------------------------------------------------------
# 3. _weighted_mean zero-weight guard
# ---------------------------------------------------------------------------

class TestWeightedMean:
    def test_unweighted(self):
        from statgpu.linear_model._penalized_cv import _weighted_mean
        vals = np.array([1.0, 2.0, 3.0])
        assert _weighted_mean(vals, None) == pytest.approx(2.0)

    def test_weighted(self):
        from statgpu.linear_model._penalized_cv import _weighted_mean
        vals = np.array([1.0, 2.0, 3.0])
        sw = np.array([1.0, 2.0, 1.0])
        expected = (1*1 + 2*2 + 3*1) / 4
        assert _weighted_mean(vals, sw) == pytest.approx(expected)

    def test_zero_weights_fallback(self):
        """When sum(weights)==0, should fall back to unweighted mean."""
        from statgpu.linear_model._penalized_cv import _weighted_mean
        vals = np.array([1.0, 2.0, 3.0])
        sw = np.array([0.0, 0.0, 0.0])
        assert _weighted_mean(vals, sw) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# 4. _evaluate_loss_numpy with loss-specific params
# ---------------------------------------------------------------------------

class TestEvaluateLossNumpy:
    def test_logistic(self):
        from statgpu.linear_model._penalized_cv import _evaluate_loss_numpy
        from statgpu.linear_model._penalized import _resolve_loss_name
        loss_fn = _resolve_loss_name("logistic")
        rng = np.random.default_rng(42)
        X = rng.standard_normal((20, 3))
        y = (rng.standard_normal(20) > 0).astype(float)
        coef = np.array([0.1, -0.2, 0.05])
        intercept = 0.0
        val = _evaluate_loss_numpy("logistic", loss_fn, X, y, coef, intercept, True)
        assert np.isfinite(val)
        assert val > 0

    def test_nb_custom_alpha(self):
        """NB with custom alpha should use that alpha, not default."""
        from statgpu.linear_model._penalized_cv import _evaluate_loss_numpy
        from statgpu.glm_core._negative_binomial import NegativeBinomialLoss
        loss_fn = NegativeBinomialLoss(alpha=2.0)
        rng = np.random.default_rng(42)
        X = rng.standard_normal((20, 3))
        y = rng.poisson(2, 20).astype(float)
        coef = np.array([0.1, -0.05, 0.02])
        intercept = 0.5
        val = _evaluate_loss_numpy("negative_binomial", loss_fn, X, y, coef, intercept, True)
        assert np.isfinite(val)

    def test_tweedie_custom_power(self):
        """Tweedie with custom power should use that power, not default."""
        from statgpu.linear_model._penalized_cv import _evaluate_loss_numpy
        from statgpu.glm_core._tweedie import TweedieLoss
        loss_fn = TweedieLoss(power=1.7)
        rng = np.random.default_rng(42)
        X = rng.standard_normal((20, 3))
        y = rng.standard_exponential(20) + 0.1
        coef = np.array([0.1, -0.05, 0.02])
        intercept = 0.5
        val = _evaluate_loss_numpy("tweedie", loss_fn, X, y, coef, intercept, True)
        assert np.isfinite(val)

    def test_squared_error_weighted(self):
        """Weighted squared error evaluation."""
        from statgpu.linear_model._penalized_cv import _evaluate_loss_numpy
        from statgpu.linear_model._penalized import _resolve_loss_name
        loss_fn = _resolve_loss_name("squared_error")
        rng = np.random.default_rng(42)
        X = rng.standard_normal((20, 3))
        y = X @ np.array([1.0, -1.0, 0.5]) + rng.standard_normal(20) * 0.1
        coef = np.array([0.9, -0.9, 0.4])
        intercept = 0.0
        sw = np.ones(20)
        val_w = _evaluate_loss_numpy("squared_error", loss_fn, X, y, coef, intercept, True, sample_weight=sw)
        val_uw = _evaluate_loss_numpy("squared_error", loss_fn, X, y, coef, intercept, True)
        assert_allclose(val_w, val_uw, rtol=1e-10)


# ---------------------------------------------------------------------------
# 5. Weighted R² score on PenalizedGeneralizedLinearModel
# ---------------------------------------------------------------------------

class TestWeightedR2:
    def test_unweighted_r2(self):
        """Basic R² without weights should be close to 1 for good fit."""
        from statgpu.linear_model._penalized import PenalizedGeneralizedLinearModel
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 3))
        y = X @ np.array([1.0, -1.0, 0.5]) + rng.standard_normal(100) * 0.01
        m = PenalizedGeneralizedLinearModel(loss='squared_error', penalty='l2', alpha=1e-6)
        m.fit(X, y)
        r2 = m.score(X, y)
        assert r2 > 0.99

    def test_uniform_weighted_r2_matches_unweighted(self):
        """Uniform weights should give same R² as unweighted."""
        from statgpu.linear_model._penalized import PenalizedGeneralizedLinearModel
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 3))
        y = X @ np.array([1.0, -1.0, 0.5]) + rng.standard_normal(100) * 0.01
        m = PenalizedGeneralizedLinearModel(loss='squared_error', penalty='l2', alpha=1e-6)
        m.fit(X, y)
        r2_unweighted = m.score(X, y)
        r2_uniform = m.score(X, y, sample_weight=np.ones(100))
        assert_allclose(r2_uniform, r2_unweighted, rtol=1e-10)

    def test_weighted_r2_finite(self):
        """Weighted R² should be finite and in reasonable range."""
        from statgpu.linear_model._penalized import PenalizedGeneralizedLinearModel
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 3))
        y = X @ np.array([1.0, -1.0, 0.5]) + rng.standard_normal(100) * 0.1
        sw = rng.uniform(0.5, 2.0, 100)
        m = PenalizedGeneralizedLinearModel(loss='squared_error', penalty='l2', alpha=1e-6)
        m.fit(X, y, sample_weight=sw)
        r2 = m.score(X, y, sample_weight=sw)
        assert np.isfinite(r2)
        assert r2 > 0.5

    def test_zero_sum_weights_returns_zero(self):
        """When all weights are zero, score should return 0.0."""
        from statgpu.linear_model._penalized import PenalizedGeneralizedLinearModel
        rng = np.random.default_rng(42)
        X = rng.standard_normal((20, 3))
        y = rng.standard_normal(20)
        m = PenalizedGeneralizedLinearModel(loss='squared_error', penalty='l2', alpha=1e-6)
        m.fit(X, y)
        r2 = m.score(X, y, sample_weight=np.zeros(20))
        assert r2 == 0.0


# ---------------------------------------------------------------------------
# 6. PenalizedGLM_CV.score() delegation
# ---------------------------------------------------------------------------

class TestCVScoreDelegation:
    def test_cv_score_after_fit(self):
        """PenalizedGLM_CV.score() should delegate to refit estimator."""
        from statgpu.linear_model._penalized_cv import PenalizedGLM_CV
        rng = np.random.default_rng(42)
        X = rng.standard_normal((80, 5))
        y = X @ np.array([1, -1, 0.5, 0, 0]) + rng.standard_normal(80) * 0.1
        m = PenalizedGLM_CV(loss='squared_error', penalty='l2', cv=3, n_alphas=5)
        m.fit(X, y)
        r2 = m.score(X, y)
        assert np.isfinite(r2)
        assert r2 > 0.5

    def test_cv_score_with_sample_weight(self):
        """PenalizedGLM_CV.score(X, y, sample_weight=w) should work."""
        from statgpu.linear_model._penalized_cv import PenalizedGLM_CV
        rng = np.random.default_rng(42)
        X = rng.standard_normal((80, 5))
        y = X @ np.array([1, -1, 0.5, 0, 0]) + rng.standard_normal(80) * 0.1
        sw = rng.uniform(0.5, 2.0, 80)
        m = PenalizedGLM_CV(loss='squared_error', penalty='l2', cv=3, n_alphas=5)
        m.fit(X, y, sample_weight=sw)
        r2 = m.score(X, y, sample_weight=sw)
        assert np.isfinite(r2)


# ---------------------------------------------------------------------------
# 7. cv_splits parameter
# ---------------------------------------------------------------------------

class TestCvSplits:
    def test_custom_splits(self):
        """Custom cv_splits should be used instead of kfold_indices."""
        from statgpu.linear_model._penalized_cv import PenalizedGLM_CV
        rng = np.random.default_rng(42)
        X = rng.standard_normal((80, 5))
        y = X @ np.array([1, -1, 0.5, 0, 0]) + rng.standard_normal(80) * 0.1
        # Create 3 custom folds
        n = len(y)
        idx = np.arange(n)
        folds = [
            (np.concatenate([idx[:27], idx[53:]]), idx[27:53]),
            (np.concatenate([idx[:13], idx[40:]]), idx[13:40]),
            (np.concatenate([idx[:53], idx[67:]]), idx[53:67]),
        ]
        m = PenalizedGLM_CV(
            loss='squared_error', penalty='l2', cv=3,
            cv_splits=folds, n_alphas=5,
        )
        m.fit(X, y)
        assert hasattr(m, 'estimator_')

    def test_cv_splits_generator(self):
        """Generator-style cv_splits should be consumed into a list."""
        from statgpu.linear_model._penalized_cv import PenalizedGLM_CV
        rng = np.random.default_rng(42)
        X = rng.standard_normal((80, 5))
        y = X @ np.array([1, -1, 0.5, 0, 0]) + rng.standard_normal(80) * 0.1

        def gen_folds(n, cv=3):
            fold_size = n // cv
            idx = np.arange(n)
            for i in range(cv):
                val_start = i * fold_size
                val_end = (i + 1) * fold_size if i < cv - 1 else n
                val_idx = idx[val_start:val_end]
                train_idx = np.concatenate([idx[:val_start], idx[val_end:]])
                yield train_idx, val_idx

        m = PenalizedGLM_CV(
            loss='squared_error', penalty='l2', cv=3,
            cv_splits=gen_folds(len(y), 3), n_alphas=5,
        )
        m.fit(X, y)
        assert hasattr(m, 'estimator_')

    def test_cv_splits_none_uses_default(self):
        """cv_splits=None should use kfold_indices as before."""
        from statgpu.linear_model._penalized_cv import PenalizedGLM_CV
        rng = np.random.default_rng(42)
        X = rng.standard_normal((80, 5))
        y = X @ np.array([1, -1, 0.5, 0, 0]) + rng.standard_normal(80) * 0.1
        m = PenalizedGLM_CV(
            loss='squared_error', penalty='l2', cv=3,
            cv_splits=None, n_alphas=5,
        )
        m.fit(X, y)
        assert hasattr(m, 'estimator_')


# ---------------------------------------------------------------------------
# 8. _build_cv_cache helper
# ---------------------------------------------------------------------------

class TestBuildCvCache:
    def test_non_squared_error_returns_none(self):
        from statgpu.linear_model._penalized_cv import PenalizedGLM_CV
        m = PenalizedGLM_CV(loss='poisson', penalty='l1', n_alphas=5)
        cache, L = m._build_cv_cache("poisson", "cpu", np.zeros((5, 3)), np.zeros(5), None)
        assert cache is None
        assert L is None

    def test_cpu_returns_none(self):
        from statgpu.linear_model._penalized_cv import PenalizedGLM_CV
        m = PenalizedGLM_CV(loss='squared_error', penalty='l2', n_alphas=5)
        cache, L = m._build_cv_cache("squared_error", "cpu", np.zeros((5, 3)), np.zeros(5), None)
        assert cache is None
        assert L is None


# ---------------------------------------------------------------------------
# 9. Dispatch table — integration test
# ---------------------------------------------------------------------------

class TestDispatchTableIntegration:
    """Test that the dispatch table correctly routes different loss+penalty combos."""

    def test_l1_poisson_cv(self):
        """Poisson + L1 should work via dispatch table."""
        from statgpu.linear_model._penalized_cv import PenalizedGLM_CV
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 5))
        eta = X @ np.array([0.5, -0.3, 0.1, 0, 0])
        y = rng.poisson(np.exp(np.clip(eta, -5, 5)))
        m = PenalizedGLM_CV(
            loss='poisson', penalty='l1', cv=3, n_alphas=5,
            device='cpu',
        )
        m.fit(X, y)
        assert hasattr(m, 'estimator_')
        assert hasattr(m, 'alpha_')

    def test_elasticnet_gamma_cv(self):
        """Gamma + ElasticNet should work via dispatch table."""
        from statgpu.linear_model._penalized_cv import PenalizedGLM_CV
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 5))
        eta = X @ np.array([0.3, -0.2, 0.1, 0, 0])
        y = np.exp(eta) + 0.01
        m = PenalizedGLM_CV(
            loss='gamma', penalty='elasticnet', cv=3, n_alphas=5,
            device='cpu', l1_ratio=0.5,
        )
        m.fit(X, y)
        assert hasattr(m, 'estimator_')

    def test_scad_squared_error_cv(self):
        """Squared error + SCAD should work."""
        from statgpu.linear_model._penalized_cv import PenalizedGLM_CV
        rng = np.random.default_rng(42)
        X = rng.standard_normal((80, 5))
        y = X @ np.array([1, -1, 0.5, 0, 0]) + rng.standard_normal(80) * 0.1
        m = PenalizedGLM_CV(
            loss='squared_error', penalty='scad', cv=3, n_alphas=5,
            device='cpu',
        )
        m.fit(X, y)
        assert hasattr(m, 'estimator_')


# ---------------------------------------------------------------------------
# 10. _lasso_cv cache_key_eff fix
# ---------------------------------------------------------------------------

class TestLassoCvCacheKey:
    def test_lasso_cv_fit_works(self):
        """LassoCV.fit() should work correctly with cache_key_eff."""
        from statgpu.linear_model._lasso_cv import LassoCV
        rng = np.random.default_rng(42)
        X = rng.standard_normal((80, 5))
        y = X @ np.array([1, -1, 0.5, 0, 0]) + rng.standard_normal(80) * 0.1
        m = LassoCV(cv=3, n_alphas=5)
        m.fit(X, y)
        assert hasattr(m, 'alpha_')
        assert hasattr(m, 'coef_')
        assert np.isfinite(m.alpha_)
        assert m.alpha_ > 0


# ---------------------------------------------------------------------------
# 11. cv_splits length ≠ cv (Fix 1: all_scores shape bug)
# ---------------------------------------------------------------------------

class TestCvSplitsLengthMismatch:
    def test_more_folds_than_cv(self):
        """cv_splits with more folds than cv parameter should work."""
        from statgpu.linear_model._penalized_cv import PenalizedGLM_CV
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 5))
        y = X @ np.array([1, -1, 0.5, 0, 0]) + rng.standard_normal(100) * 0.1
        # 7 folds but cv=3
        n = len(y)
        idx = np.arange(n)
        fold_size = n // 7
        folds = []
        for i in range(7):
            val_s = i * fold_size
            val_e = (i + 1) * fold_size if i < 6 else n
            folds.append((np.concatenate([idx[:val_s], idx[val_e:]]), idx[val_s:val_e]))
        m = PenalizedGLM_CV(
            loss='squared_error', penalty='l2', cv=3,
            cv_splits=folds, n_alphas=5,
        )
        m.fit(X, y)
        assert hasattr(m, 'estimator_')
        assert m.cv_results_['all_scores'].shape[0] == 7

    def test_fewer_folds_than_cv(self):
        """cv_splits with fewer folds than cv parameter should work."""
        from statgpu.linear_model._penalized_cv import PenalizedGLM_CV
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 5))
        y = X @ np.array([1, -1, 0.5, 0, 0]) + rng.standard_normal(100) * 0.1
        # 2 folds but cv=5
        n = len(y)
        idx = np.arange(n)
        folds = [
            (idx[50:], idx[:50]),
            (idx[:50], idx[50:]),
        ]
        m = PenalizedGLM_CV(
            loss='squared_error', penalty='l2', cv=5,
            cv_splits=folds, n_alphas=5,
        )
        m.fit(X, y)
        assert hasattr(m, 'estimator_')
        assert m.cv_results_['all_scores'].shape[0] == 2


# ---------------------------------------------------------------------------
# 12. Non-uniform weight fallback warning (Fix 3)
# ---------------------------------------------------------------------------

class TestNonUniformWeightWarning:
    def test_logistic_path_warns(self):
        """Non-uniform weight with _logistic_sparse_cv_path should warn."""
        from statgpu.linear_model._penalized_cv import _logistic_sparse_cv_path
        rng = np.random.default_rng(42)
        X = rng.standard_normal((80, 5))
        y = (rng.standard_normal(80) > 0).astype(float)
        sw = rng.uniform(0.5, 2.0, 80)
        alphas = np.array([0.1, 0.01])
        with pytest.warns(RuntimeWarning, match="non-uniform sample_weight"):
            result = _logistic_sparse_cv_path(
                X, y, alphas, "l1", 0.5, 100, 1e-4, "cpu",
                sample_weight=sw,
            )
        assert result is None

    def test_squared_error_path_warns(self):
        """Non-uniform weight with _squared_error_sparse_cv_path should warn."""
        from statgpu.linear_model._penalized_cv import _squared_error_sparse_cv_path
        rng = np.random.default_rng(42)
        X = rng.standard_normal((80, 5))
        y = rng.standard_normal(80)
        sw = rng.uniform(0.5, 2.0, 80)
        alphas = np.array([0.1, 0.01])
        with pytest.warns(RuntimeWarning, match="non-uniform sample_weight"):
            result = _squared_error_sparse_cv_path(
                X, y, alphas, "l1", 0.5, 100, 1e-4, "cpu",
                sample_weight=sw,
            )
        assert result is None

    def test_glm_sparse_path_warns(self):
        """Non-uniform weight with _glm_sparse_cv_path should warn."""
        from statgpu.linear_model._penalized_cv import _glm_sparse_cv_path
        rng = np.random.default_rng(42)
        X = rng.standard_normal((80, 5))
        y = rng.poisson(2, 80).astype(float)
        sw = rng.uniform(0.5, 2.0, 80)
        alphas = np.array([0.1, 0.01])
        with pytest.warns(RuntimeWarning, match="non-uniform sample_weight"):
            result = _glm_sparse_cv_path(
                "poisson", X, y, alphas, "l1", 0.5, 100, 1e-4, "cpu",
                sample_weight=sw,
            )
        assert result is None


# ---------------------------------------------------------------------------
# 13. _LOSS_EVAL_DISPATCH consistency with _LOSS_VALLOSS_FNS (Fix 4)
# ---------------------------------------------------------------------------

class TestLossDispatchConsistency:
    def test_dispatch_uses_val_functions(self):
        """_LOSS_EVAL_DISPATCH should use the same _val_* functions as _LOSS_VALLOSS_FNS."""
        from statgpu.linear_model._penalized_cv import (
            _LOSS_EVAL_DISPATCH, _LOSS_VALLOSS_FNS,
        )
        # All GLM losses (not squared_error) should use the same function objects
        for loss_name in ("logistic", "poisson", "gamma", "inverse_gaussian",
                          "negative_binomial", "tweedie"):
            eval_fn, uses_design = _LOSS_EVAL_DISPATCH[loss_name]
            val_fn = _LOSS_VALLOSS_FNS[loss_name]
            assert eval_fn is val_fn, f"{loss_name}: _LOSS_EVAL_DISPATCH uses different fn than _LOSS_VALLOSS_FNS"
            assert not uses_design

    def test_squared_error_uses_ps(self):
        """squared_error should use _ps_squared_error (unique signature)."""
        from statgpu.linear_model._penalized_cv import _LOSS_EVAL_DISPATCH
        eval_fn, uses_design = _LOSS_EVAL_DISPATCH["squared_error"]
        assert uses_design
        assert eval_fn.__name__ == "_ps_squared_error"

    def test_eval_dispatch_matches_valloss_numerically(self):
        """Evaluate via _LOSS_EVAL_DISPATCH and _LOSS_VALLOSS_FNS should give same result."""
        from statgpu.linear_model._penalized_cv import (
            _LOSS_EVAL_DISPATCH, _LOSS_VALLOSS_FNS,
        )
        rng = np.random.default_rng(42)
        eta = rng.standard_normal(50)
        y = rng.standard_normal(50)
        for loss_name in ("logistic", "poisson", "gamma", "inverse_gaussian",
                          "negative_binomial", "tweedie"):
            eval_fn, _ = _LOSS_EVAL_DISPATCH[loss_name]
            val_fn = _LOSS_VALLOSS_FNS[loss_name]
            result_eval = eval_fn(eta, y)
            result_val = val_fn(eta, y)
            assert_allclose(result_eval, result_val, rtol=1e-12,
                            err_msg=f"{loss_name}: dispatch vs valloss mismatch")


# ---------------------------------------------------------------------------
# 14. _evaluate_loss_numpy fallback warning
# ---------------------------------------------------------------------------

class TestEvaluateLossFallback:
    def test_unknown_loss_warns(self):
        """Unknown loss name should warn about ignoring sample weights."""
        from statgpu.linear_model._penalized_cv import _evaluate_loss_numpy
        from statgpu.linear_model._penalized import _resolve_loss_name
        loss_fn = _resolve_loss_name("squared_error")
        rng = np.random.default_rng(42)
        X = rng.standard_normal((10, 3))
        y = rng.standard_normal(10)
        coef = np.zeros(3)
        with pytest.warns(RuntimeWarning, match="not in dispatch table"):
            _evaluate_loss_numpy("unknown_loss", loss_fn, X, y, coef, 0.0, True,
                                 sample_weight=np.ones(10))


# ---------------------------------------------------------------------------
# 12. _sigmoid (from _array_ops) clips extreme values
# ---------------------------------------------------------------------------

class TestStableSigmoid:
    def test_extreme_values(self):
        """Sigmoid should return 0/1 for extreme inputs without overflow."""
        from statgpu.backends._array_ops import _sigmoid
        x_np = np.array([-1000.0, -500.0, 0.0, 500.0, 1000.0])
        result = _sigmoid(x_np)
        assert_allclose(result, [0.0, 0.0, 0.5, 1.0, 1.0], atol=1e-6)


# ---------------------------------------------------------------------------
# 15. SCAD/MCP val_sample_weight support
# ---------------------------------------------------------------------------

class TestScadMcpValWeight:
    def test_scad_mcp_accepts_val_sample_weight(self):
        """_scad_mcp_cv_path should accept val_sample_weight parameter."""
        from statgpu.linear_model._penalized_cv import _scad_mcp_cv_path
        rng = np.random.default_rng(42)
        X = rng.standard_normal((80, 5))
        y = X @ np.array([1, -1, 0.5, 0, 0]) + rng.standard_normal(80) * 0.1
        X_val = rng.standard_normal((20, 5))
        y_val = X_val @ np.array([1, -1, 0.5, 0, 0]) + rng.standard_normal(20) * 0.1
        sw_val = rng.uniform(0.5, 2.0, 20)
        alphas = np.array([0.1, 0.05])

        # No sample_weight (training) — only val_sample_weight
        result = _scad_mcp_cv_path(
            "squared_error", X, y, alphas, "scad", 0.5, 100, 1e-4, "cpu",
            X_val=X_val, y_val=y_val, val_sample_weight=sw_val,
        )
        assert result is not None
        assert result["scores"] is not None
        assert len(result["scores"]) == 2
        assert all(np.isfinite(result["scores"]))

    def test_scad_mcp_weighted_vs_unweighted(self):
        """Weighted and unweighted SCAD/MCP validation should give different scores."""
        from statgpu.linear_model._penalized_cv import _scad_mcp_cv_path
        rng = np.random.default_rng(42)
        X = rng.standard_normal((80, 5))
        y = X @ np.array([1, -1, 0.5, 0, 0]) + rng.standard_normal(80) * 0.1
        X_val = rng.standard_normal((20, 5))
        y_val = X_val @ np.array([1, -1, 0.5, 0, 0]) + rng.standard_normal(20) * 0.1
        alphas = np.array([0.1])

        # Unweighted
        r1 = _scad_mcp_cv_path(
            "squared_error", X, y, alphas, "scad", 0.5, 100, 1e-4, "cpu",
            X_val=X_val, y_val=y_val,
        )
        # Heavily weighted (first half high weight, second half low)
        sw_val = np.concatenate([np.full(10, 10.0), np.full(10, 0.1)])
        r2 = _scad_mcp_cv_path(
            "squared_error", X, y, alphas, "scad", 0.5, 100, 1e-4, "cpu",
            X_val=X_val, y_val=y_val, val_sample_weight=sw_val,
        )
        # Scores should differ because weighting changes the mean
        assert not np.allclose(r1["scores"], r2["scores"], rtol=1e-3)

    def test_scad_mcp_non_uniform_sample_weight_warns(self):
        """Non-uniform sample_weight should warn and return None."""
        from statgpu.linear_model._penalized_cv import _scad_mcp_cv_path
        rng = np.random.default_rng(42)
        X = rng.standard_normal((80, 5))
        y = X @ np.array([1, -1, 0.5, 0, 0]) + rng.standard_normal(80) * 0.1
        sw = rng.uniform(0.5, 2.0, 80)
        alphas = np.array([0.1])
        with pytest.warns(RuntimeWarning, match="non-uniform sample_weight"):
            result = _scad_mcp_cv_path(
                "squared_error", X, y, alphas, "scad", 0.5, 100, 1e-4, "cpu",
                sample_weight=sw,
            )
        assert result is None

    def test_scad_mcp_uniform_sample_weight_accepted(self):
        """Uniform sample_weight should be accepted (no warning)."""
        from statgpu.linear_model._penalized_cv import _scad_mcp_cv_path
        rng = np.random.default_rng(42)
        X = rng.standard_normal((80, 5))
        y = X @ np.array([1, -1, 0.5, 0, 0]) + rng.standard_normal(80) * 0.1
        X_val = rng.standard_normal((20, 5))
        y_val = X_val @ np.array([1, -1, 0.5, 0, 0]) + rng.standard_normal(20) * 0.1
        sw = np.ones(80) * 2.0  # uniform
        alphas = np.array([0.1])
        # No warning expected for uniform weights
        result = _scad_mcp_cv_path(
            "squared_error", X, y, alphas, "scad", 0.5, 100, 1e-4, "cpu",
            X_val=X_val, y_val=y_val, sample_weight=sw,
        )
        assert result is not None
        assert result["scores"] is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
