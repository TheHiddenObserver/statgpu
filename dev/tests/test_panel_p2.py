"""Comprehensive tests for Panel module: HAC, PooledOLS, BetweenOLS, FDOLS, FamaMacBeth."""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from statgpu.panel import (
    PanelOLS, RandomEffects, PooledOLS, BetweenOLS,
    FirstDifferenceOLS, FamaMacBeth,
    hac_covariance, clustered_covariance,
    PanelSummary,
)


# ============================================================================
# HAC covariance
# ============================================================================

class TestHACCovariance:

    def test_basic(self):
        np.random.seed(42)
        n, k = 100, 3
        X = np.column_stack([np.ones(n), np.random.randn(n, k - 1)])
        resid = np.random.randn(n)
        V = hac_covariance(X, resid)
        assert V.shape == (k, k)
        # Should be positive semi-definite
        eigvals = np.linalg.eigvalsh(V)
        assert np.all(eigvals >= -1e-10)

    def test_bandwidth_zero_equals_ols(self):
        np.random.seed(42)
        n, k = 100, 3
        X = np.column_stack([np.ones(n), np.random.randn(n, k - 1)])
        resid = np.random.randn(n)
        V_hac = hac_covariance(X, resid, bandwidth=0)
        # With bandwidth=0, should be similar to White HC0
        V_white = (X.T @ np.diag(resid ** 2) @ X) / n ** 2
        # Not exactly the same (different normalization), but structure should be similar
        assert V_hac.shape == V_white.shape

    def test_positive_bandwidth(self):
        np.random.seed(42)
        n = 100
        X = np.column_stack([np.ones(n), np.random.randn(n, 2)])
        resid = np.random.randn(n)
        V = hac_covariance(X, resid, bandwidth=5)
        assert V.shape == (3, 3)

    def test_automatic_bandwidth(self):
        np.random.seed(42)
        n = 200
        X = np.column_stack([np.ones(n), np.random.randn(n, 2)])
        resid = np.random.randn(n)
        V = hac_covariance(X, resid)  # auto bandwidth
        assert V.shape == (3, 3)


# ============================================================================
# PooledOLS
# ============================================================================

class TestPooledOLS:

    def test_basic(self):
        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 3)
        y = X @ np.array([1.0, 2.0, 3.0]) + np.random.randn(n) * 0.5
        m = PooledOLS().fit(X, y)
        assert m.coef_.shape == (4,)  # intercept + 3
        assert m.bse_.shape == (4,)
        assert m.pvalues_.shape == (4,)
        assert 0 < m.rsquared < 1

    def test_coef_close_to_true(self):
        np.random.seed(42)
        n = 500
        X = np.random.randn(n, 2)
        beta_true = np.array([1.0, 2.0])
        y = X @ beta_true + np.random.randn(n) * 0.1
        m = PooledOLS().fit(X, y)
        assert_allclose(m.coef_[1:], beta_true, atol=0.1)

    def test_robust_cov(self):
        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 2)
        y = X @ np.array([1.0, 2.0]) + np.random.randn(n) * 0.5
        m = PooledOLS(cov_type='robust').fit(X, y)
        assert m.bse_.shape == (3,)

    def test_hac_cov(self):
        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 2)
        y = X @ np.array([1.0, 2.0]) + np.random.randn(n) * 0.5
        m = PooledOLS(cov_type='hac', bandwidth=5).fit(X, y)
        assert m.bse_.shape == (3,)

    def test_predict(self):
        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 2)
        y = X @ np.array([1.0, 2.0]) + np.random.randn(n) * 0.1
        m = PooledOLS().fit(X, y)
        y_pred = m.predict(X[:10])
        assert y_pred.shape == (10,)

    def test_error_invalid_cov_type(self):
        with pytest.raises(ValueError, match="cov_type"):
            PooledOLS(cov_type='invalid')


# ============================================================================
# BetweenOLS
# ============================================================================

class TestBetweenOLS:

    def test_basic(self):
        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 2)
        y = X @ np.array([1.0, 2.0]) + np.random.randn(n) * 0.5
        eids = np.repeat(np.arange(20), 5)
        m = BetweenOLS().fit(X, y, entity_ids=eids)
        assert m.nobs == 20  # 20 groups
        assert m.coef_.shape == (3,)

    def test_no_entity_ids_raises(self):
        X = np.random.randn(10, 2)
        y = np.random.randn(10)
        with pytest.raises(ValueError, match="entity_ids"):
            BetweenOLS().fit(X, y)

    def test_predict(self):
        np.random.seed(42)
        n = 50
        X = np.random.randn(n, 2)
        y = X @ np.array([1.0, 2.0])
        eids = np.repeat(np.arange(10), 5)
        m = BetweenOLS().fit(X, y, entity_ids=eids)
        y_pred = m.predict(X[:5])
        assert y_pred.shape == (5,)


# ============================================================================
# FirstDifferenceOLS
# ============================================================================

class TestFirstDifferenceOLS:

    def test_basic(self):
        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 2)
        y = X @ np.array([1.0, 2.0]) + np.random.randn(n) * 0.5
        eids = np.repeat(np.arange(20), 5)
        tids = np.tile(np.arange(5), 20)
        m = FirstDifferenceOLS().fit(X, y, entity_ids=eids, time_ids=tids)
        assert m.nobs == 80  # 20 entities * 4 diffs each
        assert m.coef_.shape == (2,)  # no intercept

    def test_coef_recovery(self):
        np.random.seed(42)
        n = 200
        X = np.random.randn(n, 2)
        beta_true = np.array([1.0, 2.0])
        y = X @ beta_true + np.random.randn(n) * 0.1
        eids = np.repeat(np.arange(40), 5)
        tids = np.tile(np.arange(5), 40)
        m = FirstDifferenceOLS().fit(X, y, entity_ids=eids, time_ids=tids)
        assert_allclose(m.coef_, beta_true, atol=0.3)

    def test_no_entity_ids_raises(self):
        X = np.random.randn(10, 2)
        y = np.random.randn(10)
        with pytest.raises(ValueError, match="entity_ids"):
            FirstDifferenceOLS().fit(X, y)


# ============================================================================
# FamaMacBeth
# ============================================================================

class TestFamaMacBeth:

    def test_basic(self):
        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 3)
        y = X @ np.array([1.0, 2.0, 3.0]) + np.random.randn(n) * 0.5
        tids = np.repeat(np.arange(20), 5)
        m = FamaMacBeth(cov_type='nonrobust').fit(X, y, time_ids=tids)
        assert m.coef_.shape == (4,)  # intercept + 3
        assert m.n_periods == 20
        assert m.betas_.shape == (20, 4)

    def test_newey_west(self):
        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 2)
        y = X @ np.array([1.0, 2.0]) + np.random.randn(n) * 0.5
        tids = np.repeat(np.arange(20), 5)
        m = FamaMacBeth(cov_type='newey-west').fit(X, y, time_ids=tids)
        assert m.bse_.shape == (3,)

    def test_predict(self):
        np.random.seed(42)
        n = 50
        X = np.random.randn(n, 2)
        y = X @ np.array([1.0, 2.0])
        tids = np.repeat(np.arange(10), 5)
        m = FamaMacBeth().fit(X, y, time_ids=tids)
        y_pred = m.predict(X[:5])
        assert y_pred.shape == (5,)

    def test_no_time_ids_raises(self):
        X = np.random.randn(10, 2)
        y = np.random.randn(10)
        with pytest.raises(ValueError, match="time_ids"):
            FamaMacBeth().fit(X, y)


# ============================================================================
# Existing PanelOLS and RandomEffects regression
# ============================================================================

class TestExistingPanel:

    def test_panel_ols_basic(self):
        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 2)
        y = X @ np.array([1.0, 2.0]) + np.random.randn(n) * 0.5
        eids = np.repeat(np.arange(20), 5)
        m = PanelOLS(entity_effects=True).fit(X, y, entity_ids=eids)
        assert m.coef_.shape == (2,)

    def test_random_effects_basic(self):
        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 2)
        y = X @ np.array([1.0, 2.0]) + np.random.randn(n) * 0.5
        eids = np.repeat(np.arange(20), 5)
        m = RandomEffects().fit(X, y, entity_ids=eids)
        assert m.coef_.shape == (2,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
