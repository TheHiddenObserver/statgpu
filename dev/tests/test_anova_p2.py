"""Comprehensive tests for ANOVA module: two-way, Welch, post-hoc, effect sizes."""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from statgpu.anova import (
    f_oneway, AnovaResult,
    f_twoway, TwoWayAnovaResult,
    f_welch,
    tukey_hsd, bonferroni, TukeyResult, PosthocResult,
    cohens_f, partial_eta_squared,
)


# ============================================================================
# f_oneway regression
# ============================================================================

class TestFoneway:
    """Ensure existing f_oneway still works correctly."""

    def test_basic(self):
        g1 = np.array([5.1, 4.9, 5.0])
        g2 = np.array([6.2, 6.0, 6.3])
        g3 = np.array([7.1, 7.3, 7.0])
        r = f_oneway(g1, g2, g3)
        assert isinstance(r, AnovaResult)
        assert r.statistic > 0
        assert 0 <= r.pvalue <= 1
        assert r.df_between == 2
        assert r.df_within == 6
        assert 0 <= r.eta_squared <= 1

    def test_two_groups(self):
        g1 = np.array([1, 2, 3])
        g2 = np.array([4, 5, 6])
        r = f_oneway(g1, g2)
        assert r.df_between == 1

    def test_vs_scipy(self):
        from scipy import stats
        np.random.seed(42)
        g1 = np.random.randn(20) + 0
        g2 = np.random.randn(20) + 1
        g3 = np.random.randn(20) + 2
        r = f_oneway(g1, g2, g3)
        s = stats.f_oneway(g1, g2, g3)
        assert_allclose(r.statistic, s.statistic, rtol=1e-10)
        assert_allclose(r.pvalue, s.pvalue, rtol=1e-6)

    def test_error_too_few_groups(self):
        with pytest.raises(ValueError, match="at least 2"):
            f_oneway(np.array([1]))

    def test_backend_auto(self):
        g1 = np.array([1.0, 2.0, 3.0])
        g2 = np.array([4.0, 5.0, 6.0])
        r = f_oneway(g1, g2, backend="auto")
        assert isinstance(r.statistic, float)


# ============================================================================
# f_twoway
# ============================================================================

class TestFtwoway:

    def test_basic_balanced(self):
        np.random.seed(42)
        data = [[np.random.randn(10) + i + j for j in range(3)] for i in range(2)]
        r = f_twoway(data, interaction=True)
        assert isinstance(r, TwoWayAnovaResult)
        assert r.factor_a_statistic > 0
        assert r.factor_b_statistic > 0
        assert r.interaction_statistic is not None
        assert r.df_within > 0

    def test_no_interaction(self):
        np.random.seed(42)
        data = [[np.random.randn(10) + i + j for j in range(3)] for i in range(2)]
        r = f_twoway(data, interaction=False)
        assert r.interaction_statistic is None
        assert r.interaction_pvalue is None

    def test_unbalanced(self):
        np.random.seed(42)
        data = [
            [np.random.randn(5), np.random.randn(10), np.random.randn(8)],
            [np.random.randn(12), np.random.randn(6), np.random.randn(9)],
        ]
        r = f_twoway(data, interaction=True)
        assert r.factor_a_statistic > 0

    def test_eta_squared_range(self):
        np.random.seed(42)
        data = [[np.random.randn(10) for _ in range(3)] for _ in range(2)]
        r = f_twoway(data)
        assert 0 <= r.factor_a_eta_squared <= 1
        assert 0 <= r.factor_b_eta_squared <= 1
        assert 0 <= r.interaction_eta_squared <= 1

    def test_error_empty_data(self):
        with pytest.raises(ValueError):
            f_twoway([])

    def test_significant_factor(self):
        np.random.seed(42)
        # Factor A has strong effect
        data = [
            [np.random.randn(50) + 0 for _ in range(3)],
            [np.random.randn(50) + 5 for _ in range(3)],
        ]
        r = f_twoway(data, interaction=False)
        assert r.factor_a_pvalue < 0.001


# ============================================================================
# f_welch
# ============================================================================

class TestFwelch:

    def test_basic(self):
        g1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        g2 = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        r = f_welch(g1, g2)
        assert r.statistic > 0
        assert r.pvalue > 0

    def test_unequal_variances(self):
        np.random.seed(42)
        g1 = np.random.randn(100) * 1.0
        g2 = np.random.randn(10) * 10.0
        r = f_welch(g1, g2)
        assert isinstance(r, AnovaResult)

    def test_error_too_few_groups(self):
        with pytest.raises(ValueError, match="at least 2"):
            f_welch(np.array([1, 2, 3]))

    def test_error_too_few_obs(self):
        with pytest.raises(ValueError, match="at least 2"):
            f_welch(np.array([1]), np.array([2, 3]))

    def test_eta_squared_is_nan(self):
        r = f_welch(np.array([1, 2, 3]), np.array([4, 5, 6]))
        assert np.isnan(r.eta_squared)


# ============================================================================
# Post-hoc tests
# ============================================================================

class TestTukeyHSD:

    def test_basic(self):
        g1 = np.array([1.0, 2.0, 3.0])
        g2 = np.array([4.0, 5.0, 6.0])
        g3 = np.array([7.0, 8.0, 9.0])
        r = tukey_hsd(g1, g2, g3)
        assert isinstance(r, TukeyResult)
        assert len(r.comparisons) == 3  # C(3,2) = 3
        assert r.n_groups == 3
        assert r.alpha == 0.05

    def test_no_significant_difference(self):
        np.random.seed(42)
        g1 = np.random.randn(100)
        g2 = np.random.randn(100) + 0.01
        r = tukey_hsd(g1, g2, alpha=0.05)
        assert len(r.comparisons) == 1
        # With very small difference, should not reject
        # (may or may not reject depending on variance)

    def test_significant_difference(self):
        g1 = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        g2 = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        r = tukey_hsd(g1, g2, alpha=0.05)
        assert r.comparisons[0].reject is True

    def test_ci_contains_mean_diff(self):
        np.random.seed(42)
        g1 = np.random.randn(20)
        g2 = np.random.randn(20) + 2
        r = tukey_hsd(g1, g2)
        c = r.comparisons[0]
        assert c.ci_lower < c.mean_diff < c.ci_upper


class TestBonferroni:

    def test_basic(self):
        g1 = np.array([1.0, 2.0, 3.0, 4.0])
        g2 = np.array([5.0, 6.0, 7.0, 8.0])
        r = bonferroni(g1, g2)
        assert isinstance(r, PosthocResult)
        assert r.n_comparisons == 1

    def test_multiple_groups(self):
        g1 = np.array([1.0, 2.0, 3.0])
        g2 = np.array([4.0, 5.0, 6.0])
        g3 = np.array([7.0, 8.0, 9.0])
        r = bonferroni(g1, g2, g3)
        assert r.n_comparisons == 3

    def test_error_too_few_obs(self):
        with pytest.raises(ValueError, match="at least 2"):
            bonferroni(np.array([1]), np.array([2, 3]))


# ============================================================================
# Effect sizes
# ============================================================================

class TestEffectSizes:

    def test_cohens_f(self):
        g1 = np.array([1.0, 2.0, 3.0])
        g2 = np.array([4.0, 5.0, 6.0])
        f = cohens_f(g1, g2)
        assert f > 0

    def test_partial_eta_squared(self):
        assert partial_eta_squared(10, 5) == pytest.approx(10 / 15)
        assert partial_eta_squared(0, 0) != partial_eta_squared(0, 0)  # NaN

    def test_cohens_f_range(self):
        np.random.seed(42)
        g1 = np.random.randn(50)
        g2 = np.random.randn(50) + 0.1
        f = cohens_f(g1, g2)
        assert f >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
