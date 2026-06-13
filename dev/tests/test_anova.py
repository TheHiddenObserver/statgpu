"""
Tests for one-way ANOVA (statgpu.anova.f_oneway).

Validates against scipy.stats.f_oneway for numerical agreement.
"""

import numpy as np
import pytest
import scipy.stats as sps

from statgpu.anova import f_oneway, AnovaResult


# ---------------------------------------------------------------------------
# Basic tests
# ---------------------------------------------------------------------------

class TestFonewayBasic:
    """Core ANOVA tests against scipy reference."""

    def test_three_groups(self):
        """Three groups with distinct means."""
        rng = np.random.RandomState(42)
        g1 = rng.normal(loc=5.0, scale=1.0, size=30)
        g2 = rng.normal(loc=7.0, scale=1.0, size=30)
        g3 = rng.normal(loc=9.0, scale=1.0, size=30)

        sp_result = sps.f_oneway(g1, g2, g3)
        sg_result = f_oneway(g1, g2, g3)

        assert isinstance(sg_result, AnovaResult)
        assert abs(sg_result.statistic - sp_result.statistic) < 1e-10
        assert abs(sg_result.pvalue - sp_result.pvalue) < 1e-10

    def test_two_groups_equals_t_squared(self):
        """Two-group ANOVA should equal the square of the independent t-test."""
        rng = np.random.RandomState(0)
        g1 = rng.normal(loc=0.0, scale=1.0, size=25)
        g2 = rng.normal(loc=1.5, scale=1.0, size=25)

        sp_result = sps.f_oneway(g1, g2)
        sg_result = f_oneway(g1, g2)

        assert abs(sg_result.statistic - sp_result.statistic) < 1e-10
        assert abs(sg_result.pvalue - sp_result.pvalue) < 1e-10

        # Also check relation to t-test
        t_stat, t_pval = sps.ttest_ind(g1, g2)
        assert abs(sg_result.statistic - t_stat ** 2) < 1e-10
        assert abs(sg_result.pvalue - t_pval) < 1e-10

    def test_precision_vs_scipy(self):
        """Numerical precision: abs diff < 1e-10 for F and p."""
        rng = np.random.RandomState(123)
        g1 = rng.normal(loc=0, scale=1, size=50)
        g2 = rng.normal(loc=2, scale=1, size=40)
        g3 = rng.normal(loc=4, scale=1.5, size=60)

        sp_result = sps.f_oneway(g1, g2, g3)
        sg_result = f_oneway(g1, g2, g3)

        assert abs(sg_result.statistic - sp_result.statistic) < 1e-10
        assert abs(sg_result.pvalue - sp_result.pvalue) < 1e-10


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestFonewayEdgeCases:
    """Edge cases: identical means, very different means, df values."""

    def test_all_groups_same_mean(self):
        """All groups identical distribution -> F ~ 0, p ~ 1."""
        rng = np.random.RandomState(7)
        g1 = rng.normal(loc=5.0, scale=1.0, size=20)
        g2 = rng.normal(loc=5.0, scale=1.0, size=20)
        g3 = rng.normal(loc=5.0, scale=1.0, size=20)

        sg_result = f_oneway(g1, g2, g3)
        sp_result = sps.f_oneway(g1, g2, g3)

        # F should be small and p should be large
        assert sg_result.statistic < 5.0
        assert sg_result.pvalue > 0.05
        assert abs(sg_result.statistic - sp_result.statistic) < 1e-10
        assert abs(sg_result.pvalue - sp_result.pvalue) < 1e-10

    def test_very_different_means(self):
        """Very different means -> F large, p very small."""
        rng = np.random.RandomState(99)
        g1 = rng.normal(loc=0.0, scale=0.1, size=20)
        g2 = rng.normal(loc=100.0, scale=0.1, size=20)

        sg_result = f_oneway(g1, g2)
        sp_result = sps.f_oneway(g1, g2)

        assert sg_result.statistic > 1000
        assert sg_result.pvalue < 1e-15
        # For very large F stats, use relative tolerance
        rel_err = abs(sg_result.statistic - sp_result.statistic) / sp_result.statistic
        assert rel_err < 1e-10
        assert abs(sg_result.pvalue - sp_result.pvalue) < 1e-10

    def test_degrees_of_freedom(self):
        """Check df_between and df_within are correct."""
        g1 = np.array([1.0, 2.0, 3.0])
        g2 = np.array([4.0, 5.0, 6.0, 7.0])
        g3 = np.array([8.0, 9.0])

        result = f_oneway(g1, g2, g3)

        assert result.df_between == 2.0  # k - 1 = 3 - 1
        assert result.df_within == 6.0   # N - k = 9 - 3

    def test_eta_squared(self):
        """eta_squared should be between 0 and 1."""
        rng = np.random.RandomState(55)
        g1 = rng.normal(loc=0, scale=1, size=30)
        g2 = rng.normal(loc=3, scale=1, size=30)
        g3 = rng.normal(loc=6, scale=1, size=30)

        result = f_oneway(g1, g2, g3)
        assert 0.0 <= result.eta_squared <= 1.0
        # With very different means, eta_squared should be large
        assert result.eta_squared > 0.5

    def test_eta_squared_identical_groups(self):
        """eta_squared should be small when groups are similar."""
        rng = np.random.RandomState(11)
        g1 = rng.normal(loc=5.0, scale=1.0, size=50)
        g2 = rng.normal(loc=5.0, scale=1.0, size=50)

        result = f_oneway(g1, g2)
        assert result.eta_squared < 0.1


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestFonewayValidation:
    """Test input validation and error handling."""

    def test_too_few_groups(self):
        """Must supply at least 2 groups."""
        with pytest.raises(ValueError, match="at least 2 groups"):
            f_oneway(np.array([1.0, 2.0]))

    def test_empty_group(self):
        """Each group must have at least 1 observation."""
        with pytest.raises(ValueError, match="at least 1 observation"):
            f_oneway(np.array([1.0, 2.0]), np.array([]))

    def test_insufficient_total_observations(self):
        """Total N must exceed k."""
        # k=2 groups, each with 1 obs => N=2, N <= k
        with pytest.raises(ValueError, match="total observations"):
            f_oneway(np.array([1.0]), np.array([2.0]))


# ---------------------------------------------------------------------------
# Backend tests
# ---------------------------------------------------------------------------

class TestFonewayBackends:
    """Test different backend specifications."""

    def test_numpy_backend_explicit(self):
        """Explicit numpy backend."""
        rng = np.random.RandomState(42)
        g1 = rng.normal(loc=5.0, scale=1.0, size=20)
        g2 = rng.normal(loc=7.0, scale=1.0, size=20)

        result = f_oneway(g1, g2, backend="numpy")
        sp_result = sps.f_oneway(g1, g2)

        assert abs(result.statistic - sp_result.statistic) < 1e-10
        assert abs(result.pvalue - sp_result.pvalue) < 1e-10

    def test_auto_backend_with_numpy(self):
        """Auto backend with numpy arrays should work like numpy."""
        rng = np.random.RandomState(42)
        g1 = rng.normal(loc=5.0, scale=1.0, size=20)
        g2 = rng.normal(loc=7.0, scale=1.0, size=20)

        result = f_oneway(g1, g2, backend="auto")
        sp_result = sps.f_oneway(g1, g2)

        assert abs(result.statistic - sp_result.statistic) < 1e-10


# ---------------------------------------------------------------------------
# Various group counts
# ---------------------------------------------------------------------------

class TestFonewayGroupCounts:
    """Test with varying numbers of groups."""

    def test_many_groups(self):
        """ANOVA with 5 groups."""
        rng = np.random.RandomState(42)
        groups = [rng.normal(loc=i * 2, scale=1.0, size=30) for i in range(5)]

        sp_result = sps.f_oneway(*groups)
        sg_result = f_oneway(*groups)

        assert abs(sg_result.statistic - sp_result.statistic) < 1e-10
        assert abs(sg_result.pvalue - sp_result.pvalue) < 1e-10
        assert sg_result.df_between == 4.0
        assert sg_result.df_within == 145.0

    def test_unequal_group_sizes(self):
        """Groups with very different sizes."""
        rng = np.random.RandomState(42)
        g1 = rng.normal(loc=0, scale=1, size=10)
        g2 = rng.normal(loc=1, scale=1, size=100)
        g3 = rng.normal(loc=2, scale=1, size=500)

        sp_result = sps.f_oneway(g1, g2, g3)
        sg_result = f_oneway(g1, g2, g3)

        assert abs(sg_result.statistic - sp_result.statistic) < 1e-10
        assert abs(sg_result.pvalue - sp_result.pvalue) < 1e-10
