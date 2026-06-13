"""
Tests for panel data models (PanelOLS, RandomEffects).

Tests compare against linearmodels.panel when available, and also
perform basic sanity checks that work standalone.

Note: Uses device='cpu' as default for reliable testing in CI.
      GPU testing happens on the remote benchmark server.
"""

import numpy as np
import pytest

from statgpu.panel import PanelOLS, RandomEffects
from statgpu.panel._utils import within_transform, demean_variables
from statgpu.panel._covariance import clustered_covariance


# Default device for tests (overridable via env var for GPU testing)
import os
_DEVICE = os.environ.get('STATGPU_TEST_DEVICE', 'cpu')


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_panel(n_entities=50, n_periods=10, k=3, seed=42):
    """Generate synthetic balanced panel data.

    Returns
    -------
    y, X, entity_ids, time_ids : ndarray
        Outcome, regressors, entity labels, time labels.
    beta_true : ndarray
        True slope coefficients.
    """
    rng = np.random.RandomState(seed)
    n = n_entities * n_periods
    entity_ids = np.repeat(np.arange(n_entities), n_periods)
    time_ids = np.tile(np.arange(n_periods), n_entities)

    X = rng.randn(n, k)
    beta_true = np.array([1.5, -0.7, 0.3])[:k]
    alpha_i = rng.randn(n_entities).repeat(n_periods)  # entity FE
    y = X @ beta_true + alpha_i + 0.5 * rng.randn(n)
    return y, X, entity_ids, time_ids, beta_true


def _make_unbalanced_panel(n_entities=30, min_T=5, max_T=12, k=2, seed=99):
    """Generate synthetic unbalanced panel data."""
    rng = np.random.RandomState(seed)
    beta_true = np.array([2.0, -1.0])[:k]

    y_list, X_list, ent_list, time_list = [], [], [], []
    for i in range(n_entities):
        Ti = rng.randint(min_T, max_T + 1)
        Xi = rng.randn(Ti, k)
        alpha_i = rng.randn()
        yi = Xi @ beta_true + alpha_i + 0.3 * rng.randn(Ti)
        y_list.append(yi)
        X_list.append(Xi)
        ent_list.append(np.full(Ti, i))
        time_list.append(np.arange(Ti))

    y = np.concatenate(y_list)
    X = np.vstack(X_list)
    entity_ids = np.concatenate(ent_list)
    time_ids = np.concatenate(time_list)
    return y, X, entity_ids, time_ids, beta_true


# ---------------------------------------------------------------------------
# within_transform tests
# ---------------------------------------------------------------------------

class TestWithinTransform:
    def test_basic_demeaning(self):
        """Group means should be zero after within transform."""
        y = np.array([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
        groups = np.array([0, 0, 0, 1, 1, 1])
        y_w = within_transform(y, groups)
        for g in np.unique(groups):
            assert np.abs(np.mean(y_w[groups == g])) < 1e-12

    def test_single_group(self):
        """Single group: result should be y - mean(y)."""
        y = np.array([1.0, 2.0, 3.0])
        groups = np.array([0, 0, 0])
        y_w = within_transform(y, groups)
        expected = y - np.mean(y)
        np.testing.assert_allclose(y_w, expected, atol=1e-12)

    def test_preserves_shape(self):
        y = np.random.randn(100)
        groups = np.repeat(np.arange(20), 5)
        y_w = within_transform(y, groups)
        assert y_w.shape == y.shape

    def test_multiple_groups(self):
        """Each group should be independently demeaned."""
        rng = np.random.RandomState(11)
        y = rng.randn(60)
        groups = np.repeat(np.arange(6), 10)
        y_w = within_transform(y, groups)
        for g in range(6):
            grp_mean = np.mean(y_w[groups == g])
            assert abs(grp_mean) < 1e-12


# ---------------------------------------------------------------------------
# PanelOLS tests
# ---------------------------------------------------------------------------

class TestPanelOLS:
    def test_entity_fe_coef(self):
        """Entity FE should recover true slopes."""
        y, X, eids, tids, beta = _make_panel()
        model = PanelOLS(entity_effects=True, cov_type='nonrobust',
                         device=_DEVICE)
        model.fit(X, y, entity_ids=eids)
        np.testing.assert_allclose(model.coef_, beta, atol=0.1)

    def test_two_way_fe_coef(self):
        """Two-way FE should recover true slopes."""
        y, X, eids, tids, beta = _make_panel()
        model = PanelOLS(
            entity_effects=True, time_effects=True, cov_type='nonrobust',
            device=_DEVICE,
        )
        model.fit(X, y, entity_ids=eids, time_ids=tids)
        np.testing.assert_allclose(model.coef_, beta, atol=0.15)

    def test_no_fe_ols(self):
        """With no FE, should produce standard OLS."""
        rng = np.random.RandomState(7)
        X = rng.randn(200, 2)
        beta = np.array([1.0, -1.0])
        y = X @ beta + 0.1 * rng.randn(200)

        model = PanelOLS(device=_DEVICE)
        model.fit(X, y)
        np.testing.assert_allclose(model.coef_, beta, atol=0.05)

    def test_robust_se(self):
        """HC1 SE should differ from nonrobust."""
        y, X, eids, _, _ = _make_panel()
        m1 = PanelOLS(entity_effects=True, cov_type='nonrobust',
                       device=_DEVICE)
        m1.fit(X, y, entity_ids=eids)

        m2 = PanelOLS(entity_effects=True, cov_type='robust',
                       device=_DEVICE)
        m2.fit(X, y, entity_ids=eids)

        # SEs should generally differ
        assert not np.allclose(m1.bse_, m2.bse_, rtol=1e-3)

    def test_clustered_se(self):
        """Clustered SE should differ from nonrobust."""
        y, X, eids, _, _ = _make_panel()
        m1 = PanelOLS(entity_effects=True, cov_type='nonrobust',
                       device=_DEVICE)
        m1.fit(X, y, entity_ids=eids)

        m2 = PanelOLS(entity_effects=True, cov_type='clustered',
                       device=_DEVICE)
        m2.fit(X, y, entity_ids=eids, cluster=eids)

        assert not np.allclose(m1.bse_, m2.bse_, rtol=1e-3)

    def test_predict(self):
        y, X, eids, _, beta = _make_panel()
        model = PanelOLS(entity_effects=True, device=_DEVICE)
        model.fit(X, y, entity_ids=eids)
        y_hat = model.predict(X)
        assert y_hat.shape == (len(y),)

    def test_conf_int_contains_true(self):
        """95% CI should contain true parameter for large samples."""
        y, X, eids, _, beta = _make_panel(n_entities=200, n_periods=20)
        model = PanelOLS(entity_effects=True, cov_type='nonrobust',
                         device=_DEVICE)
        model.fit(X, y, entity_ids=eids)
        for i in range(len(beta)):
            assert model.conf_int_[i, 0] <= beta[i] <= model.conf_int_[i, 1]

    def test_within_r_squared(self):
        """Within R-squared should be between 0 and 1."""
        y, X, eids, _, _ = _make_panel()
        model = PanelOLS(entity_effects=True, device=_DEVICE)
        model.fit(X, y, entity_ids=eids)
        assert 0.0 <= model.rsquared_within <= 1.0

    def test_unbalanced_panel(self):
        """Should work on unbalanced panels."""
        y, X, eids, _, beta = _make_unbalanced_panel()
        model = PanelOLS(entity_effects=True, device=_DEVICE)
        model.fit(X, y, entity_ids=eids)
        np.testing.assert_allclose(model.coef_, beta, atol=0.15)

    def test_pvalues_in_range(self):
        y, X, eids, _, _ = _make_panel()
        model = PanelOLS(entity_effects=True, device=_DEVICE)
        model.fit(X, y, entity_ids=eids)
        assert np.all(model.pvalues_ >= 0)
        assert np.all(model.pvalues_ <= 1)

    def test_tvalues_significant(self):
        """For well-powered data, true coefficients should be significant."""
        y, X, eids, _, beta = _make_panel(n_entities=100, n_periods=20)
        model = PanelOLS(entity_effects=True, device=_DEVICE)
        model.fit(X, y, entity_ids=eids)
        # |t| > 2 for true non-zero coefficients
        for i in range(len(beta)):
            if abs(beta[i]) > 0.1:
                assert abs(model.tvalues_[i]) > 1.5

    def test_summary_runs(self, capsys):
        """summary() should not raise."""
        y, X, eids, _, _ = _make_panel(n_entities=20, n_periods=5)
        model = PanelOLS(entity_effects=True, device=_DEVICE)
        model.fit(X, y, entity_ids=eids)
        model.summary()
        out = capsys.readouterr().out
        assert 'PanelOLS' in out

    def test_cluster_2d_two_way(self):
        """Passing 2-column cluster should trigger two-way clustering."""
        y, X, eids, tids, _ = _make_panel(n_entities=50, n_periods=10)
        model = PanelOLS(entity_effects=True, cov_type='clustered',
                         device=_DEVICE)
        cluster_2d = np.column_stack([eids, tids])
        model.fit(X, y, entity_ids=eids, cluster=cluster_2d)
        assert model.bse_ is not None
        assert len(model.bse_) == X.shape[1]

    def test_entity_fe_missing_ids_raises(self):
        """entity_effects=True without entity_ids should raise."""
        y = np.random.randn(50)
        X = np.random.randn(50, 2)
        model = PanelOLS(entity_effects=True, device=_DEVICE)
        with pytest.raises(ValueError, match="entity_ids is required"):
            model.fit(X, y)

    def test_clustered_missing_cluster_raises(self):
        """cov_type='clustered' without cluster should raise."""
        y = np.random.randn(50)
        X = np.random.randn(50, 2)
        eids = np.repeat(np.arange(10), 5)
        model = PanelOLS(entity_effects=True, cov_type='clustered',
                         device=_DEVICE)
        with pytest.raises(ValueError, match="cluster is required"):
            model.fit(X, y, entity_ids=eids)


# ---------------------------------------------------------------------------
# RandomEffects tests
# ---------------------------------------------------------------------------

class TestRandomEffects:
    def test_coef_recovery(self):
        """RE should recover true slopes."""
        y, X, eids, _, beta = _make_panel(n_entities=100, n_periods=20)
        model = RandomEffects(device=_DEVICE)
        model.fit(X, y, entity_ids=eids)
        np.testing.assert_allclose(model.coef_, beta, atol=0.15)

    def test_variance_components(self):
        """sigma2_e and sigma2_a should be non-negative."""
        y, X, eids, _, _ = _make_panel()
        model = RandomEffects(device=_DEVICE)
        model.fit(X, y, entity_ids=eids)
        assert model.variance_components_['sigma2_e'] >= 0
        assert model.variance_components_['sigma2_a'] >= 0

    def test_theta_in_range(self):
        """theta should be in [0, 1)."""
        y, X, eids, _, _ = _make_panel()
        model = RandomEffects(device=_DEVICE)
        model.fit(X, y, entity_ids=eids)
        assert 0.0 <= model.theta_ < 1.0

    def test_predict(self):
        y, X, eids, _, _ = _make_panel()
        model = RandomEffects(device=_DEVICE)
        model.fit(X, y, entity_ids=eids)
        y_hat = model.predict(X)
        assert y_hat.shape == (len(y),)

    def test_pvalues_in_range(self):
        y, X, eids, _, _ = _make_panel()
        model = RandomEffects(device=_DEVICE)
        model.fit(X, y, entity_ids=eids)
        assert np.all(model.pvalues_ >= 0)
        assert np.all(model.pvalues_ <= 1)

    def test_conf_int_contains_true(self):
        """95% CI should contain true parameter for large samples."""
        y, X, eids, _, beta = _make_panel(n_entities=200, n_periods=20)
        model = RandomEffects(device=_DEVICE)
        model.fit(X, y, entity_ids=eids)
        for i in range(len(beta)):
            assert model.conf_int_[i, 0] <= beta[i] <= model.conf_int_[i, 1]

    def test_summary_runs(self, capsys):
        y, X, eids, _, _ = _make_panel(n_entities=20, n_periods=5)
        model = RandomEffects(device=_DEVICE)
        model.fit(X, y, entity_ids=eids)
        model.summary()
        out = capsys.readouterr().out
        assert 'RandomEffects' in out

    def test_unbalanced_panel(self):
        """Should work on unbalanced panels."""
        y, X, eids, _, beta = _make_unbalanced_panel()
        model = RandomEffects(device=_DEVICE)
        model.fit(X, y, entity_ids=eids)
        np.testing.assert_allclose(model.coef_, beta, atol=0.3)

    def test_missing_entity_ids_raises(self):
        """entity_ids=None should raise."""
        y = np.random.randn(50)
        X = np.random.randn(50, 2)
        model = RandomEffects(device=_DEVICE)
        with pytest.raises(ValueError, match="entity_ids is required"):
            model.fit(X, y)


# ---------------------------------------------------------------------------
# Clustered covariance tests
# ---------------------------------------------------------------------------

class TestClusteredCovariance:
    def test_shape(self):
        n, k = 100, 3
        X = np.random.randn(n, k)
        resid = np.random.randn(n)
        clusters = np.repeat(np.arange(20), 5)
        V = clustered_covariance(X, resid, clusters)
        assert V.shape == (k, k)

    def test_symmetric(self):
        n, k = 100, 3
        X = np.random.randn(n, k)
        resid = np.random.randn(n)
        clusters = np.repeat(np.arange(20), 5)
        V = clustered_covariance(X, resid, clusters)
        np.testing.assert_allclose(V, V.T, atol=1e-12)

    def test_positive_diagonal(self):
        n, k = 100, 3
        X = np.random.randn(n, k)
        resid = np.random.randn(n)
        clusters = np.repeat(np.arange(20), 5)
        V = clustered_covariance(X, resid, clusters)
        assert np.all(np.diag(V) >= 0)


# ---------------------------------------------------------------------------
# linearmodels comparison (optional)
# ---------------------------------------------------------------------------

class TestLinearmodelsComparison:
    """Compare with linearmodels.panel if available."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_linearmodels(self):
        pytest.importorskip("linearmodels")

    def test_entity_fe_vs_linearmodels(self):
        """PanelOLS entity FE should match linearmodels PanelOLS."""
        from linearmodels.panel import PanelOLS as LMPanelOLS
        import pandas as pd

        y, X, eids, tids, beta = _make_panel(
            n_entities=100, n_periods=10, k=3, seed=123
        )

        # Create multi-index for linearmodels (entity/time must be numeric or date-like)
        mi = pd.MultiIndex.from_arrays([eids.astype(int), tids.astype(int)],
                                        names=['entity', 'time'])

        df_y = pd.DataFrame({'y': y}, index=mi)
        df_X = pd.DataFrame(X, columns=['x1', 'x2', 'x3'], index=mi)

        # linearmodels
        lm_model = LMPanelOLS(df_y, df_X, entity_effects=True)
        lm_result = lm_model.fit(cov_type='unadjusted')

        # statgpu
        sg_model = PanelOLS(entity_effects=True, cov_type='nonrobust',
                            device=_DEVICE)
        sg_model.fit(X, y, entity_ids=eids)

        # Coefficients should match closely
        np.testing.assert_allclose(
            sg_model.coef_, lm_result.params.values, atol=1e-4
        )

        # Standard errors should match closely (numerical precision varies)
        np.testing.assert_allclose(
            sg_model.bse_, lm_result.std_errors.values, rtol=1e-3, atol=1e-4
        )

    def test_random_effects_vs_linearmodels(self):
        """RandomEffects should match linearmodels RandomEffects."""
        from linearmodels.panel import RandomEffects as LMRandomEffects
        import pandas as pd

        y, X, eids, tids, beta = _make_panel(
            n_entities=100, n_periods=10, k=3, seed=456
        )

        mi = pd.MultiIndex.from_arrays([eids.astype(int), tids.astype(int)],
                                        names=['entity', 'time'])

        df_y = pd.DataFrame({'y': y}, index=mi)
        df_X = pd.DataFrame(X, columns=['x1', 'x2', 'x3'], index=mi)

        # linearmodels
        lm_model = LMRandomEffects(df_y, df_X)
        lm_result = lm_model.fit()

        # statgpu
        sg_model = RandomEffects(device=_DEVICE)
        sg_model.fit(X, y, entity_ids=eids)

        # Coefficients should match closely (numerical precision varies)
        np.testing.assert_allclose(
            sg_model.coef_, lm_result.params.values, rtol=1e-3, atol=1e-3
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
