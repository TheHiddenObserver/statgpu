"""DBSCAN edge case tests: min_samples=1, high-dim path, Cython fallback.
CoxPH Efron tie correctness tests."""

import numpy as np
import pytest


class TestCoxEfronTies:
    """CoxPH Efron tied-event correctness tests."""

    def test_efron_tied_events_fit(self):
        """Efron should produce finite coefficients with many tied event times."""
        from statgpu.survival import CoxPH

        np.random.seed(42)
        n, p = 100, 3
        X = np.random.randn(n, p)

        # Create data with many tied event times
        time = np.random.choice([1, 2, 3, 4, 5], size=n).astype(np.float64)
        event = np.ones(n, dtype=np.float64)

        model = CoxPH(ties='efron', device='cpu', max_iter=50)
        model.fit(X, time, event)

        # Should produce finite coefficients
        assert np.all(np.isfinite(model.coef_)), "Coefficients should be finite"
        assert np.linalg.norm(model.coef_) > 0, "Coefficients should be non-zero"

    def test_efron_tied_reference_parity(self):
        """Efron with tied events should match statsmodels reference."""
        pytest.importorskip("statsmodels")
        from statgpu.survival import CoxPH
        from statsmodels.duration.hazard_regression import PHReg

        np.random.seed(42)
        n, p = 100, 3
        X = np.random.randn(n, p)

        # Many tied event times (d_g > 1 for several groups)
        time = np.random.choice([1, 2, 3, 4, 5], size=n).astype(np.float64)
        event = np.ones(n, dtype=np.float64)
        # Add some censoring
        event[np.random.choice(n, size=20, replace=False)] = 0

        # statgpu
        model_sg = CoxPH(ties='efron', device='cpu', max_iter=100)
        model_sg.fit(X, time, event)

        # statsmodels
        model_sm = PHReg(time, X, event, ties='efron')
        result_sm = model_sm.fit(maxiter=100, tol=1e-6)

        # Coefficients should be close
        np.testing.assert_allclose(
            model_sg.coef_, result_sm.params, rtol=5e-2, atol=0.1,
            err_msg=f"Efron tied-event coefficients differ from statsmodels: "
                    f"statgpu={model_sg.coef_}, statsmodels={result_sm.params}"
        )

    def test_efron_unique_times_matches_breslow(self):
        """When no ties, Efron and Breslow should give similar results."""
        from statgpu.survival import CoxPH

        np.random.seed(42)
        n, p = 50, 3
        X = np.random.randn(n, p)

        # All unique event times (no ties)
        time = np.arange(n, dtype=np.float64) + np.random.uniform(0, 0.1, n)
        event = np.ones(n, dtype=np.float64)

        # Breslow
        model_b = CoxPH(ties='breslow', device='cpu', max_iter=50)
        model_b.fit(X, time, event)

        # Efron
        model_e = CoxPH(ties='efron', device='cpu', max_iter=50)
        model_e.fit(X, time, event)

        # Coefficients should be very close when no ties
        np.testing.assert_allclose(model_e.coef_, model_b.coef_, rtol=1e-3,
                                    err_msg="Efron coef should match Breslow when no ties")


class TestDBSCANMinSamples1:
    """DBSCAN(min_samples=1) should return singleton clusters for isolated points."""

    def test_isolated_points_min_samples_1(self):
        """Each isolated point should be its own cluster when min_samples=1."""
        from statgpu.unsupervised._dbscan import DBSCAN

        # Points far apart (no neighbors within eps)
        X = np.array([[0, 0], [10, 10], [20, 20], [30, 30]], dtype=np.float64)
        db = DBSCAN(eps=0.1, min_samples=1)
        db.fit(X)

        # Each point should be its own cluster (not noise)
        assert len(set(db.labels_)) == 4, f"Expected 4 clusters, got {set(db.labels_)}"
        assert -1 not in db.labels_, "No points should be noise"
        assert len(db.core_sample_indices_) == 4, "All points should be core"

    def test_min_samples_1_with_clusters(self):
        """Points within eps should form one cluster when min_samples=1."""
        from statgpu.unsupervised._dbscan import DBSCAN

        X = np.array([[0, 0], [0.01, 0.01], [10, 10], [10.01, 10.01]], dtype=np.float64)
        db = DBSCAN(eps=0.5, min_samples=1)
        db.fit(X)

        # Two clusters: {0,1} and {2,3}
        assert len(set(db.labels_)) == 2
        assert db.labels_[0] == db.labels_[1]
        assert db.labels_[2] == db.labels_[3]
        assert db.labels_[0] != db.labels_[2]


class TestDBSCANHighDim:
    """DBSCAN high-dimensional path (n_features > 12) vs sklearn."""

    def test_high_dim_sklearn_parity(self):
        """High-dim DBSCAN should match sklearn for min_samples=2."""
        from statgpu.unsupervised._dbscan import DBSCAN
        from sklearn.cluster import DBSCAN as SklearnDBSCAN

        np.random.seed(42)
        X = np.random.randn(100, 20)  # 20 features > 12 threshold

        # statgpu
        db = DBSCAN(eps=0.5, min_samples=2)
        db.fit(X)
        labels_statgpu = db.labels_

        # sklearn
        db_sk = SklearnDBSCAN(eps=0.5, min_samples=2)
        db_sk.fit(X)
        labels_sklearn = db_sk.labels_

        # Compare number of clusters (ignoring noise label mapping)
        n_clusters_statgpu = len(set(labels_statgpu)) - (1 if -1 in labels_statgpu else 0)
        n_clusters_sklearn = len(set(labels_sklearn)) - (1 if -1 in labels_sklearn else 0)
        assert n_clusters_statgpu == n_clusters_sklearn, \
            f"Cluster count mismatch: {n_clusters_statgpu} vs {n_clusters_sklearn}"

    def test_high_dim_min_samples_3(self):
        """High-dim DBSCAN with min_samples=3 should match sklearn."""
        from statgpu.unsupervised._dbscan import DBSCAN
        from sklearn.cluster import DBSCAN as SklearnDBSCAN

        np.random.seed(42)
        X = np.random.randn(100, 15)

        db = DBSCAN(eps=0.5, min_samples=3)
        db.fit(X)

        db_sk = SklearnDBSCAN(eps=0.5, min_samples=3)
        db_sk.fit(X)

        n_clusters_statgpu = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
        n_clusters_sklearn = len(set(db_sk.labels_)) - (1 if -1 in db_sk.labels_ else 0)
        assert n_clusters_statgpu == n_clusters_sklearn


class TestDBSCANCythonFallback:
    """Test DBSCAN pure Python fallback when Cython is not available."""

    def test_fallback_labels_from_edges(self):
        """_labels_from_edges should produce correct labels."""
        from statgpu.unsupervised._dbscan import DBSCAN

        # Create simple test data
        X = np.array([[0, 0], [0.1, 0], [10, 10], [10.1, 10]], dtype=np.float64)
        db = DBSCAN(eps=0.5, min_samples=2)

        # Force use of pure Python fallback by calling _labels_from_edges directly
        counts = np.array([1, 1, 1, 1], dtype=np.int64)  # each point has 1 neighbor
        row_idx = np.array([0, 1, 2, 3, 1, 0, 3, 2], dtype=np.int64)
        col_idx = np.array([1, 0, 3, 2, 0, 1, 2, 3], dtype=np.int64)

        labels, core_indices = db._labels_from_edges(4, counts, row_idx, col_idx)

        # Points 0,1 should be in same cluster, points 2,3 in same cluster
        assert labels[0] == labels[1]
        assert labels[2] == labels[3]
        assert labels[0] != labels[2]


class TestQuantileSCADScaling:
    """Quantile + SCAD/MCP penalty scaling should be consistent across n."""

    def test_sparsity_stable_across_n(self):
        """Sparsity should not weaken as n increases."""
        from statgpu.linear_model.penalized._penalized_quantile import PenalizedQuantileRegression

        np.random.seed(42)
        p = 10
        beta_true = np.zeros(p)
        beta_true[0] = 3.0
        beta_true[3] = -2.5
        beta_true[6] = 1.5

        active_sets = []
        for n in [100, 500, 1000]:
            X = np.random.randn(n, p)
            y = X @ beta_true + np.random.randn(n) * 0.5

            model = PenalizedQuantileRegression(quantile=0.5, penalty='scad', alpha=0.1)
            model.fit(X, y)
            active = tuple(sorted(np.where(np.abs(model.coef_) > 0.05)[0]))
            active_sets.append(active)

        # All n should produce the same active set
        assert active_sets[0] == active_sets[1] == active_sets[2], \
            f"Active sets differ across n: {active_sets}"
        assert active_sets[0] == (0, 3, 6), f"Expected (0,3,6), got {active_sets[0]}"


class TestQuantileSCADParity:
    """Quantile + SCAD/MCP: proximal IRLS-CD vs FISTA-LLA objective parity."""

    def test_proximal_irls_vs_fista_lla(self):
        """Proximal IRLS-CD objective should be close to FISTA-LLA."""
        from statgpu.solvers._proximal_irls_quantile import proximal_irls_quantile_solver
        from statgpu.solvers._fista_lla import fista_lla_path
        from statgpu.losses._quantile import QuantileLoss
        from statgpu.penalties._scad import SCADPenalty

        np.random.seed(42)
        n, p = 200, 10
        X = np.random.randn(n, p).astype(np.float64)
        beta_true = np.zeros(p)
        beta_true[0] = 3.0
        beta_true[3] = -2.5
        beta_true[6] = 1.5
        y = (X @ beta_true + np.random.randn(n) * 0.5).astype(np.float64)

        cn = np.maximum(np.sqrt(np.sum(X**2, axis=0)), 1e-20)
        Xs = X * (np.sqrt(n) / cn)
        yc = y - np.mean(y)
        lam = float(np.max(np.abs(Xs.T @ yc / n)))

        for alpha in [0.05, 0.1]:
            loss = QuantileLoss(0.5)
            penalty = SCADPenalty(alpha=alpha)
            ap = np.geomspace(max(lam, alpha * 1.1), alpha, 3)

            # Proximal IRLS-CD
            c1, _, _ = proximal_irls_quantile_solver(
                loss, penalty, X, y, ap, max_lla_per_step=2, max_iter=200,
                tol=1e-6, fit_intercept=True)

            # FISTA-LLA
            c2, _, _ = fista_lla_path(
                loss, penalty, X, y, ap, max_lla_per_step=2, tol=1e-6,
                max_iter=[100, 200, 500], fit_intercept=True)

            # Active set should match
            a1 = set(np.where(np.abs(c1) > 0.05)[0])
            a2 = set(np.where(np.abs(c2) > 0.05)[0])
            assert a1 == a2, f"alpha={alpha}: active sets differ: {a1} vs {a2}"

            # Coefficients should be close
            np.testing.assert_allclose(
                c1, c2, rtol=0.15, atol=0.3,
                err_msg=f"alpha={alpha}: coef divergence beyond tolerance")

    def test_sample_weight_cpu(self):
        """Weighted quantile SCAD on CPU should differ from unweighted."""
        from statgpu.linear_model.penalized._penalized_quantile import PenalizedQuantileRegression

        np.random.seed(42)
        n, p = 200, 10
        X = np.random.randn(n, p)
        beta_true = np.zeros(p)
        beta_true[0] = 3.0
        beta_true[5] = -2.5
        y = X @ beta_true + np.random.randn(n) * 0.5

        # Unweighted
        m1 = PenalizedQuantileRegression(quantile=0.5, penalty='scad', alpha=0.1)
        m1.fit(X, y)

        # Weighted: upweight first half
        sw = np.ones(n)
        sw[:n//2] = 5.0
        m2 = PenalizedQuantileRegression(quantile=0.5, penalty='scad', alpha=0.1)
        m2.fit(X, y, sample_weight=sw)

        # Weighted fit should differ from unweighted
        diff = np.max(np.abs(m1.coef_ - m2.coef_))
        assert diff > 0.01, f"Weighted fit should differ from unweighted, got diff={diff}"


class TestScoreSampleWeight:
    """Score methods should handle sample_weight."""

    def test_quantile_score_weighted(self):
        """Quantile score with sample_weight should give weighted result."""
        from statgpu.linear_model.penalized._penalized_quantile import PenalizedQuantileRegression

        np.random.seed(42)
        n = 200
        X = np.random.randn(n, 5)
        y = X @ np.array([1.0, 0, -0.5, 0, 0.3]) + np.random.randn(n) * 0.5

        m = PenalizedQuantileRegression(quantile=0.5, penalty='scad', alpha=0.1)
        m.fit(X, y)

        s1 = m.score(X, y)
        sw = np.ones(n)
        sw[:50] = 10.0
        s2 = m.score(X, y, sample_weight=sw)

        # Weighted score should differ from unweighted
        assert abs(s1 - s2) > 1e-6, f"Weighted score should differ from unweighted, diff={abs(s1-s2):.8f}"


class TestCrossBackendParity:
    """Verify solver produces consistent results across backends."""

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    def test_quantile_scad_cross_backend(self, backend):
        """Quantile+SCAD should give same active set on torch vs numpy."""
        from statgpu.linear_model.penalized._penalized_quantile import PenalizedQuantileRegression

        np.random.seed(42)
        n, p = 100, 10
        X_np = np.random.randn(n, p)
        beta_true = np.zeros(p)
        beta_true[0] = 3.0; beta_true[5] = -2.5
        y_np = X_np @ beta_true + np.random.randn(n) * 0.5

        if backend == "torch":
            import torch
            X = torch.tensor(X_np, dtype=torch.float64)
            y = torch.tensor(y_np, dtype=torch.float64)
        else:
            X, y = X_np, y_np

        m = PenalizedQuantileRegression(quantile=0.5, penalty='scad', alpha=0.1)
        m.fit(X, y)
        active = sorted(np.where(np.abs(m.coef_) > 0.05)[0])
        assert active == [0, 5], f"backend={backend}: active={active}"

        m_np = PenalizedQuantileRegression(quantile=0.5, penalty='scad', alpha=0.1)
        m_np.fit(X_np, y_np)
        if backend == "torch":
            d = np.max(np.abs(m.coef_ - m_np.coef_))
            assert d < 0.15, f"torch vs numpy coef diff too large: {d}"


class TestCupySmoke:
    """Basic CuPy smoke tests (skip if cupy not available)."""

    @staticmethod
    def _cupy_available():
        try:
            import cupy; return True
        except ImportError:
            return False

    @pytest.mark.skipif(not _cupy_available(), reason="CuPy not installed")
    def test_quantile_scad_cupy(self):
        """Quantile+SCAD should work on CuPy."""
        from statgpu.linear_model.penalized._penalized_quantile import PenalizedQuantileRegression
        import cupy as cp

        np.random.seed(42)
        n, p = 100, 10
        X_np = np.random.randn(n, p).astype(np.float64)
        beta_true = np.zeros(p)
        beta_true[0] = 3.0; beta_true[5] = -2.5
        y_np = (X_np @ beta_true + np.random.randn(n) * 0.5).astype(np.float64)

        X = cp.asarray(X_np); y = cp.asarray(y_np)
        m = PenalizedQuantileRegression(quantile=0.5, penalty='scad', alpha=0.1)
        m.fit(X, y)
        active = sorted(np.where(np.abs(m.coef_) > 0.05)[0])
        assert active == [0, 5], f"CuPy active={active}"

    @pytest.mark.skipif(not _cupy_available(), reason="CuPy not installed")
    def test_huber_scad_cupy(self):
        """Huber+SCAD should work on CuPy."""
        from statgpu.linear_model.penalized._penalized_robust import PenalizedRobustRegression
        import cupy as cp

        np.random.seed(42)
        n, p = 100, 10
        X_np = np.random.randn(n, p).astype(np.float64)
        beta_true = np.zeros(p)
        beta_true[0] = 3.0; beta_true[5] = -2.5
        y_np = (X_np @ beta_true + np.random.randn(n) * 0.5).astype(np.float64)

        X = cp.asarray(X_np); y = cp.asarray(y_np)
        m = PenalizedRobustRegression(loss='huber', penalty='scad', alpha=0.1)
        m.fit(X, y)
        active = sorted(np.where(np.abs(m.coef_) > 0.05)[0])
        assert active == [0, 5], f"CuPy Huber active={active}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
