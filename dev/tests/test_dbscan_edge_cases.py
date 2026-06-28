"""DBSCAN edge case tests: min_samples=1, high-dim path, Cython fallback."""

import numpy as np
import pytest


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
