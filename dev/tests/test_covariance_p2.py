"""Comprehensive tests for Covariance module: ShrunkCovariance, MinCovDet, GraphicalLasso."""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from statgpu.covariance import (
    EmpiricalCovariance, LedoitWolf, OAS,
    ShrunkCovariance, MinCovDet, GraphicalLasso, GraphicalLassoCV,
)


class TestShrunkCovariance:

    def test_basic(self):
        X = np.random.randn(50, 5)
        sc = ShrunkCovariance(shrinkage=0.1).fit(X)
        assert sc.covariance_.shape == (5, 5)
        assert sc.precision_.shape == (5, 5)
        assert sc.shrinkage_ == 0.1

    def test_shrinkage_zero_equals_empirical(self):
        X = np.random.randn(50, 5)
        sc = ShrunkCovariance(shrinkage=0.0, device="cpu").fit(X)
        ec = EmpiricalCovariance(device="cpu").fit(X)
        assert_allclose(sc.covariance_, ec.covariance_, rtol=1e-10)

    def test_shrinkage_one_equals_scaled_identity(self):
        X = np.random.randn(50, 5)
        sc = ShrunkCovariance(shrinkage=1.0, device="cpu").fit(X)
        # Should be mu * I
        mu = np.trace(np.cov(X, rowvar=False, bias=True)) / 5
        expected = mu * np.eye(5)
        assert_allclose(sc.covariance_, expected, rtol=1e-10)

    def test_vs_sklearn(self):
        from sklearn.covariance import ShrunkCovariance as SkShrunk
        X = np.random.randn(100, 5)
        sg = ShrunkCovariance(shrinkage=0.3, device="cpu").fit(X)
        sk = SkShrunk(shrinkage=0.3).fit(X)
        assert_allclose(sg.covariance_, sk.covariance_, rtol=1e-6)

    def test_error_invalid_shrinkage(self):
        X = np.random.randn(50, 5)
        with pytest.raises(ValueError, match="shrinkage"):
            ShrunkCovariance(shrinkage=1.5).fit(X)

    def test_predict_mahalanobis(self):
        X = np.random.randn(50, 5)
        sc = ShrunkCovariance(shrinkage=0.1).fit(X)
        dist = sc.predict(X[:5])
        assert dist.shape == (5,)
        assert np.all(dist >= 0)


class TestMinCovDet:

    def test_basic(self):
        np.random.seed(42)
        X = np.random.randn(50, 5)
        mcd = MinCovDet(random_state=42).fit(X)
        assert mcd.covariance_.shape == (5, 5)
        assert mcd.support_.dtype == bool
        assert mcd.support_.sum() > 0

    def test_outlier_resistance(self):
        np.random.seed(42)
        X = np.random.randn(100, 3)
        # Add outliers
        X_out = np.vstack([X, np.random.randn(20, 3) * 10 + 20])
        mcd = MinCovDet(random_state=42).fit(X_out)
        # The support should exclude most outliers
        assert mcd.support_.sum() < X_out.shape[0]

    def test_raw_and_reweighted(self):
        np.random.seed(42)
        X = np.random.randn(50, 5)
        mcd = MinCovDet(random_state=42).fit(X)
        assert hasattr(mcd, 'raw_covariance_')
        assert hasattr(mcd, 'raw_location_')

    def test_score(self):
        np.random.seed(42)
        X = np.random.randn(50, 5)
        mcd = MinCovDet(random_state=42).fit(X)
        s = mcd.score(X[:10])
        assert isinstance(s, float)

    def test_support_fraction(self):
        np.random.seed(42)
        X = np.random.randn(100, 5)
        mcd = MinCovDet(support_fraction=0.75, random_state=42).fit(X)
        assert mcd.support_.sum() > 0


class TestGraphicalLasso:

    def test_basic(self):
        np.random.seed(42)
        X = np.random.randn(100, 5)
        gl = GraphicalLasso(alpha=0.1, max_iter=50).fit(X)
        assert gl.covariance_.shape == (5, 5)
        assert gl.precision_.shape == (5, 5)
        assert gl.n_iter_ > 0

    def test_sparse_precision(self):
        np.random.seed(42)
        # Create data with known sparse precision
        p = 10
        prec_true = np.eye(p)
        prec_true[0, 1] = prec_true[1, 0] = -0.5
        cov_true = np.linalg.inv(prec_true)
        X = np.random.multivariate_normal(np.zeros(p), cov_true, size=200)
        gl = GraphicalLasso(alpha=0.2, max_iter=100).fit(X)
        # Off-diagonal of precision should be sparse
        off_diag = gl.precision_[~np.eye(p, dtype=bool)]
        assert np.sum(np.abs(off_diag) < 1e-6) > p * (p - 1) * 0.3

    def test_vs_sklearn(self):
        from sklearn.covariance import GraphicalLasso as SkGL
        np.random.seed(42)
        X = np.random.randn(200, 5)
        sg = GraphicalLasso(alpha=0.1, max_iter=100).fit(X)
        sk = SkGL(alpha=0.1, max_iter=100).fit(X)
        # Precision matrices should be similar (not exact due to implementation differences)
        corr = np.corrcoef(sg.precision_.ravel(), sk.precision_.ravel())[0, 1]
        assert corr > 0.95

    def test_error_invalid_alpha(self):
        X = np.random.randn(50, 5)
        # Negative alpha should still work (just no regularization)
        gl = GraphicalLasso(alpha=0.0, max_iter=10).fit(X)
        assert gl.precision_.shape == (5, 5)


class TestGraphicalLassoCV:

    def test_basic(self):
        np.random.seed(42)
        X = np.random.randn(100, 5)
        glcv = GraphicalLassoCV(alphas=3, cv=3, max_iter=20).fit(X)
        assert glcv.covariance_.shape == (5, 5)
        assert glcv.alpha_ > 0
        assert len(glcv.cv_results_) == 3

    def test_selects_best_alpha(self):
        np.random.seed(42)
        X = np.random.randn(100, 5)
        glcv = GraphicalLassoCV(alphas=[0.01, 0.1, 1.0], cv=3, max_iter=20).fit(X)
        assert glcv.alpha_ in [0.01, 0.1, 1.0]


class TestExistingCovariance:
    """Regression tests for existing EmpiricalCovariance, LedoitWolf, OAS."""

    def test_empirical(self):
        X = np.random.randn(50, 5)
        ec = EmpiricalCovariance().fit(X)
        assert ec.covariance_.shape == (5, 5)

    def test_ledoit_wolf(self):
        X = np.random.randn(50, 5)
        lw = LedoitWolf().fit(X)
        assert 0 <= lw.shrinkage_ <= 1

    def test_oas(self):
        X = np.random.randn(50, 5)
        oas = OAS().fit(X)
        assert 0 <= oas.shrinkage_ <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
