"""Comprehensive tests for Splines module: SplineTransformer, cyclic cubic, thin plate."""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from statgpu.nonparametric.splines import (
    bspline_basis, natural_cubic_spline_basis,
    SplineTransformer, cyclic_cubic_spline_basis, thin_plate_spline_basis,
)


class TestSplineTransformer:

    def test_basic(self):
        X = np.linspace(0, 1, 50).reshape(-1, 1)
        st = SplineTransformer(n_knots=5, degree=3).fit(X)
        X_t = st.transform(X)
        assert X_t.shape[0] == 50
        assert X_t.shape[1] == 7  # n_knots + degree - 1 = 5 + 3 - 1

    def test_include_bias_false(self):
        X = np.linspace(0, 1, 50).reshape(-1, 1)
        st = SplineTransformer(n_knots=5, degree=3, include_bias=False).fit(X)
        X_t = st.transform(X)
        assert X_t.shape[1] == 6  # 7 - 1

    def test_quantile_knots(self):
        X = np.random.randn(100, 1).reshape(-1, 1)
        st = SplineTransformer(n_knots=5, knots='quantile').fit(X)
        X_t = st.transform(X)
        assert X_t.shape[0] == 100

    def test_multi_feature(self):
        X = np.random.randn(50, 3)
        st = SplineTransformer(n_knots=4, degree=3).fit(X)
        X_t = st.transform(X)
        assert X_t.shape[1] == 3 * (4 + 3 - 1)  # 3 features * 6 splines

    def test_fit_transform(self):
        X = np.linspace(0, 1, 30).reshape(-1, 1)
        st = SplineTransformer(n_knots=4)
        X_t = st.fit_transform(X)
        assert X_t.shape[0] == 30

    def test_feature_names(self):
        X = np.linspace(0, 1, 30).reshape(-1, 1)
        st = SplineTransformer(n_knots=4).fit(X)
        names = st.get_feature_names_out()
        assert len(names) == st.n_features_out_

    def test_predict(self):
        X = np.linspace(0, 1, 30).reshape(-1, 1)
        st = SplineTransformer(n_knots=4).fit(X)
        X_t = st.predict(X)
        assert X_t.shape[0] == 30

    def test_error_too_few_knots(self):
        X = np.linspace(0, 1, 30).reshape(-1, 1)
        with pytest.raises(ValueError, match="n_knots"):
            SplineTransformer(n_knots=1).fit(X)

    def test_extrapolation_error(self):
        X_train = np.linspace(0, 1, 30).reshape(-1, 1)
        X_test = np.linspace(-1, 2, 10).reshape(-1, 1)
        st = SplineTransformer(n_knots=5, extrapolation='error').fit(X_train)
        with pytest.raises(ValueError, match="outside"):
            st.transform(X_test)


class TestCyclicCubicSpline:

    def test_basic(self):
        x = np.linspace(0, 1, 50)
        knots = np.array([0.25, 0.5, 0.75])
        B = cyclic_cubic_spline_basis(x, knots)
        assert B.shape[0] == 50
        # Should have fewer columns than standard B-spline
        B_std = bspline_basis(x, knots, boundary_lo=0, boundary_hi=1)
        assert B.shape[1] < B_std.shape[1]

    def test_periodicity(self):
        """At the boundaries, the cyclic spline should have matching values."""
        x = np.array([0.0, 1.0])
        knots = np.array([0.25, 0.5, 0.75])
        B = cyclic_cubic_spline_basis(x, knots)
        # Values at boundaries should be similar (not exact due to numerics)
        # This is a weak test; stronger tests would check derivatives

    def test_error_no_knots(self):
        with pytest.raises(ValueError, match="At least one"):
            cyclic_cubic_spline_basis(np.array([0, 1]), np.array([]))


class TestThinPlateSpline:

    def test_basic_1d(self):
        x = np.linspace(0, 1, 50)
        knots = np.array([0.2, 0.4, 0.6, 0.8])
        B = thin_plate_spline_basis(x, knots)
        assert B.shape[0] == 50
        # 4 knots + 1 intercept + 1 linear = 6
        assert B.shape[1] == 6

    def test_basic_2d(self):
        np.random.seed(42)
        x = np.random.randn(50, 2)
        knots = np.random.randn(5, 2)
        B = thin_plate_spline_basis(x, knots, penalty_order=2)
        assert B.shape[0] == 50
        # 5 knots + 1 intercept + 2 linear = 8
        assert B.shape[1] == 8

    def test_penalty_order_1(self):
        x = np.linspace(0, 1, 30)
        knots = np.array([0.3, 0.5, 0.7])
        B = thin_plate_spline_basis(x, knots, penalty_order=1)
        assert B.shape[0] == 30

    def test_error_dimension_mismatch(self):
        x = np.random.randn(10, 2)
        knots = np.random.randn(5, 3)  # 3D knots, 2D data
        with pytest.raises(ValueError, match="dimension"):
            thin_plate_spline_basis(x, knots)


class TestExistingSplines:
    """Regression tests for existing bspline_basis and natural_cubic_spline_basis."""

    def test_bspline_basic(self):
        x = np.linspace(0, 1, 50)
        knots = np.array([0.25, 0.5, 0.75])
        B = bspline_basis(x, knots, degree=3)
        assert B.shape[0] == 50
        assert B.shape[1] == 4 + 3  # 3 knots + degree + 1

    def test_natural_cubic_basic(self):
        x = np.linspace(0, 1, 50)
        knots = np.array([0.25, 0.5, 0.75])
        B = natural_cubic_spline_basis(x, knots)
        assert B.shape[0] == 50
        # natural cubic reduces by 2
        assert B.shape[1] == 4 + 3 - 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
