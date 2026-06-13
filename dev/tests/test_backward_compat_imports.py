# -*- coding: utf-8 -*-
"""Regression tests for backward-compatible import paths.

After module reorganization (nonparametric/semiparametric subpackages),
old import paths must continue to work via backward-compat shims.
"""
import pytest


class TestKernelMethodsBackwardCompat:
    """Old path: statgpu.kernel_methods -> statgpu.nonparametric.kernel_methods"""

    def test_import_kernel_ridge(self):
        from statgpu.kernel_methods import KernelRidge
        assert KernelRidge is not None

    def test_import_kernel_ridge_cv(self):
        from statgpu.kernel_methods import KernelRidgeCV
        assert KernelRidgeCV is not None

    def test_import_rbf_kernel(self):
        from statgpu.kernel_methods import rbf_kernel
        assert callable(rbf_kernel)

    def test_import_polynomial_kernel(self):
        from statgpu.kernel_methods import polynomial_kernel
        assert callable(polynomial_kernel)


class TestSplinesBackwardCompat:
    """Old path: statgpu.splines -> statgpu.nonparametric.splines + statgpu.semiparametric"""

    def test_import_bspline_basis(self):
        from statgpu.splines import bspline_basis
        assert callable(bspline_basis)

    def test_import_natural_cubic_spline_basis(self):
        from statgpu.splines import natural_cubic_spline_basis
        assert callable(natural_cubic_spline_basis)

    def test_import_gam(self):
        from statgpu.splines import GAM
        assert GAM is not None

    def test_star_import_includes_gam(self):
        import statgpu.splines as sp
        assert "GAM" in sp.__all__


class TestNonparametricKDEBackwardCompat:
    """Old path: statgpu.nonparametric._kde -> statgpu.nonparametric.kernel_smoothing._kde"""

    def test_import_kde(self):
        from statgpu.nonparametric._kde import KernelDensityEstimator
        assert KernelDensityEstimator is not None

    def test_import_kde_from_nonparametric(self):
        from statgpu.nonparametric import KernelDensityEstimator
        assert KernelDensityEstimator is not None


class TestNonparametricBandwidthBackwardCompat:
    """Old path: statgpu.nonparametric._bandwidth_selection -> kernel_smoothing._bandwidth_selection"""

    def test_import_bandwidth(self):
        from statgpu.nonparametric._bandwidth_selection import select_bandwidth
        assert callable(select_bandwidth)


class TestNonparametricKernelRegressionBackwardCompat:
    """Old path: statgpu.nonparametric._kernel_regression -> kernel_smoothing._kernel_regression"""

    def test_import_kernel_regression(self):
        from statgpu.nonparametric._kernel_regression import KernelRegression
        assert KernelRegression is not None


class TestNonparametricKernelCommonBackwardCompat:
    """Old path: statgpu.nonparametric._kernel_common -> kernel_smoothing._kernel_common"""

    def test_import_kernel_common(self):
        from statgpu.nonparametric._kernel_common import _normalize_regression_name
        assert callable(_normalize_regression_name)
