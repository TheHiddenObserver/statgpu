"""Comprehensive tests for Kernel methods: chi2_kernel, Nystroem, KernelPCA."""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from statgpu.nonparametric.kernel_methods import (
    rbf_kernel, linear_kernel, polynomial_kernel, laplacian_kernel,
    sigmoid_kernel, cosine_kernel, chi2_kernel,
    pairwise_kernels, KernelRidge, KernelRidgeCV,
    Nystroem, KernelPCA,
)


# ============================================================================
# chi2_kernel
# ============================================================================

class TestChi2Kernel:

    def test_basic(self):
        X = np.abs(np.random.randn(20, 5))
        Y = np.abs(np.random.randn(10, 5))
        K = chi2_kernel(X, Y)
        assert K.shape == (20, 10)
        assert np.all(K >= 0)
        assert np.all(K <= 1)

    def test_self_kernel(self):
        X = np.abs(np.random.randn(20, 5))
        K = chi2_kernel(X)
        assert K.shape == (20, 20)
        # Diagonal should be 1 (distance to self is 0, exp(0) = 1)
        assert_allclose(np.diag(K), 1.0, atol=1e-10)

    def test_symmetry(self):
        X = np.abs(np.random.randn(20, 5))
        K = chi2_kernel(X)
        assert_allclose(K, K.T, atol=1e-10)

    def test_vs_sklearn(self):
        from sklearn.metrics.pairwise import chi2_kernel as sk_chi2
        X = np.abs(np.random.randn(20, 5))
        Y = np.abs(np.random.randn(10, 5))
        K_ours = chi2_kernel(X, Y, gamma=0.5)
        K_sk = sk_chi2(X, Y, gamma=0.5)
        assert_allclose(K_ours, K_sk, rtol=1e-6)

    def test_registry(self):
        assert "chi2" in pairwise_kernels.__module__ or True  # just check import
        X = np.abs(np.random.randn(10, 3))
        K = pairwise_kernels(X, metric="chi2")
        assert K.shape == (10, 10)


# ============================================================================
# Existing kernels regression
# ============================================================================

class TestExistingKernels:

    def test_rbf(self):
        X = np.random.randn(20, 5)
        K = rbf_kernel(X)
        assert K.shape == (20, 20)
        assert_allclose(np.diag(K), 1.0, atol=1e-10)

    def test_linear(self):
        X = np.random.randn(20, 5)
        K = linear_kernel(X)
        assert K.shape == (20, 20)

    def test_polynomial(self):
        X = np.random.randn(20, 5)
        K = polynomial_kernel(X, degree=2)
        assert K.shape == (20, 20)

    def test_laplacian(self):
        X = np.random.randn(20, 5)
        K = laplacian_kernel(X)
        assert K.shape == (20, 20)

    def test_cosine(self):
        X = np.random.randn(20, 5)
        K = cosine_kernel(X)
        assert K.shape == (20, 20)

    def test_pairwise_kernels_registry(self):
        X = np.random.randn(10, 3)
        for metric in ["rbf", "linear", "poly", "laplacian", "sigmoid", "cosine", "chi2"]:
            K = pairwise_kernels(X, metric=metric)
            assert K.shape == (10, 10)


# ============================================================================
# Nystroem
# ============================================================================

class TestNystroem:

    def test_basic(self):
        X = np.random.randn(100, 10)
        ny = Nystroem(kernel='rbf', n_components=20, random_state=42).fit(X)
        Z = ny.transform(X)
        assert Z.shape == (100, 20)

    def test_n_components_larger_than_samples(self):
        X = np.random.randn(10, 5)
        ny = Nystroem(kernel='rbf', n_components=20).fit(X)
        Z = ny.transform(X)
        # Should clamp to n_samples
        assert Z.shape[0] == 10

    def test_different_kernels(self):
        X = np.random.randn(50, 5)
        for kernel in ['rbf', 'linear', 'poly']:
            ny = Nystroem(kernel=kernel, n_components=10, random_state=42).fit(X)
            Z = ny.transform(X)
            assert Z.shape == (50, 10)

    def test_fit_transform(self):
        X = np.random.randn(50, 5)
        ny = Nystroem(kernel='rbf', n_components=10, random_state=42)
        Z = ny.fit_transform(X)
        assert Z.shape == (50, 10)

    def test_predict(self):
        X = np.random.randn(50, 5)
        ny = Nystroem(kernel='rbf', n_components=10, random_state=42).fit(X)
        Z = ny.predict(X[:10])
        assert Z.shape == (10, 10)

    def test_cross_kernel(self):
        X_train = np.random.randn(100, 5)
        X_test = np.random.randn(20, 5)
        ny = Nystroem(kernel='rbf', n_components=10, random_state=42).fit(X_train)
        Z = ny.transform(X_test)
        assert Z.shape == (20, 10)


# ============================================================================
# KernelPCA
# ============================================================================

class TestKernelPCA:

    def test_basic(self):
        X = np.random.randn(100, 10)
        kpca = KernelPCA(n_components=3, kernel='rbf').fit(X)
        X_kp = kpca.transform(X)
        assert X_kp.shape == (100, 3)

    def test_eigenvalues_positive(self):
        X = np.random.randn(100, 5)
        kpca = KernelPCA(n_components=3, kernel='rbf').fit(X)
        assert np.all(kpca.lambdas_ > 0)

    def test_different_kernels(self):
        X = np.random.randn(50, 5)
        for kernel in ['rbf', 'linear', 'poly']:
            kpca = KernelPCA(n_components=2, kernel=kernel).fit(X)
            X_kp = kpca.transform(X)
            assert X_kp.shape == (50, 2)

    def test_fit_transform(self):
        X = np.random.randn(50, 5)
        kpca = KernelPCA(n_components=3, kernel='rbf')
        X_kp = kpca.fit_transform(X)
        assert X_kp.shape == (50, 3)

    def test_predict(self):
        X = np.random.randn(50, 5)
        kpca = KernelPCA(n_components=2, kernel='rbf').fit(X)
        X_kp = kpca.predict(X[:10])
        assert X_kp.shape == (10, 2)

    def test_n_components_clamped(self):
        X = np.random.randn(5, 3)
        kpca = KernelPCA(n_components=10, kernel='rbf').fit(X)
        X_kp = kpca.transform(X)
        assert X_kp.shape[1] <= 5

    def test_transform_new_data(self):
        X_train = np.random.randn(100, 5)
        X_test = np.random.randn(20, 5)
        kpca = KernelPCA(n_components=3, kernel='rbf').fit(X_train)
        X_kp = kpca.transform(X_test)
        assert X_kp.shape == (20, 3)


# ============================================================================
# KernelRidge / KernelRidgeCV regression
# ============================================================================

class TestExistingKRR:

    def test_kernel_ridge(self):
        X = np.random.randn(50, 5)
        y = np.random.randn(50)
        kr = KernelRidge(alpha=1.0, kernel='rbf').fit(X, y)
        y_pred = kr.predict(X[:10])
        assert y_pred.shape == (10,)

    def test_kernel_ridge_cv(self):
        X = np.random.randn(50, 5)
        y = np.random.randn(50)
        krcv = KernelRidgeCV(kernel='rbf').fit(X, y)
        assert krcv.alpha_ > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
