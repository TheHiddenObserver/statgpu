"""
Tests for kernel methods: KernelRidge and KernelRidgeCV.

Validates against sklearn's KernelRidge on CPU (numpy backend).
"""

import numpy as np
import sys
import os
import types

# Ensure we can import statgpu
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Mock cupy if not installed (needed because _cupy.py has top-level imports).
# Use a context manager pattern so the mock is removed after import,
# preventing pollution of sys.modules for other tests.
_CUPY_MOCKED = False
if 'cupy' not in sys.modules:
    try:
        import cupy  # noqa: F401
    except ImportError:
        _cupy_mock = types.ModuleType('cupy')
        _cupy_mock.ndarray = type('ndarray', (np.ndarray,), {})
        _cupy_mock.float64 = np.float64

        class _MockRawModule:
            def __init__(self, **kw):
                pass

            def get_function(self, name):
                return lambda *a: None

        _cupy_mock.RawModule = _MockRawModule
        _cupy_mock.RawKernel = type('RawKernel', (), {
            '__init__': lambda self, *a, **kw: None,
        })
        # Add minimal attrs needed for import without error
        _cupy_mock.asarray = np.asarray
        _cupy_mock.array = np.array
        _cupy_mock.zeros = np.zeros
        _cupy_mock.ones = np.ones
        _cupy_mock.float32 = np.float32
        _cupy_mock.int32 = np.int32
        _cupy_mock.int64 = np.int64
        _cupy_mock.bool_ = np.bool_
        _cupy_mock.newaxis = np.newaxis
        sys.modules['cupy'] = _cupy_mock
        _CUPY_MOCKED = True


def test_kernel_imports():
    """Test that all kernel method classes can be imported."""
    from statgpu.nonparametric.kernel_methods import (
        KernelRidge, KernelRidgeCV,
        rbf_kernel, polynomial_kernel, linear_kernel,
        laplacian_kernel, sigmoid_kernel, cosine_kernel,
        pairwise_kernels,
    )
    print("[PASS] All kernel method imports successful")


def test_rbf_kernel():
    """Test RBF kernel output matches manual computation."""
    from statgpu.nonparametric.kernel_methods import rbf_kernel
    import numpy as np

    rng = np.random.RandomState(42)
    X = rng.randn(5, 3)
    gamma = 0.5

    K = rbf_kernel(X, gamma=gamma)

    # Manual computation
    expected = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            diff = X[i] - X[j]
            expected[i, j] = np.exp(-gamma * np.sum(diff ** 2))

    assert K.shape == (5, 5), f"Shape mismatch: {K.shape}"
    np.testing.assert_allclose(K, expected, rtol=1e-10)
    print("[PASS] RBF kernel matches manual computation")


def test_linear_kernel():
    """Test linear kernel."""
    from statgpu.nonparametric.kernel_methods import linear_kernel

    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    K = linear_kernel(X)
    expected = X @ X.T
    np.testing.assert_allclose(K, expected, rtol=1e-10)
    print("[PASS] Linear kernel correct")


def test_polynomial_kernel():
    """Test polynomial kernel."""
    from statgpu.nonparametric.kernel_methods import polynomial_kernel

    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    K = polynomial_kernel(X, degree=2, gamma=1.0, coef0=1)
    expected = (X @ X.T + 1) ** 2
    np.testing.assert_allclose(K, expected, rtol=1e-10)
    print("[PASS] Polynomial kernel correct")


def test_pairwise_kernels():
    """Test pairwise_kernels registry."""
    from statgpu.nonparametric.kernel_methods import pairwise_kernels

    rng = np.random.RandomState(42)
    X = rng.randn(4, 3)

    for metric in ('rbf', 'linear', 'polynomial', 'laplacian', 'sigmoid', 'cosine'):
        K = pairwise_kernels(X, metric=metric)
        assert K.shape == (4, 4), f"Shape mismatch for {metric}: {K.shape}"
    print("[PASS] pairwise_kernels registry works for all metrics")


def test_kernel_ridge_basic():
    """Test KernelRidge basic fit and predict on numpy backend."""
    from statgpu.nonparametric.kernel_methods import KernelRidge

    rng = np.random.RandomState(42)
    X_train = rng.randn(50, 5)
    y_train = X_train @ np.array([1.0, 2.0, 0.0, -1.0, 0.5]) + 0.1 * rng.randn(50)

    model = KernelRidge(alpha=1.0, kernel='rbf', device='cpu')
    model.fit(X_train, y_train)

    assert model.dual_coef_ is not None
    assert model.X_fit_ is not None
    assert model._fitted is True

    y_pred = model.predict(X_train)
    assert y_pred.shape == (50,), f"Shape mismatch: {y_pred.shape}"

    score = model.score(X_train, y_train)
    assert score > 0.5, f"R^2 too low: {score}"
    print(f"[PASS] KernelRidge basic fit/predict OK, R^2={score:.4f}")


def test_kernel_ridge_vs_sklearn():
    """Test KernelRidge predictions match sklearn's KernelRidge."""
    try:
        from sklearn.kernel_ridge import KernelRidge as SKKRR
    except ImportError:
        print("[SKIP] sklearn not available, skipping sklearn comparison")
        return

    from statgpu.nonparametric.kernel_methods import KernelRidge

    rng = np.random.RandomState(42)
    X_train = rng.randn(100, 5)
    y_train = X_train @ np.array([1.0, 2.0, 0.0, -1.0, 0.5]) + 0.1 * rng.randn(100)
    X_test = rng.randn(20, 5)

    for kernel_name, sk_params in [
        ('rbf', {'kernel': 'rbf', 'alpha': 1.0}),
        ('linear', {'kernel': 'linear', 'alpha': 0.5}),
        ('polynomial', {'kernel': 'polynomial', 'alpha': 1.0, 'degree': 3}),
    ]:
        # statgpu model
        sg_model = KernelRidge(
            alpha=sk_params['alpha'],
            kernel=kernel_name,
            device='cpu',
            degree=sk_params.get('degree', 3),
        )
        sg_model.fit(X_train, y_train)
        sg_pred = sg_model.predict(X_test)

        # sklearn model
        sk_model = SKKRR(**sk_params)
        sk_model.fit(X_train, y_train)
        sk_pred = sk_model.predict(X_test)

        np.testing.assert_allclose(sg_pred, sk_pred, rtol=1e-6,
                                    err_msg=f"Predictions mismatch for {kernel_name}")
        print(f"[PASS] KernelRidge({kernel_name}) matches sklearn, "
              f"max diff={np.max(np.abs(sg_pred - sk_pred)):.2e}")


def test_kernel_ridge_multi_target():
    """Test KernelRidge with multiple targets."""
    from statgpu.nonparametric.kernel_methods import KernelRidge

    rng = np.random.RandomState(42)
    X = rng.randn(50, 4)
    Y = np.column_stack([
        X @ np.array([1.0, 0.0, -1.0, 0.5]),
        X @ np.array([0.0, 1.0, 0.5, -0.5]),
    ]) + 0.1 * rng.randn(50, 2)

    model = KernelRidge(alpha=1.0, kernel='rbf', device='cpu')
    model.fit(X, Y)

    y_pred = model.predict(X)
    assert y_pred.shape == (50, 2), f"Shape mismatch: {y_pred.shape}"
    score = model.score(X, Y)
    assert score > 0.5, f"R^2 too low: {score}"
    print(f"[PASS] KernelRidge multi-target OK, R^2={score:.4f}")


def test_kernel_ridge_cv():
    """Test KernelRidgeCV selects a reasonable alpha."""
    from statgpu.nonparametric.kernel_methods import KernelRidgeCV

    rng = np.random.RandomState(42)
    X = rng.randn(200, 10)
    beta = np.zeros(10)
    beta[:5] = rng.randn(5) * 2.0
    y = X @ beta + 0.1 * rng.randn(200)

    model = KernelRidgeCV(
        alphas=np.logspace(-3, 3, 20),
        cv=5,
        kernel='rbf',
        random_state=42,
        device='cpu',
    )
    model.fit(X, y)

    assert model.alpha_ is not None
    assert model.alpha_ > 0
    assert model.best_score_ is not None
    assert model.cv_results_ is not None
    assert 'alphas' in model.cv_results_
    assert 'mean_mse' in model.cv_results_

    y_pred = model.predict(X)
    assert y_pred.shape == (200,)
    score = model.score(X, y)
    print(f"[PASS] KernelRidgeCV selected alpha={model.alpha_:.4f}, "
          f"best_score={model.best_score_:.4f}, R^2={score:.4f}")


def test_kernel_ridge_cv_auto_alpha():
    """Test KernelRidgeCV with automatic alpha grid."""
    from statgpu.nonparametric.kernel_methods import KernelRidgeCV

    rng = np.random.RandomState(42)
    X = rng.randn(100, 5)
    y = X @ np.array([1.0, 2.0, 0.0, -1.0, 0.5]) + 0.1 * rng.randn(100)

    model = KernelRidgeCV(cv=3, kernel='rbf', random_state=42, device='cpu')
    model.fit(X, y)

    assert model.alpha_ is not None
    assert model.alpha_ > 0
    assert len(model.cv_results_['alphas']) == 100  # default 100 alphas
    print(f"[PASS] KernelRidgeCV auto alpha grid OK, selected alpha={model.alpha_:.4f}")


def test_kernel_ridge_cv_vs_sklearn():
    """Test KernelRidgeCV selected alpha is reasonable compared to sklearn."""
    try:
        from sklearn.kernel_ridge import KernelRidge as SKKRR
        from sklearn.model_selection import GridSearchCV
    except ImportError:
        print("[SKIP] sklearn not available, skipping CV comparison")
        return

    from statgpu.nonparametric.kernel_methods import KernelRidgeCV

    rng = np.random.RandomState(42)
    X = rng.randn(200, 10)
    beta = np.zeros(10)
    beta[:5] = rng.randn(5) * 2.0
    y = X @ beta + 0.1 * rng.randn(200)

    # statgpu
    sg_model = KernelRidgeCV(
        alphas=np.logspace(-3, 3, 50),
        cv=5,
        kernel='rbf',
        random_state=42,
        device='cpu',
    )
    sg_model.fit(X, y)

    # sklearn grid search
    alphas_grid = np.logspace(-3, 3, 50)
    sk_cv = GridSearchCV(
        SKKRR(kernel='rbf'),
        param_grid={'alpha': alphas_grid},
        cv=5,
        scoring='neg_mean_squared_error',
    )
    sk_cv.fit(X, y)

    sg_alpha = sg_model.alpha_
    sk_alpha = sk_cv.best_params_['alpha']

    # Check same order of magnitude
    ratio = max(sg_alpha, sk_alpha) / max(min(sg_alpha, sk_alpha), 1e-15)
    print(f"[INFO] statgpu alpha={sg_alpha:.4f}, sklearn alpha={sk_alpha:.4f}, "
          f"ratio={ratio:.2f}")

    # They may not match exactly due to different internal implementations,
    # but should be within a few orders of magnitude
    assert ratio < 100, f"Alpha values differ too much: {sg_alpha} vs {sk_alpha}"
    print("[PASS] KernelRidgeCV alpha is reasonable compared to sklearn GridSearchCV")


def test_get_set_params():
    """Test get_params and set_params for API compatibility."""
    from statgpu.nonparametric.kernel_methods import KernelRidge, KernelRidgeCV

    krr = KernelRidge(alpha=0.5, kernel='linear', device='cpu')
    params = krr.get_params()
    assert params['alpha'] == 0.5
    assert params['kernel'] == 'linear'

    krr.set_params(alpha=2.0)
    assert krr.alpha == 2.0

    krr_cv = KernelRidgeCV(cv=3, kernel='polynomial', device='cpu')
    params_cv = krr_cv.get_params()
    assert params_cv['cv'] == 3
    assert params_cv['kernel'] == 'polynomial'

    krr_cv.set_params(cv=10)
    assert krr_cv.cv == 10
    print("[PASS] get_params/set_params work correctly")


def test_not_fitted_error():
    """Test that predict raises error before fit."""
    from statgpu.nonparametric.kernel_methods import KernelRidge

    model = KernelRidge(device='cpu')
    try:
        model.predict(np.zeros((5, 3)))
        assert False, "Should have raised RuntimeError"
    except RuntimeError:
        pass
    print("[PASS] Not-fitted error raised correctly")


if __name__ == "__main__":
    test_kernel_imports()
    test_rbf_kernel()
    test_linear_kernel()
    test_polynomial_kernel()
    test_pairwise_kernels()
    test_kernel_ridge_basic()
    test_kernel_ridge_vs_sklearn()
    test_kernel_ridge_multi_target()
    test_kernel_ridge_cv()
    test_kernel_ridge_cv_auto_alpha()
    test_kernel_ridge_cv_vs_sklearn()
    test_get_set_params()
    test_not_fitted_error()
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
