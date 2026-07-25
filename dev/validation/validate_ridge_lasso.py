"""
Validation script for Ridge and Lasso implementations.
Compares against sklearn and statsmodels using explicit objective mappings.
"""

import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

# Try importing our implementations
try:
    from statgpu.linear_model import Ridge, Lasso
    STATGPU_AVAILABLE = True
except ImportError as e:
    print(f"statgpu not available: {e}")
    STATGPU_AVAILABLE = False

# Try importing sklearn
sklearn_available = False
try:
    from sklearn.linear_model import Ridge as SklearnRidge
    from sklearn.linear_model import Lasso as SklearnLasso
    sklearn_available = True
except ImportError:
    print("sklearn not available")

# Try importing statsmodels
statsmodels_available = False
try:
    import statsmodels.api as sm
    statsmodels_available = True
except ImportError:
    print("statsmodels not available")

# Try importing cupy for GPU
gpu_available = False
try:
    import cupy as cp
    gpu_available = True
    print("GPU (CuPy) available")
except ImportError:
    print("GPU (CuPy) not available")


def generate_data(n_samples=1000, n_features=10, noise=0.1, random_state=42):
    """Generate synthetic regression data."""
    np.random.seed(random_state)
    X = np.random.randn(n_samples, n_features)
    true_coef = np.random.randn(n_features) * 2
    y = X @ true_coef + noise * np.random.randn(n_samples)
    return X, y, true_coef


def validate_ridge():
    """Validate Ridge implementation under the average-loss objective."""
    print("\n" + "="*80)
    print("RIDGE REGRESSION VALIDATION")
    print("="*80)

    X, y, true_coef = generate_data(n_samples=1000, n_features=10)
    alpha = 1.0

    # Fit our implementation
    print("\n--- statgpu Ridge ---")
    ridge_sg = Ridge(alpha=alpha, fit_intercept=True, device='cpu')
    ridge_sg.fit(X, y)
    print(f"Coefficients (first 5): {ridge_sg.coef_[:5]}")
    print(f"Intercept: {ridge_sg.intercept_:.6f}")
    print(f"R²: {ridge_sg.rsquared:.6f}")
    print(f"AIC: {ridge_sg.aic:.4f}, BIC: {ridge_sg.bic:.4f}")

    # Compare with sklearn after mapping the objective scale.
    if sklearn_available:
        sklearn_alpha = X.shape[0] * alpha
        print(f"\n--- sklearn Ridge (alpha={sklearn_alpha:g}) ---")
        print("Mapping: sklearn_alpha = n_samples * statgpu_alpha")
        ridge_sk = SklearnRidge(alpha=sklearn_alpha, fit_intercept=True)
        ridge_sk.fit(X, y)
        print(f"Coefficients (first 5): {ridge_sk.coef_[:5]}")
        print(f"Intercept: {ridge_sk.intercept_:.6f}")
        print(f"R² (score): {ridge_sk.score(X, y):.6f}")

        coef_diff = np.abs(ridge_sg.coef_ - ridge_sk.coef_)
        intercept_diff = abs(ridge_sg.intercept_ - ridge_sk.intercept_)
        print("\n--- Comparison ---")
        print(f"Max coefficient difference: {np.max(coef_diff):.2e}")
        print(f"Intercept difference: {intercept_diff:.2e}")

        if np.max(coef_diff) < 1e-6 and intercept_diff < 1e-6:
            print("✓ Ridge coefficients match sklearn under the mapped objective!")
        else:
            print("✗ Ridge coefficients differ from sklearn under the mapped objective")

    # Compare with statsmodels
    if statsmodels_available:
        print("\n--- statsmodels OLS (no regularization) ---")
        X_with_const = sm.add_constant(X)
        model_sm = sm.OLS(y, X_with_const)
        results_sm = model_sm.fit()
        print(f"R²: {results_sm.rsquared:.6f}")
        print(f"AIC: {results_sm.aic:.4f}, BIC: {results_sm.bic:.4f}")


def validate_lasso():
    """Validate Lasso implementation."""
    print("\n" + "="*80)
    print("LASSO REGRESSION VALIDATION")
    print("="*80)

    X, y, true_coef = generate_data(n_samples=1000, n_features=10)
    alpha = 0.1

    print("\n--- statgpu Lasso ---")
    lasso_sg = Lasso(alpha=alpha, fit_intercept=True, max_iter=2000, device='cpu')
    lasso_sg.fit(X, y)
    print(f"Coefficients (first 5): {lasso_sg.coef_[:5]}")
    print(f"Intercept: {lasso_sg.intercept_:.6f}")
    print(f"Non-zero coefficients: {np.sum(np.abs(lasso_sg.coef_) > 1e-10)}")
    print(f"Iterations: {lasso_sg.n_iter_}")
    print(f"R²: {lasso_sg.rsquared:.6f}")
    print(f"AIC: {lasso_sg.aic:.4f}, BIC: {lasso_sg.bic:.4f}")

    if sklearn_available:
        print("\n--- sklearn Lasso ---")
        lasso_sk = SklearnLasso(alpha=alpha, fit_intercept=True, max_iter=2000)
        lasso_sk.fit(X, y)
        print(f"Coefficients (first 5): {lasso_sk.coef_[:5]}")
        print(f"Intercept: {lasso_sk.intercept_:.6f}")
        print(f"Non-zero coefficients: {np.sum(np.abs(lasso_sk.coef_) > 1e-10)}")
        print(f"Iterations: {lasso_sk.n_iter_}")
        print(f"R² (score): {lasso_sk.score(X, y):.6f}")

        coef_diff = np.abs(lasso_sg.coef_ - lasso_sk.coef_)
        intercept_diff = abs(lasso_sg.intercept_ - lasso_sk.intercept_)
        print("\n--- Comparison ---")
        print(f"Max coefficient difference: {np.max(coef_diff):.2e}")
        print(f"Intercept difference: {intercept_diff:.2e}")

        if np.max(coef_diff) < 1e-4 and intercept_diff < 1e-4:
            print("✓ Lasso coefficients match sklearn!")
        else:
            print("✗ Lasso coefficients differ from sklearn (expected due to algorithm differences)")


def benchmark_gpu():
    """Benchmark GPU vs CPU performance."""
    print("\n" + "="*80)
    print("GPU BENCHMARK")
    print("="*80)

    if not gpu_available:
        print("GPU not available, skipping benchmark")
        return

    sizes = [
        (1000, 50),
        (5000, 100),
        (10000, 200),
        (50000, 500),
    ]

    print(f"\n{'Size':<15} {'Model':<10} {'CPU (ms)':<12} {'GPU (ms)':<12} {'Speedup':<10}")
    print("-" * 65)

    for n_samples, n_features in sizes:
        X, y, _ = generate_data(n_samples, n_features)

        ridge_cpu = Ridge(alpha=1.0, device='cpu')
        t0 = time.perf_counter()
        ridge_cpu.fit(X, y)
        cpu_time = (time.perf_counter() - t0) * 1000

        ridge_gpu = Ridge(alpha=1.0, device='cuda')
        t0 = time.perf_counter()
        ridge_gpu.fit(X, y)
        gpu_time = (time.perf_counter() - t0) * 1000

        speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
        print(f"{n_samples}x{n_features:<8} {'Ridge':<10} {cpu_time:<12.2f} {gpu_time:<12.2f} {speedup:<10.2f}x")

        if n_samples <= 10000:
            lasso_cpu = Lasso(alpha=0.1, max_iter=500, device='cpu')
            t0 = time.perf_counter()
            lasso_cpu.fit(X, y)
            cpu_time = (time.perf_counter() - t0) * 1000

            lasso_gpu = Lasso(alpha=0.1, max_iter=500, device='cuda')
            t0 = time.perf_counter()
            lasso_gpu.fit(X, y)
            gpu_time = (time.perf_counter() - t0) * 1000

            speedup = cpu_time / gpu_time if gpu_time > 0 else float('inf')
            print(f"{n_samples}x{n_features:<8} {'Lasso':<10} {cpu_time:<12.2f} {gpu_time:<12.2f} {speedup:<10.2f}x")


def test_summary():
    """Test summary output."""
    print("\n" + "="*80)
    print("SUMMARY OUTPUT TEST")
    print("="*80)

    X, y, _ = generate_data(n_samples=200, n_features=5)

    print("\n--- Ridge Summary ---")
    ridge = Ridge(alpha=1.0, device='cpu')
    ridge.fit(X, y)
    ridge.summary()

    print("\n--- Lasso Summary ---")
    lasso = Lasso(alpha=0.1, device='cpu')
    lasso.fit(X, y)
    lasso.summary()


def test_api_compliance():
    """Test sklearn-like API compliance."""
    print("\n" + "="*80)
    print("API COMPLIANCE TEST")
    print("="*80)

    X, y, _ = generate_data(n_samples=200, n_features=5)
    X_test, y_test, _ = generate_data(n_samples=100, n_features=5, random_state=43)

    ridge = Ridge(alpha=1.0, device='cpu')
    ridge.fit(X, y)
    y_pred = ridge.predict(X_test)
    score = ridge.score(X_test, y_test)

    print("\n--- Ridge API ---")
    print("fit() works: ✓")
    print(f"predict() shape: {y_pred.shape}")
    print(f"score() R²: {score:.6f}")
    print(f"coef_ shape: {ridge.coef_.shape}")
    print(f"intercept_: {ridge.intercept_:.6f}")

    lasso = Lasso(alpha=0.1, device='cpu')
    lasso.fit(X, y)
    y_pred = lasso.predict(X_test)
    score = lasso.score(X_test, y_test)

    print("\n--- Lasso API ---")
    print("fit() works: ✓")
    print(f"predict() shape: {y_pred.shape}")
    print(f"score() R²: {score:.6f}")
    print(f"coef_ shape: {lasso.coef_.shape}")
    print(f"intercept_: {lasso.intercept_:.6f}")
    print(f"n_iter_: {lasso.n_iter_}")


if __name__ == "__main__":
    print("StatGPU Ridge & Lasso Validation")
    print(f"NumPy version: {np.__version__}")

    if STATGPU_AVAILABLE:
        validate_ridge()
        validate_lasso()
        test_summary()
        test_api_compliance()
        benchmark_gpu()
    else:
        print("statgpu not available - cannot run validation")
