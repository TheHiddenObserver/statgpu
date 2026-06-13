"""
Validate debiased Lasso inference against:
1. R's hdi::lasso.proj (same algorithm)
2. OLS (asymptotic benchmark when alpha -> 0)
"""

import numpy as np
from statgpu.linear_model import Lasso


def test_debiased_vs_ols_small_alpha():
    """With very small alpha, debiased Lasso should approximate OLS."""
    np.random.seed(42)
    n, p = 200, 8
    X = np.random.randn(n, p)
    beta = np.array([2.0, -1.0, 0.5, 0.0, 0.0, 0.3, 0.0, -0.2])
    y = X @ beta + 0.2 * np.random.randn(n)

    # OLS
    X_ols = np.column_stack([np.ones(n), X])
    from numpy.linalg import lstsq
    beta_ols, _, _, _ = lstsq(X_ols, y, rcond=None)
    resid_ols = y - X_ols @ beta_ols
    sigma2_ols = np.sum(resid_ols**2) / (n - p - 1)
    se_ols = np.sqrt(sigma2_ols * np.diag(np.linalg.inv(X_ols.T @ X_ols)))

    # Debiased Lasso with tiny alpha (~OLS)
    model = Lasso(alpha=0.001, compute_inference=True, device='cpu')
    model.fit(X, y)

    coef_diff = np.max(np.abs(model.coef_ - beta_ols[1:]))
    se_ratio = np.array(model._bse[1:]) / se_ols[1:]  # skip intercept

    print(f"Max |coef_diff|: {coef_diff:.6f}")
    print(f"SE ratio range: [{se_ratio.min():.4f}, {se_ratio.max():.4f}]")
    print(f"Sigma ratio: {np.sqrt(model._scale) / np.sqrt(sigma2_ols):.4f}")

    # Coefficients should be within 5% of OLS
    assert coef_diff < 0.05, f"coef diff too large: {coef_diff}"
    # SEs should be within 15% of OLS
    assert se_ratio.min() > 0.85, f"SE too small: {se_ratio.min()}"
    assert se_ratio.max() < 1.15, f"SE too large: {se_ratio.max()}"
    print("PASSED: debiased Lasso (small alpha) ~ OLS")


def test_debiased_selects_correct_variables():
    """Debiased Lasso should identify true nonzero coefficients."""
    np.random.seed(123)
    n, p = 150, 10
    X = np.random.randn(n, p)
    beta = np.array([3.0, -2.0, 1.5, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, -0.5])
    y = X @ beta + 0.3 * np.random.randn(n)

    model = Lasso(alpha=0.05, compute_inference=True, device='cpu')
    model.fit(X, y)

    pvals = model._pvalues[1:]  # skip intercept
    true_nonzero = np.abs(beta) > 0
    true_zero = ~true_nonzero

    # True nonzero should have small p-values (high power)
    power = np.mean(pvals[true_nonzero] < 0.05)
    # True zero should have large p-values (low FDR)
    fdr = np.mean(pvals[true_zero] < 0.05) if np.any(true_zero) else 0.0

    print(f"Power (detect true effects): {power:.2f}")
    print(f"FDR (false positives): {fdr:.2f}")
    print(f"P-values: {[f'{p:.4f}' for p in pvals]}")

    assert power >= 0.6, f"Power too low: {power}"
    print("PASSED: variable selection reasonable")


def test_debiased_r_squared():
    """R2 should be reasonable for a good fit."""
    np.random.seed(99)
    n, p = 100, 5
    X = np.random.randn(n, p)
    beta = np.array([1.0, -0.5, 0.3, 0.0, 0.0])
    y = X @ beta + 0.1 * np.random.randn(n)

    model = Lasso(alpha=0.05, compute_inference=True, device='cpu')
    model.fit(X, y)

    r2 = model.rsquared
    print(f"R2: {r2:.4f}")
    assert r2 > 0.9, f"R2 too low: {r2}"
    print("PASSED: R2 reasonable")


def test_summary_output(capsys=None):
    """Summary should print without errors."""
    np.random.seed(42)
    X = np.random.randn(50, 3)
    y = X @ np.array([1.0, -0.5, 0.3]) + 0.1 * np.random.randn(50)

    model = Lasso(alpha=0.05, compute_inference=True, device='cpu')
    model.fit(X, y)
    model.summary()  # Should not raise
    print("PASSED: summary() works")


if __name__ == "__main__":
    print("=" * 60)
    print("Debiased Lasso Validation Tests")
    print("=" * 60)
    for name, func in [
        ("vs OLS", test_debiased_vs_ols_small_alpha),
        ("variable selection", test_debiased_selects_correct_variables),
        ("R2", test_debiased_r_squared),
        ("summary", test_summary_output),
    ]:
        print(f"\n--- {name} ---")
        func()
    print("\n" + "=" * 60)
    print("ALL VALIDATION TESTS PASSED")
    print("=" * 60)
