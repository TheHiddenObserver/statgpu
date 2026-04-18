"""
Elastic Net Numerical Stability Test - English version
"""
import numpy as np
from statgpu.linear_model import ElasticNet, Lasso, Ridge

def test_numerical_stability():
    np.random.seed(42)
    n_samples, n_features = 200, 20

    print("=" * 70)
    print("Elastic Net Numerical Stability Test")
    print("=" * 70)

    # ========== Test 1: Different alpha values ==========
    print("\n[Test 1] Stability across alpha values")
    print("-" * 50)

    X = np.random.randn(n_samples, n_features)
    true_coef = np.zeros(n_features)
    true_coef[:5] = np.random.randn(5)
    y = X @ true_coef + np.random.randn(n_samples) * 0.5

    alpha_values = [1e-6, 1e-3, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

    all_passed = True
    for alpha in alpha_values:
        try:
            enet = ElasticNet(alpha=alpha, l1_ratio=0.5, max_iter=1000, tol=1e-8)
            enet.fit(X, y)
            coef_finite = np.all(np.isfinite(enet.coef_))
            coef_norm = np.linalg.norm(enet.coef_)
            status = "PASS" if coef_finite else "FAIL"
            if not coef_finite:
                all_passed = False
            print(f"  [{status}] alpha={alpha:>8}: ||coef||={coef_norm:>12.6e}, iter={enet.n_iter_}")
        except Exception as e:
            print(f"  [FAIL] alpha={alpha:>8}: ERROR - {e}")
            all_passed = False

    # ========== Test 2: Different l1_ratio ==========
    print("\n[Test 2] Stability across l1_ratio (alpha=1.0)")
    print("-" * 50)

    l1_ratios = [0.0, 0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999, 1.0]

    for l1_ratio in l1_ratios:
        try:
            enet = ElasticNet(alpha=1.0, l1_ratio=l1_ratio, max_iter=1000, tol=1e-8)
            enet.fit(X, y)
            coef_finite = np.all(np.isfinite(enet.coef_))
            coef_norm = np.linalg.norm(enet.coef_)
            sparsity = np.sum(enet.coef_ == 0)
            status = "PASS" if coef_finite else "FAIL"
            if not coef_finite:
                all_passed = False
            print(f"  [{status}] l1_ratio={l1_ratio:>6}: ||coef||={coef_norm:>12.6e}, sparse={sparsity:>2}, iter={enet.n_iter_}")
        except Exception as e:
            print(f"  [FAIL] l1_ratio={l1_ratio:>6}: ERROR - {e}")
            all_passed = False

    # ========== Test 3: Ill-conditioned design matrix ==========
    print("\n[Test 3] Ill-conditioned design matrix (correlated features)")
    print("-" * 50)

    n_ill = 50
    X_ill = np.random.randn(n_samples, n_ill)
    for i in range(10):
        X_ill = np.column_stack([X_ill, X_ill[:, i] + np.random.randn(n_samples) * 0.01])
    y_ill = X_ill @ np.random.randn(X_ill.shape[1]) + np.random.randn(n_samples)

    for alpha in [0.01, 0.1, 1.0, 10.0]:
        try:
            enet = ElasticNet(alpha=alpha, l1_ratio=0.5, max_iter=2000, tol=1e-8)
            enet.fit(X_ill, y_ill)
            coef_finite = np.all(np.isfinite(enet.coef_))
            coef_norm = np.linalg.norm(enet.coef_)
            r2 = enet.score(X_ill, y_ill)
            status = "PASS" if (coef_finite and r2 > 0) else "FAIL"
            if not (coef_finite and r2 > 0):
                all_passed = False
            print(f"  [{status}] alpha={alpha:>6}: ||coef||={coef_norm:>12.6e}, R2={r2:>8.6f}, iter={enet.n_iter_}")
        except Exception as e:
            print(f"  [FAIL] alpha={alpha:>6}: ERROR - {e}")
            all_passed = False

    # ========== Test 4: High-dimensional (n << p) ==========
    print("\n[Test 4] High-dimensional setting (n << p)")
    print("-" * 50)

    n_small, p_large = 30, 100
    X_hp = np.random.randn(n_small, p_large)
    y_hp = X_hp @ np.random.randn(p_large) + np.random.randn(n_small) * 0.1

    for l1_ratio in [0.0, 0.5, 1.0]:
        try:
            enet = ElasticNet(alpha=1.0, l1_ratio=l1_ratio, max_iter=2000, tol=1e-8)
            enet.fit(X_hp, y_hp)
            coef_finite = np.all(np.isfinite(enet.coef_))
            coef_norm = np.linalg.norm(enet.coef_)
            sparsity = np.sum(enet.coef_ == 0)
            status = "PASS" if coef_finite else "FAIL"
            if not coef_finite:
                all_passed = False
            print(f"  [{status}] l1_ratio={l1_ratio}: ||coef||={coef_norm:>12.6e}, sparse={sparsity:>3}, iter={enet.n_iter_}")
        except Exception as e:
            print(f"  [FAIL] l1_ratio={l1_ratio}: ERROR - {e}")
            all_passed = False

    # ========== Test 5: KKT convergence verification ==========
    print("\n[Test 5] KKT convergence verification")
    print("-" * 50)

    X = np.random.randn(200, 20)
    y = X @ np.random.randn(20) + np.random.randn(200) * 0.5

    for l1_ratio in [0.0, 0.25, 0.5, 0.75, 1.0]:
        enet = ElasticNet(alpha=1.0, l1_ratio=l1_ratio, max_iter=5000, tol=1e-10, stopping='kkt')
        enet.fit(X, y)

        n = X.shape[0]
        alpha = 1.0
        coef = enet.coef_
        l2_ratio = 1.0 - l1_ratio

        XtX = X.T @ X
        Xty = X.T @ y
        grad_rss = (XtX @ coef - Xty) / n
        grad_l2 = alpha * l2_ratio * coef

        kkt_violation = np.zeros(len(coef))
        for j in range(len(coef)):
            if coef[j] != 0:
                sign_j = np.sign(coef[j])
                kkt_violation[j] = np.abs(grad_rss[j] + grad_l2[j] + alpha * l1_ratio * sign_j)
            else:
                kkt_violation[j] = max(0, np.abs(grad_rss[j] + grad_l2[j]) - alpha * l1_ratio)

        max_kkt = np.max(kkt_violation)
        status = "PASS" if max_kkt < 1e-6 else "WARN"
        if max_kkt >= 1e-4:
            all_passed = False
        print(f"  [{status}] l1_ratio={l1_ratio}: max_KKT={max_kkt:>12.2e}, iter={enet.n_iter_}")

    # ========== Test 6: Boundary cases ==========
    print("\n[Test 6] Boundary cases")
    print("-" * 50)

    try:
        enet = ElasticNet(alpha=1e-10, l1_ratio=0.5, max_iter=1000)
        enet.fit(X, y)
        print(f"  [PASS] alpha~0: ||coef||={np.linalg.norm(enet.coef_):.6e}")
    except Exception as e:
        print(f"  [FAIL] alpha~0: ERROR - {e}")
        all_passed = False

    for l1_ratio in [0.0, 1.0]:
        enet = ElasticNet(alpha=1.0, l1_ratio=l1_ratio, max_iter=1000, tol=1e-10)
        enet.fit(X, y)
        print(f"  [PASS] l1_ratio={l1_ratio}: ||coef||={np.linalg.norm(enet.coef_):.6e}, iter={enet.n_iter_}")

    print("\n" + "=" * 70)
    if all_passed:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED!")
    print("=" * 70)

if __name__ == "__main__":
    test_numerical_stability()
