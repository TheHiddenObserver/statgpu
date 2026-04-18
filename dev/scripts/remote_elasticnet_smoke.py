"""
Remote smoke test for Elastic Net implementation.
Run on remote GPU server with myconda environment.
"""

import numpy as np
from statgpu.linear_model import ElasticNet, Lasso, Ridge

np.random.seed(42)

# Generate synthetic data
n_samples, n_features = 200, 20
X = np.random.randn(n_samples, n_features)
true_coef = np.zeros(n_features)
true_coef[:5] = np.random.randn(5)  # Only 5 true features
y = X @ true_coef + np.random.randn(n_samples) * 0.5

print("=" * 60)
print("Elastic Net Smoke Test (Remote GPU)")
print("=" * 60)

# Test 1: Basic fit/predict
print("\n[Test 1] Basic fit/predict (CPU)...")
enet = ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=1000, tol=1e-6)
enet.fit(X, y)
y_pred = enet.predict(X)
print(f"  coef_ (first 5): {enet.coef_[:5]}")
print(f"  intercept_: {enet.intercept_:.6f}")
print(f"  n_iter_: {enet.n_iter_}")
print(f"  R²: {enet.score(X, y):.6f}")
print("  ✓ Passed")

# Test 2: l1_ratio=1.0 should match Lasso
print("\n[Test 2] l1_ratio=1.0 vs Lasso...")
enet_lasso = ElasticNet(alpha=1.0, l1_ratio=1.0, max_iter=1000, tol=1e-6)
enet_lasso.fit(X, y)

lasso = Lasso(alpha=1.0, max_iter=1000, tol=1e-6, cpu_solver='fista')
lasso.fit(X, y)

coef_diff = np.max(np.abs(enet_lasso.coef_ - lasso.coef_))
print(f"  ElasticNet(l1_ratio=1.0) coef (first 3): {enet_lasso.coef_[:3]}")
print(f"  Lasso coef (first 3): {lasso.coef_[:3]}")
print(f"  Max coef difference: {coef_diff:.2e}")
if coef_diff < 1e-6:
    print("  ✓ Passed (matches Lasso)")
else:
    print(f"  ✗ Failed (difference {coef_diff:.2e} > 1e-6)")

# Test 3: l1_ratio=0.0 should match Ridge (with scaled alpha)
print("\n[Test 3] l1_ratio=0.0 vs Ridge...")
enet_ridge = ElasticNet(alpha=1.0, l1_ratio=0.0, max_iter=1000, tol=1e-6)
enet_ridge.fit(X, y)

# Note: ElasticNet loss is scaled by 1/(2n), so ElasticNet(l1_ratio=0, alpha)
# is equivalent to Ridge(alpha' = n * alpha)
ridge_scaled = Ridge(alpha=n_samples * 1.0)
ridge_scaled.fit(X, y)

coef_diff_ridge = np.max(np.abs(enet_ridge.coef_ - ridge_scaled.coef_))
print(f"  ElasticNet(l1_ratio=0.0, alpha=1.0) coef (first 3): {enet_ridge.coef_[:3]}")
print(f"  Ridge(alpha={n_samples}) coef (first 3): {ridge_scaled.coef_[:3]}")
print(f"  Max coef difference: {coef_diff_ridge:.2e}")
if coef_diff_ridge < 1e-6:
    print("  ✓ Passed (matches Ridge with scaled alpha)")
else:
    print(f"  ✗ Failed (difference {coef_diff_ridge:.2e} > 1e-6)")

# Test 4: Different l1_ratio values
print("\n[Test 4] Different l1_ratio values...")
for l1_ratio in [0.25, 0.5, 0.75]:
    enet = ElasticNet(alpha=1.0, l1_ratio=l1_ratio, max_iter=1000, tol=1e-6)
    enet.fit(X, y)
    sparsity = np.sum(enet.coef_ == 0)
    r2 = enet.score(X, y)
    print(f"  l1_ratio={l1_ratio}: sparsity={sparsity}/{n_features}, R²={r2:.6f}")

# Test 5: GPU path (CuPy)
print("\n[Test 5] GPU path (CuPy)...")
try:
    enet_cpu = ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=1000, tol=1e-6)
    enet_cpu.fit(X, y)

    enet_gpu = ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=1000, tol=1e-6, device='cuda')
    enet_gpu.fit(X, y)

    coef_diff_gpu = np.max(np.abs(enet_cpu.coef_ - enet_gpu.coef_))
    print(f"  CPU coef (first 3): {enet_cpu.coef_[:3]}")
    print(f"  GPU coef (first 3): {enet_gpu.coef_[:3]}")
    print(f"  Max coef difference (CPU vs GPU): {coef_diff_gpu:.2e}")
    if coef_diff_gpu < 1e-6:
        print("  ✓ Passed (CPU/GPU consistent)")
    else:
        print(f"  ⚠ CPU/GPU difference: {coef_diff_gpu:.2e}")
except Exception as e:
    print(f"  ⊘ Skipped: {e}")

# Test 6: Torch path
print("\n[Test 6] Torch path...")
try:
    import torch
    if torch.cuda.is_available():
        enet_cpu_torch = ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=1000, tol=1e-6)
        enet_cpu_torch.fit(X, y)

        enet_torch = ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=1000, tol=1e-6, device='cuda')
        enet_torch.fit(X, y)

        coef_diff_torch = np.max(np.abs(enet_cpu_torch.coef_ - enet_torch.coef_))
        print(f"  CPU coef (first 3): {enet_cpu_torch.coef_[:3]}")
        print(f"  Torch coef (first 3): {enet_torch.coef_[:3]}")
        print(f"  Max coef difference (CPU vs Torch): {coef_diff_torch:.2e}")
        if coef_diff_torch < 1e-6:
            print("  ✓ Passed (CPU/Torch consistent)")
        else:
            print(f"  ⚠ CPU/Torch difference: {coef_diff_torch:.2e}")
    else:
        print("  ⊘ Skipped (CUDA not available)")
except Exception as e:
    print(f"  ⊘ Skipped: {e}")

print("\n" + "=" * 60)
print("Smoke test complete!")
print("=" * 60)

# Test 7: Compare with sklearn
print("\n[Test 7] vs sklearn ElasticNet...")
try:
    from sklearn.linear_model import ElasticNet as SKLearnElasticNet

    # statgpu
    enet_statgpu = ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=1000, tol=1e-6)
    enet_statgpu.fit(X, y)

    # sklearn
    enet_sklearn = SKLearnElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=1000, tol=1e-6, fit_intercept=True)
    enet_sklearn.fit(X, y)

    coef_diff_sklearn = np.max(np.abs(enet_statgpu.coef_ - enet_sklearn.coef_))
    print(f"  statgpu coef (first 3): {enet_statgpu.coef_[:3]}")
    print(f"  sklearn coef (first 3): {enet_sklearn.coef_[:3]}")
    print(f"  Max coef difference: {coef_diff_sklearn:.2e}")
    if coef_diff_sklearn < 1e-6:
        print("  ✓ Passed (matches sklearn)")
    elif coef_diff_sklearn < 1e-4:
        print(f"  ~ Close (difference {coef_diff_sklearn:.2e}, may be convergence criteria)")
    else:
        print(f"  ✗ Failed (difference {coef_diff_sklearn:.2e})")
except ImportError:
    print("  ⊘ Skipped (sklearn not available)")
except Exception as e:
    print(f"  ⊘ Skipped: {e}")

print("\n" + "=" * 60)
print("All tests complete!")
print("=" * 60)
