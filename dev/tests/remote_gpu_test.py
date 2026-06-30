#!/usr/bin/env python3
"""GPU tests for proximal IRLS + DBSCAN + UMAP on remote server."""
import sys
sys.path.insert(0, '/root/statgpu_pkg')
import torch as t
import numpy as np
import cupy as cp

print("torch", t.__version__, "CUDA", t.cuda.is_available())
print("cupy", cp.__version__)
print()

# --- Quantile + SCAD (torch CUDA) ---
print("=== Quantile + SCAD (torch CUDA) ===")
from statgpu.solvers._proximal_irls_quantile import proximal_irls_quantile_solver
from statgpu.losses._quantile import QuantileLoss
from statgpu.penalties._scad import SCADPenalty

np.random.seed(42)
n, p = 500, 20
X_np = np.random.randn(n, p).astype(np.float64)
beta_true = np.zeros(p)
beta_true[0] = 3.0; beta_true[5] = -2.5; beta_true[10] = 1.5; beta_true[15] = -1.0
y_np = (X_np @ beta_true + np.random.randn(n) * 0.5).astype(np.float64)

X_tc = t.tensor(X_np, dtype=t.float64).cuda()
y_tc = t.tensor(y_np, dtype=t.float64).cuda()
X_cp = cp.asarray(X_np)
y_cp = cp.asarray(y_np)

loss = QuantileLoss(0.5)
penalty = SCADPenalty(alpha=0.1)
_col_norms = np.sqrt(np.sum(X_np**2, axis=0)) + 1e-20
_X_s = X_np * (np.sqrt(n) / _col_norms)
_y_c = y_np - np.mean(y_np)
_lam_max = float(np.max(np.abs(_X_s.T @ _y_c / n)))
alpha_path = np.geomspace(max(_lam_max, 0.11), 0.1, 3)

for name, X, y in [("torch CUDA", X_tc, y_tc), ("cupy CUDA", X_cp, y_cp)]:
    coef, intercept, iters = proximal_irls_quantile_solver(
        loss, penalty, X, y, alpha_path=alpha_path,
        max_lla_per_step=2, max_iter=200, tol=1e-6, fit_intercept=True)
    active = sorted(np.where(np.abs(coef) > 0.05)[0])
    ok = "OK" if active == [0, 5, 10, 15] else "FAIL"
    print(f"  [{ok}] {name}: iters={iters}, active={active}")

# --- Weighted sample (torch CUDA) ---
print()
print("=== Weighted Sample (torch CUDA) ===")
sw = np.ones(n); sw[:100] = 2.0
coef, intercept, iters = proximal_irls_quantile_solver(
    loss, penalty, X_tc, y_tc, alpha_path=alpha_path,
    max_lla_per_step=2, max_iter=200, tol=1e-6, fit_intercept=True,
    sample_weight=sw)
active = sorted(np.where(np.abs(coef) > 0.05)[0])
ok = "OK" if active == [0, 5, 10, 15] else "FAIL"
print(f"  [{ok}] weighted torch CUDA: iters={iters}, active={active}")

# --- DBSCAN CuPy ---
print()
print("=== DBSCAN (cupy) ===")
from statgpu.unsupervised._dbscan import DBSCAN
X_db = np.random.randn(200, 2).astype(np.float64) * 3
X_db[50:100] += [10, 0]
X_cp_db = cp.asarray(X_db)
db = DBSCAN(eps=0.5, min_samples=5)
db.fit(X_cp_db)
n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
n_noise = np.sum(db.labels_ == -1)
print(f"  CuPy DBSCAN: {n_clusters} clusters, {n_noise} noise")

# --- DBSCAN Torch CUDA ---
print()
print("=== DBSCAN (torch CUDA) ===")
X_tc_db = t.tensor(X_db, dtype=t.float64).cuda()
db2 = DBSCAN(eps=0.5, min_samples=5)
db2.fit(X_tc_db)
n_clusters2 = len(set(db2.labels_)) - (1 if -1 in db2.labels_ else 0)
print(f"  Torch CUDA DBSCAN: {n_clusters2} clusters")

# --- UMAP negative sampling ---
print()
print("=== UMAP (torch CUDA) ===")
from statgpu.unsupervised._umap import UMAP
X_umap = np.random.randn(100, 5).astype(np.float32)
X_tc_umap = t.tensor(X_umap, dtype=t.float64).cuda()
umap = UMAP(n_neighbors=5, n_epochs=2, device='cuda')
umap.fit(X_tc_umap)
emb = umap.embedding_
print(f"  UMAP torch CUDA: embedding shape={emb.shape}")

print()
print("=== ALL GPU TESTS PASSED ===")
