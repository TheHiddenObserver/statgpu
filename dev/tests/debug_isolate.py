# -*- coding: utf-8 -*-
"""Diagnostic script to isolate benchmark SCAD/MCP issue."""
import numpy as np
import cupy as cp
import time
from statgpu.linear_model._penalized_cv import PenalizedGLM_CV
from statgpu.linear_model._penalized import PenalizedGeneralizedLinearModel
from statgpu.backends import _to_numpy


def gen_regression(n, p, rng):
    X = rng.randn(n, p)
    y = X @ np.linspace(2, 0.5, p) + rng.randn(n) * 0.5
    return X, y


def bench(func, warmup=1, repeat=1):
    for _ in range(warmup):
        func()
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        result = func()
        times.append(time.perf_counter() - t0)
    return result, np.median(times)


N_ALPHAS = 5
N_FOLDS = 3

# === Test 1: SCAD alone ===
print("=== Test 1: SCAD alone ===")
rng = np.random.RandomState(42)
X_np, y_np = gen_regression(500, 20, rng)
X_cu = cp.asarray(X_np)
y_cu = cp.asarray(y_np)

cv_cu = PenalizedGLM_CV(loss="squared_error", penalty="scad", n_alphas=N_ALPHAS, cv=N_FOLDS, device="cuda")
cv_cu.fit(X_cu, y_cu)
cv_np = PenalizedGLM_CV(loss="squared_error", penalty="scad", n_alphas=N_ALPHAS, cv=N_FOLDS, device="cpu")
cv_np.fit(X_np, y_np)
corr = np.corrcoef(cv_np.coef_, _to_numpy(cv_cu.coef_))[0, 1]
print(f"  corr={corr:.6f}")

# === Test 2: l2 then scad ===
print("=== Test 2: l2 then scad ===")
rng = np.random.RandomState(42)
X_np, y_np = gen_regression(500, 20, rng)
X_cu = cp.asarray(X_np)
y_cu = cp.asarray(y_np)

for penalty in ["l2", "scad"]:
    cv_np = PenalizedGLM_CV(loss="squared_error", penalty=penalty, n_alphas=N_ALPHAS, cv=N_FOLDS, device="cpu")
    cv_cu = PenalizedGLM_CV(loss="squared_error", penalty=penalty, n_alphas=N_ALPHAS, cv=N_FOLDS, device="cuda")
    cv_np.fit(X_np, y_np)
    cv_cu.fit(X_cu, y_cu)
    corr = np.corrcoef(cv_np.coef_, _to_numpy(cv_cu.coef_))[0, 1]
    print(f"  {penalty}: corr={corr:.6f}")

# === Test 3: all penalties (like benchmark) ===
print("=== Test 3: all penalties ===")
rng = np.random.RandomState(42)
X_np, y_np = gen_regression(500, 20, rng)
X_cu = cp.asarray(X_np)
y_cu = cp.asarray(y_np)

for penalty in ["l2", "l1", "elasticnet", "scad", "mcp"]:
    cv_np = PenalizedGLM_CV(loss="squared_error", penalty=penalty, n_alphas=N_ALPHAS, cv=N_FOLDS, device="cpu")
    cv_cu = PenalizedGLM_CV(loss="squared_error", penalty=penalty, n_alphas=N_ALPHAS, cv=N_FOLDS, device="cuda")
    cv_np.fit(X_np, y_np)
    cv_cu.fit(X_cu, y_cu)
    corr = np.corrcoef(cv_np.coef_, _to_numpy(cv_cu.coef_))[0, 1]
    print(f"  {penalty}: corr={corr:.6f}")

# === Test 4: with bench() warmup (like benchmark) ===
print("=== Test 4: with bench warmup ===")
rng = np.random.RandomState(42)
X_np, y_np = gen_regression(500, 20, rng)
X_cu = cp.asarray(X_np)
y_cu = cp.asarray(y_np)

# Run bench for l2 first
_, t = bench(lambda: PenalizedGLM_CV(loss="squared_error", penalty="l2", n_alphas=N_ALPHAS, cv=N_FOLDS, device="cuda").fit(X_cu, y_cu))
print(f"  l2 bench done: {t*1000:.1f}ms")

# Then run scad
cv_cu = PenalizedGLM_CV(loss="squared_error", penalty="scad", n_alphas=N_ALPHAS, cv=N_FOLDS, device="cuda")
cv_cu.fit(X_cu, y_cu)
cv_np = PenalizedGLM_CV(loss="squared_error", penalty="scad", n_alphas=N_ALPHAS, cv=N_FOLDS, device="cpu")
cv_np.fit(X_np, y_np)
corr = np.corrcoef(cv_np.coef_, _to_numpy(cv_cu.coef_))[0, 1]
print(f"  scad after l2 bench: corr={corr:.6f}")

# === Test 5: with bench warmup for all penalties ===
print("=== Test 5: bench warmup all penalties ===")
rng = np.random.RandomState(42)
X_np, y_np = gen_regression(500, 20, rng)
X_cu = cp.asarray(X_np)
y_cu = cp.asarray(y_np)

for penalty in ["l2", "l1", "elasticnet", "scad", "mcp"]:
    _, t = bench(lambda p=penalty: PenalizedGLM_CV(loss="squared_error", penalty=p, n_alphas=N_ALPHAS, cv=N_FOLDS, device="cuda").fit(X_cu, y_cu))
    cv_np = PenalizedGLM_CV(loss="squared_error", penalty=penalty, n_alphas=N_ALPHAS, cv=N_FOLDS, device="cpu")
    cv_cu = PenalizedGLM_CV(loss="squared_error", penalty=penalty, n_alphas=N_ALPHAS, cv=N_FOLDS, device="cuda")
    cv_np.fit(X_np, y_np)
    cv_cu.fit(X_cu, y_cu)
    corr = np.corrcoef(cv_np.coef_, _to_numpy(cv_cu.coef_))[0, 1]
    print(f"  {penalty}: corr={corr:.6f} time={t*1000:.1f}ms")
