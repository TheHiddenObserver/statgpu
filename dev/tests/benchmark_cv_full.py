# -*- coding: utf-8 -*-
"""Full CV benchmark: statgpu PenalizedGLM_CV across all loss x penalty combos."""
import time
import warnings
import numpy as np
import subprocess
import sys

warnings.filterwarnings("ignore")


def _to_numpy(arr):
    if hasattr(arr, "get"):
        return arr.get()
    if hasattr(arr, "cpu"):
        return arr.cpu().numpy()
    return np.asarray(arr)


def coef_corr(a, b):
    a, b = _to_numpy(a).ravel(), _to_numpy(b).ravel()
    if np.std(a) < 1e-15 or np.std(b) < 1e-15:
        return 1.0 if np.allclose(a, b, atol=1e-10) else 0.0
    return float(np.corrcoef(a, b)[0, 1])


def coef_l2(a, b):
    return float(np.linalg.norm(_to_numpy(a).ravel() - _to_numpy(b).ravel()))


def bench(func, warmup=1, repeat=3):
    for _ in range(warmup):
        func()
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        result = func()
        times.append(time.perf_counter() - t0)
    return result, np.median(times)


print("=" * 90)
print("FULL CV BENCHMARK: PenalizedGLM_CV — all loss x penalty combos")
print("=" * 90)

import cupy as cp
import torch
from sklearn.metrics import r2_score, accuracy_score
from statgpu.linear_model._penalized_cv import PenalizedGLM_CV
from statgpu.linear_model._penalized import PenalizedGeneralizedLinearModel

gpu_name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
print(f"GPU: {gpu_name}")
print(f"CuPy {cp.__version__} | Torch {torch.__version__} | NumPy {np.__version__}")

# Data generators
def gen_regression(n, p, rng):
    X = rng.randn(n, p)
    y = X @ np.linspace(2, 0.5, p) + rng.randn(n) * 0.5
    return X, y

def gen_classification(n, p, rng):
    X = rng.randn(n, p)
    y = (X @ np.linspace(2, 0.5, p) + rng.randn(n) > 0).astype(float)
    return X, y

def gen_count(n, p, rng):
    X = rng.randn(n, p)
    mu = np.exp(X @ np.linspace(0.5, 0.1, p))
    y = rng.poisson(mu).astype(float)
    y = np.maximum(y, 1.0)
    return X, y

def gen_positive(n, p, rng):
    X = rng.randn(n, p)
    y = np.abs(X @ np.linspace(2, 0.5, p)) + 0.1
    return X, y

LOSSES = ["squared_error", "logistic", "poisson", "gamma"]
PENALTIES = ["l2", "l1", "elasticnet", "scad", "mcp"]
N_ALPHAS = 20

# ─── Test each loss x penalty ─────────────────────────────────────────
for loss in LOSSES:
    print(f"\n{'=' * 90}")
    print(f"  LOSS: {loss}")
    print(f"{'=' * 90}")

    rng = np.random.RandomState(42)
    n, p = 500, 20

    if loss == "squared_error":
        X_np, y_np = gen_regression(n, p, rng)
    elif loss == "logistic":
        X_np, y_np = gen_classification(n, p, rng)
    elif loss == "poisson":
        X_np, y_np = gen_count(n, p, rng)
    elif loss == "gamma":
        X_np, y_np = gen_positive(n, p, rng)
    else:
        continue

    X_cu = cp.asarray(X_np)
    y_cu = cp.asarray(y_np)
    X_to = torch.tensor(X_np, dtype=torch.float64, device="cuda")
    y_to = torch.tensor(y_np, dtype=torch.float64, device="cuda")

    for penalty in PENALTIES:
        if loss == "logistic" and penalty == "elasticnet":
            l1_ratio = 0.5
        elif penalty == "elasticnet":
            l1_ratio = 0.5
        else:
            l1_ratio = 0.5

        # Fit on numpy
        try:
            cv_np = PenalizedGLM_CV(
                loss=loss, penalty=penalty, n_alphas=N_ALPHAS,
                l1_ratio=l1_ratio, cv=3, device="cpu",
            )
            _, t_np = bench(lambda: PenalizedGLM_CV(
                loss=loss, penalty=penalty, n_alphas=N_ALPHAS,
                l1_ratio=l1_ratio, cv=3, device="cpu",
            ).fit(X_np, y_np))
            cv_np.fit(X_np, y_np)
            coef_np = cv_np.coef_
            alpha_np = cv_np.alpha_
        except Exception as e:
            print(f"  {penalty:15s} numpy: ERROR - {e}")
            continue

        # Fit on cupy
        try:
            cv_cu = PenalizedGLM_CV(
                loss=loss, penalty=penalty, n_alphas=N_ALPHAS,
                l1_ratio=l1_ratio, cv=3, device="cuda",
            )
            _, t_cu = bench(lambda: PenalizedGLM_CV(
                loss=loss, penalty=penalty, n_alphas=N_ALPHAS,
                l1_ratio=l1_ratio, cv=3, device="cuda",
            ).fit(X_cu, y_cu))
            cv_cu.fit(X_cu, y_cu)
            coef_cu = _to_numpy(cv_cu.coef_)
        except Exception as e:
            print(f"  {penalty:15s} cupy:  ERROR - {e}")
            continue

        # Fit on torch
        try:
            cv_to = PenalizedGLM_CV(
                loss=loss, penalty=penalty, n_alphas=N_ALPHAS,
                l1_ratio=l1_ratio, cv=3, device="torch",
            )
            _, t_to = bench(lambda: PenalizedGLM_CV(
                loss=loss, penalty=penalty, n_alphas=N_ALPHAS,
                l1_ratio=l1_ratio, cv=3, device="torch",
            ).fit(X_to, y_to))
            cv_to.fit(X_to, y_to)
            coef_to = _to_numpy(cv_to.coef_)
        except Exception as e:
            print(f"  {penalty:15s} torch: ERROR - {e}")
            continue

        # Compare
        corr_cu = coef_corr(coef_np, coef_cu)
        corr_to = coef_corr(coef_np, coef_to)
        l2_cu = coef_l2(coef_np, coef_cu)
        l2_to = coef_l2(coef_np, coef_to)

        print(f"  {penalty:15s} corr_cu={corr_cu:.6f} corr_to={corr_to:.6f} "
              f"L2_cu={l2_cu:.2e} L2_to={l2_to:.2e} "
              f"time: np={t_np*1000:.1f}ms cu={t_cu*1000:.1f}ms to={t_to*1000:.1f}ms")

print("\n" + "=" * 90)
print("DONE")
print("=" * 90)
