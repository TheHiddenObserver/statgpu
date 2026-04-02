"""
Benchmark Lasso inference on GPU:

Compare inference_method='cpu_ols_inference' (CPU-sided t-distribution inference)
vs inference_method='gpu_ols_inference' (GPU-sided inference, avoid residual/design transfer).
"""

from __future__ import annotations

import argparse
import time
from typing import Dict, Tuple
import sys
from pathlib import Path

import numpy as np

# Ensure local repo imports when running `python examples/...`
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from statgpu.linear_model import Lasso
from statgpu._config import cuda_available, set_device


def exact_lipschitz_L(X: np.ndarray) -> float:
    # For intercept=True, FISTA uses centered X:
    #   L = lambda_max(Xc^T Xc) / n
    n_samples = X.shape[0]
    Xc = X - X.mean(axis=0)
    XtX = Xc.T @ Xc
    w = np.linalg.eigvalsh(XtX)
    return float(w[-1] / n_samples)


def sync_cuda() -> None:
    import cupy as cp

    cp.cuda.runtime.deviceSynchronize()


def run_once(
    *,
    X_gpu,
    y_gpu,
    lipschitz_L: float,
    inference_method: str,
    args: argparse.Namespace,
) -> Tuple[Lasso, float]:
    set_device("cuda")

    model = Lasso(
        alpha=args.alpha,
        fit_intercept=True,
        max_iter=args.max_iter,
        tol=args.tol,
        stopping=args.stopping,
        inference_method=inference_method,
        n_bootstrap=args.n_bootstrap,
        bootstrap_random_state=args.bootstrap_random_state,
        device="cuda",
        solver=args.solver_gpu,
        cpu_solver=args.cpu_solver,
        lipschitz_L=lipschitz_L,
        compute_inference=True,
        admm_rho=args.admm_rho,
    )

    t0 = time.perf_counter()
    model.fit(X_gpu, y_gpu)
    sync_cuda()
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return model, elapsed_ms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=200000)
    parser.add_argument("--n_features", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--max_iter", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--solver_gpu", type=str, default="fista", choices=["fista", "admm"])
    parser.add_argument("--cpu_solver", type=str, default="fista", choices=["coordinate_descent", "fista"])
    parser.add_argument("--stopping", type=str, default="kkt", choices=["kkt", "coef_delta"])

    parser.add_argument("--n_bootstrap", type=int, default=200)
    parser.add_argument("--bootstrap_random_state", type=int, default=0)
    parser.add_argument("--admm_rho", type=float, default=1.0)

    parser.add_argument("--warmup_runs", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()

    if not cuda_available():
        raise RuntimeError("CUDA not available on this machine.")

    rng = np.random.default_rng(args.seed)
    X = rng.normal(size=(args.n_samples, args.n_features)).astype(np.float64)
    beta = rng.normal(size=args.n_features) * 2.0
    y = X @ beta + 5.0 + rng.normal(scale=0.1, size=args.n_samples)

    lipschitz_L = exact_lipschitz_L(X)
    print("=" * 80)
    print("Lasso inference benchmark (GPU SE computation)")
    print("=" * 80)
    print(f"Data: {args.n_samples} x {args.n_features} | alpha={args.alpha} | tol={args.tol} | max_iter={args.max_iter}")
    print(f"solver_gpu={args.solver_gpu} | stopping={args.stopping} | lipschitz_L={lipschitz_L:.6g}")

    methods = ["cpu_ols_inference", "gpu_ols_inference"]
    results: Dict[str, Dict] = {}

    import cupy as cp
    set_device("cuda")
    X_gpu = cp.asarray(X)
    y_gpu = cp.asarray(y)
    sync_cuda()

    for method in methods:
        # warmup
        for _ in range(max(0, args.warmup_runs)):
            m, _ = run_once(
                X_gpu=X_gpu,
                y_gpu=y_gpu,
                lipschitz_L=lipschitz_L,
                inference_method=method,
                args=args,
            )
            del m

        times = []
        m_last = None
        for _ in range(max(1, args.repeats)):
            m, elapsed_ms = run_once(
                X_gpu=X_gpu,
                y_gpu=y_gpu,
                lipschitz_L=lipschitz_L,
                inference_method=method,
                args=args,
            )
            times.append(elapsed_ms)
            m_last = m

        bse = m_last._bse
        tvalues = m_last._tvalues
        pvalues = m_last._pvalues
        conf_int = m_last._conf_int

        results[method] = {
            "time_ms_mean": float(np.mean(times)),
            "time_ms": times,
            "n_iter": int(m_last.n_iter_),
            "coef_inf": m_last.coef_.copy(),
            "intercept_inf": float(m_last.intercept_),
            "bse_head": bse[:5].copy(),
            "tvalues_head": tvalues[:5].copy(),
            "pvalues_head": pvalues[:5].copy(),
            "conf_int_head": conf_int[:3].copy(),
            "bse": bse,
            "tvalues": tvalues,
            "pvalues": pvalues,
            "conf_int": conf_int,
        }

        print("-" * 80)
        print(f"{method}: time_ms_mean={results[method]['time_ms_mean']:.2f} | n_iter={results[method]['n_iter']}")

    # Accuracy check between methods
    m1 = results["cpu_ols_inference"]
    m2 = results["gpu_ols_inference"]
    coef_diff = float(np.max(np.abs(m1["coef_inf"] - m2["coef_inf"])))
    intercept_diff = abs(m1["intercept_inf"] - m2["intercept_inf"])
    bse_diff = float(np.max(np.abs(m1["bse"] - m2["bse"])))
    tvalues_diff = float(np.max(np.abs(m1["tvalues"] - m2["tvalues"])))
    pvalues_diff = float(np.max(np.abs(m1["pvalues"] - m2["pvalues"])))
    conf_int_diff = float(np.max(np.abs(m1["conf_int"] - m2["conf_int"])))
    print("-" * 80)
    print("Accuracy check (cpu_ols_inference vs gpu_ols_inference):")
    print(f"  coef L_inf diff: {coef_diff:.3e}")
    print(f"  intercept abs diff: {intercept_diff:.3e}")
    print(f"  bse L_inf diff: {bse_diff:.3e}")
    print(f"  tvalues L_inf diff: {tvalues_diff:.3e}")
    print(f"  pvalues L_inf diff: {pvalues_diff:.3e}")
    print(f"  conf_int L_inf diff: {conf_int_diff:.3e}")


if __name__ == "__main__":
    main()

