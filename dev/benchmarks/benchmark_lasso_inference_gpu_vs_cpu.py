"""Benchmark canonical post-selection OLS across CPU and CuPy.

This script measures the complete Lasso fit + ``post_selection_ols`` inference
transaction on NumPy CPU and CuPy CUDA and reports numerical parity.  It is not
an inference-only benchmark: PR #138 intentionally removed hardware identity
from ``inference_method``, so there is no longer a meaningful
``cpu_ols_inference`` versus ``gpu_ols_inference`` method comparison.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Dict, Tuple

import numpy as np

# Ensure local repo imports when running `python dev/benchmarks/...`
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from statgpu._config import cuda_available, set_device
from statgpu.linear_model import Lasso


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
    X,
    y,
    device: str,
    lipschitz_L: float,
    args: argparse.Namespace,
) -> Tuple[Lasso, float]:
    set_device(device)
    model = Lasso(
        alpha=args.alpha,
        fit_intercept=True,
        max_iter=args.max_iter,
        tol=args.tol,
        stopping=args.stopping,
        inference_method="post_selection_ols",
        device=device,
        solver=args.solver,
        lipschitz_L=lipschitz_L,
        compute_inference=True,
        admm_rho=args.admm_rho,
    )

    if device == "cuda":
        sync_cuda()
    t0 = time.perf_counter()
    model.fit(X, y)
    if device == "cuda":
        sync_cuda()
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return model, elapsed_ms


def _result_snapshot(model: Lasso, times) -> Dict:
    result = model._inference_result
    if result is None or result.method != "post_selection_ols":
        raise RuntimeError("post_selection_ols inference result is missing")
    return {
        "time_ms_mean": float(np.mean(times)),
        "time_ms": list(times),
        "n_iter": int(model.n_iter_),
        "coef": np.asarray(model.coef_, dtype=np.float64).copy(),
        "intercept": float(model.intercept_),
        "params": np.asarray(model._params, dtype=np.float64).copy(),
        "bse": np.asarray(model._bse, dtype=np.float64).copy(),
        "tvalues": np.asarray(model._tvalues, dtype=np.float64).copy(),
        "pvalues": np.asarray(model._pvalues, dtype=np.float64).copy(),
        "conf_int": np.asarray(model._conf_int, dtype=np.float64).copy(),
        "selected_feature_indices": list(
            result.metadata.get("selected_feature_indices", [])
        ),
        "numerical_backend": result.metadata.get("numerical_backend"),
        "numerical_device": result.metadata.get("numerical_device"),
    }


def _max_error(left, right) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.shape != b.shape:
        raise RuntimeError(f"shape mismatch in parity check: {a.shape} != {b.shape}")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise RuntimeError("non-finite value encountered in parity check")
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=200000)
    parser.add_argument("--n_features", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--max_iter", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--solver", type=str, default="fista", choices=["fista", "admm"])
    parser.add_argument("--stopping", type=str, default="kkt", choices=["kkt", "coef_delta"])
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
    print("Lasso post_selection_ols benchmark (CPU vs CuPy, end-to-end)")
    print("=" * 80)
    print(
        f"Data: {args.n_samples} x {args.n_features} | alpha={args.alpha} | "
        f"tol={args.tol} | max_iter={args.max_iter}"
    )
    print(
        f"solver={args.solver} | stopping={args.stopping} | "
        f"lipschitz_L={lipschitz_L:.6g}"
    )

    import cupy as cp

    set_device("cuda")
    X_gpu = cp.asarray(X)
    y_gpu = cp.asarray(y)
    sync_cuda()

    cases = {
        "cpu": (X, y),
        "cuda": (X_gpu, y_gpu),
    }
    results: Dict[str, Dict] = {}

    for device, (X_case, y_case) in cases.items():
        for _ in range(max(0, args.warmup_runs)):
            warm_model, _ = run_once(
                X=X_case,
                y=y_case,
                device=device,
                lipschitz_L=lipschitz_L,
                args=args,
            )
            del warm_model

        times = []
        last = None
        for _ in range(max(1, args.repeats)):
            last, elapsed_ms = run_once(
                X=X_case,
                y=y_case,
                device=device,
                lipschitz_L=lipschitz_L,
                args=args,
            )
            times.append(elapsed_ms)

        results[device] = _result_snapshot(last, times)
        print("-" * 80)
        print(
            f"{device}: time_ms_mean={results[device]['time_ms_mean']:.2f} | "
            f"n_iter={results[device]['n_iter']} | "
            f"backend={results[device]['numerical_backend']} | "
            f"numerical_device={results[device]['numerical_device']}"
        )

    cpu = results["cpu"]
    cuda = results["cuda"]
    print("-" * 80)
    print("Numerical parity (CPU vs CuPy post_selection_ols):")
    print(f"  coef L_inf diff: {_max_error(cpu['coef'], cuda['coef']):.3e}")
    print(f"  intercept abs diff: {abs(cpu['intercept'] - cuda['intercept']):.3e}")
    print(f"  params L_inf diff: {_max_error(cpu['params'], cuda['params']):.3e}")
    print(f"  bse L_inf diff: {_max_error(cpu['bse'], cuda['bse']):.3e}")
    print(f"  tvalues L_inf diff: {_max_error(cpu['tvalues'], cuda['tvalues']):.3e}")
    print(f"  pvalues L_inf diff: {_max_error(cpu['pvalues'], cuda['pvalues']):.3e}")
    print(f"  conf_int L_inf diff: {_max_error(cpu['conf_int'], cuda['conf_int']):.3e}")
    print(f"  CPU selected: {cpu['selected_feature_indices']}")
    print(f"  CUDA selected: {cuda['selected_feature_indices']}")
    print(
        "Timing scope is the complete penalized fit plus post-selection inference; "
        "it is not an inference-only speedup claim."
    )


if __name__ == "__main__":
    main()
