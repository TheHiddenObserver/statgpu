"""
Benchmark GPU memory cleanup behavior across models.

Compares gpu_memory_cleanup=False vs True for:
- fit time (ms)
- CuPy memory pool used/total bytes after fit
- CuPy memory pool used/total bytes after explicit final cleanup
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# Ensure local repo imports when running `python dev/benchmarks/...`
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from statgpu._config import cuda_available, set_device
from statgpu.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from statgpu.survival import CoxPH


def _sync_cuda() -> None:
    import cupy as cp

    cp.cuda.runtime.deviceSynchronize()


def _pool_stats() -> Tuple[int, int]:
    import cupy as cp

    mp = cp.get_default_memory_pool()
    return int(mp.used_bytes()), int(mp.total_bytes())


def _reset_pools() -> None:
    import cupy as cp

    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    _sync_cuda()


def _fit_one(model_name: str, cleanup: bool, data: Dict[str, Any]) -> Dict[str, Any]:
    _reset_pools()
    t0 = time.perf_counter()

    if model_name == "linear":
        m = LinearRegression(device="cuda", compute_inference=True, gpu_memory_cleanup=cleanup)
        m.fit(data["Xg"], data["y_reg_g"])
    elif model_name == "ridge":
        m = Ridge(alpha=1.0, device="cuda", gpu_memory_cleanup=cleanup)
        m.fit(data["Xg"], data["y_reg_g"])
    elif model_name == "lasso":
        m = Lasso(
            alpha=0.1,
            device="cuda",
            solver="fista",
            cpu_solver="fista",
            stopping="kkt",
            tol=1e-4,
            max_iter=2000,
            compute_inference=True,
            inference_method="gpu_ols_inference",
            gpu_memory_cleanup=cleanup,
        )
        m.fit(data["Xg"], data["y_reg_g"])
    elif model_name == "logistic":
        m = LogisticRegression(
            device="cuda",
            max_iter=200,
            tol=1e-4,
            compute_inference=True,
            gpu_memory_cleanup=cleanup,
        )
        m.fit(data["Xg"], data["y_bin_g"])
    elif model_name == "cox":
        m = CoxPH(
            device="cuda",
            ties="breslow",
            tol=1e-7,
            max_iter=100,
            compute_inference=False,
            gpu_memory_cleanup=cleanup,
        )
        m.fit(data["Xg"], data["time_g"], data["event_g"])
    else:
        raise ValueError(f"Unknown model: {model_name}")

    _sync_cuda()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    used_after_fit, total_after_fit = _pool_stats()

    # Cleanup for normalized post-state inspection
    _reset_pools()
    used_after_reset, total_after_reset = _pool_stats()

    return {
        "model": model_name,
        "cleanup": cleanup,
        "fit_ms": float(elapsed_ms),
        "used_after_fit": used_after_fit,
        "total_after_fit": total_after_fit,
        "used_after_reset": used_after_reset,
        "total_after_reset": total_after_reset,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=20000)
    parser.add_argument("--n_features", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--models",
        type=str,
        default="linear,ridge,lasso,logistic,cox",
        help="Comma-separated subset of: linear,ridge,lasso,logistic,cox",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup_runs", type=int, default=1)
    args = parser.parse_args()

    if not cuda_available():
        raise RuntimeError("CUDA not available on this machine.")

    import cupy as cp

    set_device("cuda")
    rng = np.random.default_rng(args.seed)

    X = rng.normal(size=(args.n_samples, args.n_features)).astype(np.float64)
    beta = rng.normal(size=args.n_features)
    y_reg = X @ beta + rng.normal(scale=0.1, size=args.n_samples)

    logits = X @ beta * 0.2
    probs = 1.0 / (1.0 + np.exp(-logits))
    y_bin = (rng.uniform(size=args.n_samples) < probs).astype(np.float64)

    # Cox synthetic target
    base_hazard = 0.01
    risk = np.exp(X @ (beta * 0.1))
    u = np.clip(rng.uniform(size=args.n_samples), 1e-8, 1 - 1e-8)
    time_arr = -np.log(u) / (base_hazard * risk)
    censor = rng.exponential(scale=np.median(time_arr), size=args.n_samples)
    event = (time_arr <= censor).astype(np.int32)
    time_obs = np.minimum(time_arr, censor)

    data = {
        "Xg": cp.asarray(X),
        "y_reg_g": cp.asarray(y_reg),
        "y_bin_g": cp.asarray(y_bin),
        "time_g": cp.asarray(time_obs),
        "event_g": cp.asarray(event),
    }
    _sync_cuda()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    valid = {"linear", "ridge", "lasso", "logistic", "cox"}
    bad = [m for m in models if m not in valid]
    if bad:
        raise ValueError(f"Invalid models: {bad}; valid={sorted(valid)}")

    print("=" * 96)
    print(f"GPU memory cleanup benchmark | n={args.n_samples}, p={args.n_features}, repeats={args.repeats}")
    print("=" * 96)

    rows: List[Dict[str, Any]] = []
    for model_name in models:
        for cleanup in (False, True):
            # Warmup (not recorded)
            for _ in range(max(0, args.warmup_runs)):
                _fit_one(model_name=model_name, cleanup=cleanup, data=data)
            for _ in range(max(1, args.repeats)):
                rows.append(_fit_one(model_name=model_name, cleanup=cleanup, data=data))

    # Aggregate
    def _mean(k: str, model_name: str, cleanup: bool) -> float:
        vals = [r[k] for r in rows if r["model"] == model_name and r["cleanup"] == cleanup]
        return float(np.mean(vals)) if vals else float("nan")

    print(
        f"{'model':<12} {'cleanup':<8} {'fit_ms':>10} {'pool_used_fit':>14} "
        f"{'pool_total_fit':>14} {'pool_used_reset':>16} {'pool_total_reset':>17}"
    )
    print("-" * 96)

    for model_name in models:
        for cleanup in (False, True):
            print(
                f"{model_name:<12} {str(cleanup):<8} "
                f"{_mean('fit_ms', model_name, cleanup):>10.2f} "
                f"{int(_mean('used_after_fit', model_name, cleanup)):>14} "
                f"{int(_mean('total_after_fit', model_name, cleanup)):>14} "
                f"{int(_mean('used_after_reset', model_name, cleanup)):>16} "
                f"{int(_mean('total_after_reset', model_name, cleanup)):>17}"
            )

    print("-" * 96)
    print("Interpretation:")
    print("- pool_total_fit drops with cleanup=True => pooled VRAM is returned earlier.")
    print("- fit_ms may increase with cleanup=True due to more allocations next fit.")


if __name__ == "__main__":
    main()

