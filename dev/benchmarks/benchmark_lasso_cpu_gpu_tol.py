"""
Lasso CPU vs GPU benchmark at the same tol.

This script compares CPU vs GPU at the same `alpha/tol/max_iter`.

You can optionally pre-construct GPU data outside `fit()` to avoid host<->device
roundtrips during timing (torch-like workflow).
"""

from __future__ import annotations

import argparse
import time
from typing import Optional, Tuple

import numpy as np

# Make local `statgpu` import work even when the package isn't installed.
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _maybe_synchronize_cuda() -> None:
    """Synchronize GPU so timing is closer to wall time."""
    try:
        import cupy as cp  # noqa: F401

        # Device 0 is the common default; if user's setup uses other devices,
        # CuPy will respect the current device context.
        cp.cuda.runtime.deviceSynchronize()
    except Exception:
        # If CuPy isn't available or sync fails, just continue.
        pass


def _generate_data(
    n_samples: int,
    n_features: int,
    noise: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    true_coef = rng.normal(size=n_features) * 2.0
    true_intercept = 5.0
    y = X @ true_coef + true_intercept + rng.normal(scale=noise, size=n_samples)
    return X, y


def _fit_lasso(
    X,
    y,
    *,
    device: str,
    alpha: float,
    tol: float,
    max_iter: int,
    solver: str,
    cpu_solver: Optional[str],
    lipschitz_L: Optional[float],
    compute_inference: bool,
    warmup_runs: int,
    repeats: int,
) -> dict:
    # Local import: allow running even if GPU deps are absent.
    from statgpu.linear_model import Lasso

    # Ensure consistent device selection for the model's internal logic.
    from statgpu._config import set_device

    set_device(device)

    # Note: CPU implementation uses `cpu_solver` (not `solver`).
    model = Lasso(
        alpha=alpha,
        fit_intercept=True,
        max_iter=max_iter,
        tol=tol,
        device=device,
        solver=solver,
        cpu_solver=cpu_solver or "coordinate_descent",
        compute_inference=compute_inference,
        lipschitz_L=lipschitz_L,
    )

    warmup_runs = max(0, int(warmup_runs))
    repeats = max(1, int(repeats))

    # Warmup runs: helps remove first-call cuBLAS/CuPy overhead (esp. on GPU).
    for _ in range(warmup_runs):
        _maybe_synchronize_cuda()
        model.fit(X, y)
        _maybe_synchronize_cuda()

    times_ms = []
    for _ in range(repeats):
        _maybe_synchronize_cuda()
        t0 = time.perf_counter()
        model.fit(X, y)
        _maybe_synchronize_cuda()
        times_ms.append((time.perf_counter() - t0) * 1000)

    elapsed_ms = float(np.mean(times_ms)) if times_ms else float("nan")

    return {
        "elapsed_ms": elapsed_ms,
        "n_iter": int(model.n_iter_) if model.n_iter_ is not None else None,
        "coef": np.asarray(model.coef_).copy(),
        "intercept": float(model.intercept_),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=5000)
    parser.add_argument("--n_features", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--max_iter", type=int, default=2000)
    parser.add_argument("--noise", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--gpu_data_mode",
        type=str,
        default="transfer_excluded",
        choices=["transfer_excluded", "gpu_construct"],
        help="How to prepare GPU data before calling fit().",
    )
    parser.add_argument(
        "--cpu_solver",
        type=str,
        default="coordinate_descent",
        choices=["coordinate_descent", "fista"],
        help="CPU optimization algorithm.",
    )
    parser.add_argument("--solver_gpu", type=str, default="fista", choices=["fista", "admm"])
    parser.add_argument(
        "--lipschitz_L",
        type=float,
        default=None,
        help="Optional Lipschitz constant override for GPU FISTA. If provided, GPU skips eigvalsh().",
    )
    parser.add_argument(
        "--compute_inference",
        action="store_true",
        help="Whether to compute full inference statistics (can add noticeable CPU time).",
    )
    parser.add_argument(
        "--gpu_warmup_runs",
        type=int,
        default=1,
        help="Number of GPU warmup fits (discarded). Removes one-time cuBLAS/CuPy overhead.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of measured fits per device (averaged).",
    )
    args = parser.parse_args()

    X, y = _generate_data(
        n_samples=args.n_samples,
        n_features=args.n_features,
        noise=args.noise,
        seed=args.seed,
    )

    from statgpu._config import cuda_available

    has_gpu = cuda_available()
    print("=" * 80)
    print("Lasso CPU vs GPU (same tol)")
    print("=" * 80)
    print(f"Data: {args.n_samples} x {args.n_features} | noise={args.noise} | seed={args.seed}")
    print(f"Params: alpha={args.alpha} | tol={args.tol} | max_iter={args.max_iter}")
    print(f"GPU solver: {args.solver_gpu}")
    print(f"compute_inference: {args.compute_inference}")
    print(f"GPU available: {has_gpu}")

    # CPU
    res_cpu = _fit_lasso(
        X,
        y,
        device="cpu",
        alpha=args.alpha,
        tol=args.tol,
        max_iter=args.max_iter,
        solver=args.solver_gpu,
        cpu_solver=args.cpu_solver,
        lipschitz_L=args.lipschitz_L,
        compute_inference=args.compute_inference,
        warmup_runs=0,
        repeats=args.repeats,
    )

    print("-" * 80)
    print("CPU result")
    print(f"  time_ms:  {res_cpu['elapsed_ms']:.2f}")
    print(f"  n_iter:   {res_cpu['n_iter']}")
    print(f"  intercept:{res_cpu['intercept']:.6f}")

    if not has_gpu:
        print("-" * 80)
        print("GPU skipped (no CUDA available).")
        return

    # GPU
    # Note: when gpu_data_mode='transfer_excluded', we transfer once outside `fit()`
    # and pass CuPy arrays in, so BaseEstimator won't bounce data back to CPU.
    if args.gpu_data_mode == "gpu_construct":
        import cupy as cp

        rng = cp.random.RandomState(args.seed)
        X_gpu = rng.normal(size=(args.n_samples, args.n_features))
        true_coef_gpu = rng.normal(size=args.n_features) * 2.0
        true_intercept_gpu = 5.0
        y_gpu = X_gpu @ true_coef_gpu + true_intercept_gpu + rng.normal(
            scale=args.noise, size=args.n_samples
        )

        # Warm once to reduce first-kernel noise.
        _ = X_gpu @ (cp.zeros(args.n_features))
        cp.cuda.runtime.deviceSynchronize()
    else:
        import cupy as cp

        # Transfer once outside fit() so timing is compute-only-ish.
        X_gpu = cp.asarray(X)
        y_gpu = cp.asarray(y)
        cp.cuda.runtime.deviceSynchronize()

    res_gpu = _fit_lasso(
        X_gpu,
        y_gpu,
        device="cuda",
        alpha=args.alpha,
        tol=args.tol,
        max_iter=args.max_iter,
        solver=args.solver_gpu,
        cpu_solver=args.cpu_solver,
        lipschitz_L=args.lipschitz_L,
        compute_inference=args.compute_inference,
        warmup_runs=args.gpu_warmup_runs,
        repeats=args.repeats,
    )

    diff = res_cpu["coef"] - res_gpu["coef"]
    l_inf = float(np.max(np.abs(diff)))
    l2_rel = float(np.linalg.norm(diff) / (np.linalg.norm(res_cpu["coef"]) + 1e-30))
    intercept_diff = abs(res_cpu["intercept"] - res_gpu["intercept"])

    print("-" * 80)
    print("GPU result")
    print(f"  time_ms:  {res_gpu['elapsed_ms']:.2f}")
    print(f"  n_iter:   {res_gpu['n_iter']}")
    print(f"  intercept:{res_gpu['intercept']:.6f}")

    speedup = res_cpu["elapsed_ms"] / max(res_gpu["elapsed_ms"], 1e-30)
    print("-" * 80)
    print("Comparison (CPU vs GPU)")
    if args.gpu_data_mode == "gpu_construct":
        print("  Note: gpu_data_mode='gpu_construct' uses different CPU/GPU datasets; "
              "coef diffs are not a strict same-data comparison.")
    print(f"  speedup (CPU/GPU): {speedup:.2f}x")
    print(f"  coef L_inf: {l_inf:.3e}")
    print(f"  coef L2_rel: {l2_rel:.3e}")
    print(f"  intercept abs diff: {intercept_diff:.3e}")

    # Non-zero counts (roughly reflects sparsity).
    nz_cpu = int(np.sum(np.abs(res_cpu["coef"]) > 1e-10))
    nz_gpu = int(np.sum(np.abs(res_gpu["coef"]) > 1e-10))
    print(f"  nonzeros: CPU={nz_cpu} | GPU={nz_gpu}")


if __name__ == "__main__":
    main()

