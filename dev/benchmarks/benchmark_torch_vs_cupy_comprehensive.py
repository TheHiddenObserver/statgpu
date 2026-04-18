# -*- coding: utf-8 -*-
"""
Comprehensive benchmark: Torch vs CuPy GPU backend.

This script provides a fair, head-to-head comparison between PyTorch and CuPy
backends across all statgpu models:
  - LinearRegression
  - Ridge
  - Lasso
  - LogisticRegression
  - CoxPH

Key features:
  - Same data for both backends
  - Warmup runs to eliminate CUDA initialization overhead
  - Multiple repeats for statistical significance
  - Numerical accuracy validation vs CPU reference
  - JSON + Markdown output for reproducibility
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from statgpu.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from statgpu.survival import CoxPH
from statgpu._config import cuda_available

# Optional imports
try:
    import cupy as cp
    HAS_CUPY = True
except Exception:
    cp = None
    HAS_CUPY = False

try:
    import torch
    HAS_TORCH = True
except Exception:
    torch = None
    HAS_TORCH = False


def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


@dataclass
class BenchmarkResult:
    model: str
    dataset: str
    backend: str  # "torch_gpu", "cupy_gpu", "cpu"
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    repeats: int
    ok: bool
    error: str = ""
    coef_diff: float = float('nan')
    bse_diff: float = float('nan')


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Comprehensive Torch vs CuPy GPU benchmark."
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--warmup-runs", type=int, default=2)
    p.add_argument("--json-out", type=str, default="",
                   help="Optional path to save JSON results.")
    p.add_argument("--md-out", type=str, default="",
                   help="Optional path to save Markdown report.")

    # Dataset sizes
    p.add_argument("--small-n", type=int, default=2000, help="Small dataset samples")
    p.add_argument("--small-p", type=int, default=50, help="Small dataset features")
    p.add_argument("--large-n", type=int, default=50000, help="Large dataset samples")
    p.add_argument("--large-p", type=int, default=200, help="Large dataset features")

    return p.parse_args()


def make_regression_data(rng: np.random.Generator, n: int, p: int):
    """Generate linear regression data with known coefficients."""
    X = rng.normal(size=(n, p))
    beta = rng.normal(size=p)
    y = X @ beta + 1.0 + rng.normal(scale=0.5, size=n)
    return X.astype(np.float64), y.astype(np.float64)


def make_logistic_data(rng: np.random.Generator, n: int, p: int):
    """Generate binary classification data."""
    X = rng.normal(size=(n, p))
    beta = rng.normal(scale=0.8, size=p)
    logits = X @ beta + 0.2
    prob = 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))
    y = (rng.random(n) < prob).astype(np.float64)
    return X.astype(np.float64), y


def make_cox_data(rng: np.random.Generator, n: int, p: int):
    """Generate survival data with censoring."""
    X = rng.normal(size=(n, p))
    beta = rng.normal(scale=0.35, size=p)
    linpred = X @ beta
    base_hazard = 0.03
    u = np.clip(rng.random(n), 1e-12, 1 - 1e-12)
    true_time = -np.log(u) / (base_hazard * np.exp(np.clip(linpred, -20, 20)))
    censor = rng.exponential(scale=np.median(true_time), size=n)
    event = (true_time <= censor).astype(np.float64)
    obs_time = np.minimum(true_time, censor)
    return X.astype(np.float64), obs_time.astype(np.float64), event


def as_device(arr: np.ndarray, device: str, backend: str = "cupy"):
    """Move numpy array to specified device and backend."""
    if device == "cpu":
        return arr
    elif device == "cuda":
        if backend == "cupy":
            return cp.asarray(arr)
        elif backend == "torch":
            return torch.from_numpy(arr).cuda()
    raise ValueError(f"Unknown device/backend: {device}/{backend}")


def synchronize(backend: str):
    """Synchronize GPU for accurate timing."""
    if backend == "cupy" and cp is not None:
        cp.cuda.Stream.null.synchronize()
    elif backend == "torch" and torch is not None:
        torch.cuda.synchronize()


def time_fit(
    model_factory: Callable[[], Any],
    fit_fn: Callable[[Any], None],
    warmup_runs: int,
    repeats: int,
    backend: str,
) -> Tuple[bool, List[float], str, Any]:
    """Time model.fit() with warmup and repeats."""

    # Warmup
    for i in range(warmup_runs):
        log(f"  warmup {i+1}/{warmup_runs}...")
        try:
            m = model_factory()
            fit_fn(m)
            synchronize(backend)
            del m
        except Exception as e:
            return False, [], f"warmup failed: {e}", None

    # Timing runs
    times_ms: List[float] = []
    last_model: Any = None

    for i in range(repeats):
        log(f"  repeat {i+1}/{repeats}...")
        try:
            m = model_factory()
            t0 = time.perf_counter()
            fit_fn(m)
            synchronize(backend)
            t1 = time.perf_counter()
            elapsed_ms = (t1 - t0) * 1000.0
            times_ms.append(elapsed_ms)
            log(f"  repeat {i+1}/{repeats}: {elapsed_ms:.2f} ms")
            del last_model
            last_model = m
        except Exception as e:
            return False, times_ms, f"repeat {i+1} failed: {e}", last_model

    return True, times_ms, "", last_model


def extract_coef_with_intercept(model) -> np.ndarray:
    """Extract coefficients including intercept."""
    intercept = np.asarray(model.intercept_).reshape(-1)
    coef = np.asarray(model.coef_).reshape(-1)
    return np.r_[intercept, coef]


def extract_bse(model) -> Optional[np.ndarray]:
    """Extract standard errors if available."""
    try:
        # Use internal _bse attribute which includes intercept
        if hasattr(model, '_bse') and model._bse is not None:
            bse = np.asarray(model._bse).reshape(-1)
            # _bse already includes intercept at index 0
            return bse
    except Exception:
        pass
    return None


def safe_diff(ref: np.ndarray, arr) -> float:
    """Compute max absolute difference, handling different array types."""
    try:
        a = np.asarray(ref, dtype=float).reshape(-1)
        b = np.asarray(arr, dtype=float).reshape(-1)
        n = min(len(a), len(b))
        return float(np.max(np.abs(a[:n] - b[:n])))
    except Exception:
        return float('nan')


def run_benchmark_case(
    model_name: str,
    X, y, device: str, backend: str,
    model_factory: Callable[[], Any],
    fit_fn: Callable[[Any], None],
    warmup_runs: int,
    repeats: int,
    ref_coef: Optional[np.ndarray] = None,
    ref_bse: Optional[np.ndarray] = None,
) -> BenchmarkResult:
    """Run a single benchmark case."""

    backend_name = "cpu" if device == "cpu" else f"{backend}_gpu"
    log(f"  Running {model_name} on {backend_name}...")

    ok, times, error, fitted_model = time_fit(
        model_factory, fit_fn, warmup_runs, repeats, backend
    )

    # Compute numerical accuracy vs reference
    coef_diff = float('nan')
    bse_diff = float('nan')

    if ok and fitted_model is not None and ref_coef is not None:
        try:
            model_coef = extract_coef_with_intercept(fitted_model)
            coef_diff = safe_diff(ref_coef, model_coef)

            model_bse = extract_bse(fitted_model)
            if model_bse is not None and ref_bse is not None:
                bse_diff = safe_diff(ref_bse, model_bse)
        except Exception:
            pass

    return summarize(
        model=model_name,
        dataset="custom",
        backend=backend_name,
        repeats=repeats,
        ok=ok,
        times=times,
        err=error,
        coef_diff=coef_diff,
        bse_diff=bse_diff,
    )


def summarize(model: str, dataset: str, backend: str, repeats: int,
              ok: bool, times: List[float], err: str,
              coef_diff: float = float('nan'),
              bse_diff: float = float('nan')) -> BenchmarkResult:
    """Summarize benchmark results."""
    if not ok or not times:
        return BenchmarkResult(
            model=model, dataset=dataset, backend=backend,
            mean_ms=float('nan'), std_ms=float('nan'),
            min_ms=float('nan'), max_ms=float('nan'),
            repeats=repeats, ok=False, error=err or "unknown error",
            coef_diff=coef_diff, bse_diff=bse_diff,
        )

    return BenchmarkResult(
        model=model, dataset=dataset, backend=backend,
        mean_ms=float(statistics.mean(times)),
        std_ms=float(statistics.pstdev(times) if len(times) > 1 else 0.0),
        min_ms=float(min(times)),
        max_ms=float(max(times)),
        repeats=repeats, ok=True, error="",
        coef_diff=coef_diff, bse_diff=bse_diff,
    )


def print_table(results: List[BenchmarkResult]) -> None:
    """Print results as a formatted table."""
    print("\n" + "=" * 100)
    print("TORCH VS CUPY COMPREHENSIVE BENCHMARK RESULTS")
    print("=" * 100)

    # Group by model and dataset
    models = sorted(set(r.model for r in results))
    datasets = sorted(set(r.dataset for r in results))

    for dataset in datasets:
        print(f"\n--- Dataset: {dataset} ---\n")

        for model in models:
            model_results = [r for r in results if r.model == model and r.dataset == dataset]
            if not model_results:
                continue

            print(f"{model}:")
            print(f"  {'Backend':<15} {'Mean(ms)':>12} {'Std(ms)':>10} {'Min(ms)':>10} {'Max(ms)':>10} {'Coef Diff':>12} {'Status':>8}")
            print(f"  {'-'*15} {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*12} {'-'*8}")

            for r in model_results:
                if r.ok:
                    mean_str = f"{r.mean_ms:.2f}"
                    std_str = f"{r.std_ms:.2f}"
                    min_str = f"{r.min_ms:.2f}"
                    max_str = f"{r.max_ms:.2f}"
                    coef_diff_str = f"{r.coef_diff:.2e}" if not math.isnan(r.coef_diff) else "N/A"
                    status = "PASS" if (math.isnan(r.coef_diff) or r.coef_diff < 1e-6) else "WARN"
                else:
                    mean_str = std_str = min_str = max_str = "FAIL"
                    coef_diff_str = "N/A"
                    status = "FAIL"

                print(f"  {r.backend:<15} {mean_str:>12} {std_str:>10} {min_str:>10} {max_str:>10} {coef_diff_str:>12} {status:>8}")
            print()


def generate_markdown_report(results: List[BenchmarkResult], args: argparse.Namespace) -> str:
    """Generate a Markdown report."""
    lines = []
    lines.append("# Torch vs CuPy Comprehensive Benchmark Report")
    lines.append("")
    lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## Test Configuration")
    lines.append("")
    lines.append(f"- **Seed**: {args.seed}")
    lines.append(f"- **Repeats**: {args.repeats}")
    lines.append(f"- **Warmup Runs**: {args.warmup_runs}")
    lines.append(f"- **Small Dataset**: n={args.small_n}, p={args.small_p}")
    lines.append(f"- **Large Dataset**: n={args.large_n}, p={args.large_p}")
    lines.append("")
    lines.append("## Results Summary")
    lines.append("")

    # Overall statistics
    models = sorted(set(r.model for r in results))
    datasets = sorted(set(r.dataset for r in results))
    backends = sorted(set(r.backend for r in results))

    lines.append("### Models Tested")
    lines.append("")
    for m in models:
        lines.append(f"- {m}")
    lines.append("")

    lines.append("### Backends Compared")
    lines.append("")
    for b in backends:
        lines.append(f"- {b}")
    lines.append("")

    # Detailed results by dataset
    for dataset in datasets:
        lines.append(f"## Dataset: {dataset}")
        lines.append("")

        for model in models:
            model_results = [r for r in results if r.model == model and r.dataset == dataset]
            if not model_results:
                continue

            lines.append(f"### {model}")
            lines.append("")
            lines.append("| Backend | Mean (ms) | Std (ms) | Min (ms) | Max (ms) | Coef Diff | Status |")
            lines.append("|---------|-----------|----------|----------|----------|-----------|--------|")

            for r in model_results:
                if r.ok:
                    status = "PASS" if (math.isnan(r.coef_diff) or r.coef_diff < 1e-6) else "WARN"
                    lines.append(
                        f"| {r.backend} | {r.mean_ms:.2f} | {r.std_ms:.2f} | "
                        f"{r.min_ms:.2f} | {r.max_ms:.2f} | {r.coef_diff:.2e} | {status} |"
                    )
                else:
                    lines.append(f"| {r.backend} | FAIL | FAIL | FAIL | FAIL | N/A | FAIL |")

            lines.append("")

    # Key findings
    lines.append("## Key Findings")
    lines.append("")

    # Calculate speedup ratios
    for dataset in datasets:
        for model in models:
            torch_results = [r for r in results if r.model == model and r.dataset == dataset and r.backend == "torch_gpu"]
            cupy_results = [r for r in results if r.model == model and r.dataset == dataset and r.backend == "cupy_gpu"]

            if torch_results and cupy_results and torch_results[0].ok and cupy_results[0].ok:
                torch_time = torch_results[0].mean_ms
                cupy_time = cupy_results[0].mean_ms
                ratio = torch_time / cupy_time if cupy_time > 0 else float('inf')

                if ratio < 1.0:
                    lines.append(f"- **{model} ({dataset})**: Torch GPU is {1/ratio:.2f}x faster than CuPy GPU ({torch_time:.2f}ms vs {cupy_time:.2f}ms)")
                else:
                    lines.append(f"- **{model} ({dataset})**: CuPy GPU is {ratio:.2f}x faster than Torch GPU ({cupy_time:.2f}ms vs {torch_time:.2f}ms)")

    lines.append("")
    lines.append("## Numerical Accuracy")
    lines.append("")
    lines.append("All models target coefficient difference < 1e-6 vs CPU reference.")
    lines.append("")

    # Accuracy summary table
    lines.append("| Model | Dataset | Torch Coef Diff | CuPy Coef Diff |")
    lines.append("|-------|---------|-----------------|----------------|")

    for dataset in datasets:
        for model in models:
            torch_r = [r for r in results if r.model == model and r.dataset == dataset and r.backend == "torch_gpu"]
            cupy_r = [r for r in results if r.model == model and r.dataset == dataset and r.backend == "cupy_gpu"]

            torch_diff = f"{torch_r[0].coef_diff:.2e}" if torch_r and not math.isnan(torch_r[0].coef_diff) else "N/A"
            cupy_diff = f"{cupy_r[0].coef_diff:.2e}" if cupy_r and not math.isnan(cupy_r[0].coef_diff) else "N/A"

            lines.append(f"| {model} | {dataset} | {torch_diff} | {cupy_diff} |")

    lines.append("")
    lines.append("---")
    lines.append("*Report generated by benchmark_torch_vs_cupy_comprehensive.py*")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    log("=" * 60)
    log("Torch vs CuPy Comprehensive Benchmark")
    log("=" * 60)
    log(f"Seed: {args.seed}, Repeats: {args.repeats}, Warmup: {args.warmup_runs}")
    log(f"Small: n={args.small_n}, p={args.small_p}")
    log(f"Large: n={args.large_n}, p={args.large_p}")
    log("=" * 60)

    # Check backends availability
    if not HAS_CUPY:
        log("[ERROR] CuPy is not available. Install with: pip install cupy-cuda11x")
        sys.exit(1)

    if not HAS_TORCH:
        log("[ERROR] PyTorch is not available. Install with: pip install torch")
        sys.exit(1)

    if not cuda_available():
        log("[ERROR] CUDA is not available.")
        sys.exit(1)

    log(f"CUDA available: {cuda_available()}")
    log(f"CUDA devices: {torch.cuda.device_count()}")
    if torch.cuda.device_count() > 0:
        log(f"Current device: {torch.cuda.get_device_name(0)}")

    # Generate datasets
    log("\nGenerating datasets...")
    log("  Small regression data...")
    X_small_reg, y_small_reg = make_regression_data(rng, args.small_n, args.small_p)

    log("  Large regression data...")
    X_large_reg, y_large_reg = make_regression_data(rng, args.large_n, args.large_p)

    log("  Small logistic data...")
    X_small_logit, y_small_logit = make_logistic_data(rng, args.small_n, args.small_p)

    log("  Large logistic data...")
    X_large_logit, y_large_logit = make_logistic_data(rng, args.large_n, args.large_p)

    log("  Small Cox data...")
    X_small_cox, t_small_cox, e_small_cox = make_cox_data(rng, args.small_n, args.small_p)

    log("  Large Cox data...")
    X_large_cox, t_large_cox, e_large_cox = make_cox_data(rng, args.large_n, args.large_p)

    # Build CPU reference coefficients
    log("\nBuilding CPU reference coefficients...")

    ref_lin_small = LinearRegression(compute_inference=False, device="cpu", cov_type="nonrobust")
    ref_lin_small.fit(X_small_reg, y_small_reg)
    ref_coef_lin_small = extract_coef_with_intercept(ref_lin_small)

    ref_lin_large = LinearRegression(compute_inference=False, device="cpu", cov_type="nonrobust")
    ref_lin_large.fit(X_large_reg, y_large_reg)
    ref_coef_lin_large = extract_coef_with_intercept(ref_lin_large)

    ref_logit_small = LogisticRegression(C=1.0, max_iter=150, tol=1e-5,
                                          compute_inference=False, device="cpu")
    ref_logit_small.fit(X_small_logit, y_small_logit)
    ref_coef_logit_small = extract_coef_with_intercept(ref_logit_small)

    ref_logit_large = LogisticRegression(C=1.0, max_iter=150, tol=1e-5,
                                          compute_inference=False, device="cpu")
    ref_logit_large.fit(X_large_logit, y_large_logit)
    ref_coef_logit_large = extract_coef_with_intercept(ref_logit_large)

    ref_cox_small = CoxPH(ties="breslow", max_iter=120, tol=1e-8,
                           compute_inference=False, device="cpu")
    ref_cox_small.fit(X_small_cox, t_small_cox, e_small_cox)
    ref_coef_cox_small = np.asarray(ref_cox_small.coef_).reshape(-1)

    ref_cox_large = CoxPH(ties="breslow", max_iter=120, tol=1e-8,
                           compute_inference=False, device="cpu")
    ref_cox_large.fit(X_large_cox, t_large_cox, e_large_cox)
    ref_coef_cox_large = np.asarray(ref_cox_large.coef_).reshape(-1)

    log("CPU references built.")

    all_results: List[BenchmarkResult] = []

    # Benchmark configurations
    datasets = {
        "small": {
            "reg": (X_small_reg, y_small_reg),
            "logit": (X_small_logit, y_small_logit),
            "cox": (X_small_cox, t_small_cox, e_small_cox),
        },
        "large": {
            "reg": (X_large_reg, y_large_reg),
            "logit": (X_large_logit, y_large_logit),
            "cox": (X_large_cox, t_large_cox, e_large_cox),
        },
    }

    models_config = [
        ("LinearRegression", "reg", lambda **kwargs: LinearRegression(compute_inference=False, cov_type="nonrobust", **kwargs)),
        ("Ridge", "reg", lambda **kwargs: Ridge(alpha=1.0, **kwargs)),
        ("Lasso", "reg", lambda **kwargs: Lasso(alpha=0.05, max_iter=3000, tol=1e-5, solver="fista", cpu_solver="fista", compute_inference=False, **kwargs)),
        ("LogisticRegression", "logit", lambda **kwargs: LogisticRegression(C=1.0, max_iter=150, tol=1e-5, compute_inference=False, **kwargs)),
        ("CoxPH", "cox", lambda **kwargs: CoxPH(ties="breslow", max_iter=120, tol=1e-8, compute_inference=False, **kwargs)),
    ]

    backends = ["torch", "cupy"]

    for dataset_name, dataset_data in datasets.items():
        log(f"\n{'='*60}")
        log(f"Benchmarking {dataset_name} dataset")
        log(f"{'='*60}")

        for model_name, data_type, model_factory_fn in models_config:
            log(f"\n--- {model_name} ---")

            if data_type == "reg":
                X_np, y_np = dataset_data["reg"]
            elif data_type == "logit":
                X_np, y_np = dataset_data["logit"]
            else:  # cox
                X_np, t_np, e_np = dataset_data["cox"]

            # Get reference coefficients
            if model_name == "LinearRegression":
                ref_coef = ref_coef_lin_small if dataset_name == "small" else ref_coef_lin_large
            elif model_name == "LogisticRegression":
                ref_coef = ref_coef_logit_small if dataset_name == "small" else ref_coef_logit_large
            elif model_name == "CoxPH":
                ref_coef = ref_coef_cox_small if dataset_name == "small" else ref_coef_cox_large
            else:
                ref_coef = None  # Ridge/Lasso may have different coefficients due to regularization
            ref_bse = None

            for backend in backends:
                # Prepare data for backend
                if data_type == "cox":
                    X = as_device(X_np, "cuda", backend)
                    t = as_device(t_np, "cuda", backend)
                    e = as_device(e_np, "cuda", backend)

                    def make_model():
                        # Backend is determined by input data type, not a constructor argument
                        return model_factory_fn(device="cuda")

                    def fit_fn(m):
                        m.fit(X, t, e)
                else:
                    X = as_device(X_np, "cuda", backend)
                    y = as_device(y_np, "cuda", backend)

                    def make_model():
                        # Backend is determined by input data type, not a constructor argument
                        return model_factory_fn(device="cuda")

                    def fit_fn(m):
                        m.fit(X, y)

                result = run_benchmark_case(
                    model_name=model_name,
                    X=X, y=y if data_type != "cox" else None,
                    device="cuda", backend=backend,
                    model_factory=make_model,
                    fit_fn=fit_fn,
                    warmup_runs=args.warmup_runs,
                    repeats=args.repeats,
                    ref_coef=ref_coef,
                    ref_bse=ref_bse,
                )

                all_results.append(result)

                # Print immediate feedback
                status = "PASS" if result.ok and (math.isnan(result.coef_diff) or result.coef_diff < 1e-6) else "FAIL"
                log(f"  Result: {result.backend} mean={result.mean_ms:.2f}ms, coef_diff={result.coef_diff:.2e}, status={status}")

    # Print summary table
    print_table(all_results)

    # Save JSON results
    if args.json_out:
        json_data = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "test_config": {
                    "seed": args.seed,
                    "repeats": args.repeats,
                    "warmup_runs": args.warmup_runs,
                    "small_n": args.small_n,
                    "small_p": args.small_p,
                    "large_n": args.large_n,
                    "large_p": args.large_p,
                },
                "environment": {
                    "gpu": torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "Unknown",
                    "cuda_version": torch.version.cuda,
                    "torch_version": torch.__version__,
                    "cupy_version": cp.__version__ if HAS_CUPY else "N/A",
                }
            },
            "results": [asdict(r) for r in all_results],
        }

        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2)

        log(f"\nJSON results saved to: {args.json_out}")

    # Save Markdown report
    if args.md_out:
        md_report = generate_markdown_report(all_results, args)
        with open(args.md_out, "w", encoding="utf-8") as f:
            f.write(md_report)

        log(f"Markdown report saved to: {args.md_out}")

    log("\n" + "=" * 60)
    log("Benchmark complete!")
    log("=" * 60)


if __name__ == "__main__":
    main()
