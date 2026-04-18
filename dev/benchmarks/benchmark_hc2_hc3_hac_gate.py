# -*- coding: utf-8 -*-
"""
HC2/HC3/HAC covariance benchmark gate.

This script provides standardized benchmarking for robust covariance estimation
across statsmodels, statgpu CPU, and statgpu GPU backends.

Results are saved as JSON for auditability and non-regression checks.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _try_import_statsmodels():
    """Try to import statsmodels, return None if unavailable."""
    try:
        import statsmodels.api as sm
        return sm
    except Exception:
        return None


def _try_import_cupy():
    """Try to import CuPy, return None if unavailable."""
    try:
        import cupy as cp
        if int(cp.cuda.runtime.getDeviceCount()) > 0:
            return cp
        return None
    except Exception:
        return None


def _try_import_torch():
    """Try to import PyTorch, return None if unavailable."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch
        return None
    except Exception:
        return None


def _make_regression_data(seed: int, n: int, p: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate regression data with known coefficients."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta = rng.uniform(-2, 2, size=p)
    y = X @ beta + rng.normal(scale=0.5, size=n)
    return X, y, beta


def _make_classification_data(seed: int, n: int, p: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate binary classification data."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta = rng.uniform(-1, 1, size=p)
    logits = X @ beta
    prob = 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))
    y = (rng.random(n) < prob).astype(np.float64)
    return X, y, beta


@dataclass
class BenchmarkResult:
    """Single benchmark result."""
    model: str
    cov_type: str
    backend: str
    time_ms: float
    coef_max_diff: Optional[float] = None
    bse_max_diff: Optional[float] = None
    pvalue_max_diff: Optional[float] = None
    error: Optional[str] = None


def _run_statsmodels_linear(X: np.ndarray, y: np.ndarray, cov_type: str, hac_maxlags: int = None) -> Tuple[float, Dict]:
    """Run statsmodels OLS with specified covariance type."""
    import statsmodels.api as sm

    X_design = np.column_stack([np.ones(X.shape[0]), X])

    t0 = time.perf_counter()
    if cov_type == "hac":
        model = sm.OLS(y, X_design).fit(cov_type=cov_type, cov_kwargs={"maxlags": hac_maxlags})
    else:
        model = sm.OLS(y, X_design).fit(cov_type=cov_type)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return elapsed_ms, {
        "coef": model.params[1:],
        "bse": model.bse[1:],
        "pvalues": model.pvalues[1:],
    }


def _run_statsmodels_logistic(X: np.ndarray, y: np.ndarray, cov_type: str, hac_maxlags: int = None) -> Tuple[float, Dict]:
    """Run statsmodels Logit with specified covariance type."""
    import statsmodels.api as sm

    X_design = np.column_stack([np.ones(X.shape[0]), X])

    t0 = time.perf_counter()
    if cov_type == "hac":
        model = sm.Logit(y, X_design).fit(cov_type=cov_type, cov_kwargs={"maxlags": hac_maxlags}, disp=0)
    else:
        model = sm.Logit(y, X_design).fit(cov_type=cov_type, disp=0)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return elapsed_ms, {
        "coef": model.params[1:],
        "bse": model.bse[1:],
        "pvalues": model.pvalues[1:],
    }


def _run_statgpu_linear(X, y, cov_type: str, device: str, hac_maxlags: int = None):
    """Run statgpu LinearRegression with specified covariance type."""
    from statgpu.linear_model import LinearRegression

    kwargs = {"cov_type": cov_type, "device": device}
    if cov_type == "hac" and hac_maxlags is not None:
        kwargs["hac_maxlags"] = hac_maxlags

    model = LinearRegression(**kwargs)

    t0 = time.perf_counter()
    model.fit(X, y)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # _bse and _pvalues include intercept at index 0, same as statsmodels
    # Exclude intercept to match coef_ which excludes it
    bse = model._bse[1:] if model.fit_intercept else model._bse
    pvalues = model._pvalues[1:] if model.fit_intercept else model._pvalues

    return elapsed_ms, {
        "coef": model.coef_,
        "bse": bse,
        "pvalues": pvalues,
    }


def _run_statgpu_logistic(X, y, cov_type: str, device: str, hac_maxlags: int = None):
    """Run statgpu LogisticRegression with specified covariance type."""
    from statgpu.linear_model import LogisticRegression

    kwargs = {"cov_type": cov_type, "device": device}
    if cov_type == "hac" and hac_maxlags is not None:
        kwargs["hac_maxlags"] = hac_maxlags

    model = LogisticRegression(**kwargs)

    t0 = time.perf_counter()
    model.fit(X, y)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # _bse and _pvalues include intercept at index 0, same as statsmodels
    # Exclude intercept to match coef_ which excludes it
    bse = model._bse[1:] if model.fit_intercept else model._bse
    pvalues = model._pvalues[1:] if model.fit_intercept else model._pvalues

    return elapsed_ms, {
        "coef": model.coef_,
        "bse": bse,
        "pvalues": pvalues,
    }


def _max_diff(a: np.ndarray, b: np.ndarray) -> float:
    """Compute max absolute difference between two arrays."""
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def run_benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    """Run full benchmark suite."""

    print("=" * 80)
    print("HC2/HC3/HAC Covariance Benchmark Gate")
    print("=" * 80)

    # Check backends availability
    sm_available = _try_import_statsmodels() is not None
    cupy_available = _try_import_cupy() is not None
    torch_available = _try_import_torch() is not None

    print(f"statsmodels available: {sm_available}")
    print(f"CuPy GPU available: {cupy_available}")
    print(f"Torch GPU available: {torch_available}")
    print("=" * 80)

    results: List[BenchmarkResult] = []
    reference_results = {}

    # Generate data
    np.random.seed(args.seed)

    if args.model == "linear":
        X, y, true_coef = _make_regression_data(args.seed, args.n, args.p)
    else:
        X, y, true_coef = _make_classification_data(args.seed, args.n, args.p)

    print(f"\nData: n={args.n}, p={args.p}, model={args.model}")

    # Run benchmarks for each covariance type
    for cov_type in args.cov_types:
        print(f"\n--- {cov_type} ---")

        # statsmodels (reference)
        if sm_available and not args.skip_statsmodels:
            try:
                if args.model == "linear":
                    sm_time, sm_result = _run_statsmodels_linear(
                        X, y, cov_type, args.hac_maxlags
                    )
                else:
                    sm_time, sm_result = _run_statsmodels_logistic(
                        X, y, cov_type, args.hac_maxlags
                    )
                print(f"  statsmodels: {sm_time:.2f} ms")

                reference_results[cov_type] = sm_result

                results.append(BenchmarkResult(
                    model=args.model,
                    cov_type=cov_type,
                    backend="statsmodels",
                    time_ms=sm_time,
                ))
            except Exception as e:
                print(f"  statsmodels FAILED: {e}")
                results.append(BenchmarkResult(
                    model=args.model,
                    cov_type=cov_type,
                    backend="statsmodels",
                    time_ms=float('nan'),
                    error=str(e),
                ))
        else:
            print("  statsmodels: SKIPPED")

        # statgpu CPU
        try:
            if args.model == "linear":
                cpu_time, cpu_result = _run_statgpu_linear(
                    X, y, cov_type, device="cpu", hac_maxlags=args.hac_maxlags
                )
            else:
                cpu_time, cpu_result = _run_statgpu_logistic(
                    X, y, cov_type, device="cpu", hac_maxlags=args.hac_maxlags
                )
            print(f"  statgpu CPU: {cpu_time:.2f} ms")

            # Compare vs statsmodels
            coef_diff = bse_diff = pvalue_diff = None
            if cov_type in reference_results:
                ref = reference_results[cov_type]
                coef_diff = _max_diff(ref["coef"], cpu_result["coef"])
                bse_diff = _max_diff(ref["bse"], cpu_result["bse"])
                pvalue_diff = _max_diff(ref["pvalues"], cpu_result["pvalues"])

            results.append(BenchmarkResult(
                model=args.model,
                cov_type=cov_type,
                backend="statgpu_cpu",
                time_ms=cpu_time,
                coef_max_diff=coef_diff,
                bse_max_diff=bse_diff,
                pvalue_max_diff=pvalue_diff,
            ))
        except Exception as e:
            print(f"  statgpu CPU FAILED: {e}")
            results.append(BenchmarkResult(
                model=args.model,
                cov_type=cov_type,
                backend="statgpu_cpu",
                time_ms=float('nan'),
                error=str(e),
            ))

        # statgpu GPU (CuPy)
        if cupy_available and not args.skip_gpu:
            try:
                if args.model == "linear":
                    gpu_time, gpu_result = _run_statgpu_linear(
                        X, y, cov_type, device="cuda", hac_maxlags=args.hac_maxlags
                    )
                else:
                    gpu_time, gpu_result = _run_statgpu_logistic(
                        X, y, cov_type, device="cuda", hac_maxlags=args.hac_maxlags
                    )
                print(f"  statgpu CuPy GPU: {gpu_time:.2f} ms")

                # Compare vs statsmodels
                coef_diff = bse_diff = pvalue_diff = None
                if cov_type in reference_results:
                    ref = reference_results[cov_type]
                    coef_diff = _max_diff(ref["coef"], gpu_result["coef"])
                    bse_diff = _max_diff(ref["bse"], gpu_result["bse"])
                    pvalue_diff = _max_diff(ref["pvalues"], gpu_result["pvalues"])

                results.append(BenchmarkResult(
                    model=args.model,
                    cov_type=cov_type,
                    backend="statgpu_cupy_gpu",
                    time_ms=gpu_time,
                    coef_max_diff=coef_diff,
                    bse_max_diff=bse_diff,
                    pvalue_max_diff=pvalue_diff,
                ))
            except Exception as e:
                print(f"  statgpu CuPy GPU FAILED: {e}")
                results.append(BenchmarkResult(
                    model=args.model,
                    cov_type=cov_type,
                    backend="statgpu_cupy_gpu",
                    time_ms=float('nan'),
                    error=str(e),
                ))
        else:
            print("  statgpu CuPy GPU: SKIPPED")

        # statgpu GPU (Torch)
        if torch_available and not args.skip_torch:
            try:
                if args.model == "linear":
                    gpu_time, gpu_result = _run_statgpu_linear(
                        X, y, cov_type, device="cuda", hac_maxlags=args.hac_maxlags
                    )
                else:
                    gpu_time, gpu_result = _run_statgpu_logistic(
                        X, y, cov_type, device="cuda", hac_maxlags=args.hac_maxlags
                    )
                print(f"  statgpu Torch GPU: {gpu_time:.2f} ms")

                # Compare vs statsmodels
                coef_diff = bse_diff = pvalue_diff = None
                if cov_type in reference_results:
                    ref = reference_results[cov_type]
                    coef_diff = _max_diff(ref["coef"], gpu_result["coef"])
                    bse_diff = _max_diff(ref["bse"], gpu_result["bse"])
                    pvalue_diff = _max_diff(ref["pvalues"], gpu_result["pvalues"])

                results.append(BenchmarkResult(
                    model=args.model,
                    cov_type=cov_type,
                    backend="statgpu_torch_gpu",
                    time_ms=gpu_time,
                    coef_max_diff=coef_diff,
                    bse_max_diff=bse_diff,
                    pvalue_max_diff=pvalue_diff,
                ))
            except Exception as e:
                print(f"  statgpu Torch GPU FAILED: {e}")
                results.append(BenchmarkResult(
                    model=args.model,
                    cov_type=cov_type,
                    backend="statgpu_torch_gpu",
                    time_ms=float('nan'),
                    error=str(e),
                ))
        else:
            print("  statgpu Torch GPU: SKIPPED")

    # Build output
    output = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "repo_root": str(REPO_ROOT),
        },
        "config": {
            "model": args.model,
            "n": args.n,
            "p": args.p,
            "seed": args.seed,
            "cov_types": args.cov_types,
            "hac_maxlags": args.hac_maxlags,
            "repeats": args.repeats,
        },
        "environment": {
            "statsmodels_available": sm_available,
            "cupy_available": cupy_available,
            "torch_available": torch_available,
        },
        "results": [asdict(r) for r in results],
    }

    # Compute summary statistics
    output["summary"] = compute_summary(results)

    return output


def compute_summary(results: List[BenchmarkResult]) -> Dict[str, Any]:
    """Compute summary statistics and pass/fail status."""

    # Group by cov_type
    by_cov_type: Dict[str, List[BenchmarkResult]] = {}
    for r in results:
        if r.cov_type not in by_cov_type:
            by_cov_type[r.cov_type] = []
        by_cov_type[r.cov_type].append(r)

    summary = {"covariance_types": {}}

    for cov_type, cov_results in by_cov_type.items():
        cov_summary = {"backends": {}}

        for r in cov_results:
            if np.isnan(r.time_ms):
                continue

            backend_summary = {
                "time_ms": r.time_ms,
                "coef_max_diff": r.coef_max_diff,
                "bse_max_diff": r.bse_max_diff,
                "pvalue_max_diff": r.pvalue_max_diff,
            }

            # Pass/fail check
            passed = True
            failure_reasons = []

            if r.coef_max_diff is not None and r.coef_max_diff > 1e-6:
                passed = False
                failure_reasons.append(f"coef_diff {r.coef_max_diff:.2e} > 1e-6")
            if r.bse_max_diff is not None and r.bse_max_diff > 1e-3:
                passed = False
                failure_reasons.append(f"bse_diff {r.bse_max_diff:.2e} > 1e-3")
            if r.pvalue_max_diff is not None and r.pvalue_max_diff > 5e-2:
                passed = False
                failure_reasons.append(f"pvalue_diff {r.pvalue_max_diff:.2e} > 5e-2")

            backend_summary["passed"] = passed
            backend_summary["failure_reasons"] = failure_reasons

            cov_summary["backends"][r.backend] = backend_summary

        summary["covariance_types"][cov_type] = cov_summary

    # Overall pass/fail
    all_passed = all(
        b.get("passed", True)
        for cov in summary["covariance_types"].values()
        for b in cov["backends"].values()
    )
    summary["overall_passed"] = all_passed

    return summary


def print_summary(output: Dict[str, Any]) -> None:
    """Print human-readable summary."""
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)

    summary = output["summary"]
    print(f"\nOverall: {'✅ PASS' if summary['overall_passed'] else '❌ FAIL'}")

    for cov_type, cov_summary in summary["covariance_types"].items():
        print(f"\n{cov_type}:")
        for backend, b_summary in cov_summary["backends"].items():
            status = "✅" if b_summary.get("passed", True) else "❌"
            print(f"  {backend}: time={b_summary['time_ms']:.2f}ms {status}")
            if b_summary.get("failure_reasons"):
                for reason in b_summary["failure_reasons"]:
                    print(f"    - {reason}")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description="HC2/HC3/HAC covariance benchmark gate")
    parser.add_argument("--model", type=str, default="linear", choices=["linear", "logistic"])
    parser.add_argument("--n", type=int, default=8000)
    parser.add_argument("--p", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cov-types", type=str, nargs="+", default=["hc2", "hc3", "hac"])
    parser.add_argument("--hac-maxlags", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--skip-statsmodels", action="store_true")
    parser.add_argument("--skip-gpu", action="store_true")
    parser.add_argument("--skip-torch", action="store_true")
    parser.add_argument("--json-out", type=str, default="", help="Output JSON file")

    args = parser.parse_args()

    # Run benchmark
    output = run_benchmark(args)

    # Print summary
    print_summary(output)

    # Save JSON
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nJSON saved to: {args.json_out}")


if __name__ == "__main__":
    main()
