#!/usr/bin/env python3
"""StatGPU benchmark runner for PR79 validation.

Runs LinearRegression, Ridge, PooledOLS, and CoxPH on NumPy/CuPy/Torch
and produces raw benchmark JSON records.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# Ensure project root is on path
_project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(_project_root))

from dev.benchmarks.pr79.runners.common import (
    make_case_id,
    make_method_config_id,
    make_raw_run,
    record_environment,
    safe_run,
    synchronized_time,
)


# ---------------------------------------------------------------------------
# Backend helpers
# ---------------------------------------------------------------------------


def _backend_inputs(X, y, backend, sample_weight=None, **extra):
    """Convert numpy inputs to the target backend."""
    if backend == "cupy":
        import cupy as cp
        X_d = cp.asarray(X)
        y_d = cp.asarray(y)
        sw_d = cp.asarray(sample_weight) if sample_weight is not None else None
    elif backend == "torch":
        import torch
        X_d = torch.as_tensor(X, dtype=torch.float64, device="cuda")
        y_d = torch.as_tensor(y, dtype=torch.float64, device="cuda")
        sw_d = torch.as_tensor(sample_weight, dtype=torch.float64, device="cuda") if sample_weight is not None else None
    else:
        X_d, y_d, sw_d = X, y, sample_weight
    extra_d = {}
    for k, v in extra.items():
        if backend == "cupy":
            import cupy as cp
            extra_d[k] = cp.asarray(v) if isinstance(v, np.ndarray) else v
        elif backend == "torch":
            import torch
            extra_d[k] = torch.as_tensor(v, dtype=torch.int64, device="cuda") if isinstance(v, np.ndarray) else v
        else:
            extra_d[k] = v
    return X_d, y_d, sw_d, extra_d


def _to_np(arr):
    """Safely convert any backend array to numpy."""
    if arr is None:
        return None
    try:
        if hasattr(arr, "get"):
            import cupy as cp
            return cp.asnumpy(arr)
    except Exception:
        pass
    try:
        if hasattr(arr, "cpu") and hasattr(arr, "detach"):
            return arr.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(arr)


def _extract_results(model, X_test, y_test=None) -> Dict[str, Any]:
    """Extract standard results from a fitted model."""
    r: Dict[str, Any] = {}
    for attr in ["coef_", "intercept_", "rank_", "rsquared", "aic", "bic",
                 "_df_model", "_df_resid", "_bse", "_pvalues", "_log_likelihood",
                 "_var_matrix", "n_iter_", "alpha_", "_converged"]:
        val = getattr(model, attr, None)
        if val is not None:
            val_np = _to_np(val)
            r[attr] = val_np.tolist() if hasattr(val_np, "tolist") else float(val_np) if np.isscalar(val_np) else val_np

    # Predictions
    if hasattr(model, "predict") and X_test is not None:
        pred = model.predict(_to_backend(X_test, model))
        pred_np = _to_np(pred)
        r["prediction_summary"] = {
            "mean": float(np.mean(pred_np)),
            "std": float(np.std(pred_np)),
            "shape": list(pred_np.shape),
        }
    return r


def _to_backend(X, model):
    """Convert X to model's backend."""
    import cupy as cp
    if hasattr(model, "coef_") and hasattr(model.coef_, "device") and hasattr(model.coef_, "is_cuda"):
        if model.coef_.is_cuda:
            import torch
            return torch.as_tensor(X, dtype=torch.float64, device="cuda")
    try:
        from statgpu.backends import _is_cupy_array
        if _is_cupy_array(getattr(model, "coef_", None)):
            return cp.asarray(X)
    except Exception:
        pass
    return X


def _device_from_model(model) -> str:
    """Heuristic: detect which backend the model used."""
    coef = getattr(model, "coef_", None)
    if coef is None:
        return "numpy"
    try:
        from statgpu.backends import _is_cupy_array
        if _is_cupy_array(coef):
            return "cupy"
    except Exception:
        pass
    try:
        import torch
        if isinstance(coef, torch.Tensor) and coef.is_cuda:
            return "torch"
    except Exception:
        pass
    return "numpy"


# ---------------------------------------------------------------------------
# Benchmark functions
# ---------------------------------------------------------------------------


def bench_linear(
    X: np.ndarray,
    y: np.ndarray,
    backend: str = "numpy",
    cov_type: str = "nonrobust",
    fit_intercept: bool = True,
    compute_inference: bool = True,
    sample_weight: Optional[np.ndarray] = None,
    n_warmup: int = 2,
    n_measured: int = 5,
) -> List[Dict[str, Any]]:
    """Benchmark LinearRegression on one backend."""
    from statgpu.linear_model import LinearRegression

    X_d, y_d, sw_d, _ = _backend_inputs(X, y, backend, sample_weight=sample_weight)
    device_str = {"numpy": "cpu", "cupy": "cuda", "torch": "torch"}[backend]
    runs = []

    params = {
        "fit_intercept": fit_intercept,
        "cov_type": cov_type,
        "compute_inference": compute_inference,
        "device": device_str,
    }

    for i in range(n_warmup + n_measured):
        model = LinearRegression(**params)
        result, elapsed = synchronized_time(
            model.fit, X_d, y_d,
            sample_weight=sw_d,
        )
        if i >= n_warmup:
            runs.append({
                "iteration": i - n_warmup,
                "fit_time_s": round(elapsed, 6),
                "results": _extract_results(model, X, y),
                "backend_detected": _device_from_model(model),
            })
    return runs


def bench_ridge(
    X: np.ndarray,
    y: np.ndarray,
    backend: str = "numpy",
    alpha: float = 1.0,
    solver: str = "auto",
    fit_intercept: bool = True,
    sample_weight: Optional[np.ndarray] = None,
    n_warmup: int = 2,
    n_measured: int = 5,
) -> List[Dict[str, Any]]:
    """Benchmark Ridge on one backend."""
    from statgpu.linear_model import Ridge

    X_d, y_d, sw_d, _ = _backend_inputs(X, y, backend, sample_weight=sample_weight)
    device_str = {"numpy": "cpu", "cupy": "cuda", "torch": "torch"}[backend]
    runs = []

    for i in range(n_warmup + n_measured):
        model = Ridge(alpha=alpha, solver=solver, fit_intercept=fit_intercept, device=device_str)
        result, elapsed = synchronized_time(
            model.fit, X_d, y_d,
            sample_weight=sw_d,
        )
        if i >= n_warmup:
            runs.append({
                "iteration": i - n_warmup,
                "fit_time_s": round(elapsed, 6),
                "results": _extract_results(model, X, y),
            })
    return runs


def bench_pooled_ols(
    X: np.ndarray,
    y: np.ndarray,
    entity: np.ndarray,
    time_idx: np.ndarray,
    backend: str = "numpy",
    cov_type: str = "nonrobust",
    n_warmup: int = 2,
    n_measured: int = 5,
) -> List[Dict[str, Any]]:
    """Benchmark PooledOLS on one backend."""
    from statgpu.panel import PooledOLS

    X_d, y_d, _, extra = _backend_inputs(X, y, backend)
    device_str = {"numpy": "cpu", "cupy": "cuda", "torch": "torch"}[backend]
    runs = []

    for i in range(n_warmup + n_measured):
        model = PooledOLS(cov_type=cov_type, device=device_str)
        result, elapsed = synchronized_time(
            model.fit, X_d, y_d,
            cluster=entity if cov_type == "clustered" else None,
            time_index=time_idx if cov_type == "hac" else None,
        )
        if i >= n_warmup:
            runs.append({
                "iteration": i - n_warmup,
                "fit_time_s": round(elapsed, 6),
                "results": _extract_results(model, X, y),
            })
    return runs


def bench_coxph(
    X: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    backend: str = "numpy",
    ties: str = "efron",
    penalty: float = 0.0,
    compute_inference: bool = True,
    entry: Optional[np.ndarray] = None,
    n_warmup: int = 2,
    n_measured: int = 5,
) -> List[Dict[str, Any]]:
    """Benchmark CoxPH on one backend."""
    from statgpu.survival import CoxPH

    X_d, _, _, _ = _backend_inputs(X, np.zeros_like(time), backend)
    device_str = {"numpy": "cpu", "cupy": "cuda", "torch": "torch"}[backend]
    runs = []

    for i in range(n_warmup + n_measured):
        model = CoxPH(
            ties=ties, penalty=penalty, compute_inference=compute_inference,
            device=device_str, compute_cindex=False, tol=1e-6, max_iter=30,
        )
        result, elapsed = synchronized_time(
            model.fit, X_d,
            time=time, event=event, entry=entry,
        )
        if i >= n_warmup:
            runs.append({
                "iteration": i - n_warmup,
                "fit_time_s": round(elapsed, 6),
                "results": _extract_results(model, X, None),
            })
    return runs


# ---------------------------------------------------------------------------
# Main smoke test
# ---------------------------------------------------------------------------


def run_smoke(output_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Run a smoke test covering Linear, Ridge, Panel, and CoxPH across backends."""
    from dev.benchmarks.pr79.generators.linear import (
        case_params_linear,
        case_params_linear_rank_def,
        case_params_linear_weighted,
        generate_coxph_simple,
        generate_coxph_ties,
        generate_linear_full_rank,
        generate_linear_rank_deficient,
        generate_linear_weighted,
        generate_panel_balanced,
    )

    env = record_environment()
    runs: List[Dict[str, Any]] = []
    backends = ["numpy", "cupy", "torch"]
    git_sha = _get_git_sha()

    print(f"PR79 Benchmark Smoke Test — SHA: {git_sha}")
    print(f"Backends: {backends}")
    print()

    # --- Linear full-rank ---
    print("=== Linear full-rank ===")
    cp = case_params_linear()
    X, y, beta = generate_linear_full_rank()
    case_id = make_case_id(cp)
    for b in backends:
        try:
            bench_runs = bench_linear(X, y, backend=b)
            for br in bench_runs:
                mc = {"model_id": "LinearRegression", "cov_type": "nonrobust",
                      "compute_inference": True, "backend": b}
                runs.append(make_raw_run(
                    f"linear-fr-{b}", case_id, make_method_config_id(mc),
                    "LinearRegression", "statgpu", b, mc,
                    {"fit_warm_s": br["fit_time_s"]}, br["results"],
                ))
            print(f"  {b}: {bench_runs[0]['fit_time_s']*1000:.1f} ms, rank={bench_runs[0]['results'].get('rank_')}")
        except Exception as exc:
            print(f"  {b}: FAILED — {exc}")

    # --- Linear rank-deficient ---
    print("=== Linear rank-deficient ===")
    cp = case_params_linear_rank_def()
    X, y, _ = generate_linear_rank_deficient()
    case_id = make_case_id(cp)
    for b in backends:
        try:
            bench_runs = bench_linear(X, y, backend=b, cov_type="hc1")
            for br in bench_runs:
                mc = {"model_id": "LinearRegression", "cov_type": "hc1",
                      "compute_inference": True, "backend": b}
                runs.append(make_raw_run(
                    f"linear-rd-{b}", case_id, make_method_config_id(mc),
                    "LinearRegression", "statgpu", b, mc,
                    {"fit_warm_s": br["fit_time_s"]}, br["results"],
                ))
            r = bench_runs[0]["results"]
            print(f"  {b}: rank={r.get('rank_')}, df_resid={r.get('_df_resid')}")
        except Exception as exc:
            print(f"  {b}: FAILED — {exc}")

    # --- Linear weighted ---
    print("=== Linear weighted ===")
    cp = case_params_linear_weighted()
    X, y, _, weights = generate_linear_weighted()
    case_id = make_case_id(cp)
    for b in backends:
        try:
            bench_runs = bench_linear(X, y, backend=b, sample_weight=weights)
            for br in bench_runs:
                mc = {"model_id": "LinearRegression", "cov_type": "nonrobust",
                      "compute_inference": True, "weighted": True, "backend": b}
                runs.append(make_raw_run(
                    f"linear-wt-{b}", case_id, make_method_config_id(mc),
                    "LinearRegression", "statgpu", b, mc,
                    {"fit_warm_s": br["fit_time_s"]}, br["results"],
                ))
            print(f"  {b}: {bench_runs[0]['fit_time_s']*1000:.1f} ms")
        except Exception as exc:
            print(f"  {b}: FAILED — {exc}")

    # --- CoxPH simple ---
    print("=== CoxPH Efron simple ===")
    X, time_, event, _ = generate_coxph_simple()
    cp = {"domain": "survival", "n_samples": 200, "n_features": 4, "seed": 42, "ties": "efron"}
    case_id = make_case_id(cp)
    for b in backends:
        try:
            bench_runs = bench_coxph(X, time_, event, backend=b, ties="efron")
            for br in bench_runs:
                mc = {"model_id": "CoxPH", "ties": "efron", "compute_inference": True, "backend": b}
                runs.append(make_raw_run(
                    f"cox-efron-{b}", case_id, make_method_config_id(mc),
                    "CoxPH", "statgpu", b, mc,
                    {"fit_warm_s": br["fit_time_s"]}, br["results"],
                ))
            r = bench_runs[0]["results"]
            print(f"  {b}: {bench_runs[0]['fit_time_s']*1000:.1f} ms, ll={r.get('_log_likelihood')}")
        except Exception as exc:
            print(f"  {b}: FAILED — {exc}")

    # --- Panel PooledOLS ---
    print("=== Panel PooledOLS ===")
    X, y, entity, time_idx, _ = generate_panel_balanced()
    cp = {"domain": "panel", "n_entities": 30, "n_periods": 5, "n_features": 3, "seed": 42}
    case_id = make_case_id(cp)
    for b in backends:
        try:
            bench_runs = bench_pooled_ols(X, y, entity, time_idx, backend=b)
            for br in bench_runs:
                mc = {"model_id": "PooledOLS", "cov_type": "nonrobust", "backend": b}
                runs.append(make_raw_run(
                    f"pooled-{b}", case_id, make_method_config_id(mc),
                    "PooledOLS", "statgpu", b, mc,
                    {"fit_warm_s": br["fit_time_s"]}, br["results"],
                ))
            print(f"  {b}: {bench_runs[0]['fit_time_s']*1000:.1f} ms")
        except Exception as exc:
            print(f"  {b}: FAILED — {exc}")

    # Summary
    print(f"\nTotal runs: {len(runs)}")
    passed = sum(1 for r in runs if r["status"] == "success")
    failed = sum(1 for r in runs if r["status"] != "success")
    print(f"Passed: {passed}, Failed: {failed}")

    if output_path:
        output = {
            "source_schema_version": "pr79-benchmark-source-1.0",
            "benchmark_session_id": f"pr79-{git_sha[:7]}-smoke",
            "git_sha": git_sha,
            "environment": env,
            "runs": runs,
        }
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"Saved to {output_path}")

    return runs


def _get_git_sha() -> str:
    try:
        import subprocess
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, timeout=5
        ).strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "results/pr79/smoke/smoke_benchmark.json"
    run_smoke(out)
