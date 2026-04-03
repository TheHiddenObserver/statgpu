"""
Large-scale runtime benchmark across all current statgpu methods.

Covers:
  - LinearRegression
  - Ridge
  - Lasso
  - LogisticRegression
  - CoxPH

The script separates data construction from fit timing:
  1) build NumPy data once
  2) optionally move data to GPU once
  3) benchmark model.fit(...) only
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from statgpu.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from statgpu.survival import CoxPH
from statgpu._config import cuda_available


try:
    import cupy as cp

    HAS_CUPY = True
except Exception:
    cp = None
    HAS_CUPY = False


def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


@dataclass
class CaseResult:
    model: str
    device: str
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    repeats: int
    ok: bool
    error: str = ""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark all statgpu methods at larger scales.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmup-runs", type=int, default=1)
    p.add_argument("--devices", type=str, default="cpu,cuda", help="Comma separated: cpu,cuda")
    p.add_argument(
        "--compute-inference",
        action="store_true",
        help="If set, include inference computations in timing (usually slower).",
    )
    p.add_argument(
        "--gpu-memory-cleanup",
        action="store_true",
        help="Enable gpu_memory_cleanup in model constructors.",
    )
    p.add_argument("--json-out", type=str, default="", help="Optional path to save JSON results.")

    # Sizes (intentionally large but still practical defaults)
    p.add_argument("--n-reg", type=int, default=60000, help="Rows for linear/ridge/lasso.")
    p.add_argument("--p-reg", type=int, default=64, help="Cols for linear/ridge/lasso.")
    p.add_argument("--n-logit", type=int, default=80000, help="Rows for logistic.")
    p.add_argument("--p-logit", type=int, default=48, help="Cols for logistic.")
    p.add_argument("--n-cox", type=int, default=50000, help="Rows for CoxPH.")
    p.add_argument("--p-cox", type=int, default=24, help="Cols for CoxPH.")
    return p.parse_args()


def make_regression_data(rng: np.random.Generator, n: int, p: int) -> Tuple[np.ndarray, np.ndarray]:
    X = rng.normal(size=(n, p))
    beta = rng.normal(size=p)
    y = X @ beta + 1.0 + rng.normal(scale=0.5, size=n)
    return X.astype(np.float64), y.astype(np.float64)


def make_logistic_data(rng: np.random.Generator, n: int, p: int) -> Tuple[np.ndarray, np.ndarray]:
    X = rng.normal(size=(n, p))
    beta = rng.normal(scale=0.8, size=p)
    logits = X @ beta + 0.2
    prob = 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))
    y = (rng.random(n) < prob).astype(np.float64)
    return X.astype(np.float64), y


def make_cox_data(
    rng: np.random.Generator, n: int, p: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def as_device(arr: np.ndarray, device: str):
    if device == "cuda":
        return cp.asarray(arr)
    return arr


def time_fit(
    factory: Callable[[], Any],
    fit_call: Callable[[Any], None],
    warmup_runs: int,
    repeats: int,
    model_name: str = "",
) -> Tuple[bool, List[float], str]:
    try:
        for i in range(warmup_runs):
            log(f"  warmup {i + 1}/{warmup_runs} ...")
            m = factory()
            fit_call(m)
            del m
            if HAS_CUPY and cp is not None:
                cp.cuda.Stream.null.synchronize()
    except Exception as e:
        return False, [], f"warmup failed: {type(e).__name__}: {e}"

    times_ms: List[float] = []
    for i in range(repeats):
        try:
            log(f"  repeat {i + 1}/{repeats} ...")
            m = factory()
            t0 = time.perf_counter()
            fit_call(m)
            if HAS_CUPY and cp is not None:
                cp.cuda.Stream.null.synchronize()
            t1 = time.perf_counter()
            elapsed_ms = (t1 - t0) * 1000.0
            times_ms.append(elapsed_ms)
            log(f"  repeat {i + 1}/{repeats} done: {elapsed_ms:.1f} ms")
            del m
        except Exception as e:
            return False, times_ms, f"repeat failed: {type(e).__name__}: {e}"
    return True, times_ms, ""


def summarize(model: str, device: str, repeats: int, ok: bool, times: List[float], err: str) -> CaseResult:
    if not ok or not times:
        return CaseResult(
            model=model,
            device=device,
            mean_ms=math.nan,
            std_ms=math.nan,
            min_ms=math.nan,
            max_ms=math.nan,
            repeats=repeats,
            ok=False,
            error=err or "unknown error",
        )
    return CaseResult(
        model=model,
        device=device,
        mean_ms=float(statistics.mean(times)),
        std_ms=float(statistics.pstdev(times) if len(times) > 1 else 0.0),
        min_ms=float(min(times)),
        max_ms=float(max(times)),
        repeats=repeats,
        ok=True,
        error="",
    )


def print_table(rows: List[CaseResult]) -> None:
    print("\n=== Large-Scale Runtime Benchmark (fit only) ===")
    print(f"{'model':<20} {'device':<8} {'mean_ms':>12} {'std_ms':>10} {'min_ms':>10} {'max_ms':>10} {'ok':>6}")
    for r in rows:
        if r.ok:
            print(
                f"{r.model:<20} {r.device:<8} "
                f"{r.mean_ms:>12.2f} {r.std_ms:>10.2f} {r.min_ms:>10.2f} {r.max_ms:>10.2f} {'yes':>6}"
            )
        else:
            print(f"{r.model:<20} {r.device:<8} {'-':>12} {'-':>10} {'-':>10} {'-':>10} {'no':>6}")
            print(f"  error: {r.error}")


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    log("==============================================")
    log(f"benchmark_all_methods_large_scale.py starting")
    log(f"  devices={args.devices}  repeats={args.repeats}  warmup={args.warmup_runs}")
    log(f"  regression:  n={args.n_reg}, p={args.p_reg}")
    log(f"  logistic:    n={args.n_logit}, p={args.p_logit}")
    log(f"  cox:         n={args.n_cox}, p={args.p_cox}")
    log(f"  compute_inference={args.compute_inference}  gpu_memory_cleanup={args.gpu_memory_cleanup}")
    log("==============================================")

    requested_devices = [d.strip().lower() for d in args.devices.split(",") if d.strip()]
    valid_devices = []
    for d in requested_devices:
        if d not in ("cpu", "cuda"):
            raise ValueError(f"Unsupported device '{d}', expected 'cpu' or 'cuda'")
        if d == "cuda":
            if not HAS_CUPY:
                log("[skip] cuda requested but CuPy is unavailable.")
                continue
            if not cuda_available():
                log("[skip] cuda requested but CUDA runtime is unavailable.")
                continue
        valid_devices.append(d)
    if not valid_devices:
        raise RuntimeError("No valid devices to run.")

    log(f"Active devices: {valid_devices}")

    # Build data once (outside timing)
    log("Generating regression data ...")
    X_reg_np, y_reg_np = make_regression_data(rng, args.n_reg, args.p_reg)
    log(f"  X_reg {X_reg_np.shape}, y_reg {y_reg_np.shape}")

    log("Generating logistic data ...")
    X_log_np, y_log_np = make_logistic_data(rng, args.n_logit, args.p_logit)
    log(f"  X_log {X_log_np.shape}, y_log {y_log_np.shape}")

    log("Generating Cox data ...")
    X_cox_np, t_cox_np, e_cox_np = make_cox_data(rng, args.n_cox, args.p_cox)
    log(f"  X_cox {X_cox_np.shape}")

    rows: List[CaseResult] = []

    for device in valid_devices:
        log(f"----------------------------------------------")
        log(f"[device={device}] transferring data ...")
        X_reg = as_device(X_reg_np, device)
        y_reg = as_device(y_reg_np, device)
        X_log = as_device(X_log_np, device)
        y_log = as_device(y_log_np, device)
        X_cox = as_device(X_cox_np, device)
        t_cox = as_device(t_cox_np, device)
        e_cox = as_device(e_cox_np, device)

        common_kwargs: Dict[str, Any] = {
            "device": device,
            "gpu_memory_cleanup": bool(args.gpu_memory_cleanup),
        }

        cases: List[Tuple[str, Callable[[], Any], Callable[[Any], None]]] = [
            (
                "LinearRegression",
                lambda ck=common_kwargs: LinearRegression(
                    compute_inference=bool(args.compute_inference),
                    cov_type="nonrobust",
                    **ck,
                ),
                lambda m, X=X_reg, y=y_reg: m.fit(X, y),
            ),
            (
                "Ridge",
                lambda ck=common_kwargs: Ridge(alpha=1.0, **ck),
                lambda m, X=X_reg, y=y_reg: m.fit(X, y),
            ),
            (
                "Lasso",
                lambda ck=common_kwargs: Lasso(
                    alpha=0.05,
                    max_iter=3000,
                    tol=1e-5,
                    solver="fista",
                    cpu_solver="fista",
                    compute_inference=bool(args.compute_inference),
                    **ck,
                ),
                lambda m, X=X_reg, y=y_reg: m.fit(X, y),
            ),
            (
                "LogisticRegression",
                lambda ck=common_kwargs: LogisticRegression(
                    C=1.0,
                    max_iter=150,
                    tol=1e-5,
                    compute_inference=bool(args.compute_inference),
                    cov_type="nonrobust",
                    **ck,
                ),
                lambda m, X=X_log, y=y_log: m.fit(X, y),
            ),
            (
                "CoxPH",
                lambda ck=common_kwargs: CoxPH(
                    ties="breslow",
                    max_iter=120,
                    tol=1e-8,
                    compute_inference=bool(args.compute_inference),
                    **ck,
                ),
                lambda m, X=X_cox, t=t_cox, e=e_cox: m.fit(X, t, e),
            ),
        ]

        log(f"[device={device}] running {len(cases)} models ...")
        for idx, (name, factory, fit_call) in enumerate(cases, 1):
            log(f"[device={device}] ({idx}/{len(cases)}) {name} ...")
            t_start = time.perf_counter()
            ok, times, err = time_fit(factory, fit_call, args.warmup_runs, args.repeats, name)
            elapsed = time.perf_counter() - t_start
            result = summarize(name, device, args.repeats, ok, times, err)
            rows.append(result)
            if result.ok:
                log(f"[device={device}] ({idx}/{len(cases)}) {name} DONE — "
                    f"mean={result.mean_ms:.1f}ms  std={result.std_ms:.1f}ms  "
                    f"min={result.min_ms:.1f}ms  max={result.max_ms:.1f}ms  "
                    f"(total wall {elapsed:.1f}s)")
            else:
                log(f"[device={device}] ({idx}/{len(cases)}) {name} FAILED: {result.error}")

    print_table(rows)

    if args.json_out:
        out_path = Path(args.json_out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": vars(args),
            "results": [r.__dict__ for r in rows],
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        log(f"Saved JSON: {out_path}")

    log("==============================================")
    log("benchmark_all_methods_large_scale.py complete")
    log("==============================================")


if __name__ == "__main__":
    main()
