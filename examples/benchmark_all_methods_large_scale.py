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
    coef_diff: float = math.nan   # max-abs diff vs statgpu-cpu reference (nan = not compared)


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
    p.add_argument(
        "--include-external",
        action="store_true",
        help="Also benchmark statsmodels and sklearn and compare coefficients vs statgpu-cpu.",
    )

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


def time_external_fit(
    full_fit_fn: Callable[[], Any],
    warmup_runs: int,
    repeats: int,
) -> Tuple[bool, List[float], str, Any]:
    """
    Timing helper for external frameworks (statsmodels / sklearn).

    Unlike time_fit, the callable does the full fit and *returns* the fitted
    object so the caller can extract coefficients for comparison.
    """
    try:
        for i in range(warmup_runs):
            log(f"  warmup {i + 1}/{warmup_runs} ...")
            full_fit_fn()
    except Exception as e:
        return False, [], f"warmup failed: {type(e).__name__}: {e}", None

    times_ms: List[float] = []
    last_result: Any = None
    for i in range(repeats):
        try:
            log(f"  repeat {i + 1}/{repeats} ...")
            t0 = time.perf_counter()
            last_result = full_fit_fn()
            t1 = time.perf_counter()
            elapsed_ms = (t1 - t0) * 1000.0
            times_ms.append(elapsed_ms)
            log(f"  repeat {i + 1}/{repeats} done: {elapsed_ms:.1f} ms")
        except Exception as e:
            return False, times_ms, f"repeat failed: {type(e).__name__}: {e}", last_result
    return True, times_ms, "", last_result


def summarize(model: str, device: str, repeats: int, ok: bool, times: List[float], err: str,
              coef_diff: float = math.nan) -> CaseResult:
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
            coef_diff=coef_diff,
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
        coef_diff=coef_diff,
    )


def print_table(rows: List[CaseResult]) -> None:
    has_diff = any(not math.isnan(r.coef_diff) for r in rows)
    hdr = (f"{'model':<22} {'device':<24} {'mean_ms':>10} {'std_ms':>8} "
           f"{'min_ms':>8} {'max_ms':>8} {'ok':>4}")
    if has_diff:
        hdr += f"  {'coef_diff':>10}  note"
    print("\n=== Large-Scale Runtime Benchmark (fit only) ===")
    print(hdr)
    for r in rows:
        if r.ok:
            line = (f"{r.model:<22} {r.device:<24} "
                    f"{r.mean_ms:>10.2f} {r.std_ms:>8.2f} "
                    f"{r.min_ms:>8.2f} {r.max_ms:>8.2f} {'yes':>4}")
        else:
            line = (f"{r.model:<22} {r.device:<24} "
                    f"{'—':>10} {'—':>8} {'—':>8} {'—':>8} {'no':>4}")
        if has_diff:
            if math.isnan(r.coef_diff):
                line += f"  {'—':>10}"
            else:
                flag = "  [WARN: large diff]" if r.coef_diff > 0.05 else ""
                line += f"  {r.coef_diff:>10.2e}{flag}"
        print(line)
        if not r.ok:
            print(f"  error: {r.error}")
    if has_diff:
        print("  coef_diff = max|coef_statgpu_cpu - coef_external|; [WARN] if > 0.05")


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

    # ── External frameworks (statsmodels / sklearn) ────────────────────────────
    if args.include_external:
        log("----------------------------------------------")
        log("[external] Building statgpu-cpu reference coefficients ...")

        def _safe_diff(ref: np.ndarray, ext) -> float:
            """Max absolute difference between two coefficient vectors."""
            try:
                a = np.asarray(ref, dtype=float).reshape(-1)
                b = np.asarray(ext, dtype=float).reshape(-1)
                n = min(len(a), len(b))
                return float(np.max(np.abs(a[:n] - b[:n])))
            except Exception:
                return math.nan

        # One-off statgpu-cpu fits to get reference coefficients.
        # LogisticRegression uses two refs: C=1.0 (for sklearn) and C=1e6 (for statsmodels).
        _r_lin = LinearRegression(compute_inference=False, device="cpu", cov_type="nonrobust")
        _r_lin.fit(X_reg_np, y_reg_np)
        ref_lin = np.r_[_r_lin.intercept_, _r_lin.coef_]

        _r_rid = Ridge(alpha=1.0, device="cpu")
        _r_rid.fit(X_reg_np, y_reg_np)
        ref_rid = np.r_[_r_rid.intercept_, _r_rid.coef_]

        _r_las = Lasso(alpha=0.05, max_iter=3000, tol=1e-5, solver="fista",
                       cpu_solver="fista", compute_inference=False, device="cpu")
        _r_las.fit(X_reg_np, y_reg_np)
        ref_las = np.r_[_r_las.intercept_, _r_las.coef_]

        # Regularized (C=1.0, alpha_statgpu=0.5) for sklearn comparison (sklearn C=2.0 → alpha=0.5).
        _r_log_reg = LogisticRegression(C=1.0, max_iter=150, tol=1e-5,
                                        compute_inference=False, cov_type="nonrobust", device="cpu")
        _r_log_reg.fit(X_log_np, y_log_np)
        ref_log_reg = np.r_[_r_log_reg.intercept_, _r_log_reg.coef_]

        # Unregularized (C=1e6) for statsmodels.Logit comparison.
        _r_log_unr = LogisticRegression(C=1e6, max_iter=300, tol=1e-8,
                                         compute_inference=False, cov_type="nonrobust", device="cpu")
        _r_log_unr.fit(X_log_np, y_log_np)
        ref_log_unr = np.r_[_r_log_unr.intercept_, _r_log_unr.coef_]

        _r_cox = CoxPH(ties="breslow", max_iter=120, tol=1e-8,
                       compute_inference=False, device="cpu")
        _r_cox.fit(X_cox_np, t_cox_np, e_cox_np)
        ref_cox = _r_cox.coef_.copy()

        log("[external] References built.")

        # ── statsmodels ──────────────────────────────────────────────────────
        try:
            import statsmodels.api as _sm
            import statsmodels.duration.api as _smd

            log("[external] statsmodels found.")

            # (name, device_label, fit_fn, coef_extractor, ref_array, note)
            sm_cases = [
                (
                    "LinearRegression", "statsmodels.OLS",
                    lambda: _sm.OLS(y_reg_np, _sm.add_constant(X_reg_np)).fit(disp=0),
                    lambda r: r.params,       # [intercept, coef...]
                    ref_lin, "",
                ),
                (
                    "LogisticRegression", "statsmodels.Logit",
                    # Unregularized; compare against statgpu C=1e6 ref (≈no-reg)
                    lambda: _sm.Logit(y_log_np, _sm.add_constant(X_log_np)).fit(
                        disp=0, maxiter=300, method="newton"),
                    lambda r: r.params,
                    ref_log_unr, "no reg → compared vs statgpu C=1e6",
                ),
                (
                    "CoxPH", "statsmodels.PHReg",
                    lambda: _smd.PHReg(
                        t_cox_np, X_cox_np, status=e_cox_np.astype(bool),
                        ties="breslow",
                    ).fit(disp=0),
                    lambda r: r.params,       # coefs only, no intercept for Cox
                    ref_cox, "",
                ),
            ]

            for idx, (name, fw, fit_fn, coef_fn, ref, note) in enumerate(sm_cases, 1):
                log(f"[external] ({idx}/{len(sm_cases)}) {fw} {name} ...")
                t_wall = time.perf_counter()
                ok, times, err, result = time_external_fit(fit_fn, args.warmup_runs, args.repeats)
                elapsed = time.perf_counter() - t_wall
                cd = _safe_diff(ref, coef_fn(result)) if ok and result is not None else math.nan
                res = summarize(name, fw, args.repeats, ok, times, err, cd)
                rows.append(res)
                if res.ok:
                    diff_str = f"coef_diff={cd:.2e}" if not math.isnan(cd) else "coef_diff=n/a"
                    log(f"[external] ({idx}/{len(sm_cases)}) {fw} {name} DONE — "
                        f"mean={res.mean_ms:.1f}ms  {diff_str}  (wall {elapsed:.1f}s)"
                        + (f"  [{note}]" if note else ""))
                else:
                    log(f"[external] ({idx}/{len(sm_cases)}) {fw} {name} FAILED: {res.error}")

        except ImportError as exc:
            log(f"[external] statsmodels not available ({exc}), skipping.")

        # ── sklearn ──────────────────────────────────────────────────────────
        try:
            from sklearn.linear_model import LinearRegression as _SkLin
            from sklearn.linear_model import Ridge as _SkRidge
            from sklearn.linear_model import Lasso as _SkLasso
            from sklearn.linear_model import LogisticRegression as _SkLogit

            log("[external] sklearn found.")

            # Note on LogisticRegression convention:
            #   statgpu uses alpha = 1/(2C) → C=1.0 gives alpha=0.5
            #   sklearn  uses alpha = 1/C   → C=2.0 gives alpha=0.5
            # We use sklearn C=2.0 to match statgpu C=1.0.
            sk_cases = [
                (
                    "LinearRegression", "sklearn.LinearRegression",
                    lambda: _SkLin().fit(X_reg_np, y_reg_np),
                    lambda r: np.r_[r.intercept_, r.coef_],
                    ref_lin, "",
                ),
                (
                    "Ridge", "sklearn.Ridge(α=1.0)",
                    lambda: _SkRidge(alpha=1.0).fit(X_reg_np, y_reg_np),
                    lambda r: np.r_[r.intercept_, r.coef_],
                    ref_rid, "",
                ),
                (
                    "Lasso", "sklearn.Lasso(α=0.05)",
                    lambda: _SkLasso(alpha=0.05, max_iter=3000, tol=1e-5).fit(X_reg_np, y_reg_np),
                    lambda r: np.r_[r.intercept_, r.coef_],
                    ref_las, "",
                ),
                (
                    "LogisticRegression", "sklearn.LogisticReg(C=2)",
                    # C=2.0 → alpha=0.5, matching statgpu C=1.0 → alpha=0.5
                    lambda: _SkLogit(
                        C=2.0, solver="lbfgs", max_iter=150, tol=1e-5,
                    ).fit(X_log_np, y_log_np.astype(int)),
                    lambda r: np.r_[r.intercept_[0], r.coef_[0]],
                    ref_log_reg, "C=2.0 matches statgpu C=1.0 (alpha=0.5)",
                ),
                # CoxPH: no sklearn API → omitted
            ]

            for idx, (name, fw, fit_fn, coef_fn, ref, note) in enumerate(sk_cases, 1):
                log(f"[external] ({idx}/{len(sk_cases)}) {fw} {name} ...")
                t_wall = time.perf_counter()
                ok, times, err, result = time_external_fit(fit_fn, args.warmup_runs, args.repeats)
                elapsed = time.perf_counter() - t_wall
                cd = _safe_diff(ref, coef_fn(result)) if ok and result is not None else math.nan
                res = summarize(name, fw, args.repeats, ok, times, err, cd)
                rows.append(res)
                if res.ok:
                    diff_str = f"coef_diff={cd:.2e}" if not math.isnan(cd) else "coef_diff=n/a"
                    log(f"[external] ({idx}/{len(sk_cases)}) {fw} {name} DONE — "
                        f"mean={res.mean_ms:.1f}ms  {diff_str}  (wall {elapsed:.1f}s)"
                        + (f"  [{note}]" if note else ""))
                else:
                    log(f"[external] ({idx}/{len(sk_cases)}) {fw} {name} FAILED: {res.error}")

        except ImportError as exc:
            log(f"[external] sklearn not available ({exc}), skipping.")

        # Reprint the full table now that external rows are appended.
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
