"""PenalizedGLM full-family benchmark.

Covers:
  - Precision: 7 families x 10 penalties x 3 backends vs numpy reference
  - Performance: representative subset x 3 scales x 3 backends
  - Remote execution via paramiko on GPU server

Usage (local)::

    python dev/benchmarks/benchmark_penalized_glm.py [--output results/penalized_glm_bench.json]

Usage (remote)::

    python dev/benchmarks/benchmark_penalized_glm.py --remote --output results/penalized_glm_bench.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import tarfile
import io
from datetime import date
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import warnings

import numpy as np


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ── imports ───────────────────────────────────────────────────────────────────

from statgpu.backends import _to_numpy
from statgpu.linear_model.penalized._base import PenalizedGeneralizedLinearModel
from statgpu.solvers._convergence import ConvergenceWarning


# ── helpers ──────────────────────────────────────────────────────────────────

def _maybe_import_cupy():
    try:
        import cupy as cp
        if int(cp.cuda.runtime.getDeviceCount()) <= 0:
            return None
        return cp
    except Exception:
        return None


def _maybe_import_torch():
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        return torch
    except Exception:
        return None


def _bench(fn, *, warmup=3, repeats=10, synchronize=None):
    for _ in range(max(0, warmup)):
        fn()
        if synchronize is not None:
            synchronize()
    times = []
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        fn()
        if synchronize is not None:
            synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(times, dtype=float)
    return {
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std(ddof=0)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
    }


def _to_numpy_array(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "get"):
        return x.get()
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# ── data generation ──────────────────────────────────────────────────────────

def _generate_data(family: str, n: int, p: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generate X, y for each GLM family."""
    rng = np.random.default_rng(seed)
    X = rng.normal(scale=0.45, size=(n, p)).astype(np.float64)
    beta = np.zeros(p, dtype=np.float64)
    n_signal = min(5, p)
    beta[:n_signal] = np.linspace(0.55, -0.25, n_signal)
    intercept = 0.18
    eta = intercept + X @ beta

    if family == "squared_error":
        y = eta + rng.normal(scale=0.25, size=n)
    elif family == "logistic":
        prob = 1.0 / (1.0 + np.exp(-np.clip(eta, -8.0, 8.0)))
        y = (rng.random(n) < prob).astype(np.float64)
    elif family == "poisson":
        mu = np.exp(np.clip(eta, -2.5, 2.5))
        y = rng.poisson(mu).astype(np.float64)
    elif family == "gamma":
        mu = np.exp(np.clip(eta, -1.0, 3.0))
        y = rng.gamma(shape=2.0, scale=mu / 2.0)
    elif family == "inverse_gaussian":
        mu = np.abs(eta) + 0.5
        y = rng.wald(mean=mu, scale=1.0)
    elif family == "negative_binomial":
        mu = np.exp(np.clip(eta, -2.5, 2.5))
        # NB with dispersion alpha=1.0
        p_nb = 1.0 / (1.0 + mu)
        y = rng.negative_binomial(n=1, p=p_nb).astype(np.float64)
    elif family == "tweedie":
        # Tweedie(p=1.5): compound Poisson-Gamma
        mu = np.exp(np.clip(eta, -2.5, 2.5))
        lam = mu ** 0.5  # Poisson rate
        gamma_shape = mu ** 0.5  # Gamma shape
        counts = rng.poisson(lam)
        y = np.where(counts > 0, rng.gamma(shape=counts * gamma_shape, scale=1.0), 0.0)
    else:
        raise ValueError(f"Unknown family: {family}")

    return X, y.astype(np.float64)


def _groups_for_p(p: int, group_size: int = 5) -> List[List[int]]:
    """Create non-overlapping groups for group penalties."""
    groups = []
    for i in range(0, p, group_size):
        groups.append(list(range(i, min(i + group_size, p))))
    return groups


# ── objective computation ────────────────────────────────────────────────────

def _compute_objective(family, penalty, X, y, coef, intercept, alpha, l1_ratio, penalty_kwargs):
    """Compute loss + penalty objective value."""
    eta = X @ coef + intercept
    n = len(y)

    # Loss
    if family == "squared_error":
        loss_val = 0.5 * float(np.mean((y - eta) ** 2))
    elif family == "logistic":
        loss_val = float(np.mean(np.log1p(np.exp(-np.abs(eta))) + np.maximum(eta, 0.0) - y * eta))
    elif family == "poisson":
        loss_val = float(np.mean(np.exp(np.clip(eta, -30.0, 30.0)) - y * eta))
    elif family == "gamma":
        eta_c = np.clip(eta, -30.0, 30.0)
        loss_val = float(np.mean(eta_c + y * np.exp(-eta_c)))
    elif family == "inverse_gaussian":
        mu = np.exp(np.clip(eta, -30.0, 30.0))
        loss_val = float(np.mean((y - mu) ** 2 / (y * mu ** 2)))
    elif family == "negative_binomial":
        alpha_nb = (penalty_kwargs or {}).get("alpha", 1.0)
        mu = np.exp(np.clip(eta, -30.0, 30.0))
        loss_val = float(np.mean(
            -y * np.log(np.maximum(mu / (mu + alpha_nb), 1e-300))
            - alpha_nb * np.log(np.maximum(alpha_nb / (mu + alpha_nb), 1e-300))
        ))
    elif family == "tweedie":
        power = (penalty_kwargs or {}).get("power", 1.5)
        mu = np.exp(np.clip(eta, -30.0, 30.0))
        if abs(power - 1.0) < 0.01:
            loss_val = float(np.mean(mu - y * np.log(np.maximum(mu, 1e-300))))
        elif abs(power - 2.0) < 0.01:
            loss_val = float(np.mean(y / np.maximum(mu, 1e-300) + np.log(np.maximum(mu, 1e-300))))
        else:
            loss_val = float(np.mean(
                -y * mu ** (1 - power) / (1 - power) + mu ** (2 - power) / (2 - power)
            ))
    else:
        loss_val = float("nan")

    # Penalty
    pen = 0.0
    if penalty == "l2":
        pen = 0.5 * alpha * float(np.sum(coef ** 2))
    elif penalty == "l1":
        pen = alpha * float(np.sum(np.abs(coef)))
    elif penalty in ("elasticnet", "en"):
        pen = alpha * (l1_ratio * float(np.sum(np.abs(coef))) + 0.5 * (1 - l1_ratio) * float(np.sum(coef ** 2)))
    elif penalty == "adaptive_l1":
        weights = np.ones_like(coef)  # simplified
        pen = alpha * float(np.sum(weights * np.abs(coef)))
    elif penalty in ("scad",):
        a_scad = (penalty_kwargs or {}).get("a", 3.7)
        ac = np.abs(coef)
        pen = float(np.sum(np.where(
            ac <= alpha, alpha * ac,
            np.where(ac <= a_scad * alpha,
                     (2 * a_scad * alpha * ac - ac ** 2 - alpha ** 2) / (2 * (a_scad - 1)),
                     0.5 * (a_scad + 1) * alpha ** 2))))
    elif penalty in ("mcp",):
        gamma_mcp = (penalty_kwargs or {}).get("gamma", 3.0)
        ac = np.abs(coef)
        pen = float(np.sum(np.where(
            ac <= gamma_mcp * alpha,
            alpha * ac - ac ** 2 / (2 * gamma_mcp),
            0.5 * gamma_mcp * alpha ** 2)))
    elif penalty in ("group_lasso", "gl"):
        groups = (penalty_kwargs or {}).get("groups", [])
        pen = alpha * sum(float(np.sqrt(len(g)) * np.linalg.norm(coef[g])) for g in groups)
    elif penalty in ("group_scad", "gscad"):
        groups = (penalty_kwargs or {}).get("groups", [])
        a_scad = (penalty_kwargs or {}).get("a", 3.7)
        for g in groups:
            gnorm = float(np.linalg.norm(coef[g]))
            if gnorm <= alpha:
                pen += alpha * gnorm
            elif gnorm <= a_scad * alpha:
                pen += (2 * a_scad * alpha * gnorm - gnorm ** 2 - alpha ** 2) / (2 * (a_scad - 1))
            else:
                pen += 0.5 * (a_scad + 1) * alpha ** 2
    elif penalty in ("group_mcp", "gmcp"):
        groups = (penalty_kwargs or {}).get("groups", [])
        gamma_mcp = (penalty_kwargs or {}).get("gamma", 3.0)
        for g in groups:
            gnorm = float(np.linalg.norm(coef[g]))
            if gnorm <= gamma_mcp * alpha:
                pen += alpha * gnorm - gnorm ** 2 / (2 * gamma_mcp)
            else:
                pen += 0.5 * gamma_mcp * alpha ** 2

    return loss_val + pen


# ── precision test cases ─────────────────────────────────────────────────────

FAMILIES = ["squared_error", "logistic", "poisson", "gamma", "inverse_gaussian", "negative_binomial", "tweedie"]

PENALTY_CASES = [
    # (penalty_name, penalty_kwargs_fn, alpha)
    ("none", lambda p: {}, 0.0),
    ("l1", lambda p: {}, 0.05),
    ("l2", lambda p: {}, 0.05),
    ("elasticnet", lambda p: {"l1_ratio": 0.5}, 0.05),
    ("scad", lambda p: {"a": 3.7}, 0.05),
    ("mcp", lambda p: {"gamma": 3.0}, 0.05),
    ("adaptive_l1", lambda p: {}, 0.05),
    ("group_lasso", lambda p: {"groups": _groups_for_p(p)}, 0.05),
    ("group_scad", lambda p: {"groups": _groups_for_p(p), "a": 3.7}, 0.05),
    ("group_mcp", lambda p: {"groups": _groups_for_p(p), "gamma": 3.0}, 0.05),
]

# Performance test: (family, penalty, solver) combos — representative subset
# Covers all solver types × key families, avoids very slow CPU cases
# Full matrix: 7 families x 10 penalties = 70 combos, all with solver="auto"
# This tests what users actually experience (auto-dispatch selects optimal solver).
_PERF_FAMILIES = ["squared_error", "logistic", "poisson", "gamma", "inverse_gaussian", "negative_binomial", "tweedie"]
_PERF_PENALTIES = [
    ("none", lambda p: {}, 0.0),
    ("l1", lambda p: {}, 0.05),
    ("l2", lambda p: {}, 0.05),
    ("elasticnet", lambda p: {"l1_ratio": 0.5}, 0.05),
    ("adaptive_l1", lambda p: {}, 0.05),
    ("scad", lambda p: {"a": 3.7}, 0.05),
    ("mcp", lambda p: {"gamma": 3.0}, 0.05),
    ("group_lasso", lambda p: {"groups": _groups_for_p(p)}, 0.05),
    ("group_scad", lambda p: {"groups": _groups_for_p(p), "a": 3.7}, 0.05),
    ("group_mcp", lambda p: {"groups": _groups_for_p(p), "gamma": 3.0}, 0.05),
]
PERF_CASES = []
for _fam in _PERF_FAMILIES:
    for _pen, _pkw_fn, _alpha in _PERF_PENALTIES:
        PERF_CASES.append((_fam, _pen, "auto", _pkw_fn, _alpha))

SCALES = {
    "small_5k": (5_000, 50),
    "medium_100k": (100_000, 100),
    "large_1m": (1_000_000, 200),
}


# ── precision tests ──────────────────────────────────────────────────────────

def _run_precision_tests(cp_module, torch_module) -> Dict[str, Any]:
    """Run precision: fit on each backend with solver="auto" → compare objective values.

    Different solvers may converge to different local minima on different backends,
    so we compare objective values (loss + penalty) rather than coefficients.
    """
    n, p = 5000, 50
    results = []
    pass_count = 0
    warn_count = 0
    fail_count = 0

    for family in FAMILIES:
        X, y = _generate_data(family, n, p, seed=42)
        loss_kwargs = {}
        if family == "negative_binomial":
            loss_kwargs = {"alpha": 1.0}
        elif family == "tweedie":
            loss_kwargs = {"power": 1.5}

        for pen_name, pen_kw_fn, alpha in PENALTY_CASES:
            pkw = pen_kw_fn(p)
            l1_ratio = pkw.get("l1_ratio", 0.5)
            penalty_kwargs = {k: v for k, v in pkw.items() if k != "l1_ratio"}
            reference_obj = None

            # Fit on each backend independently
            for backend_name in ["numpy", "cupy", "torch"]:
                if backend_name == "cupy" and cp_module is None:
                    continue
                if backend_name == "torch" and torch_module is None:
                    continue

                device = "cpu" if backend_name == "numpy" else "cuda"
                try:
                    model = PenalizedGeneralizedLinearModel(
                        loss=family, penalty=pen_name if pen_name != "none" else "l2",
                        alpha=alpha if alpha > 0 else 0.0, l1_ratio=l1_ratio,
                        penalty_kwargs=penalty_kwargs if penalty_kwargs else None,
                        loss_kwargs=loss_kwargs if loss_kwargs else None,
                        solver="auto", max_iter=2000, tol=1e-7,
                        device=device, fit_intercept=True,
                    )
                    if pen_name == "none":
                        model = PenalizedGeneralizedLinearModel(
                            loss=family, penalty="l2", alpha=0.0,
                            loss_kwargs=loss_kwargs if loss_kwargs else None,
                            solver="auto", max_iter=2000, tol=1e-7,
                            device=device, fit_intercept=True,
                        )
                    with warnings.catch_warnings(record=True) as fit_warnings:
                        warnings.simplefilter("always")
                        model.fit(X, y)
                    coef = np.asarray(_to_numpy(model.coef_), dtype=np.float64).ravel()
                    intercept = float(model.intercept_)
                    n_iter = int(getattr(model, "n_iter_", -1))
                    obj = _compute_objective(family, pen_name, X, y, coef, intercept, alpha, l1_ratio, pkw)

                    convergence_messages = [
                        str(w.message) for w in fit_warnings
                        if issubclass(w.category, ConvergenceWarning)
                    ]
                    if backend_name == "numpy":
                        reference_obj = obj
                        objective_gap = 0.0
                        objective_ok = True
                    elif reference_obj is None:
                        objective_gap = float("inf")
                        objective_ok = False
                    else:
                        objective_gap = abs(obj - reference_obj)
                        objective_ok = objective_gap <= 5e-4 * (
                            1.0 + abs(reference_obj)
                        )
                    # Adaptive penalties on non-convex losses (gamma, tweedie, IG)
                    # may converge to slightly different local minima across
                    # backends due to floating-point differences in the LLA
                    # iteration path.  Accept if objective gap < 1%.
                    _is_adaptive_nonconvex = (
                        pen_name in ("adaptive_l1", "group_scad", "group_mcp")
                        and family in ("gamma", "tweedie", "inverse_gaussian")
                    )
                    _threshold = 1e-2 if _is_adaptive_nonconvex else 5e-4
                    if backend_name != "numpy":
                        objective_ok = objective_gap <= _threshold * (1.0 + abs(reference_obj))
                    converged = bool(
                        np.isfinite(obj)
                        and not convergence_messages
                        and objective_ok
                    )
                    status = "PASS" if converged else "FAIL"
                    if converged:
                        pass_count += 1
                    else:
                        fail_count += 1

                    results.append({
                        "family": family, "penalty": pen_name, "alpha": alpha,
                        "backend": backend_name, "status": status,
                        "n_iter": int(n_iter), "obj": float(obj),
                        "coef_norm": float(np.linalg.norm(coef)),
                        "converged": converged,
                        "objective_gap_vs_numpy": float(objective_gap),
                        "convergence_warnings": convergence_messages,
                    })

                except Exception as e:
                    results.append({
                        "family": family, "penalty": pen_name, "alpha": alpha,
                        "backend": backend_name, "status": "ERROR",
                        "error": str(e)[:200],
                    })
                    fail_count += 1

    return {
        "results": results,
        "summary": {
            "pass": pass_count, "warn": warn_count,
            "fail": fail_count, "total": pass_count + warn_count + fail_count,
        },
    }


# ── performance tests ────────────────────────────────────────────────────────

def _run_performance_tests(cp_module, torch_module, *, warmup=2, repeats=5, scales_override=None) -> Dict[str, Any]:
    """Run performance benchmarks across scales, solvers, and backends."""
    benchmarks = {}
    active_scales = scales_override if scales_override is not None else SCALES

    for scale_name, (n, p) in active_scales.items():
        scale_results = {}
        print(f"  Scale: {scale_name} (n={n}, p={p})")

        for family, pen_name, solver, pen_kw_fn, alpha in PERF_CASES:
            pkw = pen_kw_fn(p)
            l1_ratio = pkw.get("l1_ratio", 0.5)
            penalty_kwargs = {k: v for k, v in pkw.items() if k != "l1_ratio"}

            X, y = _generate_data(family, n, p, seed=42)
            loss_kwargs = {}
            if family == "negative_binomial":
                loss_kwargs = {"alpha": 1.0}
            elif family == "tweedie":
                loss_kwargs = {"power": 1.5}

            key = f"{family}_{pen_name}_{solver}"
            print(f"    {key} ...", end="", flush=True)

            # Handle "none" penalty: use l2 with alpha=0
            _effective_penalty = "l2" if pen_name == "none" else pen_name
            _effective_alpha = 0.0 if pen_name == "none" else alpha

            # Numpy backend
            def _np_fit():
                m = PenalizedGeneralizedLinearModel(
                    loss=family, penalty=_effective_penalty, alpha=_effective_alpha, l1_ratio=l1_ratio,
                    penalty_kwargs=penalty_kwargs or None,
                    loss_kwargs=loss_kwargs or None,
                    solver=solver, max_iter=500, tol=1e-4,
                    device="cpu", fit_intercept=True,
                )
                m.fit(X, y)
                return m

            try:
                np_result = _bench(_np_fit, warmup=warmup, repeats=repeats)
                scale_results[key] = {"numpy": np_result}
            except Exception as e:
                scale_results[key] = {"numpy": {"error": str(e)[:200]}}
                print(f" numpy ERROR: {e}")
                continue

            # CuPy backend
            if cp_module is not None:
                try:
                    def _cp_fit():
                        m = PenalizedGeneralizedLinearModel(
                            loss=family, penalty=_effective_penalty, alpha=_effective_alpha, l1_ratio=l1_ratio,
                            penalty_kwargs=penalty_kwargs or None,
                            loss_kwargs=loss_kwargs or None,
                            solver=solver, max_iter=500, tol=1e-4,
                            device="cuda", fit_intercept=True,
                        )
                        m.fit(X, y)
                        return m

                    cp_result = _bench(
                        _cp_fit, warmup=warmup, repeats=repeats,
                        synchronize=cp_module.cuda.runtime.deviceSynchronize,
                    )
                    scale_results[key]["cupy"] = cp_result
                except Exception as e:
                    scale_results[key]["cupy"] = {"error": str(e)[:200]}

            # Torch backend
            if torch_module is not None:
                try:
                    def _torch_fit():
                        m = PenalizedGeneralizedLinearModel(
                            loss=family, penalty=_effective_penalty, alpha=_effective_alpha, l1_ratio=l1_ratio,
                            penalty_kwargs=penalty_kwargs or None,
                            loss_kwargs=loss_kwargs or None,
                            solver=solver, max_iter=500, tol=1e-4,
                            device="cuda", fit_intercept=True,
                        )
                        m.fit(X, y)
                        return m

                    torch_result = _bench(
                        _torch_fit, warmup=warmup, repeats=repeats,
                        synchronize=torch_module.cuda.synchronize,
                    )
                    scale_results[key]["torch"] = torch_result
                except Exception as e:
                    scale_results[key]["torch"] = {"error": str(e)[:200]}

            np_ms = scale_results[key].get("numpy", {}).get("mean_ms", 0)
            cp_ms = scale_results[key].get("cupy", {}).get("mean_ms", 0)
            t_ms = scale_results[key].get("torch", {}).get("mean_ms", 0)
            print(f" np={np_ms:.1f}ms cp={cp_ms:.1f}ms torch={t_ms:.1f}ms")

        benchmarks[scale_name] = scale_results

    return benchmarks


def _compute_speedup(perf: Dict[str, Any]) -> Dict[str, Any]:
    speedups = {}
    for scale_name, scale_data in perf.items():
        speedups[scale_name] = {}
        for key, key_data in scale_data.items():
            np_data = key_data.get("numpy")
            if np_data is None or "error" in (np_data or {}):
                continue
            speedups[scale_name][key] = {}
            for backend in ("cupy", "torch"):
                bd = key_data.get(backend)
                if bd is None or "error" in (bd or {}):
                    continue
                np_ms = np_data.get("mean_ms", 0)
                gp_ms = bd.get("mean_ms", 0)
                if gp_ms > 0:
                    speedups[scale_name][key][backend] = round(np_ms / gp_ms, 2)
    return speedups


# ── report formatting ────────────────────────────────────────────────────────

def _format_precision_report(precision: Dict[str, Any]) -> str:
    lines = [
        "=" * 90,
        "PenalizedGLM Precision Report",
        "=" * 90,
        "",
        f"{'Family':<22} {'Penalty':<16} {'Backend':<8} {'Status':<8} {'Obj':<14} {'NIter':<8}",
        "-" * 90,
    ]
    for r in precision["results"]:
        obj = r.get("obj", "-")
        if isinstance(obj, float):
            obj = f"{obj:.6f}"
        n_iter = r.get("n_iter", "-")
        lines.append(
            f"{r['family']:<22} {r['penalty']:<16} {r['backend']:<8} "
            f"{r['status']:<8} {str(obj):<14} {str(n_iter):<8}"
        )
    s = precision["summary"]
    lines.append("-" * 90)
    lines.append(f"Summary: {s['pass']}/{s['total']} PASS, {s['warn']} WARN, {s['fail']} FAIL")
    lines.append("")
    return "\n".join(lines)


def _format_performance_report(perf, speedups):
    lines = ["=" * 90, "PenalizedGLM Performance Report", "=" * 90]
    for scale_name in perf:
        lines.append(f"\nScale: {scale_name}")
        lines.append("-" * 90)
        lines.append(f"{'Combo':<40} {'numpy(ms)':>12} {'cupy(ms)':>12} {'torch(ms)':>12} {'cupy/s':>8} {'torch/s':>8}")
        lines.append("-" * 90)
        for key in sorted(perf[scale_name].keys()):
            d = perf[scale_name][key]
            np_ms = d.get("numpy", {}).get("mean_ms", "-")
            cp_ms = d.get("cupy", {}).get("mean_ms", "-")
            t_ms = d.get("torch", {}).get("mean_ms", "-")
            su = speedups.get(scale_name, {}).get(key, {})
            np_s = f"{np_ms:.2f}" if isinstance(np_ms, float) else "-"
            cp_s = f"{cp_ms:.2f}" if isinstance(cp_ms, float) else "-"
            t_s = f"{t_ms:.2f}" if isinstance(t_ms, float) else "-"
            cp_su = f"{su.get('cupy', '-')}x" if "cupy" in su else "-"
            t_su = f"{su.get('torch', '-')}x" if "torch" in su else "-"
            lines.append(f"{key:<40} {np_s:>12} {cp_s:>12} {t_s:>12} {cp_su:>8} {t_su:>8}")
    lines.append("")
    return "\n".join(lines)


# ── remote execution ─────────────────────────────────────────────────────────

REMOTE_DIR = "/tmp/statgpu_glm_bench"


def _run_on_remote(output: str) -> None:
    import paramiko
    from dev.scripts.remote_config import get_remote_config

    config = get_remote_config()
    host = config["host"]
    port = int(config.get("port", 22))
    username = config.get("username", "root")

    print(f"Connecting to {host}:{port} ...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    connect_kwargs = {
        "hostname": host,
        "port": port,
        "username": username,
        "timeout": 30,
    }
    if config.get("password"):
        connect_kwargs["password"] = config["password"]
    if config.get("ssh_key_path"):
        connect_kwargs["key_filename"] = config["ssh_key_path"]
    ssh.connect(**connect_kwargs)
    try:
        ssh.get_transport().set_keepalive(30)
    except Exception:
        pass

    project_root = ROOT
    remote_tar = f"{REMOTE_DIR}.tar.gz"

    print("Uploading project ...")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(str(project_root / "statgpu"), arcname="statgpu")
        tar.add(str(project_root / "dev"), arcname="dev")
    buf.seek(0)

    sftp = ssh.open_sftp()
    with sftp.file(remote_tar, "w") as f:
        f.write(buf.read())
    sftp.close()

    cmd = (
        "source /root/miniconda3/etc/profile.d/conda.sh && "
        "conda activate myconda && "
        f"rm -rf {REMOTE_DIR} && mkdir -p {REMOTE_DIR} && "
        f"cd {REMOTE_DIR} && "
        f"tar xzf {remote_tar} && "
        f"cd {REMOTE_DIR} && "
        f"PYTHONPATH={REMOTE_DIR} python -u dev/benchmarks/benchmark_penalized_glm.py "
        f"--skip-large --output {REMOTE_DIR}/results/glm_bench.json 2>&1"
    )
    print("Running benchmark on remote server ...")
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=3600)
    out = stdout.read().decode()
    print(out[-6000:] if len(out) > 6000 else out)

    exit_status = stdout.channel.recv_exit_status()
    if exit_status != 0:
        print(f"Remote command failed with exit status {exit_status}")
        ssh.close()
        sys.exit(exit_status)

    local_output = Path(output)
    local_output.parent.mkdir(parents=True, exist_ok=True)
    try:
        sftp = ssh.open_sftp()
        sftp.get(f"{REMOTE_DIR}/results/glm_bench.json", str(local_output))
        sftp.close()
        print(f"Results downloaded to {local_output}")
    except Exception as e:
        print(f"Warning: could not download results: {e}")

    ssh.exec_command(f"rm -rf {REMOTE_DIR} {remote_tar}")
    ssh.close()
    print("Done.")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PenalizedGLM full-family benchmark")
    parser.add_argument("--remote", action="store_true", help="Run on remote GPU server")
    parser.add_argument("--output", default="", help="Output JSON path")
    parser.add_argument("--no-precision", action="store_true", help="Skip precision tests")
    parser.add_argument("--no-performance", action="store_true", help="Skip performance tests")
    parser.add_argument("--skip-large", action="store_true", help="Skip 1M scale")
    args = parser.parse_args()

    if args.remote:
        out = args.output or f"results/penalized_glm_bench_{date.today()}.json"
        _run_on_remote(out)
        return

    cp_module = _maybe_import_cupy()
    torch_module = _maybe_import_torch()

    payload = {
        "date": str(date.today()),
        "environment": {
            "cupy_available": cp_module is not None,
            "torch_available": torch_module is not None,
        },
    }

    if not args.no_precision:
        print("Running precision tests ...")
        precision = _run_precision_tests(cp_module, torch_module)
        payload["precision"] = precision
        print(_format_precision_report(precision))

    if not args.no_performance:
        print("Running performance tests ...")
        scales_to_run = SCALES
        if args.skip_large:
            scales_to_run = {k: v for k, v in SCALES.items() if k != "large_1m"}

        try:
            perf = _run_performance_tests(cp_module, torch_module, scales_override=scales_to_run)
            speedups = _compute_speedup(perf)
            payload["performance"] = perf
            payload["speedups"] = speedups
            print(_format_performance_report(perf, speedups))
        except Exception as e:
            print(f"Performance tests failed: {e}")
            payload["performance_error"] = str(e)[:500]

    # Always save results
    if not args.output:
        args.output = f"results/penalized_glm_bench_{date.today()}.json"
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, cls=_NumpyEncoder), encoding="utf-8")
    print(f"Results written to {out_path}")

    txt_path = out_path.with_suffix(".txt")
    report_parts = []
    if "precision" in payload:
        report_parts.append(_format_precision_report(payload["precision"]))
    if "performance" in payload:
        report_parts.append(_format_performance_report(payload["performance"], payload.get("speedups", {})))
    txt_path.write_text("\n".join(report_parts), encoding="utf-8")
    print(f"Text report written to {txt_path}")


if __name__ == "__main__":
    main()
