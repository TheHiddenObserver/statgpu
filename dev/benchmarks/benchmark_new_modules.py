"""Comprehensive benchmark for Panel Data, GAM, and ANOVA modules.

Tests:
- Panel: 6 estimators × 3 backends × multiple cov types
- GAM: n_splines × n_features × 3 backends
- ANOVA: 4 functions × 3 backends × multiple group configs

Saves structured JSON results compatible with existing benchmark format.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime

import numpy as np

# ---------------------------------------------------------------------------
# Remote server
# ---------------------------------------------------------------------------
REMOTE_HOST = "hz-4.matpool.com"
REMOTE_PORT = 28838
REMOTE_USER = "root"
REMOTE_PASS = "q06qj[{K8[[gj5yB"
REMOTE_PYTHON = "/root/miniconda3/envs/myconda/bin/python"
REMOTE_WORKSPACE = "/root/statgpu"


# ---------------------------------------------------------------------------
# Data generators
# ---------------------------------------------------------------------------
def make_panel_data(n_entities, n_times, n_vars, seed=42):
    """Generate balanced panel data."""
    rng = np.random.RandomState(seed)
    n = n_entities * n_times
    X = rng.randn(n, n_vars)
    beta = rng.randn(n_vars) * 0.5
    entity_ids = np.repeat(np.arange(n_entities), n_times)
    time_ids = np.tile(np.arange(n_times), n_entities)
    # Fixed effects
    alpha_i = rng.randn(n_entities) * 0.3
    y = X @ beta + alpha_i[entity_ids] + rng.randn(n) * 0.5
    return X, y, entity_ids, time_ids


def make_gam_data(n, n_features, seed=42):
    """Generate data with nonlinear signal."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, n_features)
    y = np.zeros(n)
    for j in range(n_features):
        if j % 3 == 0:
            y += np.sin(X[:, j] * 2)
        elif j % 3 == 1:
            y += 0.5 * X[:, j] ** 2
        else:
            y += 0.3 * X[:, j]
    y += rng.randn(n) * 0.3
    return X, y


def make_anova_data(n_per_group, n_groups, seed=42):
    """Generate groups with different means."""
    rng = np.random.RandomState(seed)
    groups = [rng.randn(n_per_group) + i * 0.5 for i in range(n_groups)]
    return groups


def make_twoway_data(n_per_cell, n_a, n_b, seed=42):
    """Generate two-way ANOVA data."""
    rng = np.random.RandomState(seed)
    data = []
    for i in range(n_a):
        row = []
        for j in range(n_b):
            row.append(rng.randn(n_per_cell) + i * 0.3 + j * 0.2)
        data.append(row)
    return data


# ---------------------------------------------------------------------------
# Benchmark functions
# ---------------------------------------------------------------------------
def bench_panel(backend, n_entities, n_times, n_vars, n_repeat=3):
    """Benchmark all panel estimators."""
    from statgpu.panel import (
        PooledOLS, PanelOLS, RandomEffects, BetweenOLS,
        FirstDifferenceOLS, FamaMacBeth,
    )

    X, y, eids, tids = make_panel_data(n_entities, n_times, n_vars)
    device = "cpu" if backend == "numpy" else "cuda"

    estimators = {
        "PooledOLS": lambda: PooledOLS(device=device),
        "PooledOLS_hac": lambda: PooledOLS(cov_type="hac", bandwidth=5, device=device),
        "PanelOLS_entity": lambda: PanelOLS(entity_effects=True, device=device),
        "PanelOLS_two_way": lambda: PanelOLS(entity_effects=True, time_effects=True, device=device),
        "RandomEffects": lambda: RandomEffects(device=device),
        "BetweenOLS": lambda: BetweenOLS(device=device),
        "FirstDifferenceOLS": lambda: FirstDifferenceOLS(device=device),
        "FamaMacBeth": lambda: FamaMacBeth(device=device),
    }

    results = {}
    for name, make_est in estimators.items():
        times = []
        coef = None
        err = None
        for _ in range(n_repeat):
            try:
                est = make_est()
                t0 = time.perf_counter()
                if name == "FamaMacBeth":
                    est.fit(X, y, time_ids=tids)
                elif name in ("PanelOLS_entity", "PanelOLS_two_way", "RandomEffects", "BetweenOLS", "FirstDifferenceOLS"):
                    est.fit(X, y, entity_ids=eids, time_ids=tids)
                else:
                    est.fit(X, y)
                elapsed = time.perf_counter() - t0
                times.append(elapsed)
                coef = est.params.copy() if hasattr(est, 'params') else None
            except Exception as e:
                err = str(e)[:200]
                break

        if times:
            results[name] = {
                "time": float(np.median(times)),
                "coef_norm": float(np.linalg.norm(coef)) if coef is not None else None,
            }
        else:
            results[name] = {"time": None, "error": err}

    return results


def bench_gam(backend, n, n_features, n_splines, n_repeat=3):
    """Benchmark GAM."""
    from statgpu.semiparametric import GAM

    X, y = make_gam_data(n, n_features)
    device = "cpu" if backend == "numpy" else "cuda"

    times = []
    coef = None
    err = None
    for _ in range(n_repeat):
        try:
            gam = GAM(n_splines=n_splines, lam=1.0, device=device)
            t0 = time.perf_counter()
            gam.fit(X, y)
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            coef = gam.coef_.copy()
        except Exception as e:
            err = str(e)[:200]
            break

    if times:
        return {
            "time": float(np.median(times)),
            "coef_norm": float(np.linalg.norm(coef)) if coef is not None else None,
            "edf": float(gam.edf_) if hasattr(gam, 'edf_') else None,
        }
    return {"time": None, "error": err}


def bench_anova(backend, n_per_group, n_groups, n_repeat=10):
    """Benchmark ANOVA functions."""
    from statgpu.anova import f_oneway, f_twoway, f_welch, tukey_hsd, bonferroni

    groups = make_anova_data(n_per_group, n_groups)
    twoway_data = make_twoway_data(n_per_group, 3, 4)

    results = {}

    # f_oneway
    times = []
    for _ in range(n_repeat):
        try:
            t0 = time.perf_counter()
            r = f_oneway(*groups, backend=backend)
            times.append(time.perf_counter() - t0)
        except Exception as e:
            results["f_oneway"] = {"time": None, "error": str(e)[:200]}
            break
    if times:
        results["f_oneway"] = {
            "time": float(np.median(times)),
            "statistic": float(r.statistic),
            "pvalue": float(r.pvalue),
        }

    # f_twoway
    times = []
    for _ in range(n_repeat):
        try:
            t0 = time.perf_counter()
            r = f_twoway(twoway_data, interaction=True, backend=backend)
            times.append(time.perf_counter() - t0)
        except Exception as e:
            results["f_twoway"] = {"time": None, "error": str(e)[:200]}
            break
    if times:
        results["f_twoway"] = {
            "time": float(np.median(times)),
            "factor_a_stat": float(r.factor_a_statistic),
            "factor_b_stat": float(r.factor_b_statistic),
        }

    # f_welch
    times = []
    for _ in range(n_repeat):
        try:
            t0 = time.perf_counter()
            r = f_welch(*groups, backend=backend)
            times.append(time.perf_counter() - t0)
        except Exception as e:
            results["f_welch"] = {"time": None, "error": str(e)[:200]}
            break
    if times:
        results["f_welch"] = {
            "time": float(np.median(times)),
            "statistic": float(r.statistic),
        }

    # tukey_hsd
    times = []
    for _ in range(n_repeat):
        try:
            t0 = time.perf_counter()
            r = tukey_hsd(*groups, backend=backend)
            times.append(time.perf_counter() - t0)
        except Exception as e:
            results["tukey_hsd"] = {"time": None, "error": str(e)[:200]}
            break
    if times:
        results["tukey_hsd"] = {
            "time": float(np.median(times)),
            "n_comparisons": len(r.comparisons),
        }

    # bonferroni
    times = []
    for _ in range(n_repeat):
        try:
            t0 = time.perf_counter()
            r = bonferroni(*groups, backend=backend)
            times.append(time.perf_counter() - t0)
        except Exception as e:
            results["bonferroni"] = {"time": None, "error": str(e)[:200]}
            break
    if times:
        results["bonferroni"] = {
            "time": float(np.median(times)),
            "n_comparisons": len(r.comparisons),
        }

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="New modules benchmark")
    parser.add_argument("--module", choices=["panel", "gam", "anova", "all"], default="all")
    parser.add_argument("--backend", choices=["numpy", "cupy", "torch", "all"], default="all")
    parser.add_argument("--scale", choices=["small", "medium", "large", "all"], default="all")
    parser.add_argument("--output", "-o", type=str, default=None)
    args = parser.parse_args()

    backends = ["numpy", "cupy", "torch"] if args.backend == "all" else [args.backend]
    modules = ["panel", "gam", "anova"] if args.module == "all" else [args.module]

    results = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "environment": {"backends": backends},
        "benchmarks": {},
    }

    # Panel benchmark
    if "panel" in modules:
        print("=== Panel Data Benchmark ===", flush=True)
        scales = {
            "small": (50, 10, 5),      # 500 obs
            "medium": (500, 20, 10),    # 10K obs
            "large": (2000, 50, 20),    # 100K obs
        }
        if args.scale == "all":
            run_scales = scales
        else:
            run_scales = {args.scale: scales[args.scale]}

        for scale_name, (n_ent, n_time, n_var) in run_scales.items():
            print(f"\n  Scale: {scale_name} ({n_ent}x{n_time}={n_ent*n_time} obs, {n_var} vars)", flush=True)
            for backend in backends:
                print(f"    Backend: {backend}...", flush=True)
                try:
                    res = bench_panel(backend, n_ent, n_time, n_var)
                    key = f"panel_{scale_name}_{backend}"
                    results["benchmarks"][key] = res
                    for name, v in res.items():
                        t = v.get("time")
                        print(f"      {name}: {t:.4f}s" if t else f"      {name}: FAIL", flush=True)
                except Exception as e:
                    print(f"      ERROR: {e}", flush=True)

    # GAM benchmark
    if "gam" in modules:
        print("\n=== GAM Benchmark ===", flush=True)
        scales = {
            "small": (1000, 3, 15),     # 1K obs, 3 features
            "medium": (10000, 5, 20),    # 10K obs, 5 features
            "large": (100000, 10, 25),   # 100K obs, 10 features
        }
        if args.scale == "all":
            run_scales = scales
        else:
            run_scales = {args.scale: scales[args.scale]}

        for scale_name, (n, nf, ns) in run_scales.items():
            print(f"\n  Scale: {scale_name} ({n} obs, {nf} features, {ns} splines)", flush=True)
            for backend in backends:
                print(f"    Backend: {backend}...", flush=True)
                try:
                    res = bench_gam(backend, n, nf, ns)
                    key = f"gam_{scale_name}_{backend}"
                    results["benchmarks"][key] = res
                    t = res.get("time")
                    print(f"      time: {t:.4f}s" if t else f"      FAIL: {res.get('error')}", flush=True)
                except Exception as e:
                    print(f"      ERROR: {e}", flush=True)

    # ANOVA benchmark
    if "anova" in modules:
        print("\n=== ANOVA Benchmark ===", flush=True)
        scales = {
            "small": (100, 5),          # 100 per group, 5 groups
            "medium": (10000, 10),       # 10K per group, 10 groups
            "large": (100000, 20),       # 100K per group, 20 groups
        }
        if args.scale == "all":
            run_scales = scales
        else:
            run_scales = {args.scale: scales[args.scale]}

        for scale_name, (npg, ng) in run_scales.items():
            print(f"\n  Scale: {scale_name} ({npg}/group, {ng} groups, total {npg*ng})", flush=True)
            for backend in backends:
                print(f"    Backend: {backend}...", flush=True)
                try:
                    res = bench_anova(backend, npg, ng)
                    key = f"anova_{scale_name}_{backend}"
                    results["benchmarks"][key] = res
                    for name, v in res.items():
                        t = v.get("time")
                        print(f"      {name}: {t:.4f}s" if t else f"      {name}: FAIL", flush=True)
                except Exception as e:
                    print(f"      ERROR: {e}", flush=True)

    # Save
    out_file = args.output or f"results/new_modules_bench_{datetime.now().strftime('%Y-%m-%d')}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_file}")
    print("DONE")


if __name__ == "__main__":
    main()
