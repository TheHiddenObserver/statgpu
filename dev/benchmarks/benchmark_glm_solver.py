"""Comprehensive GLM Solver Benchmark: family x penalty x solver.

Tests all valid solver combinations for each family-penalty pair:
- Smooth penalties (none, l2): exact, newton, irls, lbfgs, fista, fista_bb, admm
- Non-smooth penalties: fista, fista_bb, admm

Reports:
1. Precision: coef L2-diff vs numpy reference for each solver
2. Performance: speedup vs numpy for each solver
3. Best solver: fastest solver for each family-penalty-backend combination
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np

# ---------------------------------------------------------------------------
# Remote server configuration
# ---------------------------------------------------------------------------
REMOTE_HOST = "hz-4.matpool.com"
REMOTE_PORT = 28838
REMOTE_USER = "root"
REMOTE_PASS = "q06qj[{K8[[gj5yB"
REMOTE_PYTHON = "/root/miniconda3/envs/myconda/bin/python"
REMOTE_WORKSPACE = "/root/statgpu"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_encoder():
    class _NumpyEncoder(json.JSONEncoder):
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
    return _NumpyEncoder


def _generate_data(n, p, family, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p).astype(np.float64)
    coef_true = rng.randn(p).astype(np.float64) * 0.5
    eta = X @ coef_true

    if family == "squared_error":
        y = eta + rng.randn(n) * 0.5
    elif family == "logistic":
        prob = 1.0 / (1.0 + np.exp(-np.clip(eta, -20, 20)))
        y = (rng.rand(n) < prob).astype(np.float64)
    elif family == "poisson":
        mu = np.exp(np.clip(eta, -20, 5))
        y = rng.poisson(mu).astype(np.float64)
    elif family == "gamma":
        mu = np.exp(np.clip(eta, -10, 10))
        y = rng.gamma(2.0, mu / 2.0)
    elif family == "inverse_gaussian":
        mu = np.exp(np.clip(eta, -10, 10))
        y = np.abs(mu + rng.randn(n) * mu * 0.3)
        y = np.maximum(y, 1e-6)
    elif family == "negative_binomial":
        mu = np.exp(np.clip(eta, -10, 10))
        p_nb = 0.5
        r_nb = mu * p_nb / (1 - p_nb)
        r_nb = np.maximum(r_nb, 0.1)
        y = rng.negative_binomial(np.clip(r_nb, 1, 1000).astype(int), p_nb).astype(np.float64)
    elif family == "tweedie":
        mu = np.exp(np.clip(eta, -10, 10))
        y = np.abs(mu + rng.randn(n) * mu * 0.5)
        y = np.maximum(y, 1e-6)
    else:
        raise ValueError(f"Unknown family: {family}")
    return X, y


def _get_group_indices(p, n_groups=None):
    if n_groups is None:
        n_groups = max(1, p // 5)
    group_size = p // n_groups
    groups = []
    for g in range(n_groups):
        start = g * group_size
        end = start + group_size if g < n_groups - 1 else p
        groups.append(list(range(start, end)))
    return groups


def _run_solver(X, y, family, penalty, solver, backend, alpha, groups, max_iter=200):
    from statgpu.linear_model import PenalizedGLM
    from statgpu.linear_model.penalized._fit_mixin import _SOLVER_DISPATCH_TABLE

    model = PenalizedGLM(
        family=family,
        penalty=penalty,
        solver=solver,
        alpha=alpha,
        max_iter=max_iter,
        backend=backend,
    )

    fit_kw = {}
    if penalty in ("group_lasso", "group_scad", "group_mcp"):
        fit_kw["group_indices"] = groups

    t0 = time.perf_counter()
    try:
        model.fit(X, y, **fit_kw)
        elapsed = time.perf_counter() - t0
        return model.coef_, elapsed, None
    except (ValueError, TypeError, RuntimeError) as e:
        elapsed = time.perf_counter() - t0
        return None, elapsed, str(e)


# ---------------------------------------------------------------------------
# Precision benchmark
# ---------------------------------------------------------------------------
def run_precision(X, y, family, penalties, backends, alpha=0.1):
    results = []
    p = X.shape[1]
    groups = _get_group_indices(p)

    # Valid solvers per penalty category
    SMOOTH_SOLVERS = ["exact", "newton", "irls", "lbfgs", "fista", "fista_bb", "admm"]
    NONSMOOTH_SOLVERS = ["fista", "fista_bb", "admm"]

    for pen_name, pen_cls, pen_kw in penalties:
        is_smooth = pen_name in ("none", "l2")
        solvers = SMOOTH_SOLVERS if is_smooth else NONSMOOTH_SOLVERS

        for backend in backends:
            # Reference: numpy + fista
            coef_ref, t_ref, err_ref = _run_solver(
                X, y, family, pen_name, "fista", "numpy", alpha, groups
            )
            if coef_ref is None:
                results.append({
                    "family": family, "penalty": pen_name, "solver": "fista",
                    "backend": "numpy", "status": "FAIL", "error": err_ref,
                })
                continue

            for solver in solvers:
                coef, t, err = _run_solver(
                    X, y, family, pen_name, solver, backend, alpha, groups
                )
                if coef is None:
                    results.append({
                        "family": family, "penalty": pen_name, "solver": solver,
                        "backend": backend, "status": "ERROR", "error": err,
                    })
                    continue

                l2_diff = float(np.linalg.norm(coef - coef_ref))
                l2_ref = float(np.linalg.norm(coef_ref))
                rel_diff = l2_diff / max(l2_ref, 1e-12)

                results.append({
                    "family": family, "penalty": pen_name, "solver": solver,
                    "backend": backend, "l2_diff": l2_diff, "rel_diff": rel_diff,
                    "time": t, "status": "PASS",
                })

    return results


# ---------------------------------------------------------------------------
# Performance benchmark
# ---------------------------------------------------------------------------
def run_performance(X, y, family, penalties, backend, alpha=0.1, n_repeat=3):
    results = []
    p = X.shape[1]
    groups = _get_group_indices(p)

    SMOOTH_SOLVERS = ["exact", "newton", "irls", "lbfgs", "fista", "fista_bb", "admm"]
    NONSMOOTH_SOLVERS = ["fista", "fista_bb", "admm"]

    # Numpy reference
    for pen_name, pen_cls, pen_kw in penalties:
        is_smooth = pen_name in ("none", "l2")
        solvers = SMOOTH_SOLVERS if is_smooth else NONSMOOTH_SOLVERS

        # Numpy reference
        ref_times = []
        coef_ref = None
        for _ in range(n_repeat):
            coef, t, err = _run_solver(
                X, y, family, pen_name, "fista", "numpy", alpha, groups
            )
            if coef is not None:
                ref_times.append(t)
                coef_ref = coef
        ref_time = np.median(ref_times) if ref_times else float("nan")

        if coef_ref is None:
            results.append({
                "family": family, "penalty": pen_name, "backend": backend,
                "ref_backend": "numpy", "ref_time": ref_time, "solvers": {},
            })
            continue

        solver_results = {}
        for solver in solvers:
            times = []
            coef_gpu = None
            for _ in range(n_repeat):
                coef, t, err = _run_solver(
                    X, y, family, pen_name, solver, backend, alpha, groups
                )
                if coef is not None:
                    times.append(t)
                    coef_gpu = coef
            gpu_time = np.median(times) if times else float("nan")
            l2_diff = float(np.linalg.norm(coef_gpu - coef_ref)) if coef_gpu is not None else float("nan")
            speedup = ref_time / gpu_time if gpu_time > 0 and np.isfinite(gpu_time) else 0.0

            solver_results[solver] = {
                "time": gpu_time,
                "speedup": speedup,
                "l2_diff": l2_diff,
            }

        # Find best solver
        valid_solvers = {k: v for k, v in solver_results.items() if v["speedup"] > 0}
        best = max(valid_solvers, key=lambda k: valid_solvers[k]["speedup"]) if valid_solvers else None

        results.append({
            "family": family, "penalty": pen_name, "backend": backend,
            "ref_backend": "numpy", "ref_time": ref_time,
            "solvers": solver_results,
            "best_solver": best,
        })

    return results


# ---------------------------------------------------------------------------
# SSH runner
# ---------------------------------------------------------------------------
def run_on_server(bench_code, scale, backends, family_penalty_solver_cases, max_iter=200, n_repeat=3):
    import paramiko

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(REMOTE_HOST, port=REMOTE_PORT, username=REMOTE_USER, password=REMOTE_PASS, timeout=30)

    cases_json = json.dumps(family_penalty_solver_cases)
    python_code = f"""\
import json, time, sys
sys.path.insert(0, '/root/statgpu')
import numpy as np

results = {bench_code}

print(json.dumps(results, default=str))
"""

    stdin, stdout, stderr = ssh.exec_command(
        f"cd {REMOTE_WORKSPACE} && {REMOTE_PYTHON} -u -c '{python_code}'",
        timeout=3600,
    )
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    ssh.close()

    if out.strip():
        return json.loads(out.strip())
    else:
        raise RuntimeError(f"Server error:\n{err}")


# ---------------------------------------------------------------------------
# Remote precision test
# ---------------------------------------------------------------------------
PREC_SCRIPT = r'''
import json, time, sys, os
sys.path.insert(0, '/root/statgpu')
import numpy as np
from statgpu.linear_model import PenalizedGLM

FAMILIES = ["squared_error", "logistic", "poisson", "gamma", "inverse_gaussian", "negative_binomial", "tweedie"]
PENALTIES = [
    ("none", None, {}),
    ("l2", None, {}),
    ("l1", None, {}),
    ("elasticnet", None, {"l1_ratio": 0.5}),
    ("adaptive_l1", None, {}),
    ("scad", None, {}),
    ("mcp", None, {}),
    ("group_lasso", None, {}),
    ("group_scad", None, {}),
    ("group_mcp", None, {}),
]

SMOOTH_SOLVERS = ["exact", "newton", "irls", "lbfgs", "fista", "fista_bb", "admm"]
NONSMOOTH_SOLVERS = ["fista", "fista_bb", "admm"]

def gen(n, p, fam, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p)
    c = rng.randn(p) * 0.5
    e = X @ c
    if fam == "squared_error":
        y = e + rng.randn(n) * 0.5
    elif fam == "logistic":
        p1 = 1/(1+np.exp(-np.clip(e, -20, 20)))
        y = (rng.rand(n) < p1).astype(float)
    elif fam == "poisson":
        y = rng.poisson(np.exp(np.clip(e, -20, 5))).astype(float)
    elif fam == "gamma":
        mu = np.exp(np.clip(e, -10, 10))
        y = rng.gamma(2.0, mu/2.0)
    elif fam == "inverse_gaussian":
        mu = np.exp(np.clip(e, -10, 10))
        y = np.abs(mu + rng.randn(n)*mu*0.3); y = np.maximum(y, 1e-6)
    elif fam == "negative_binomial":
        mu = np.exp(np.clip(e, -10, 10))
        p_nb = 0.5; r_nb = np.maximum(mu*p_nb/(1-p_nb), 0.1)
        y = rng.negative_binomial(np.clip(r_nb, 1, 1000).astype(int), p_nb).astype(float)
    elif fam == "tweedie":
        mu = np.exp(np.clip(e, -10, 10))
        y = np.abs(mu + rng.randn(n)*mu*0.5); y = np.maximum(y, 1e-6)
    return X, y

def groups_indices(p, n_groups=None):
    if n_groups is None: n_groups = max(1, p//5)
    gs = p // n_groups
    return [list(range(g*gs, (g+1)*gs if g < n_groups-1 else p)) for g in range(n_groups)]

N, P = 10000, 50
alpha = 0.1
results = []

for fam in FAMILIES:
    X, y = gen(N, P, fam)
    grps = groups_indices(P)
    for pen_name, pen_cls, pen_kw in PENALTIES:
        is_smooth = pen_name in ("none", "l2")
        solvers = SMOOTH_SOLVERS if is_smooth else NONSMOOTH_SOLVERS

        # Reference: numpy + fista
        try:
            m = PenalizedGLM(family=fam, penalty=pen_name, solver="fista", alpha=alpha, max_iter=200, backend="numpy")
            kw = {"group_indices": grps} if pen_name in ("group_lasso","group_scad","group_mcp") else {}
            m.fit(X, y, **kw)
            coef_ref = m.coef_.copy()
        except Exception as e:
            results.append({"family": fam, "penalty": pen_name, "solver": "fista", "backend": "numpy", "status": "FAIL", "error": str(e)[:200]})
            continue

        for solver in solvers:
            try:
                m = PenalizedGLM(family=fam, penalty=pen_name, solver=solver, alpha=alpha, max_iter=200, backend="numpy")
                kw = {"group_indices": grps} if pen_name in ("group_lasso","group_scad","group_mcp") else {}
                t0 = time.perf_counter()
                m.fit(X, y, **kw)
                elapsed = time.perf_counter() - t0
                coef = m.coef_
                l2_diff = float(np.linalg.norm(coef - coef_ref))
                l2_ref = float(np.linalg.norm(coef_ref))
                results.append({
                    "family": fam, "penalty": pen_name, "solver": solver, "backend": "numpy",
                    "l2_diff": l2_diff, "rel_diff": l2_diff/max(l2_ref, 1e-12),
                    "time": elapsed, "status": "PASS",
                })
            except Exception as e:
                results.append({
                    "family": fam, "penalty": pen_name, "solver": solver, "backend": "numpy",
                    "status": "ERROR", "error": str(e)[:200],
                })

print(json.dumps(results))
'''


# ---------------------------------------------------------------------------
# Remote performance test
# ---------------------------------------------------------------------------
PERF_SCRIPT = r'''
import json, time, sys
sys.path.insert(0, '/root/statgpu')
import numpy as np
from statgpu.linear_model import PenalizedGLM

FAMILIES = ["squared_error", "logistic", "poisson", "gamma", "inverse_gaussian", "negative_binomial", "tweedie"]
PENALTIES = [
    ("none", None, {}),
    ("l2", None, {}),
    ("l1", None, {}),
    ("elasticnet", None, {"l1_ratio": 0.5}),
    ("adaptive_l1", None, {}),
    ("scad", None, {}),
    ("mcp", None, {}),
    ("group_lasso", None, {}),
    ("group_scad", None, {}),
    ("group_mcp", None, {}),
]
BACKEND = "torch"

SMOOTH_SOLVERS = ["exact", "newton", "irls", "lbfgs", "fista", "fista_bb", "admm"]
NONSMOOTH_SOLVERS = ["fista", "fista_bb", "admm"]

def gen(n, p, fam, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p)
    c = rng.randn(p) * 0.5
    e = X @ c
    if fam == "squared_error":
        y = e + rng.randn(n) * 0.5
    elif fam == "logistic":
        p1 = 1/(1+np.exp(-np.clip(e, -20, 20)))
        y = (rng.rand(n) < p1).astype(float)
    elif fam == "poisson":
        y = rng.poisson(np.exp(np.clip(e, -20, 5))).astype(float)
    elif fam == "gamma":
        mu = np.exp(np.clip(e, -10, 10))
        y = rng.gamma(2.0, mu/2.0)
    elif fam == "inverse_gaussian":
        mu = np.exp(np.clip(e, -10, 10))
        y = np.abs(mu + rng.randn(n)*mu*0.3); y = np.maximum(y, 1e-6)
    elif fam == "negative_binomial":
        mu = np.exp(np.clip(e, -10, 10))
        p_nb = 0.5; r_nb = np.maximum(mu*p_nb/(1-p_nb), 0.1)
        y = rng.negative_binomial(np.clip(r_nb, 1, 1000).astype(int), p_nb).astype(float)
    elif fam == "tweedie":
        mu = np.exp(np.clip(e, -10, 10))
        y = np.abs(mu + rng.randn(n)*mu*0.5); y = np.maximum(y, 1e-6)
    return X, y

def groups_indices(p, n_groups=None):
    if n_groups is None: n_groups = max(1, p//5)
    gs = p // n_groups
    return [list(range(g*gs, (g+1)*gs if g < n_groups-1 else p)) for g in range(n_groups)]

N, P = 100000, 50
alpha = 0.1
N_REPEAT = 3
results = []

# Warmup
try:
    import torch
    X_w = torch.randn(100, 10, device='cuda')
    _ = torch.mm(X_w, X_w.t())
    torch.cuda.synchronize()
except:
    pass

for fam in FAMILIES:
    X, y = gen(N, P, fam)
    grps = groups_indices(P)
    for pen_name, pen_cls, pen_kw in PENALTIES:
        is_smooth = pen_name in ("none", "l2")
        solvers = SMOOTH_SOLVERS if is_smooth else NONSMOOTH_SOLVERS

        # Numpy reference
        ref_times = []
        coef_ref = None
        for _ in range(N_REPEAT):
            try:
                m = PenalizedGLM(family=fam, penalty=pen_name, solver="fista", alpha=alpha, max_iter=200, backend="numpy")
                kw = {"group_indices": grps} if pen_name in ("group_lasso","group_scad","group_mcp") else {}
                t0 = time.perf_counter()
                m.fit(X, y, **kw)
                t = time.perf_counter() - t0
                ref_times.append(t)
                coef_ref = m.coef_.copy()
            except:
                pass
        ref_time = float(np.median(ref_times)) if ref_times else float("nan")

        solver_results = {}
        for solver in solvers:
            times = []
            coef_gpu = None
            for _ in range(N_REPEAT):
                try:
                    m = PenalizedGLM(family=fam, penalty=pen_name, solver=solver, alpha=alpha, max_iter=200, backend=BACKEND)
                    kw = {"group_indices": grps} if pen_name in ("group_lasso","group_scad","group_mcp") else {}
                    t0 = time.perf_counter()
                    m.fit(X, y, **kw)
                    t = time.perf_counter() - t0
                    times.append(t)
                    coef_gpu = m.coef_.copy()
                except Exception as e:
                    pass
            gpu_time = float(np.median(times)) if times else float("nan")
            l2_diff = float(np.linalg.norm(coef_gpu - coef_ref)) if coef_gpu is not None and coef_ref is not None else float("nan")
            speedup = ref_time / gpu_time if gpu_time > 0 and np.isfinite(gpu_time) else 0.0
            solver_results[solver] = {"time": gpu_time, "speedup": speedup, "l2_diff": l2_diff}

        valid = {k: v for k, v in solver_results.items() if v["speedup"] > 0}
        best = max(valid, key=lambda k: valid[k]["speedup"]) if valid else None

        results.append({
            "family": fam, "penalty": pen_name, "backend": BACKEND,
            "ref_time": ref_time, "solvers": solver_results, "best_solver": best,
        })

print(json.dumps(results))
'''


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="GLM solver benchmark (family x penalty x solver)")
    parser.add_argument("--precision", action="store_true", help="Run precision tests")
    parser.add_argument("--performance", action="store_true", help="Run performance tests")
    parser.add_argument("--all", action="store_true", help="Run both precision and performance")
    parser.add_argument("--output", "-o", type=str, default=None)
    args = parser.parse_args()

    if not args.precision and not args.performance and not args.all:
        args.all = True

    if args.all:
        args.precision = True
        args.performance = True

    ts = datetime.now().strftime("%Y-%m-%d")

    if args.precision:
        print("Running precision benchmark (all valid family x penalty x solver combos)...")
        prec_results = run_on_server(PREC_SCRIPT, "10K", ["numpy"], [])

        prec_file = args.output or f"results/glm_solver_precision_{ts}.json"
        with open(prec_file, "w") as f:
            json.dump(prec_results, f, indent=2, cls=_make_encoder())
        print(f"Precision results saved to {prec_file}")

        # Generate markdown
        md_file = prec_file.replace(".json", ".md")
        _generate_precision_markdown(prec_results, md_file)
        print(f"Markdown saved to {md_file}")

    if args.performance:
        print("Running performance benchmark (all valid family x penalty x solver combos)...")
        perf_results = run_on_server(PERF_SCRIPT, "100K", ["torch"], [])

        perf_file = args.output or f"results/glm_solver_perf_{ts}.json"
        with open(perf_file, "w") as f:
            json.dump(perf_results, f, indent=2, cls=_make_encoder())
        print(f"Performance results saved to {perf_file}")

        # Generate markdown
        md_file = perf_file.replace(".json", ".md")
        _generate_performance_markdown(perf_results, md_file)
        print(f"Markdown saved to {md_file}")


def _generate_precision_markdown(results, md_file):
    # Group by family
    families = sorted(set(r["family"] for r in results))

    with open(md_file, "w") as f:
        f.write("# GLM Solver Precision Benchmark\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d')}\n\n")
        f.write("Reference: numpy + fista solver\n\n")

        for fam in families:
            fam_results = [r for r in results if r["family"] == fam]
            f.write(f"## {fam}\n\n")

            # Group by penalty
            penalties = sorted(set(r["penalty"] for r in fam_results))
            for pen in penalties:
                pen_results = [r for r in fam_results if r["penalty"] == pen]
                f.write(f"### {pen}\n\n")
                f.write("| Solver | Backend | L2 Diff | Rel Diff | Status |\n")
                f.write("|--------|---------|---------|----------|--------|\n")
                for r in pen_results:
                    if r["status"] == "PASS":
                        f.write(f"| {r['solver']} | {r['backend']} | {r['l2_diff']:.2e} | {r['rel_diff']:.2e} | {r['status']} |\n")
                    else:
                        f.write(f"| {r['solver']} | {r['backend']} | - | - | {r['status']} |\n")
                f.write("\n")


def _generate_performance_markdown(results, md_file):
    with open(md_file, "w") as f:
        f.write("# GLM Solver Performance Benchmark\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d')}\n\n")
        f.write("Backend: torch (GPU) vs numpy (CPU)\n\n")

        # Summary table: best solver per family-penalty
        f.write("## Best Solver Summary\n\n")
        f.write("| Family | Penalty | Best Solver | Best Speedup | All Speedups |\n")
        f.write("|--------|---------|-------------|--------------|--------------|\n")

        for r in results:
            best = r.get("best_solver", "N/A")
            best_spd = r["solvers"].get(best, {}).get("speedup", 0) if best else 0
            all_spds = ", ".join(
                f"{k}:{v['speedup']:.1f}x" for k, v in r["solvers"].items()
                if v["speedup"] > 0
            )
            f.write(f"| {r['family']} | {r['penalty']} | {best} | {best_spd:.1f}x | {all_spds} |\n")

        # Detailed tables per family
        f.write("\n## Detailed Results\n\n")
        families = sorted(set(r["family"] for r in results))
        for fam in families:
            fam_results = [r for r in results if r["family"] == fam]
            f.write(f"### {fam}\n\n")
            f.write("| Penalty | Solver | GPU Time (s) | Speedup | L2 Diff |\n")
            f.write("|---------|--------|-------------|---------|--------|\n")
            for r in fam_results:
                for solver, sv in r["solvers"].items():
                    f.write(f"| {r['penalty']} | {solver} | {sv['time']:.4f} | {sv['speedup']:.1f}x | {sv['l2_diff']:.2e} |\n")


if __name__ == "__main__":
    main()
