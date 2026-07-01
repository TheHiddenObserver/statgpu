"""Remote benchmark: Panel Data, GAM, ANOVA across 3 backends."""
import json, time, sys, warnings
sys.path.insert(0, '/root/statgpu')
import numpy as np

# ---------------------------------------------------------------------------
# Data generators
# ---------------------------------------------------------------------------
def make_panel_data(n_entities, n_times, n_vars, seed=42):
    rng = np.random.RandomState(seed)
    n = n_entities * n_times
    X = rng.randn(n, n_vars)
    beta = rng.randn(n_vars) * 0.5
    entity_ids = np.repeat(np.arange(n_entities), n_times)
    time_ids = np.tile(np.arange(n_times), n_entities)
    alpha_i = rng.randn(n_entities) * 0.3
    y = X @ beta + alpha_i[entity_ids] + rng.randn(n) * 0.5
    return X, y, entity_ids, time_ids

def make_gam_data(n, n_features, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, n_features)
    y = np.zeros(n)
    for j in range(n_features):
        if j % 3 == 0: y += np.sin(X[:, j] * 2)
        elif j % 3 == 1: y += 0.5 * X[:, j] ** 2
        else: y += 0.3 * X[:, j]
    y += rng.randn(n) * 0.3
    return X, y

def make_anova_data(n_per_group, n_groups, seed=42):
    rng = np.random.RandomState(seed)
    return [rng.randn(n_per_group) + i * 0.5 for i in range(n_groups)]

def make_twoway_data(n_per_cell, n_a, n_b, seed=42):
    rng = np.random.RandomState(seed)
    return [[rng.randn(n_per_cell) + i * 0.3 + j * 0.2 for j in range(n_b)] for i in range(n_a)]

# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------
try:
    import torch
    Xw = torch.randn(100, 10, device='cuda')
    _ = torch.mm(Xw, Xw.t())
    torch.cuda.synchronize()
    print("GPU warmup done", flush=True)
except:
    print("GPU warmup failed", flush=True)

# ---------------------------------------------------------------------------
# Panel benchmark
# ---------------------------------------------------------------------------
def bench_panel():
    from statgpu.panel import (
        PooledOLS, PanelOLS, RandomEffects, BetweenOLS,
        FirstDifferenceOLS, FamaMacBeth,
    )

    scales = {
        "small": (50, 10, 5),
        "medium": (500, 20, 10),
        "large": (2000, 50, 20),
    }
    backends = [("numpy", "cpu"), ("cupy", "cuda"), ("torch", "cuda")]

    results = {}
    for scale_name, (n_ent, n_time, n_var) in scales.items():
        X, y, eids, tids = make_panel_data(n_ent, n_time, n_var)
        print(f"\n  Panel {scale_name} ({n_ent*n_time} obs):", flush=True)

        for backend_name, device in backends:
            estimators = {
                "PooledOLS": lambda d: PooledOLS(device=d),
                "PooledOLS_hac": lambda d: PooledOLS(cov_type="hac", bandwidth=5, device=d),
                "PanelOLS_entity": lambda d: PanelOLS(entity_effects=True, device=d),
                "PanelOLS_two_way": lambda d: PanelOLS(entity_effects=True, time_effects=True, device=d),
                "RandomEffects": lambda d: RandomEffects(device=d),
                "BetweenOLS": lambda d: BetweenOLS(device=d),
                "FirstDifferenceOLS": lambda d: FirstDifferenceOLS(device=d),
                "FamaMacBeth": lambda d: FamaMacBeth(device=d),
            }

            for name, make_est in estimators.items():
                times = []
                coef = None
                err = None
                for _ in range(3):
                    try:
                        est = make_est(device)
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

                key = f"panel_{scale_name}_{backend_name}_{name}"
                if times:
                    results[key] = {
                        "time": float(np.median(times)),
                        "coef_norm": float(np.linalg.norm(coef)) if coef is not None else None,
                    }
                    print(f"    {backend_name}/{name}: {np.median(times):.4f}s", flush=True)
                else:
                    results[key] = {"time": None, "error": err}
                    print(f"    {backend_name}/{name}: FAIL - {err}", flush=True)

    return results

# ---------------------------------------------------------------------------
# GAM benchmark
# ---------------------------------------------------------------------------
def bench_gam():
    from statgpu.semiparametric import GAM

    scales = {
        "small": (1000, 3, 15),
        "medium": (10000, 5, 20),
        "large": (100000, 10, 25),
    }
    backends = [("numpy", "cpu"), ("cupy", "cuda"), ("torch", "cuda")]

    results = {}
    for scale_name, (n, nf, ns) in scales.items():
        X, y = make_gam_data(n, nf)
        print(f"\n  GAM {scale_name} ({n} obs, {nf} feat, {ns} splines):", flush=True)

        for backend_name, device in backends:
            times = []
            coef = None
            err = None
            for _ in range(3):
                try:
                    gam = GAM(n_splines=ns, lam=1.0, device=device)
                    t0 = time.perf_counter()
                    gam.fit(X, y)
                    elapsed = time.perf_counter() - t0
                    times.append(elapsed)
                    coef = gam.coef_.copy()
                except Exception as e:
                    err = str(e)[:200]
                    break

            key = f"gam_{scale_name}_{backend_name}"
            if times:
                results[key] = {
                    "time": float(np.median(times)),
                    "coef_norm": float(np.linalg.norm(coef)) if coef is not None else None,
                }
                print(f"    {backend_name}: {np.median(times):.4f}s", flush=True)
            else:
                results[key] = {"time": None, "error": err}
                print(f"    {backend_name}: FAIL - {err}", flush=True)

    return results

# ---------------------------------------------------------------------------
# ANOVA benchmark
# ---------------------------------------------------------------------------
def bench_anova():
    from statgpu.anova import f_oneway, f_twoway, f_welch, tukey_hsd, bonferroni

    scales = {
        "small": (100, 5),
        "medium": (10000, 10),
        "large": (100000, 20),
    }
    backends = ["numpy", "cupy", "torch"]

    results = {}
    for scale_name, (npg, ng) in scales.items():
        groups = make_anova_data(npg, ng)
        twoway_data = make_twoway_data(npg, 3, 4)
        print(f"\n  ANOVA {scale_name} ({npg}/group, {ng} groups, total {npg*ng}):", flush=True)

        for backend in backends:
            for func_name, func, args in [
                ("f_oneway", f_oneway, groups),
                ("f_twoway", f_twoway, [twoway_data]),
                ("f_welch", f_welch, groups),
                ("tukey_hsd", tukey_hsd, groups),
                ("bonferroni", bonferroni, groups),
            ]:
                times = []
                err = None
                for _ in range(3):
                    try:
                        t0 = time.perf_counter()
                        if func_name == "f_twoway":
                            r = func(*args, interaction=True, backend=backend)
                        else:
                            r = func(*args, backend=backend)
                        times.append(time.perf_counter() - t0)
                    except Exception as e:
                        err = str(e)[:200]
                        break

                key = f"anova_{scale_name}_{backend}_{func_name}"
                if times:
                    results[key] = {"time": float(np.median(times))}
                    print(f"    {backend}/{func_name}: {np.median(times):.4f}s", flush=True)
                else:
                    results[key] = {"time": None, "error": err}
                    print(f"    {backend}/{func_name}: FAIL - {err}", flush=True)

    return results

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
print("=== New Modules Benchmark ===", flush=True)
all_results = {
    "date": "2026-06-24",
    "environment": {"gpu": "Tesla P100-SXM2-16GB"},
    "benchmarks": {},
}

print("\n--- Panel Data ---", flush=True)
all_results["benchmarks"].update(bench_panel())

print("\n--- GAM ---", flush=True)
all_results["benchmarks"].update(bench_gam())

print("\n--- ANOVA ---", flush=True)
all_results["benchmarks"].update(bench_anova())

with open("/root/statgpu/results_new_modules.json", "w") as f:
    json.dump(all_results, f, indent=2, default=str)

print(f"\nTotal: {len(all_results['benchmarks'])} results")
print("DONE")
