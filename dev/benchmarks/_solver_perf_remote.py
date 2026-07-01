"""Remote performance benchmark: family x penalty x solver x backend (numpy/cupy/torch)."""
import json, time, sys, warnings
sys.path.insert(0, '/root/statgpu')
import numpy as np
from statgpu.linear_model.penalized._base import PenalizedGeneralizedLinearModel

FAMILIES = ["squared_error", "logistic", "poisson", "gamma", "inverse_gaussian", "negative_binomial", "tweedie"]
PENALTIES = [
    ("none", "l2", 0.0, {}),
    ("l2", "l2", 0.1, {}),
    ("l1", "l1", 0.1, {}),
    ("elasticnet", "elasticnet", 0.1, {"l1_ratio": 0.5}),
    ("adaptive_l1", "adaptive_l1", 0.1, {}),
    ("scad", "scad", 0.1, {}),
    ("mcp", "mcp", 0.1, {}),
]
GROUP_PENALTIES = [
    ("group_lasso", "group_lasso", 0.1, {}),
    ("group_scad", "group_scad", 0.1, {}),
    ("group_mcp", "group_mcp", 0.1, {}),
]

SMOOTH_SOLVERS = ["exact", "newton", "irls", "lbfgs", "fista", "fista_bb", "admm"]
NONSMOOTH_SOLVERS = ["fista", "fista_bb", "admm"]

BACKENDS = [
    ("numpy", "cpu"),
    ("cupy", "cuda"),
    ("torch", "cuda"),
]

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

def groups_for_p(p, n_groups=None):
    if n_groups is None: n_groups = max(1, p//5)
    gs = p // n_groups
    return [list(range(g*gs, (g+1)*gs if g < n_groups-1 else p)) for g in range(n_groups)]

def fit_model(fam, pen_type, alpha, pen_kw, solver, device, X, y):
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        m = PenalizedGeneralizedLinearModel(
            loss=fam, penalty=pen_type, alpha=alpha,
            penalty_kwargs=pen_kw if pen_kw else None,
            solver=solver, max_iter=200, tol=1e-4,
            device=device, fit_intercept=True,
        )
        t0 = time.perf_counter()
        m.fit(X, y)
        elapsed = time.perf_counter() - t0
    return m.coef_, elapsed

N, P = 100000, 50
alpha = 0.1
N_REPEAT = 3
results = []

# Warmup GPU
try:
    import torch
    X_w = torch.randn(100, 10, device='cuda')
    _ = torch.mm(X_w, X_w.t())
    torch.cuda.synchronize()
    print("GPU warmup done", flush=True)
except Exception as e:
    print(f"GPU warmup failed: {e}", flush=True)

all_penalties = PENALTIES + GROUP_PENALTIES
total = len(FAMILIES) * len(all_penalties)
done = 0

for fam in FAMILIES:
    X, y = gen(N, P, fam)
    grps = groups_for_p(P)

    for pen_label, pen_type, pen_alpha, pen_kw in all_penalties:
        done += 1
        is_smooth = pen_label in ("none", "l2")
        solvers = SMOOTH_SOLVERS if is_smooth else NONSMOOTH_SOLVERS

        pkw = dict(pen_kw)
        if pen_label in ("group_lasso", "group_scad", "group_mcp"):
            pkw["groups"] = grps

        # Numpy reference (fista)
        ref_times = []
        coef_ref = None
        for _ in range(N_REPEAT):
            try:
                coef, t = fit_model(fam, pen_type, pen_alpha, pkw, "fista", "cpu", X, y)
                ref_times.append(t)
                coef_ref = coef.copy()
            except:
                pass
        ref_time = float(np.median(ref_times)) if ref_times else float("nan")

        backend_results = {}
        for backend_name, device in BACKENDS:
            solver_results = {}
            for solver in solvers:
                times = []
                coef_gpu = None
                err_msg = None
                for _ in range(N_REPEAT):
                    try:
                        coef, t = fit_model(fam, pen_type, pen_alpha, pkw, solver, device, X, y)
                        times.append(t)
                        coef_gpu = coef.copy()
                    except Exception as e:
                        err_msg = str(e)[:150]
                gpu_time = float(np.median(times)) if times else float("nan")
                l2_diff = float(np.linalg.norm(coef_gpu - coef_ref)) if coef_gpu is not None and coef_ref is not None else float("nan")
                speedup = ref_time / gpu_time if gpu_time > 0 and np.isfinite(gpu_time) else 0.0
                solver_results[solver] = {"time": gpu_time, "speedup": speedup, "l2_diff": l2_diff}
                if err_msg:
                    solver_results[solver]["error"] = err_msg

            valid = {k: v for k, v in solver_results.items() if v["speedup"] > 0}
            best = max(valid, key=lambda k: valid[k]["speedup"]) if valid else None
            backend_results[backend_name] = {
                "solvers": solver_results,
                "best_solver": best,
            }

        results.append({
            "family": fam, "penalty": pen_label,
            "ref_time": ref_time,
            "backends": backend_results,
        })

        # Print summary
        parts = []
        for bn in ["numpy", "cupy", "torch"]:
            br = backend_results[bn]
            best = br["best_solver"]
            if best:
                spd = br["solvers"][best]["speedup"]
                parts.append(f"{bn}:{best}({spd:.1f}x)")
        print(f"[{done}/{total}] {fam}+{pen_label} ref={ref_time:.3f}s {' | '.join(parts)}", flush=True)

with open('/root/statgpu/results_solver_perf.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nTotal: {len(results)} results")
print("DONE")
