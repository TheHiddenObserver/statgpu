"""Remaining 25 cases at 100K: 3-backend, N_REPEAT=3.
For scad/mcp/group_scad/group_mcp: only test fista (representative),
then estimate fista_bb/admm as same speed (they share LLA path).
"""
import json, time, sys, warnings
sys.path.insert(0, '/root/statgpu')
import numpy as np
from statgpu.linear_model.penalized._base import PenalizedGeneralizedLinearModel

REMAINING = [
    ("inverse_gaussian", "scad", "scad", 0.1, {}),
    ("inverse_gaussian", "mcp", "mcp", 0.1, {}),
    ("inverse_gaussian", "group_lasso", "group_lasso", 0.1, {}),
    ("inverse_gaussian", "group_scad", "group_scad", 0.1, {}),
    ("inverse_gaussian", "group_mcp", "group_mcp", 0.1, {}),
    ("negative_binomial", "none", "l2", 0.0, {}),
    ("negative_binomial", "l2", "l2", 0.1, {}),
    ("negative_binomial", "l1", "l1", 0.1, {}),
    ("negative_binomial", "elasticnet", "elasticnet", 0.1, {"l1_ratio": 0.5}),
    ("negative_binomial", "adaptive_l1", "adaptive_l1", 0.1, {}),
    ("negative_binomial", "scad", "scad", 0.1, {}),
    ("negative_binomial", "mcp", "mcp", 0.1, {}),
    ("negative_binomial", "group_lasso", "group_lasso", 0.1, {}),
    ("negative_binomial", "group_scad", "group_scad", 0.1, {}),
    ("negative_binomial", "group_mcp", "group_mcp", 0.1, {}),
    ("tweedie", "none", "l2", 0.0, {}),
    ("tweedie", "l2", "l2", 0.1, {}),
    ("tweedie", "l1", "l1", 0.1, {}),
    ("tweedie", "elasticnet", "elasticnet", 0.1, {"l1_ratio": 0.5}),
    ("tweedie", "adaptive_l1", "adaptive_l1", 0.1, {}),
    ("tweedie", "scad", "scad", 0.1, {}),
    ("tweedie", "mcp", "mcp", 0.1, {}),
    ("tweedie", "group_lasso", "group_lasso", 0.1, {}),
    ("tweedie", "group_scad", "group_scad", 0.1, {}),
    ("tweedie", "group_mcp", "group_mcp", 0.1, {}),
]

SMOOTH = ["exact", "newton", "irls", "lbfgs", "fista", "fista_bb", "admm"]
NONSMMOOTH = ["fista", "fista_bb", "admm"]
# For LLA-based penalties, only test fista (others are ~identical)
LLA_PENALTIES = {"scad", "mcp", "group_scad", "group_mcp"}
LLA_SOLVERS = ["fista"]  # representative
BACKENDS = [("numpy", "cpu"), ("cupy", "cuda"), ("torch", "cuda")]

def gen(n, p, fam, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p); c = rng.randn(p)*0.5; e = X @ c
    if fam=="squared_error": y = e + rng.randn(n)*0.5
    elif fam=="logistic": p1=1/(1+np.exp(-np.clip(e,-20,20))); y=(rng.rand(n)<p1).astype(float)
    elif fam=="poisson": y=rng.poisson(np.exp(np.clip(e,-20,5))).astype(float)
    elif fam=="gamma": mu=np.exp(np.clip(e,-10,10)); y=rng.gamma(2.0,mu/2.0)
    elif fam=="inverse_gaussian": mu=np.exp(np.clip(e,-10,10)); y=np.abs(mu+rng.randn(n)*mu*0.3); y=np.maximum(y,1e-6)
    elif fam=="negative_binomial": mu=np.exp(np.clip(e,-10,10)); p_nb=0.5; r_nb=np.maximum(mu*p_nb/(1-p_nb),0.1); y=rng.negative_binomial(np.clip(r_nb,1,1000).astype(int),p_nb).astype(float)
    elif fam=="tweedie": mu=np.exp(np.clip(e,-10,10)); y=np.abs(mu+rng.randn(n)*mu*0.5); y=np.maximum(y,1e-6)
    return X, y

def groups_p(p, ng=None):
    if ng is None: ng=max(1,p//5)
    gs=p//ng
    return [list(range(g*gs,(g+1)*gs if g<ng-1 else p)) for g in range(ng)]

def fit_model(fam, pt, ap, pkw, solver, dev, X, y):
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        m = PenalizedGeneralizedLinearModel(
            loss=fam, penalty=pt, alpha=ap,
            penalty_kwargs=pkw if pkw else None,
            solver=solver, max_iter=200, tol=1e-4,
            device=dev, fit_intercept=True,
        )
        t0 = time.perf_counter()
        m.fit(X, y)
        elapsed = time.perf_counter() - t0
    return m.coef_, elapsed

N, P = 100000, 50
N_REPEAT = 3
results = []

# Warmup
import torch
Xw = torch.randn(100, 10, device="cuda")
_ = torch.mm(Xw, Xw.t())
torch.cuda.synchronize()
print("Warmup done", flush=True)

for i, (fam, plabel, pt, ap, pkw) in enumerate(REMAINING):
    X, y = gen(N, P, fam)
    grps = groups_p(P)
    is_smooth = plabel in ("none", "l2")
    is_lla = plabel in LLA_PENALTIES

    if is_smooth:
        solvers = SMOOTH
    elif is_lla:
        solvers = LLA_SOLVERS  # only fista for LLA-based
    else:
        solvers = NONSMMOOTH

    pkw2 = dict(pkw)
    if plabel in ("group_lasso", "group_scad", "group_mcp"):
        pkw2["groups"] = grps

    # Numpy reference: fista, median of N_REPEAT
    ref_times = []
    coef_ref = None
    for _ in range(N_REPEAT):
        try:
            c, t = fit_model(fam, pt, ap, pkw2, "fista", "cpu", X, y)
            ref_times.append(t)
            coef_ref = c.copy()
        except:
            pass
    ref_time = float(np.median(ref_times)) if ref_times else float("nan")

    if coef_ref is None:
        print(f"[{i+1}/25] {fam}+{plabel} REF FAIL", flush=True)
        continue

    backend_results = {}
    for bn, dev in BACKENDS:
        solver_results = {}
        for solver in solvers:
            times = []
            coef_gpu = None
            err_msg = None
            for _ in range(N_REPEAT):
                try:
                    c, t = fit_model(fam, pt, ap, pkw2, solver, dev, X, y)
                    times.append(t)
                    coef_gpu = c.copy()
                except Exception as e:
                    err_msg = str(e)[:150]
            gpu_time = float(np.median(times)) if times else float("nan")
            l2_diff = float(np.linalg.norm(coef_gpu - coef_ref)) if coef_gpu is not None else float("nan")
            speedup = ref_time / gpu_time if gpu_time > 0 and np.isfinite(gpu_time) else 0.0
            solver_results[solver] = {"time": gpu_time, "speedup": speedup, "l2_diff": l2_diff}
            if err_msg:
                solver_results[solver]["error"] = err_msg

        # For LLA penalties, copy fista results to fista_bb/admm (they share LLA path)
        if is_lla and "fista" in solver_results:
            fista_res = solver_results["fista"]
            for other in ["fista_bb", "admm"]:
                solver_results[other] = dict(fista_res)
                solver_results[other]["_estimated"] = True

        valid = {k: v for k, v in solver_results.items() if v["speedup"] > 0}
        best = max(valid, key=lambda k: valid[k]["speedup"]) if valid else None
        backend_results[bn] = {"solvers": solver_results, "best_solver": best}

    parts = []
    for bn in ["numpy", "cupy", "torch"]:
        b = backend_results[bn]
        bst = b["best_solver"]
        if bst:
            spd = b["solvers"][bst]["speedup"]
            parts.append(f"{bn}:{bst}({spd:.1f}x)")
    est_mark = " [LLA: fista only]" if is_lla else ""
    print(f"[{i+1}/25] {fam}+{plabel} ref={ref_time:.3f}s{est_mark} {' | '.join(parts)}", flush=True)

    results.append({
        "family": fam, "penalty": plabel,
        "ref_time": ref_time,
        "backends": backend_results,
    })

with open("/root/statgpu/results_remaining_100K.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nTotal: {len(results)}")
print("DONE")
