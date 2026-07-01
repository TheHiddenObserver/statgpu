"""Remote precision benchmark: family x penalty x solver."""
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

N, P = 10000, 50
alpha = 0.1
results = []
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

        # Build penalty_kwargs with groups for group penalties
        pkw = dict(pen_kw)
        if pen_label in ("group_lasso", "group_scad", "group_mcp"):
            pkw["groups"] = grps

        # Reference: numpy + fista
        try:
            coef_ref, _ = fit_model(fam, pen_type, pen_alpha, pkw, "fista", "cpu", X, y)
            coef_ref = coef_ref.copy()
        except Exception as e:
            results.append({"family": fam, "penalty": pen_label, "solver": "fista", "backend": "numpy", "status": "FAIL", "error": str(e)[:200]})
            print(f"[{done}/{total}] {fam}+{pen_label}+fista REF FAIL: {e}", flush=True)
            continue

        for solver in solvers:
            try:
                coef, elapsed = fit_model(fam, pen_type, pen_alpha, pkw, solver, "cpu", X, y)
                l2_diff = float(np.linalg.norm(coef - coef_ref))
                l2_ref = float(np.linalg.norm(coef_ref))
                results.append({
                    "family": fam, "penalty": pen_label, "solver": solver, "backend": "numpy",
                    "l2_diff": l2_diff, "rel_diff": l2_diff/max(l2_ref, 1e-12),
                    "time": elapsed, "status": "PASS",
                })
                print(f"[{done}/{total}] {fam}+{pen_label}+{solver} PASS rel={l2_diff/max(l2_ref,1e-12):.2e} t={elapsed:.3f}s", flush=True)
            except Exception as e:
                results.append({
                    "family": fam, "penalty": pen_label, "solver": solver, "backend": "numpy",
                    "status": "ERROR", "error": str(e)[:200],
                })
                print(f"[{done}/{total}] {fam}+{pen_label}+{solver} ERROR: {e}", flush=True)

with open('/root/statgpu/results_solver_precision.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nTotal: {len(results)} results")
print("DONE")
