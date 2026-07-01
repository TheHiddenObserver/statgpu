"""
Precision and timing benchmark for loss × penalty × solver matrix.

For each combination:
- Precision: compare coefficients against reference (L-BFGS + L2 baseline)
- Timing: wall-clock time (median of 3 runs)

Runs on local machine (numpy/torch) and optionally on remote GPU.
"""

import time
import numpy as np


def timer(func, *a, n=3, **kw):
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        r = func(*a, **kw)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return r, np.median(times)


def coef_np(coef):
    if hasattr(coef, 'cpu'):
        return coef.cpu().numpy()
    return np.asarray(coef)


# ── Data ─────────────────────────────────────────────────────────────

np.random.seed(42)
n, p = 500, 5
X = np.random.randn(n, p)
true_coef = np.array([1.0, -2.0, 0.5, 0.0, 0.3])
y = X @ true_coef + np.random.randn(n) * 0.5

# Survival data
X_cox = np.random.randn(300, 3)
true_cox = np.array([0.5, -1.0, 0.3])
time_cox = np.random.exponential(1.0 / np.exp(X_cox @ true_cox))
event_cox = np.ones(300)
y_cox = {"time": time_cox, "event": event_cox}


# ── Imports ──────────────────────────────────────────────────────────

from statgpu.losses import QuantileLoss, HuberLoss, CoxPartialLikelihoodLoss
from statgpu.glm_core import get_glm_loss
from statgpu.solvers import fista_solver, fista_bb_solver, newton_solver, lbfgs_solver, admm_solver
from statgpu.penalties import (
    L2Penalty, L1Penalty, ElasticNetPenalty, SCADPenalty, MCPPenalty,
    AdaptiveL1Penalty, GroupLassoPenalty, GroupMCPPenalty, GroupSCADPenalty,
)

SOLVERS = {
    "fista": fista_solver,
    "fista_bb": fista_bb_solver,
    "newton": newton_solver,
    "lbfgs": lbfgs_solver,
    "admm": admm_solver,
}

SMOOTH_SOLVERS = {"newton", "lbfgs"}

LOSSES_CONT = {
    "squared": lambda: get_glm_loss("squared_error"),
    "huber":   lambda: HuberLoss(delta=1.0),
    "quantile": lambda: QuantileLoss(quantile=0.5),
}
LOSSES_SURV = {
    "cox_ph": lambda: CoxPartialLikelihoodLoss(ties="breslow"),
}


def make_penalties(p):
    half = p // 2
    groups = [list(range(0, half)), list(range(half, p))]
    return {
        "none": L2Penalty(0.0),
        "l2": L2Penalty(0.01),
        "l1": L1Penalty(0.01),
        "elasticnet": ElasticNetPenalty(alpha=0.01, l1_ratio=0.5),
        "scad": SCADPenalty(alpha=0.01, a=3.7),
        "mcp": MCPPenalty(alpha=0.01, gamma=3.0),
        "group_lasso": GroupLassoPenalty(alpha=0.01, groups=groups),
    }


def run_one(loss, penalty, solver_name, X, y, init_coef=None):
    """Run one combination, return (coef, time_ms) or (None, None) on failure."""
    fn = SOLVERS[solver_name]
    kwargs = {"max_iter": 300, "tol": 1e-6}
    if solver_name == "newton":
        kwargs["max_iter"] = 50
        kwargs["tol"] = 1e-8
    if solver_name == "admm":
        kwargs["max_iter"] = 150
    if init_coef is not None:
        kwargs["init_coef"] = init_coef

    try:
        (coef, n_iter), t_ms = timer(fn, loss, penalty, X, y, **kwargs)
        return coef_np(coef), t_ms, n_iter
    except (ValueError, NotImplementedError, TypeError, np.linalg.LinAlgError) as e:
        return None, None, str(e)


# ── Benchmark ────────────────────────────────────────────────────────

print("=" * 80)
print("Loss × Penalty × Solver: Precision & Timing Benchmark")
print("=" * 80)

# Reference: L-BFGS + L2 for each loss
print("\n--- Reference solutions (L-BFGS + L2) ---")
ref_coefs = {}
for name, loss_fn in LOSSES_CONT.items():
    loss = loss_fn()
    coef, t, it = run_one(loss, L2Penalty(0.0), "lbfgs", X, y)
    ref_coefs[name] = coef
    print("  %s: %.1fms (%d iter)" % (name, t, it))

loss = LOSSES_SURV["cox_ph"]()
coef, t, it = run_one(loss, L2Penalty(0.0), "lbfgs", X_cox, y_cox)
ref_coefs["cox_ph"] = coef
print("  cox_ph: %.1fms (%d iter)" % (t, it))

# Full matrix
penalties = make_penalties(p)

print("\n--- Full matrix: loss × penalty × solver ---")
print("%-10s %-12s %-10s %8s %8s %8s %10s" % ("loss", "penalty", "solver", "time_ms", "iter", "|diff|", "status"))
print("-" * 80)

for loss_name, loss_fn in LOSSES_CONT.items():
    for pen_name, penalty in penalties.items():
        for solver_name in SOLVERS:
            # Skip incompatible
            if solver_name in SMOOTH_SOLVERS and pen_name in ("l1", "scad", "mcp", "group_lasso"):
                continue

            loss = loss_fn()

            # AdaptiveL1: use L2 warm-start
            init = ref_coefs.get(loss_name) if pen_name == "adaptive_l1" else None
            if pen_name == "adaptive_l1":
                if hasattr(penalty, 'set_weights'):
                    penalty.set_weights(ref_coefs.get(loss_name, np.zeros(p)))

            coef, t, info = run_one(loss, penalty, solver_name, X, y, init_coef=init)
            if coef is None:
                print("%-10s %-12s %-10s %8s %8s %8s %10s" % (loss_name, pen_name, solver_name, "-", "-", "-", "SKIP"))
                continue

            ref = ref_coefs.get(loss_name)
            diff = np.linalg.norm(coef - ref) if ref is not None else 0.0
            status = "OK" if np.all(np.isfinite(coef)) else "FAIL"
            print("%-10s %-12s %-10s %7.0fms %8d %8.4f %10s" % (loss_name, pen_name, solver_name, t, info, diff, status))

# Cox PH
for pen_name, penalty in make_penalties(3).items():
    for solver_name in SOLVERS:
        if solver_name in SMOOTH_SOLVERS and pen_name in ("l1", "scad", "mcp", "group_lasso"):
            continue

        loss = LOSSES_SURV["cox_ph"]()
        coef, t, info = run_one(loss, penalty, solver_name, X_cox, y_cox)
        if coef is None:
            print("%-10s %-12s %-10s %8s %8s %8s %10s" % ("cox_ph", pen_name, solver_name, "-", "-", "-", "SKIP"))
            continue

        ref = ref_coefs.get("cox_ph")
        diff = np.linalg.norm(coef - ref) if ref is not None else 0.0
        status = "OK" if np.all(np.isfinite(coef)) else "FAIL"
        print("%-10s %-12s %-10s %7.0fms %8d %8.4f %10s" % ("cox_ph", pen_name, solver_name, t, info, diff, status))

print("\n" + "=" * 80)
print("Done.")
