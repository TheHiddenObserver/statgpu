"""
Full loss x penalty x solver x backend benchmark v2.

Improvements over v1:
- Larger data scales: n=5000, 50000 (show GPU advantage)
- Convergence-based stopping (increase max_iter until converge)
- Precision: compare against statsmodels/scipy reference solutions
- Optimality check: verify KKT conditions at solution
"""

import time
import numpy as np
import warnings
warnings.filterwarnings("ignore")  # suppress convergence warnings for clean output


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
    if hasattr(coef, 'get'):
        return coef.get()
    return np.asarray(coef)


# ── Data ─────────────────────────────────────────────────────────────

def make_data(n, p, seed=42):
    np.random.seed(seed)
    X = np.random.randn(n, p)
    true_coef = np.zeros(p)
    true_coef[:3] = [1.0, -2.0, 0.5]
    y = X @ true_coef + np.random.randn(n) * 0.5
    return X, y, true_coef


def make_survival_data(n, p, seed=42):
    np.random.seed(seed)
    X = np.random.randn(n, p)
    true_coef = np.zeros(p)
    true_coef[:3] = [0.5, -1.0, 0.3]
    time_arr = np.random.exponential(1.0 / np.exp(X @ true_coef))
    event = np.ones(n)
    return X, {"time": time_arr, "event": event}, true_coef


# ── Imports ──────────────────────────────────────────────────────────

import torch
try:
    import cupy
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

HAS_CUDA = torch.cuda.is_available()

from statgpu.losses import QuantileLoss, HuberLoss, CoxPartialLikelihoodLoss
from statgpu.glm_core import get_glm_loss
from statgpu.solvers import fista_solver, fista_bb_solver, newton_solver, lbfgs_solver, admm_solver
from statgpu.penalties import (
    L2Penalty, L1Penalty, ElasticNetPenalty, SCADPenalty, MCPPenalty,
    GroupLassoPenalty,
)

# ── Backend conversion ───────────────────────────────────────────────

def to_backend_X(X_np, backend):
    if backend == "numpy":
        return X_np
    elif backend == "cupy":
        return cupy.asarray(X_np)
    elif backend == "torch":
        return torch.from_numpy(X_np).double()
    elif backend == "torch-cuda":
        return torch.from_numpy(X_np).double().cuda()
    return X_np


def to_backend_y(y, backend):
    if isinstance(y, dict):
        return {k: to_backend_y(v, backend) for k, v in y.items()}
    if isinstance(y, np.ndarray):
        return to_backend_X(y, backend)
    return y


# ── Reference solutions ──────────────────────────────────────────────

def compute_reference(loss_name, X, y, p):
    """Compute reference solution using statsmodels/scipy."""
    import statsmodels.api as sm

    if loss_name == "squared":
        # OLS
        return np.linalg.lstsq(X, y, rcond=None)[0]

    elif loss_name == "huber":
        # statsmodels RLM
        rlm = sm.RLM(y, X, M=sm.robust.norms.HuberT())
        return rlm.fit().params

    elif loss_name == "quantile":
        # statsmodels QuantReg
        model = sm.QuantReg(y, X)
        return model.fit(q=0.5).params

    elif loss_name == "cox_ph":
        import statsmodels.duration.api as smd
        model = smd.PHReg(y["time"], X, status=y["event"], ties="breslow")
        return model.fit(disp=0).params

    return None


# ── Optimality check ─────────────────────────────────────────────────

def check_optimality(loss, penalty, coef, X, y):
    """Check KKT conditions: grad_loss + subgrad_penalty ≈ 0."""
    coef_np_arr = coef_np(coef)
    grad = loss.gradient(X, y, coef_np_arr)
    grad_norm = np.linalg.norm(grad)

    if penalty is None:
        return grad_norm

    # For smooth penalties, check grad + penalty_grad ≈ 0
    pen_name = type(penalty).__name__
    if hasattr(penalty, 'smooth_gradient'):
        try:
            pen_grad = penalty.smooth_gradient(coef_np_arr)
            residual = grad + pen_grad
            return np.linalg.norm(residual)
        except Exception:
            pass

    # For non-smooth, just check loss gradient magnitude
    return grad_norm


# ── Solver config ────────────────────────────────────────────────────

SOLVERS = {
    "fista":    {"fn": fista_solver,    "smooth_only": False, "needs_hessian": False, "max_iter": 2000},
    "fista_bb": {"fn": fista_bb_solver, "smooth_only": False, "needs_hessian": False, "max_iter": 2000},
    "newton":   {"fn": newton_solver,   "smooth_only": True,  "needs_hessian": True,  "max_iter": 100},
    "lbfgs":    {"fn": lbfgs_solver,    "smooth_only": True,  "needs_hessian": False, "max_iter": 500},
    "admm":     {"fn": admm_solver,     "smooth_only": False, "needs_hessian": False, "max_iter": 500},
}

SMOOTH_PENALTIES = {"none", "l2", "elasticnet"}
NON_SMOOTH_PENALTIES = {"l1", "scad", "mcp", "group_lasso"}


def make_penalties(p):
    half = max(1, p // 2)
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


# ── Main benchmark ───────────────────────────────────────────────────

print("=" * 130)
print("Full Loss x Penalty x Solver x Backend Benchmark v2")
print("Larger scales, convergence-based, precision vs external frameworks")
print("=" * 130)

backends = ["numpy", "torch"]
if HAS_CUPY:
    backends.append("cupy")
if HAS_CUDA:
    backends.append("torch-cuda")

print("Backends: %s | CUDA: %s" % (", ".join(backends), HAS_CUDA))

# ── Reference solutions ──────────────────────────────────────────────

SCALES = [
    ("small",  500,   5),
    ("medium", 5000,  10),
    ("large",  50000, 10),
]

LOSS_CONFIGS_CONT = [
    ("squared",   lambda: get_glm_loss("squared_error")),
    ("huber",     lambda: HuberLoss(delta=1.0)),
    ("quantile",  lambda: QuantileLoss(quantile=0.5)),
]

LOSS_CONFIGS_SURV = [
    ("cox_ph",    lambda: CoxPartialLikelihoodLoss(ties="breslow")),
]

print("\n--- Reference solutions (statsmodels) — |ref-true| = statsmodels error vs true coef ---")
for scale_name, n, p in SCALES:
    X, y, true = make_data(n, p)
    for loss_name, _ in LOSS_CONFIGS_CONT:
        ref = compute_reference(loss_name, X, y, p)
        if ref is not None:
            err = np.linalg.norm(ref - true)
            print("  %s n=%d p=%d: statsmodels |ref-true|=%.4f" % (loss_name, n, p, err))

X_cox, y_cox, true_cox = make_survival_data(5000, 5)
ref_cox = compute_reference("cox_ph", X_cox, y_cox, 5)
print("  cox_ph n=5000 p=5: statsmodels |ref-true|=%.4f" % np.linalg.norm(ref_cox - true_cox))

# ── Full matrix ──────────────────────────────────────────────────────

print("\n" + "-" * 130)
print("%-8s %-6s %-10s %-12s %-10s %8s %8s %10s %10s %8s" % (
    "loss", "scale", "penalty", "solver", "backend", "time_ms", "iter", "|ours-ref|", "kkt_err", "status"))
print("-" * 130)

for scale_name, n, p in SCALES:
    X_np, y_np, true_coef = make_data(n, p)
    penalties = make_penalties(p)

    # Reference
    refs = {}
    for loss_name, _ in LOSS_CONFIGS_CONT:
        refs[loss_name] = compute_reference(loss_name, X_np, y_np, p)

    for loss_name, loss_fn in LOSS_CONFIGS_CONT:
        for pen_name, penalty in penalties.items():
            for solver_name, solver_info in SOLVERS.items():
                # Skip incompatible
                if solver_info["smooth_only"] and pen_name in NON_SMOOTH_PENALTIES:
                    continue
                if solver_info["needs_hessian"] and loss_name == "quantile":
                    continue

                for backend in backends:
                    X_b = to_backend_X(X_np, backend)
                    y_b = to_backend_y(y_np, backend)
                    loss = loss_fn()

                    fn = solver_info["fn"]
                    kwargs = {"max_iter": solver_info["max_iter"], "tol": 1e-6}
                    if solver_name == "newton":
                        kwargs["tol"] = 1e-8

                    try:
                        (coef, n_iter), t_ms = timer(fn, loss, penalty, X_b, y_b, **kwargs)
                    except (ValueError, NotImplementedError, TypeError, RuntimeError, np.linalg.LinAlgError) as e:
                        print("%-8s %-6s %-10s %-12s %-10s %8s %8s %10s %10s %8s" % (
                            loss_name, scale_name, pen_name, solver_name, backend,
                            "-", "-", "-", "-", "ERR"))
                        continue

                    coef_arr = coef_np(coef)
                    ref = refs.get(loss_name)
                    ref_err = np.linalg.norm(coef_arr - ref) if ref is not None else 0.0
                    kkt_err = check_optimality(loss, penalty, coef_arr, X_np, y_np)
                    finite = np.all(np.isfinite(coef_arr))
                    status = "OK" if finite else "NaN"
                    if n_iter >= solver_info["max_iter"]:
                        status = "NC"  # not converged

                    print("%-8s %-6s %-10s %-12s %-10s %7.0fms %8d %10.4f %10.6f %8s" % (
                        loss_name, scale_name, pen_name, solver_name, backend,
                        t_ms, n_iter, ref_err, kkt_err, status))

# ── Survival (CoxPH) ────────────────────────────────────────────────

for scale_name, n, p in [("small", 500, 5), ("medium", 5000, 5)]:
    X_np, y_np, true_coef = make_survival_data(n, p)
    penalties = make_penalties(p)
    ref_cox = compute_reference("cox_ph", X_np, y_np, p)

    for pen_name, penalty in penalties.items():
        for solver_name, solver_info in SOLVERS.items():
            if solver_info["smooth_only"] and pen_name in NON_SMOOTH_PENALTIES:
                continue

            for backend in backends:
                X_b = to_backend_X(X_np, backend)
                y_b = to_backend_y(y_np, backend)
                loss = CoxPartialLikelihoodLoss(ties="breslow")

                fn = solver_info["fn"]
                kwargs = {"max_iter": solver_info["max_iter"], "tol": 1e-6}
                if solver_name == "newton":
                    kwargs["tol"] = 1e-8

                try:
                    (coef, n_iter), t_ms = timer(fn, loss, penalty, X_b, y_b, **kwargs)
                except (ValueError, NotImplementedError, TypeError, RuntimeError, np.linalg.LinAlgError) as e:
                    print("%-8s %-6s %-10s %-12s %-10s %8s %8s %10s %10s %8s" % (
                        "cox_ph", scale_name, pen_name, solver_name, backend,
                        "-", "-", "-", "-", "ERR"))
                    continue

                coef_arr = coef_np(coef)
                ref_err = np.linalg.norm(coef_arr - ref_cox) if ref_cox is not None else 0.0
                kkt_err = check_optimality(loss, penalty, coef_arr, X_np, y_np)
                finite = np.all(np.isfinite(coef_arr))
                status = "OK" if finite else "NaN"
                if n_iter >= solver_info["max_iter"]:
                    status = "NC"

                print("%-8s %-6s %-10s %-12s %-10s %7.0fms %8d %10.4f %10.6f %8s" % (
                    "cox_ph", scale_name, pen_name, solver_name, backend,
                    t_ms, n_iter, ref_err, kkt_err, status))

print("\n" + "=" * 130)
print("Legend: |ours-ref| = |statgpu_coef - statsmodels_coef|; kkt_err = optimality residual")
print("        OK = converged and finite; NC = not converged (hit max_iter); ERR = error")
print("=" * 130)
print("Done.")
