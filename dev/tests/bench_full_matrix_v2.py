"""
Full loss x penalty x solver x backend benchmark v2.

Precision metrics:
- |ours-ref|: statgpu coef vs statsmodels coef
- our_loss: loss value at statgpu solution
- ref_loss: loss value at statsmodels solution
- loss_diff: our_loss - ref_loss (negative = statgpu is better)
- kkt_err: optimality condition residual
"""

import time
import numpy as np
import warnings
warnings.filterwarnings("ignore")


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

def compute_reference(loss_name, X, y):
    """Compute reference solution using statsmodels."""
    import statsmodels.api as sm

    if loss_name == "squared":
        return np.linalg.lstsq(X, y, rcond=None)[0]

    elif loss_name == "huber":
        rlm = sm.RLM(y, X, M=sm.robust.norms.HuberT())
        return rlm.fit().params

    elif loss_name == "quantile":
        model = sm.QuantReg(y, X)
        return model.fit(q=0.5).params

    elif loss_name == "cox_ph":
        import statsmodels.duration.api as smd
        model = smd.PHReg(y["time"], X, status=y["event"], ties="breslow")
        return model.fit(disp=0).params

    return None


# ── Optimality check ─────────────────────────────────────────────────

def check_optimality(loss, penalty, coef, X, y):
    """Check KKT conditions: grad_loss + subgrad_penalty ≈ 0.

    For smooth penalties: grad_loss + grad_penalty ≈ 0
    For non-smooth penalties: check prox_fixed_point condition
        coef ≈ prox(coef - step * grad_loss, step)
    """
    coef_arr = coef_np(coef)
    grad = loss.gradient(X, y, coef_arr)

    if penalty is None:
        return np.linalg.norm(grad)

    # For smooth penalties (L2, ElasticNet): grad_loss + grad_penalty = 0
    if hasattr(penalty, 'gradient'):
        try:
            pen_grad = penalty.gradient(coef_arr)
            return np.linalg.norm(grad + pen_grad)
        except Exception:
            pass

    # For all penalties with proximal: check fixed-point condition
    # At optimum: coef = prox(coef - step * grad_loss, step)
    # residual = coef - prox(coef - step * grad, step)
    if hasattr(penalty, 'proximal'):
        try:
            L = loss.lipschitz(X, coef_arr, y) if hasattr(loss, 'lipschitz') else 1.0
            step = 1.0 / max(L, 1e-8)
            w_tilde = coef_arr - step * grad
            prox_result = penalty.proximal(w_tilde, step, backend="numpy")
            return np.linalg.norm(coef_arr - prox_result)
        except Exception:
            pass

    return np.linalg.norm(grad)


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

print("=" * 140)
print("Full Loss x Penalty x Solver x Backend Benchmark v2")
print("Precision: |ours-ref|, our_loss, ref_loss, loss_diff, kkt_err")
print("=" * 140)

backends = ["numpy", "torch"]
if HAS_CUPY:
    backends.append("cupy")
if HAS_CUDA:
    backends.append("torch-cuda")

print("Backends: %s | CUDA: %s" % (", ".join(backends), HAS_CUDA))

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

# ── Reference solutions ──────────────────────────────────────────────

print("\n--- Reference solutions (statsmodels) ---")
ref_solutions = {}
for scale_name, n, p in SCALES:
    X, y, true = make_data(n, p)
    for loss_name, loss_fn in LOSS_CONFIGS_CONT:
        ref = compute_reference(loss_name, X, y)
        if ref is not None:
            ref_solutions[(loss_name, scale_name)] = ref
            loss_obj = loss_fn()
            ref_loss = loss_obj.value(X, y, ref)
            true_loss = loss_obj.value(X, y, true)
            print("  %s n=%6d: |ref-true|=%.4f  ref_loss=%.6f  true_loss=%.6f" % (
                loss_name, n, np.linalg.norm(ref - true), ref_loss, true_loss))

X_cox, y_cox, true_cox = make_survival_data(5000, 5)
ref_cox = compute_reference("cox_ph", X_cox, y_cox)
cox_loss_obj = CoxPartialLikelihoodLoss(ties="breslow")
ref_cox_loss = cox_loss_obj.value(X_cox, y_cox, ref_cox)
true_cox_loss = cox_loss_obj.value(X_cox, y_cox, true_cox)
ref_solutions[("cox_ph", "medium")] = ref_cox
print("  cox_ph  n= 5000: |ref-true|=%.4f  ref_loss=%.6f  true_loss=%.6f" % (
    np.linalg.norm(ref_cox - true_cox), ref_cox_loss, true_cox_loss))

# ── Full matrix ──────────────────────────────────────────────────────

print("\n" + "-" * 140)
print("%-8s %-6s %-10s %-12s %-10s %7s %5s %9s %9s %9s %9s %6s" % (
    "loss", "scale", "penalty", "solver", "backend",
    "time", "iter", "|ours-ref|", "our_loss", "ref_loss", "loss_diff", "status"))
print("-" * 140)

for scale_name, n, p in SCALES:
    X_np, y_np, true_coef = make_data(n, p)
    penalties = make_penalties(p)

    for loss_name, loss_fn in LOSS_CONFIGS_CONT:
        ref_coef = ref_solutions.get((loss_name, scale_name))
        loss_obj = loss_fn()
        ref_loss_val = loss_obj.value(X_np, y_np, ref_coef) if ref_coef is not None else None

        for pen_name, penalty in penalties.items():
            for solver_name, solver_info in SOLVERS.items():
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
                    except (ValueError, NotImplementedError, TypeError, RuntimeError, np.linalg.LinAlgError):
                        print("%-8s %-6s %-10s %-12s %-10s %7s %5s %9s %9s %9s %9s %6s" % (
                            loss_name, scale_name, pen_name, solver_name, backend,
                            "-", "-", "-", "-", "-", "-", "ERR"))
                        continue

                    coef_arr = coef_np(coef)
                    ours_ref = np.linalg.norm(coef_arr - ref_coef) if ref_coef is not None else 0.0
                    our_loss_val = loss_obj.value(X_np, y_np, coef_arr)
                    loss_diff = our_loss_val - ref_loss_val if ref_loss_val is not None else 0.0
                    kkt_err = check_optimality(loss, penalty, coef_arr, X_np, y_np)
                    finite = np.all(np.isfinite(coef_arr))
                    status = "OK" if finite else "NaN"
                    if n_iter >= solver_info["max_iter"]:
                        status = "NC"

                    print("%-8s %-6s %-10s %-12s %-10s %6.0fms %5d %9.4f %9.5f %9.5f %+9.5f %6s" % (
                        loss_name, scale_name, pen_name, solver_name, backend,
                        t_ms, n_iter, ours_ref, our_loss_val, ref_loss_val, loss_diff, status))

# ── CoxPH ────────────────────────────────────────────────────────────

for scale_name, n, p in [("small", 500, 5), ("medium", 5000, 5)]:
    X_np, y_np, true_coef = make_survival_data(n, p)
    penalties = make_penalties(p)
    ref_coef = ref_solutions.get(("cox_ph", scale_name))
    cox_loss_obj = CoxPartialLikelihoodLoss(ties="breslow")
    ref_loss_val = cox_loss_obj.value(X_np, y_np, ref_coef) if ref_coef is not None else None

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
                except (ValueError, NotImplementedError, TypeError, RuntimeError, np.linalg.LinAlgError):
                    print("%-8s %-6s %-10s %-12s %-10s %7s %5s %9s %9s %9s %9s %6s" % (
                        "cox_ph", scale_name, pen_name, solver_name, backend,
                        "-", "-", "-", "-", "-", "-", "ERR"))
                    continue

                coef_arr = coef_np(coef)
                ours_ref = np.linalg.norm(coef_arr - ref_coef) if ref_coef is not None else 0.0
                our_loss_val = cox_loss_obj.value(X_np, y_np, coef_arr)
                loss_diff = our_loss_val - ref_loss_val if ref_loss_val is not None else 0.0
                kkt_err = check_optimality(loss, penalty, coef_arr, X_np, y_np)
                finite = np.all(np.isfinite(coef_arr))
                status = "OK" if finite else "NaN"
                if n_iter >= solver_info["max_iter"]:
                    status = "NC"

                print("%-8s %-6s %-10s %-12s %-10s %6.0fms %5d %9.4f %9.5f %9.5f %+9.5f %6s" % (
                    "cox_ph", scale_name, pen_name, solver_name, backend,
                    t_ms, n_iter, ours_ref, our_loss_val, ref_loss_val, loss_diff, status))

print("\n" + "=" * 140)
print("Legend:")
print("  |ours-ref| = |statgpu_coef - statsmodels_coef|")
print("  our_loss   = loss value at statgpu solution (lower is better)")
print("  ref_loss   = loss value at statsmodels solution")
print("  loss_diff  = our_loss - ref_loss (negative = statgpu found better solution)")
print("  OK = converged; NC = not converged; ERR = error")
print("=" * 140)
print("Done.")
