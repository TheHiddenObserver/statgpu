"""
Precision and timing comparison: statgpu vs external frameworks.

Compares:
1. QuantileLoss vs sklearn.QuantileRegressor (L1, L2)
2. HuberLoss vs statsmodels.RLM + R robustbase::lmrob
3. CoxPH vs statsmodels.PHReg

For each comparison:
- Precision: |statgpu_coef - external_coef|
- Timing: wall-clock time (median of 5)
- Loss value comparison: loss(statgpu_coef) vs loss(external_coef)
"""

import time
import numpy as np
import warnings
warnings.filterwarnings("ignore")


def timer(func, *a, n=5, **kw):
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
from statgpu.solvers import lbfgs_solver, fista_solver, newton_solver
from statgpu.penalties import L2Penalty, L1Penalty


# ── Backend conversion ───────────────────────────────────────────────

def to_backend(X_np, y_np, backend):
    if backend == "numpy":
        return X_np, y_np
    elif backend == "cupy":
        return cupy.asarray(X_np), cupy.asarray(y_np)
    elif backend == "torch":
        return torch.from_numpy(X_np).double(), torch.from_numpy(y_np).double()
    elif backend == "torch-cuda":
        return torch.from_numpy(X_np).double().cuda(), torch.from_numpy(y_np).double().cuda()
    return X_np, y_np


# ── Main benchmark ───────────────────────────────────────────────────

print("=" * 120)
print("statgpu vs External Frameworks: Precision & Timing")
print("=" * 120)

backends = ["numpy", "torch"]
if HAS_CUPY:
    backends.append("cupy")
if HAS_CUDA:
    backends.append("torch-cuda")

print("Backends: %s | CUDA: %s" % (", ".join(backends), HAS_CUDA))

SCALES = [
    (500, 5),
    (5000, 10),
    (50000, 10),
]

# ═══════════════════════════════════════════════════════════════════
# 1. QuantileLoss vs sklearn.QuantileRegressor
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 120)
print("1. QuantileLoss vs sklearn.QuantileRegressor")
print("=" * 120)

from sklearn.linear_model import QuantileRegressor

for n, p in SCALES:
    X, y, true = make_data(n, p)

    for tau in [0.25, 0.5, 0.75]:
        # sklearn
        try:
            def run_sklearn_qr(X, y, tau):
                qr = QuantileRegressor(quantile=tau, alpha=0, solver="highs")
                return qr.fit(X, y).coef_
            coef_sk, t_sk = timer(run_sklearn_qr, X, y, tau)
        except Exception as e:
            coef_sk, t_sk = None, None

        # statgpu (all backends)
        results = {}
        for backend in backends:
            X_b, y_b = to_backend(X, y, backend)
            loss = QuantileLoss(quantile=tau)
            try:
                (coef, n_iter), t_ms = timer(
                    fista_solver, loss, L2Penalty(0.0), X_b, y_b,
                    n=3, max_iter=2000, tol=1e-6
                )
                results[backend] = (coef_np(coef), t_ms, n_iter)
            except Exception:
                results[backend] = (None, None, None)

        # Print
        if coef_sk is not None:
            ref_loss = QuantileLoss(quantile=tau).value(X, y, coef_sk)
            print("\n  tau=%.2f n=%d p=%d:" % (tau, n, p))
            print("    sklearn:  coef=%s  time=%.0fms  loss=%.6f" % (
                np.round(coef_sk, 4), t_sk, ref_loss))

            for backend in backends:
                coef_b, t_b, it_b = results[backend]
                if coef_b is not None:
                    diff = np.linalg.norm(coef_b - coef_sk)
                    loss_b = QuantileLoss(quantile=tau).value(X, y, coef_b)
                    print("    %-10s coef=%s  time=%.0fms(%dit)  |diff|=%.6f  loss=%.6f  loss_diff=%+.6f" % (
                        backend, np.round(coef_b, 4), t_b, it_b, diff, loss_b, loss_b - ref_loss))


# ═══════════════════════════════════════════════════════════════════
# 2. HuberLoss vs statsmodels.RLM
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 120)
print("2. HuberLoss vs statsmodels.RLM")
print("=" * 120)

import statsmodels.api as sm

for n, p in SCALES:
    X, y, true = make_data(n, p)

    # statsmodels RLM
    def run_sm_rlm(X, y):
        return sm.RLM(y, X, M=sm.robust.norms.HuberT()).fit().params
    coef_sm, t_sm = timer(run_sm_rlm, X, y)
    ref_loss = HuberLoss(delta=1.0).value(X, y, coef_sm)

    # statgpu
    results = {}
    for backend in backends:
        X_b, y_b = to_backend(X, y, backend)
        loss = HuberLoss(delta=1.0)
        (coef, n_iter), t_ms = timer(
            newton_solver, loss, L2Penalty(0.0), X_b, y_b,
            n=3, max_iter=100, tol=1e-8
        )
        results[backend] = (coef_np(coef), t_ms, n_iter)

    print("\n  n=%d p=%d:" % (n, p))
    print("    sm-RLM:   coef=%s  time=%.0fms  loss=%.6f" % (
        np.round(coef_sm, 4), t_sm, ref_loss))

    for backend in backends:
        coef_b, t_b, it_b = results[backend]
        diff = np.linalg.norm(coef_b - coef_sm)
        loss_b = HuberLoss(delta=1.0).value(X, y, coef_b)
        print("    %-10s coef=%s  time=%.0fms(%dit)  |diff|=%.6f  loss=%.6f  loss_diff=%+.6f" % (
            backend, np.round(coef_b, 4), t_b, it_b, diff, loss_b, loss_b - ref_loss))


# ═══════════════════════════════════════════════════════════════════
# 3. HuberLoss vs R robustbase::lmrob
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 120)
print("3. HuberLoss vs R robustbase::lmrob (MM-estimator)")
print("=" * 120)

try:
    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri
    numpy2ri.activate()
    ro.r('library(robustbase)')
    HAS_R = True
except Exception:
    HAS_R = False
    print("  rpy2/robustbase not available, skipping")

if HAS_R:
    for n, p in SCALES:
        X, y, true = make_data(n, p)

        # R lmrob
        ro.globalenv['X'] = X
        ro.globalenv['y'] = y
        def run_lmrob():
            return np.array(ro.r('as.numeric(lmrob(y ~ X - 1)$coefficients)'))
        coef_r, t_r = timer(run_lmrob)

        # statgpu
        loss = HuberLoss(delta=1.0)
        (coef_ours, n_iter), t_ours = timer(
            newton_solver, loss, L2Penalty(0.0), X, y,
            n=3, max_iter=100, tol=1e-8
        )
        coef_ours = coef_np(coef_ours)

        diff = np.linalg.norm(coef_ours - coef_r)
        loss_ours = HuberLoss(delta=1.0).value(X, y, coef_ours)
        loss_r = HuberLoss(delta=1.0).value(X, y, coef_r)

        print("\n  n=%d p=%d:" % (n, p))
        print("    R lmrob:  coef=%s  time=%.0fms  loss=%.6f" % (np.round(coef_r, 4), t_r, loss_r))
        print("    statgpu:  coef=%s  time=%.0fms(%dit)  loss=%.6f" % (np.round(coef_ours, 4), t_ours, n_iter, loss_ours))
        print("    |diff|=%.6f  loss_diff=%+.6f" % (diff, loss_ours - loss_r))


# ═══════════════════════════════════════════════════════════════════
# 4. CoxPH vs statsmodels.PHReg
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 120)
print("4. CoxPH vs statsmodels.PHReg")
print("=" * 120)

import statsmodels.duration.api as smd

for n, p in [(500, 5), (2000, 5), (5000, 5)]:
    X, y, true = make_survival_data(n, p)

    # statsmodels PHReg
    def run_sm_cox(X, y):
        return smd.PHReg(y['time'], X, status=y['event'], ties='breslow').fit(disp=0).params
    coef_sm, t_sm = timer(run_sm_cox, X, y)

    # statgpu
    results = {}
    for backend in backends:
        X_b = X if backend == "numpy" else (
            cupy.asarray(X) if backend == "cupy" else
            torch.from_numpy(X).double() if backend == "torch" else
            torch.from_numpy(X).double().cuda()
        )
        y_b = y if backend == "numpy" else {
            'time': (cupy.asarray(y['time']) if backend == "cupy" else
                     torch.from_numpy(y['time']).double() if backend == "torch" else
                     torch.from_numpy(y['time']).double().cuda()),
            'event': (cupy.asarray(y['event']) if backend == "cupy" else
                      torch.from_numpy(y['event']).double() if backend == "torch" else
                      torch.from_numpy(y['event']).double().cuda()),
        }
        loss = CoxPartialLikelihoodLoss(ties='breslow')
        try:
            (coef, n_iter), t_ms = timer(
                newton_solver, loss, L2Penalty(0.0), X_b, y_b,
                n=3, max_iter=100, tol=1e-8
            )
            results[backend] = (coef_np(coef), t_ms, n_iter)
        except Exception as e:
            results[backend] = (None, None, str(e))

    cox_loss = CoxPartialLikelihoodLoss(ties='breslow')
    ref_loss = cox_loss.value(X, y, coef_sm)

    print("\n  n=%d p=%d:" % (n, p))
    print("    sm-PHReg: coef=%s  time=%.0fms  loss=%.6f" % (
        np.round(coef_sm, 4), t_sm, ref_loss))

    for backend in backends:
        coef_b, t_b, it_b = results[backend]
        if coef_b is not None:
            diff = np.linalg.norm(coef_b - coef_sm)
            loss_b = cox_loss.value(X, y, coef_b)
            print("    %-10s coef=%s  time=%.0fms(%dit)  |diff|=%.6f  loss=%.6f  loss_diff=%+.6f" % (
                backend, np.round(coef_b, 4), t_b, it_b, diff, loss_b, loss_b - ref_loss))


# ═══════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════

print("\n" + "=" * 120)
print("Summary")
print("=" * 120)
print("""
Precision metrics:
  |diff| = |statgpu_coef - external_coef|
  loss_diff = loss(statgpu) - loss(external)
    negative = statgpu found better solution
    positive = external found better solution (or different penalty)

Timing metrics:
  All times are wall-clock ms, median of 5 runs
  Speedup = external_time / statgpu_best_time
""")
print("Done.")
