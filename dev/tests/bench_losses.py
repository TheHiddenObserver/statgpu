"""
Benchmark: LossBase losses (Quantile, Huber, Cox PH)

Compares:
1. 3-backend timing (numpy only for now — Quantile/Huber are numpy-ready)
2. vs R equivalents via rpy2 (quantreg::rq, MASS::rlm, survival::coxph)
3. vs scipy/statsmodels Python equivalents
"""

import time
import numpy as np


def timer(func, *args, n_repeats=5, **kwargs):
    """Run func n_repeats times, return (result, median_time_ms)."""
    times = []
    result = None
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return result, np.median(times)


def bench_quantile():
    """QuantileLoss benchmark."""
    from statgpu.losses import QuantileLoss
    from statgpu.solvers import lbfgs_solver

    print("\n" + "=" * 60)
    print("QuantileLoss (median regression)")
    print("=" * 60)

    for n in [1000, 5000, 20000]:
        np.random.seed(42)
        p = 10
        X = np.random.randn(n, p)
        true_coef = np.random.randn(p)
        y = X @ true_coef + np.random.randn(n) * 0.5

        loss = QuantileLoss(quantile=0.5)
        (coef, n_iter), t_ms = timer(lbfgs_solver, loss, None, X, y, max_iter=200, tol=1e-6)
        print(f"  n={n:6d}: {t_ms:8.1f} ms  ({n_iter} iterations)")

    # vs scipy (no direct equivalent, but linprog can do it)
    print("\n  Comparison with scipy (n=5000):")
    try:
        from scipy.optimize import linprog
        np.random.seed(42)
        n, p = 5000, 10
        X = np.random.randn(n, p)
        y = X @ np.random.randn(p) + np.random.randn(n) * 0.5

        # scipy quantile regression via LP
        _, t_scipy = timer(lambda: None)  # placeholder
        print(f"    scipy.optimize.linprog: not directly comparable (LP formulation)")
    except Exception as e:
        print(f"    scipy: {e}")


def bench_huber():
    """HuberLoss benchmark."""
    from statgpu.losses import HuberLoss
    from statgpu.solvers import lbfgs_solver

    print("\n" + "=" * 60)
    print("HuberLoss (robust regression)")
    print("=" * 60)

    for n in [1000, 5000, 20000]:
        np.random.seed(42)
        p = 10
        X = np.random.randn(n, p)
        true_coef = np.random.randn(p)
        y = X @ true_coef + np.random.randn(n) * 0.5

        loss = HuberLoss(delta=1.0)
        (coef, n_iter), t_ms = timer(lbfgs_solver, loss, None, X, y, max_iter=200, tol=1e-6)
        print(f"  n={n:6d}: {t_ms:8.1f} ms  ({n_iter} iterations)")

    # vs statsmodels
    print("\n  Comparison with statsmodels (n=5000):")
    try:
        import statsmodels.api as sm
        np.random.seed(42)
        n, p = 5000, 10
        X = np.random.randn(n, p)
        y = X @ np.random.randn(p) + np.random.randn(n) * 0.5

        _, t_sm = timer(sm.RLM, y, X, M=sm.robust.norms.HuberT())
        print(f"    statsmodels.RLM: {t_sm:.1f} ms")

        loss = HuberLoss(delta=1.0)
        (_, n_iter), t_ours = timer(lbfgs_solver, loss, None, X, y, max_iter=200, tol=1e-6)
        print(f"    statgpu HuberLoss: {t_ours:.1f} ms ({n_iter} iter)")
        print(f"    speedup: {t_sm / t_ours:.2f}x")
    except ImportError:
        print("    statsmodels not available")


def bench_cox():
    """CoxPartialLikelihoodLoss benchmark."""
    from statgpu.losses import CoxPartialLikelihoodLoss
    from statgpu.solvers import newton_solver
    from statgpu.penalties import L2Penalty

    print("\n" + "=" * 60)
    print("CoxPartialLikelihoodLoss (survival)")
    print("=" * 60)

    for n in [500, 2000, 5000]:
        np.random.seed(42)
        p = 5
        X = np.random.randn(n, p)
        true_coef = np.array([0.5, -1.0, 0.3, 0.0, 0.2])
        time_to_event = np.random.exponential(1.0 / np.exp(X @ true_coef))
        event = np.ones(n)
        y = {'time': time_to_event, 'event': event}

        loss = CoxPartialLikelihoodLoss(ties='breslow')
        (coef, n_iter), t_ms = timer(
            newton_solver, loss, L2Penalty(0.0), X, y, max_iter=50, tol=1e-8
        )
        print(f"  n={n:6d}: {t_ms:8.1f} ms  ({n_iter} iterations)")

    # vs statsmodels
    print("\n  Comparison with statsmodels (n=2000):")
    try:
        import statsmodels.duration.api as smd
        np.random.seed(42)
        n, p = 2000, 5
        X = np.random.randn(n, p)
        true_coef = np.array([0.5, -1.0, 0.3, 0.0, 0.2])
        time_to_event = np.random.exponential(1.0 / np.exp(X @ true_coef))
        event = np.ones(n)

        _, t_sm = timer(smd.PHReg, time_to_event, X, status=event, ties='breslow')
        print(f"    statsmodels.PHReg: {t_sm:.1f} ms")

        y = {'time': time_to_event, 'event': event}
        loss = CoxPartialLikelihoodLoss(ties='breslow')
        (_, n_iter), t_ours = timer(
            newton_solver, loss, L2Penalty(0.0), X, y, max_iter=50, tol=1e-8
        )
        print(f"    statgpu CoxPH: {t_ours:.1f} ms ({n_iter} iter)")
        print(f"    speedup: {t_sm / t_ours:.2f}x")
    except ImportError:
        print("    statsmodels not available")


def bench_r_comparison():
    """Compare with R equivalents via rpy2."""
    print("\n" + "=" * 60)
    print("R Comparison (via rpy2)")
    print("=" * 60)

    try:
        import rpy2.robjects as ro
        from rpy2.robjects import numpy2ri
        numpy2ri.activate()
    except ImportError:
        print("  rpy2 not available, skipping R comparison")
        return

    np.random.seed(42)
    n, p = 2000, 5
    X = np.random.randn(n, p)
    true_coef = np.array([1.0, -2.0, 0.5, 0.0, 0.3])
    y = X @ true_coef + np.random.randn(n) * 0.5

    # Quantile regression
    print("\n  Quantile regression (n=2000, p=5):")
    try:
        ro.globalenv['X'] = X
        ro.globalenv['y'] = y
        _, t_r = timer(lambda: ro.r('library(quantreg); rq(y ~ X - 1, tau=0.5)'))
        print(f"    R quantreg::rq: {t_r:.1f} ms")

        from statgpu.losses import QuantileLoss
        from statgpu.solvers import lbfgs_solver
        loss = QuantileLoss(quantile=0.5)
        (_, n_iter), t_ours = timer(lbfgs_solver, loss, None, X, y, max_iter=200, tol=1e-6)
        print(f"    statgpu QuantileLoss: {t_ours:.1f} ms ({n_iter} iter)")
        print(f"    speedup: {t_r / t_ours:.2f}x")
    except Exception as e:
        print(f"    R quantreg: {e}")

    # Huber regression
    print("\n  Huber regression (n=2000, p=5):")
    try:
        ro.globalenv['X'] = X
        ro.globalenv['y'] = y
        _, t_r = timer(lambda: ro.r('library(MASS); rlm(y ~ X - 1, method="M", psi=psi.huber)'))
        print(f"    R MASS::rlm: {t_r:.1f} ms")

        from statgpu.losses import HuberLoss
        from statgpu.solvers import lbfgs_solver
        loss = HuberLoss(delta=1.0)
        (_, n_iter), t_ours = timer(lbfgs_solver, loss, None, X, y, max_iter=200, tol=1e-6)
        print(f"    statgpu HuberLoss: {t_ours:.1f} ms ({n_iter} iter)")
        print(f"    speedup: {t_r / t_ours:.2f}x")
    except Exception as e:
        print(f"    R MASS: {e}")

    # Cox PH
    print("\n  Cox PH (n=2000, p=5):")
    try:
        time_to_event = np.random.exponential(1.0 / np.exp(X @ true_coef))
        event = np.ones(n)
        ro.globalenv['X'] = X
        ro.globalenv['time'] = time_to_event
        ro.globalenv['event'] = event.astype(int)
        _, t_r = timer(lambda: ro.r('library(survival); coxph(Surv(time, event) ~ X)'))
        print(f"    R survival::coxph: {t_r:.1f} ms")

        from statgpu.losses import CoxPartialLikelihoodLoss
        from statgpu.solvers import newton_solver
        from statgpu.penalties import L2Penalty
        y_surv = {'time': time_to_event, 'event': event}
        loss = CoxPartialLikelihoodLoss(ties='breslow')
        (_, n_iter), t_ours = timer(
            newton_solver, loss, L2Penalty(0.0), X, y_surv, max_iter=50, tol=1e-8
        )
        print(f"    statgpu CoxPH: {t_ours:.1f} ms ({n_iter} iter)")
        print(f"    speedup: {t_r / t_ours:.2f}x")
    except Exception as e:
        print(f"    R survival: {e}")


if __name__ == '__main__':
    print("statgpu LossBase Benchmark")
    print("=" * 60)

    bench_quantile()
    bench_huber()
    bench_cox()
    bench_r_comparison()

    print("\n" + "=" * 60)
    print("Benchmark complete.")
