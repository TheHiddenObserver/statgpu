"""
3-backend benchmark for LossBase losses (Quantile, Huber, Cox PH).

Compares:
1. Timing: numpy / cupy / torch-CPU / torch-CUDA at multiple scales
2. Precision: vs external frameworks (statsmodels, scipy)
3. Backend consistency: same results across backends

Usage:
    # Local (numpy + torch-CPU)
    python dev/tests/bench_losses_3backend.py

    # Remote GPU (numpy + cupy + torch-CUDA)
    ssh -p 28838 root@hz-4.matpool.com "cd /root/statgpu && conda activate myconda && python dev/tests/bench_losses_3backend.py"
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


def to_backend(X_np, y_np, backend, device="cpu"):
    """Convert numpy arrays to target backend."""
    if backend == "numpy":
        return X_np, y_np
    elif backend == "cupy":
        import cupy as cp
        return cp.asarray(X_np), cp.asarray(y_np)
    elif backend == "torch":
        import torch
        dev = device if torch.cuda.is_available() else "cpu"
        X_t = torch.from_numpy(X_np).double().to(dev)
        y_t = torch.from_numpy(y_np).double().to(dev)
        return X_t, y_t
    else:
        raise ValueError(f"Unknown backend: {backend}")


def to_backend_surv(X_np, time_np, event_np, backend, device="cpu"):
    """Convert survival data to target backend."""
    if backend == "numpy":
        return X_np, {'time': time_np, 'event': event_np}
    elif backend == "cupy":
        import cupy as cp
        return cp.asarray(X_np), {'time': cp.asarray(time_np), 'event': cp.asarray(event_np)}
    elif backend == "torch":
        import torch
        dev = device if torch.cuda.is_available() else "cpu"
        X_t = torch.from_numpy(X_np).double().to(dev)
        t_t = torch.from_numpy(time_np).double().to(dev)
        e_t = torch.from_numpy(event_np).double().to(dev)
        return X_t, {'time': t_t, 'event': e_t}
    else:
        raise ValueError(f"Unknown backend: {backend}")


def coef_to_numpy(coef):
    """Extract coef as numpy array regardless of backend."""
    if hasattr(coef, 'cpu'):
        return coef.cpu().numpy()
    elif hasattr(coef, 'get'):
        return coef.get()
    return np.asarray(coef)


# ── Precision comparison ────────────────────────────────────────────

def precision_quantile():
    """Compare QuantileLoss vs statsmodels QuantReg."""
    from statgpu.losses import QuantileLoss
    from statgpu.solvers import lbfgs_solver

    print("\n" + "=" * 60)
    print("Precision: QuantileLoss vs statsmodels QuantReg")
    print("=" * 60)

    for tau in [0.25, 0.5, 0.75]:
        np.random.seed(42)
        n, p = 500, 5
        X = np.random.randn(n, p)
        true_coef = np.array([1.0, -2.0, 0.5, 0.0, 0.3])
        y = X @ true_coef + np.random.randn(n) * 0.5

        # statgpu (FISTA for non-smooth quantile loss)
        from statgpu.solvers import fista_solver
        from statgpu.penalties import L2Penalty
        loss = QuantileLoss(quantile=tau)
        coef_ours, _ = fista_solver(loss, L2Penalty(0.0), X, y, max_iter=2000, tol=1e-8)

        # statsmodels QuantReg
        try:
            import statsmodels.api as sm
            model = sm.QuantReg(y, X)
            result = model.fit(q=tau)
            coef_sm = result.params

            diff = np.linalg.norm(coef_ours - coef_sm)
            print(f"  tau={tau}: |diff|={diff:.6f}  {'OK' if diff < 0.05 else 'WARN'}")
            if diff >= 0.05:
                print(f"    statgpu:    {np.round(coef_ours, 4)}")
                print(f"    statsmodels:{np.round(coef_sm, 4)}")
        except ImportError:
            print(f"  tau={tau}: statsmodels not available")


def precision_huber():
    """Compare HuberLoss vs statsmodels."""
    from statgpu.losses import HuberLoss
    from statgpu.solvers import lbfgs_solver

    print("\n" + "=" * 60)
    print("Precision: HuberLoss vs statsmodels")
    print("=" * 60)

    np.random.seed(42)
    n, p = 500, 5
    X = np.random.randn(n, p)
    true_coef = np.array([1.0, -2.0, 0.5, 0.0, 0.3])
    y = X @ true_coef + np.random.randn(n) * 0.5

    # statgpu
    loss = HuberLoss(delta=1.0)
    coef_ours, _ = lbfgs_solver(loss, None, X, y, max_iter=200, tol=1e-8)

    # statsmodels
    try:
        import statsmodels.api as sm
        rlm_model = sm.RLM(y, X, M=sm.robust.norms.HuberT())
        rlm_result = rlm_model.fit()
        coef_sm = rlm_result.params

        print(f"  statgpu HuberLoss:    {np.round(coef_ours, 4)}")
        print(f"  statsmodels RLM:      {np.round(coef_sm, 4)}")
        print(f"  diff (statgpu-sm):    {np.round(coef_ours - coef_sm, 4)}")
        print(f"  |diff|: {np.linalg.norm(coef_ours - coef_sm):.6f}")
    except ImportError:
        print("  statsmodels not available")


def precision_cox():
    """Compare CoxPartialLikelihoodLoss vs statsmodels."""
    from statgpu.losses import CoxPartialLikelihoodLoss
    from statgpu.solvers import newton_solver
    from statgpu.penalties import L2Penalty

    print("\n" + "=" * 60)
    print("Precision: CoxPH vs statsmodels")
    print("=" * 60)

    np.random.seed(42)
    n, p = 300, 3
    X = np.random.randn(n, p)
    true_coef = np.array([0.5, -1.0, 0.3])
    time_to_event = np.random.exponential(1.0 / np.exp(X @ true_coef))
    event = np.ones(n)

    # statgpu
    loss = CoxPartialLikelihoodLoss(ties='breslow')
    y = {'time': time_to_event, 'event': event}
    coef_ours, _ = newton_solver(loss, L2Penalty(0.0), X, y, max_iter=50, tol=1e-8)

    # statsmodels
    try:
        import statsmodels.duration.api as smd
        model = smd.PHReg(time_to_event, X, status=event, ties='breslow')
        result = model.fit(disp=0)
        coef_sm = result.params

        print(f"  statgpu CoxPH:        {np.round(coef_ours, 4)}")
        print(f"  statsmodels PHReg:    {np.round(coef_sm, 4)}")
        print(f"  diff (statgpu-sm):    {np.round(coef_ours - coef_sm, 4)}")
        print(f"  |diff|: {np.linalg.norm(coef_ours - coef_sm):.6f}")
    except ImportError:
        print("  statsmodels not available")


# ── Timing ──────────────────────────────────────────────────────────

def timing_quantile():
    """QuantileLoss timing across backends."""
    from statgpu.losses import QuantileLoss
    from statgpu.solvers import fista_solver
    from statgpu.penalties import L2Penalty

    print("\n" + "=" * 60)
    print("Timing: QuantileLoss (FISTA, tol=1e-6)")
    print("=" * 60)

    backends = ["numpy", "torch"]
    try:
        import cupy
        backends.insert(1, "cupy")
    except ImportError:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            backends.append("torch-cuda")
    except ImportError:
        pass

    for n in [1000, 5000, 20000]:
        np.random.seed(42)
        p = 10
        X_np = np.random.randn(n, p)
        true_coef = np.random.randn(p)
        y_np = X_np @ true_coef + np.random.randn(n) * 0.5

        row = f"  n={n:6d}:"
        for backend in backends:
            if backend == "torch-cuda":
                X_b, y_b = to_backend(X_np, y_np, "torch", device="cuda")
            else:
                X_b, y_b = to_backend(X_np, y_np, backend)
            loss = QuantileLoss(quantile=0.5)
            (_, n_iter), t_ms = timer(fista_solver, loss, L2Penalty(0.0), X_b, y_b, max_iter=2000, tol=1e-6)
            row += f"  {backend}={t_ms:7.0f}ms({n_iter:2d}it)"
        print(row)


def timing_huber():
    """HuberLoss timing across backends."""
    from statgpu.losses import HuberLoss
    from statgpu.solvers import lbfgs_solver

    print("\n" + "=" * 60)
    print("Timing: HuberLoss (L-BFGS, tol=1e-6)")
    print("=" * 60)

    backends = ["numpy", "torch"]
    try:
        import cupy
        backends.insert(1, "cupy")
    except ImportError:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            backends.append("torch-cuda")
    except ImportError:
        pass

    for n in [1000, 5000, 20000]:
        np.random.seed(42)
        p = 10
        X_np = np.random.randn(n, p)
        true_coef = np.random.randn(p)
        y_np = X_np @ true_coef + np.random.randn(n) * 0.5

        row = f"  n={n:6d}:"
        for backend in backends:
            if backend == "torch-cuda":
                X_b, y_b = to_backend(X_np, y_np, "torch", device="cuda")
            else:
                X_b, y_b = to_backend(X_np, y_np, backend)
            loss = HuberLoss(delta=1.0)
            (_, n_iter), t_ms = timer(lbfgs_solver, loss, None, X_b, y_b, max_iter=200, tol=1e-6)
            row += f"  {backend}={t_ms:7.0f}ms({n_iter:2d}it)"
        print(row)


def timing_cox():
    """CoxPartialLikelihoodLoss timing across backends."""
    from statgpu.losses import CoxPartialLikelihoodLoss
    from statgpu.solvers import newton_solver
    from statgpu.penalties import L2Penalty

    print("\n" + "=" * 60)
    print("Timing: CoxPH (Newton, tol=1e-8)")
    print("=" * 60)

    backends = ["numpy", "torch"]
    try:
        import cupy
        backends.insert(1, "cupy")
    except ImportError:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            backends.append("torch-cuda")
    except ImportError:
        pass

    for n in [500, 2000, 5000]:
        np.random.seed(42)
        p = 5
        X_np = np.random.randn(n, p)
        true_coef = np.array([0.5, -1.0, 0.3, 0.0, 0.2])
        time_np = np.random.exponential(1.0 / np.exp(X_np @ true_coef))
        event_np = np.ones(n)

        row = f"  n={n:6d}:"
        for backend in backends:
            if backend == "torch-cuda":
                X_b, y_b = to_backend_surv(X_np, time_np, event_np, "torch", device="cuda")
            else:
                X_b, y_b = to_backend_surv(X_np, time_np, event_np, backend)
            loss = CoxPartialLikelihoodLoss(ties='breslow')
            (_, n_iter), t_ms = timer(newton_solver, loss, L2Penalty(0.0), X_b, y_b, max_iter=50, tol=1e-8)
            row += f"  {backend}={t_ms:7.0f}ms({n_iter:2d}it)"
        print(row)


# ── Backend consistency ─────────────────────────────────────────────

def consistency_check():
    """Verify all backends produce same results."""
    from statgpu.losses import QuantileLoss, HuberLoss, CoxPartialLikelihoodLoss
    from statgpu.solvers import lbfgs_solver, newton_solver
    from statgpu.penalties import L2Penalty

    print("\n" + "=" * 60)
    print("Backend Consistency Check")
    print("=" * 60)

    np.random.seed(42)
    n, p = 100, 3
    X_np = np.random.randn(n, p)
    true_coef = np.array([1.0, -2.0, 0.5])
    y_np = X_np @ true_coef + np.random.randn(n) * 0.3
    coef_np = np.zeros(p)

    backends = ["numpy", "torch"]
    try:
        import cupy
        backends.append("cupy")
    except ImportError:
        pass

    for LossClass, loss_kwargs, name in [
        (QuantileLoss, {'quantile': 0.5}, 'QuantileLoss'),
        (HuberLoss, {'delta': 1.0}, 'HuberLoss'),
    ]:
        vals = {}
        grads = {}
        for backend in backends:
            X_b, y_b = to_backend(X_np, y_np, backend)
            coef_b = to_backend(X_np, np.zeros(p), backend)[1]  # zeros on device
            loss = LossClass(**loss_kwargs)
            val = loss.value(X_b, y_b, coef_b)
            grad = loss.gradient(X_b, y_b, coef_b)
            vals[backend] = float(val) if not hasattr(val, 'item') else float(val.item())
            grads[backend] = coef_to_numpy(grad)

        ref = vals["numpy"]
        print(f"\n  {name}:")
        for backend in backends:
            diff = abs(vals[backend] - ref)
            grad_diff = np.linalg.norm(grads[backend] - grads["numpy"])
            status = "OK" if diff < 1e-10 and grad_diff < 1e-10 else "MISMATCH"
            print(f"    {backend:8s}: value_diff={diff:.2e}, grad_diff={grad_diff:.2e} [{status}]")

    # CoxPH
    X_cox_np = np.random.randn(100, 3)
    true_cox = np.array([0.5, -1.0, 0.3])
    time_np = np.random.exponential(1.0 / np.exp(X_cox_np @ true_cox))
    event_np = np.ones(100)

    vals = {}
    grads = {}
    for backend in backends:
        X_b, y_b = to_backend_surv(X_cox_np, time_np, event_np, backend)
        coef_b = to_backend(X_cox_np, np.zeros(3), backend)[1]
        loss = CoxPartialLikelihoodLoss(ties='breslow')
        val = loss.value(X_b, y_b, coef_b)
        grad = loss.gradient(X_b, y_b, coef_b)
        vals[backend] = float(val) if not hasattr(val, 'item') else float(val.item())
        grads[backend] = coef_to_numpy(grad)

    ref = vals["numpy"]
    print(f"\n  CoxPH:")
    for backend in backends:
        diff = abs(vals[backend] - ref)
        grad_diff = np.linalg.norm(grads[backend] - grads["numpy"])
        status = "OK" if diff < 1e-8 and grad_diff < 1e-8 else "MISMATCH"
        print(f"    {backend:8s}: value_diff={diff:.2e}, grad_diff={grad_diff:.2e} [{status}]")


def precision_r_comparison():
    """Compare with R equivalents via rpy2."""
    print("\n" + "=" * 60)
    print("Precision: vs R (via rpy2)")
    print("=" * 60)

    try:
        import rpy2.robjects as ro
        from rpy2.robjects import numpy2ri
        numpy2ri.activate()
    except ImportError:
        print("  rpy2 not available, skipping R comparison")
        return

    np.random.seed(42)
    n, p = 200, 3
    X = np.random.randn(n, p)
    true_coef = np.array([1.0, -2.0, 0.5])
    y = X @ true_coef + np.random.randn(n) * 0.5

    ro.globalenv['X'] = X
    ro.globalenv['y'] = y

    # Quantile regression
    try:
        from statgpu.losses import QuantileLoss
        from statgpu.solvers import lbfgs_solver
        loss = QuantileLoss(quantile=0.5)
        coef_ours, _ = lbfgs_solver(loss, None, X, y, max_iter=500, tol=1e-8)
        coef_r = np.array(ro.r('library(quantreg); as.numeric(rq(y ~ X - 1, tau=0.5)$coefficients)'))
        diff = np.linalg.norm(coef_ours - coef_r)
        print(f"  QuantileLoss vs quantreg::rq: |diff|={diff:.6f}  {'OK' if diff < 0.05 else 'WARN'}")
    except Exception as e:
        print(f"  Quantile R: {e}")

    # Huber regression
    try:
        from statgpu.losses import HuberLoss
        from statgpu.solvers import lbfgs_solver
        loss = HuberLoss(delta=1.0)
        coef_ours, _ = lbfgs_solver(loss, None, X, y, max_iter=200, tol=1e-8)
        coef_r = np.array(ro.r('library(MASS); as.numeric(rlm(y ~ X - 1, method="M", psi=psi.huber)$coefficients)'))
        diff = np.linalg.norm(coef_ours - coef_r)
        print(f"  HuberLoss vs MASS::rlm:       |diff|={diff:.6f}  {'OK' if diff < 0.05 else 'WARN'}")
    except Exception as e:
        print(f"  Huber R: {e}")

    # Cox PH
    try:
        from statgpu.losses import CoxPartialLikelihoodLoss
        from statgpu.solvers import newton_solver
        from statgpu.penalties import L2Penalty
        time_to_event = np.random.exponential(1.0 / np.exp(X @ true_coef))
        event = np.ones(n)
        ro.globalenv['time'] = time_to_event
        ro.globalenv['event'] = event.astype(int)
        y_surv = {'time': time_to_event, 'event': event}
        loss = CoxPartialLikelihoodLoss(ties='breslow')
        coef_ours, _ = newton_solver(loss, L2Penalty(0.0), X, y_surv, max_iter=50, tol=1e-8)
        coef_r = np.array(ro.r('library(survival); as.numeric(coxph(Surv(time, event) ~ X)$coefficients)'))
        diff = np.linalg.norm(coef_ours - coef_r)
        print(f"  CoxPH vs survival::coxph:     |diff|={diff:.6f}  {'OK' if diff < 0.05 else 'WARN'}")
    except Exception as e:
        print(f"  Cox R: {e}")


if __name__ == '__main__':
    print("statgpu LossBase 3-Backend Benchmark")
    print("=" * 60)

    # Precision
    precision_quantile()
    precision_huber()
    precision_cox()
    precision_r_comparison()

    # Backend consistency
    consistency_check()

    # Timing
    timing_quantile()
    timing_huber()
    timing_cox()

    print("\n" + "=" * 60)
    print("Benchmark complete.")
