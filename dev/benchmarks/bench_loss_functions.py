"""Comprehensive benchmark for Quantile, Huber, CoxPH(Breslow/Efron).

Saves structured JSON results for each loss function.
Precision is checked via gradient norm at solution (self-consistency).
"""
import json
import os
import sys
import time

import numpy as np


def make_regression_data(n, p, seed, noise="normal"):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta = rng.normal(scale=0.5, size=p)
    if noise == "normal":
        y = X @ beta + rng.normal(scale=1.0, size=n)
    elif noise == "heavy_tail":
        y = X @ beta + rng.standard_t(df=3, size=n)
    else:
        y = X @ beta + rng.normal(scale=1.0, size=n)
    return X, y, beta


def make_survival_data(n, p, seed, n_bins=None):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta = rng.normal(scale=0.3, size=p)
    lin = X @ beta
    u = np.clip(rng.random(n), 1e-12, 1 - 1e-12)
    t_true = -np.log(u) / (0.03 * np.exp(np.clip(lin, -20, 20)))
    censor = rng.exponential(scale=np.median(t_true), size=n)
    event = (t_true <= censor).astype(int)
    time_obs = np.minimum(t_true, censor)
    if n_bins is not None:
        edges = np.linspace(time_obs.min(), time_obs.max() + 1e-10, n_bins + 1)
        time_obs = np.digitize(time_obs, edges).astype(np.float64)
    return X, time_obs, event


def time_fn(fn, repeat=3):
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts) * 1000)


def check_convergence(loss, X, y, coef, ref_coef=None):
    """Check convergence.

    For smooth losses (Huber): gradient norm at solution should be ~0.
    For non-smooth losses (Quantile): objective value should be <= reference.
    """
    val, grad = loss.fused_value_and_gradient(X, y, coef)
    grad_norm = float(np.linalg.norm(grad))
    result = {"grad_norm": grad_norm, "objective": float(val)}
    if ref_coef is not None:
        ref_val = loss.value(X, y, ref_coef)
        result["ref_objective"] = float(ref_val)
        result["objective_diff"] = float(val - ref_val)
        result["objective_better_or_equal"] = bool(val <= ref_val + 1e-10)
    return result


def bench_quantile(configs):
    """Benchmark QuantileLoss via IRLS solver."""
    from statgpu.losses import get_loss

    results = []
    for n, p, seed in configs:
        X, y, beta_true = make_regression_data(n, p, seed)
        row = {"n": n, "p": p, "seed": seed, "times": {}, "convergence": {}}

        loss = get_loss("quantile", quantile=0.5)

        # sklearn reference (run first to get reference coef for objective comparison)
        coef_ref = None
        try:
            from sklearn.linear_model import QuantileRegressor as SkQR
            def fit_sk():
                qr = SkQR(quantile=0.5, solver="highs", alpha=0)
                qr.fit(X, y)
                return qr
            qr_sk = fit_sk()
            coef_ref = qr_sk.coef_
            row["times"]["sklearn"] = time_fn(lambda: fit_sk())
        except Exception:
            pass

        # statgpu CPU via IRLS
        def fit_cpu():
            coef, niter = loss.irls(X, y, max_iter=500, tol=1e-12)
            return coef, niter
        coef_cpu, niter = fit_cpu()
        row["times"]["cpu"] = time_fn(lambda: loss.irls(X, y, max_iter=500, tol=1e-12)[0])
        row["convergence"]["cpu"] = check_convergence(loss, X, y, coef_cpu, ref_coef=coef_ref)
        row["convergence"]["cpu"]["iterations"] = niter

        results.append(row)
        print("  quantile n=%d p=%d: cpu=%.1fms grad_norm=%.1e" % (
            n, p, row["times"]["cpu"], row["convergence"]["cpu"]["grad_norm"]))
    return results


def bench_huber(configs):
    """Benchmark HuberLoss via Newton solver."""
    from statgpu.losses import get_loss
    from statgpu.solvers import newton_solver
    from statgpu.penalties import L2Penalty

    results = []
    for n, p, seed in configs:
        X, y, beta_true = make_regression_data(n, p, seed, noise="heavy_tail")
        row = {"n": n, "p": p, "seed": seed, "times": {}, "convergence": {}}

        loss = get_loss("huber", delta=1.0)
        penalty = L2Penalty(0.0)

        # statgpu CPU via Newton
        def fit_cpu():
            coef, niter = newton_solver(loss, penalty, X, y, max_iter=1000, tol=1e-12)
            return coef, niter
        coef_cpu, niter = fit_cpu()
        row["times"]["cpu"] = time_fn(lambda: newton_solver(loss, penalty, X, y, max_iter=1000, tol=1e-12)[0])
        row["convergence"]["cpu"] = check_convergence(loss, X, y, coef_cpu)
        row["convergence"]["cpu"]["iterations"] = niter

        # sklearn reference (note: sklearn uses different delta/scale parameterization)
        try:
            from sklearn.linear_model import HuberRegressor as SkHR
            def fit_sk():
                hr = SkHR(max_iter=1000)
                hr.fit(X, y)
                return hr
            hr_sk = fit_sk()
            row["times"]["sklearn"] = time_fn(lambda: fit_sk())
        except Exception:
            pass

        results.append(row)
        print("  huber n=%d p=%d: cpu=%.1fms grad_norm=%.1e iters=%d" % (
            n, p, row["times"]["cpu"], row["convergence"]["cpu"]["grad_norm"], niter))
    return results


def bench_coxph(ties, configs):
    """Benchmark CoxPH with given tie method."""
    from statgpu.survival._cox import CoxPH
    import statsmodels.duration.hazard_regression as smd

    results = []
    for n, p, seed in configs:
        X, time_obs, event = make_survival_data(n, p, seed)
        n_events = int(event.sum())
        n_uft = int(len(np.unique(time_obs[event == 1])))
        row = {
            "n": n, "p": p, "seed": seed,
            "events": n_events, "uft": n_uft,
            "times": {}, "coef_diff": {},
        }

        # statgpu CPU
        sg = CoxPH(device="cpu", ties=ties, max_iter=3, compute_inference=False)
        sg.fit(X, time_obs, event)
        row["times"]["cpu"] = time_fn(
            lambda: CoxPH(device="cpu", ties=ties, max_iter=80, tol=1e-9, compute_inference=False).fit(X, time_obs, event)
        )

        # statgpu Torch GPU
        try:
            import torch
            X_g = torch.tensor(X, dtype=torch.float64, device="cuda")
            sg = CoxPH(device="cuda", ties=ties, max_iter=3, compute_inference=False)
            sg.fit(X_g, time_obs, event)
            torch.cuda.synchronize()
            def fit_torch():
                s = CoxPH(device="cuda", ties=ties, max_iter=80, tol=1e-9, compute_inference=False)
                torch.cuda.synchronize()
                s.fit(X_g, time_obs, event)
                torch.cuda.synchronize()
            row["times"]["torch_gpu"] = time_fn(fit_torch)
        except Exception:
            pass

        # statgpu CuPy GPU
        try:
            import cupy as cp
            X_c = cp.asarray(X)
            sg = CoxPH(device="cuda", ties=ties, max_iter=3, compute_inference=False)
            sg.fit(X_c, time_obs, event)
            cp.cuda.Stream.null.synchronize()
            def fit_cupy():
                s = CoxPH(device="cuda", ties=ties, max_iter=80, tol=1e-9, compute_inference=False)
                cp.cuda.Stream.null.synchronize()
                s.fit(X_c, time_obs, event)
                cp.cuda.Stream.null.synchronize()
            row["times"]["cupy_gpu"] = time_fn(fit_cupy)
        except Exception:
            pass

        # statsmodels reference
        sm_res = smd.PHReg(time_obs, X, status=event, ties=ties).fit(disp=0)
        row["times"]["statsmodels"] = time_fn(
            lambda: smd.PHReg(time_obs, X, status=event, ties=ties).fit(disp=0)
        )

        # Coef diffs vs statsmodels
        sg = CoxPH(device="cpu", ties=ties, max_iter=80, tol=1e-9, compute_inference=False)
        sg.fit(X, time_obs, event)
        row["coef_diff"]["cpu"] = float(np.max(np.abs(sg.coef_ - sm_res.params)))
        try:
            import cupy as cp
            sg = CoxPH(device="cuda", ties=ties, max_iter=80, tol=1e-9, compute_inference=False)
            sg.fit(cp.asarray(X), time_obs, event)
            c = sg.coef_
            row["coef_diff"]["cupy_gpu"] = float(np.max(np.abs(cp.asnumpy(c) - sm_res.params)))
        except Exception:
            pass

        results.append(row)
        best_gpu = min((row["times"].get(k, 1e9) for k in ["torch_gpu", "cupy_gpu"]), default=0)
        sm_t = row["times"]["statsmodels"]
        print("  coxph(%s) n=%d p=%d uft=%d: cpu=%.1fms gpu=%.1fms sm=%.1fms" % (
            ties, n, p, n_uft, row["times"]["cpu"], best_gpu, sm_t))
    return results


configs_small = [(500, 5, 310), (1000, 10, 311), (2000, 20, 312)]
configs_coxph = [(1000, 10, 311), (2000, 20, 312), (5000, 20, 313), (10000, 50, 314), (20000, 50, 315)]

backends = ["cpu"]
try:
    import torch
    if torch.cuda.is_available():
        backends.append("torch_gpu")
        GPU_NAME = torch.cuda.get_device_name(0)
except Exception:
    GPU_NAME = "N/A"
try:
    import cupy
    backends.append("cupy_gpu")
except Exception:
    pass

print("=" * 80)
print("Loss Function Benchmark Suite")
print("GPU: %s" % GPU_NAME)
print("=" * 80)

all_results = {
    "date": time.strftime("%Y-%m-%d"),
    "environment": {"gpu": GPU_NAME, "backends": backends},
}

print("\n[1/4] Quantile Loss (tau=0.5)")
all_results["quantile"] = {
    "loss": "quantile", "quantile": 0.5,
    "configs": configs_small,
    "results": bench_quantile(configs_small),
}

print("\n[2/4] Huber Loss (delta=1.0)")
all_results["huber"] = {
    "loss": "huber", "delta": 1.0,
    "configs": configs_small,
    "results": bench_huber(configs_small),
}

print("\n[3/4] CoxPH Breslow")
all_results["coxph_breslow"] = {
    "model": "CoxPH", "ties": "breslow",
    "configs": configs_coxph,
    "results": bench_coxph("breslow", configs_coxph),
}

print("\n[4/4] CoxPH Efron")
all_results["coxph_efron"] = {
    "model": "CoxPH", "ties": "efron",
    "configs": configs_coxph,
    "results": bench_coxph("efron", configs_coxph),
}

# Save
out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "results")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "loss_functions_bench_%s.json" % time.strftime("%Y-%m-%d"))
with open(out_path, "w") as f:
    json.dump(all_results, f, indent=2)
print("\nSaved: %s" % out_path)
