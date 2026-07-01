"""CoxPH Efron final comprehensive benchmark.

Tests accuracy and speed across:
- Light-ties (continuous time) and heavy-ties (120 bins)
- CPU, Torch GPU, CuPy GPU, statsmodels
- Multiple problem sizes
"""
import numpy as np
import time

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

import statsmodels.duration.hazard_regression as smd
from statgpu.survival._cox import CoxPH


def make_data(n, p, seed, n_bins=None):
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


def time_it(device, ties, X, time_obs, event, repeat=3):
    """Time CoxPH.fit() on given device, median of repeat runs."""
    if device == "cpu":
        sg = CoxPH(device="cpu", ties=ties, max_iter=3, compute_inference=False)
        sg.fit(X, time_obs, event)
        ts = []
        for _ in range(repeat):
            sg = CoxPH(device="cpu", ties=ties, max_iter=80, tol=1e-9, compute_inference=False)
            t0 = time.perf_counter()
            sg.fit(X, time_obs, event)
            ts.append(time.perf_counter() - t0)
        return np.median(ts) * 1000

    elif device == "torch_gpu":
        import torch
        X_g = torch.tensor(X, dtype=torch.float64, device="cuda")
        sg = CoxPH(device="cuda", ties=ties, max_iter=3, compute_inference=False)
        sg.fit(X_g, time_obs, event)
        torch.cuda.synchronize()
        ts = []
        for _ in range(repeat):
            sg = CoxPH(device="cuda", ties=ties, max_iter=80, tol=1e-9, compute_inference=False)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            sg.fit(X_g, time_obs, event)
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
        return np.median(ts) * 1000

    elif device == "cupy_gpu":
        import cupy as cp
        X_c = cp.asarray(X)
        sg = CoxPH(device="cuda", ties=ties, max_iter=3, compute_inference=False)
        sg.fit(X_c, time_obs, event)
        cp.cuda.Stream.null.synchronize()
        ts = []
        for _ in range(repeat):
            sg = CoxPH(device="cuda", ties=ties, max_iter=80, tol=1e-9, compute_inference=False)
            cp.cuda.Stream.null.synchronize()
            t0 = time.perf_counter()
            sg.fit(X_c, time_obs, event)
            cp.cuda.Stream.null.synchronize()
            ts.append(time.perf_counter() - t0)
        return np.median(ts) * 1000


def time_sm(X, time_obs, event, repeat=3):
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        smd.PHReg(time_obs, X, status=event, ties="efron").fit(disp=0)
        ts.append(time.perf_counter() - t0)
    return np.median(ts) * 1000


configs = [
    (1000,  10, 311),
    (2000,  20, 312),
    (5000,  20, 313),
    (10000, 50, 314),
    (20000, 50, 315),
]

# ── Header ──
print("=" * 120)
print("CoxPH Efron Final Benchmark — statgpu vs statsmodels")
print("GPU: %s" % GPU_NAME)
print("=" * 120)

# ── Accuracy ──
print()
print("[1] ACCURACY — max|coef(statgpu) - coef(statsmodels)|")
print("-" * 70)
print("%6s %4s %7s | %14s %14s" % ("n", "p", "events", "Torch_GPU", "CuPy_GPU"))
print("-" * 70)
for n, p, seed in configs:
    X, time_obs, event = make_data(n, p, seed)
    sm_res = smd.PHReg(time_obs, X, status=event, ties="efron").fit(disp=0)
    diffs = {}
    if "torch_gpu" in backends:
        import torch
        X_g = torch.tensor(X, dtype=torch.float64, device="cuda")
        sg = CoxPH(device="cuda", ties="efron", max_iter=80, tol=1e-9, compute_inference=False)
        sg.fit(X_g, time_obs, event)
        tc = sg.coef_
        if hasattr(tc, "cpu"):
            tc = tc.cpu().numpy()
        diffs["torch"] = np.max(np.abs(tc - sm_res.params))
    if "cupy_gpu" in backends:
        import cupy as cp
        X_c = cp.asarray(X)
        sg = CoxPH(device="cuda", ties="efron", max_iter=80, tol=1e-9, compute_inference=False)
        sg.fit(X_c, time_obs, event)
        cc = sg.coef_
        if hasattr(cc, "get"):
            cc = cc.get()
        diffs["cupy"] = np.max(np.abs(cc - sm_res.params))
    print("%6d %4d %7d | %14.2e %14.2e" % (
        n, p, event.sum(),
        diffs.get("torch", float("nan")),
        diffs.get("cupy", float("nan")),
    ))

# ── Light-ties Speed ──
print()
print("[2] SPEED — Light-ties (continuous time, avg_tie ~ 1)")
print("-" * 120)
gpu_cols = [b for b in backends if b != "cpu"]
header = "%6s %4s %7s | %10s" % ("n", "p", "events", "CPU")
for b in gpu_cols:
    header += " %10s" % b
header += " %10s |" % "statsmodels"
for b in gpu_cols:
    header += " %s/SM" % b
header += " CPU/SM"
print(header)
print("-" * 120)

for n, p, seed in configs:
    X, time_obs, event = make_data(n, p, seed)
    t_cpu = time_it("cpu", "efron", X, time_obs, event)
    t_gpu = {}
    for b in gpu_cols:
        t_gpu[b] = time_it(b, "efron", X, time_obs, event)
    t_sm = time_sm(X, time_obs, event)

    row = "%6d %4d %7d | %10.1f" % (n, p, event.sum(), t_cpu)
    for b in gpu_cols:
        row += " %10.1f" % t_gpu[b]
    row += " %10.1f |" % t_sm
    for b in gpu_cols:
        row += " %5.2fx" % (t_gpu[b] / t_sm)
    row += " %5.2fx" % (t_cpu / t_sm)
    print(row)

# ── Heavy-ties Speed ──
print()
print("[3] SPEED — Heavy-ties (120 bins, avg_tie ~ 70-175)")
print("-" * 120)
header = "%6s %4s %7s %7s | %10s" % ("n", "p", "events", "uft", "CPU")
for b in gpu_cols:
    header += " %10s" % b
header += " %10s |" % "statsmodels"
for b in gpu_cols:
    header += " %s/SM" % b
header += " CPU/SM"
print(header)
print("-" * 120)

for n, p, seed in configs:
    X, time_obs, event = make_data(n, p, seed, n_bins=120)
    n_uft = len(np.unique(time_obs[event == 1]))
    t_cpu = time_it("cpu", "efron", X, time_obs, event)
    t_gpu = {}
    for b in gpu_cols:
        t_gpu[b] = time_it(b, "efron", X, time_obs, event)
    t_sm = time_sm(X, time_obs, event)

    row = "%6d %4d %7d %7d | %10.1f" % (n, p, event.sum(), n_uft, t_cpu)
    for b in gpu_cols:
        row += " %10.1f" % t_gpu[b]
    row += " %10.1f |" % t_sm
    for b in gpu_cols:
        row += " %5.2fx" % (t_gpu[b] / t_sm)
    row += " %5.2fx" % (t_cpu / t_sm)
    print(row)

print("-" * 120)
print("ratio = time / statsmodels.  < 1.0 = faster than statsmodels.")
