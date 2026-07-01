"""CoxPH Efron comprehensive benchmark — save results to JSON.

Runs accuracy + speed tests across light-ties and heavy-ties,
with and without Numba, and saves structured JSON for frontend use.
"""
import json
import os
import sys
import time

import numpy as np

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


def time_coxph(device, ties, X, time_obs, event, repeat=3):
    if device == "cpu":
        sg = CoxPH(device="cpu", ties=ties, max_iter=3, compute_inference=False)
        sg.fit(X, time_obs, event)
        ts = []
        for _ in range(repeat):
            sg = CoxPH(device="cpu", ties=ties, max_iter=80, tol=1e-9, compute_inference=False)
            t0 = time.perf_counter()
            sg.fit(X, time_obs, event)
            ts.append(time.perf_counter() - t0)
        return float(np.median(ts) * 1000)

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
        return float(np.median(ts) * 1000)

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
        return float(np.median(ts) * 1000)


def time_sm(X, time_obs, event, repeat=3):
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        smd.PHReg(time_obs, X, status=event, ties="efron").fit(disp=0)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts) * 1000)


def get_coef(device, ties, X, time_obs, event):
    if device == "cpu":
        sg = CoxPH(device="cpu", ties=ties, max_iter=80, tol=1e-9, compute_inference=False)
        sg.fit(X, time_obs, event)
        return sg.coef_.tolist()
    elif device == "torch_gpu":
        import torch
        X_g = torch.tensor(X, dtype=torch.float64, device="cuda")
        sg = CoxPH(device="cuda", ties=ties, max_iter=80, tol=1e-9, compute_inference=False)
        sg.fit(X_g, time_obs, event)
        c = sg.coef_
        return c.cpu().numpy().tolist() if hasattr(c, "cpu") else c.tolist()
    elif device == "cupy_gpu":
        import cupy as cp
        X_c = cp.asarray(X)
        sg = CoxPH(device="cuda", ties=ties, max_iter=80, tol=1e-9, compute_inference=False)
        sg.fit(X_c, time_obs, event)
        c = sg.coef_
        return cp.asnumpy(c).tolist() if hasattr(c, "get") else c.tolist()


configs = [
    (1000,  10, 311),
    (2000,  20, 312),
    (5000,  20, 313),
    (10000, 50, 314),
    (20000, 50, 315),
]

# Detect available backends
backends = ["cpu"]
gpu_name = "N/A"
try:
    import torch
    if torch.cuda.is_available():
        backends.append("torch_gpu")
        gpu_name = torch.cuda.get_device_name(0)
except ImportError:
    pass
try:
    import cupy
    backends.append("cupy_gpu")
except ImportError:
    pass

from statgpu.survival._cox import _HAS_NUMBA_EFRON
numba_enabled = _HAS_NUMBA_EFRON

print("=" * 80)
print("CoxPH Efron Benchmark — saving results")
print("GPU: %s | Numba: %s" % (gpu_name, numba_enabled))
print("=" * 80)

results = {
    "meta": {
        "gpu": gpu_name,
        "numba": numba_enabled,
        "backends": backends,
        "configs": configs,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    },
    "accuracy": [],
    "light_ties": [],
    "heavy_ties": [],
}

# ── Accuracy ──
print("\n[1/3] Accuracy...")
for n, p, seed in configs:
    X, time_obs, event = make_data(n, p, seed)
    sm_res = smd.PHReg(time_obs, X, status=event, ties="efron").fit(disp=0)
    sm_coef = sm_res.params.tolist()
    row = {"n": n, "p": p, "events": int(event.sum()), "sm_coef": sm_coef, "diffs": {}}
    for dev in backends:
        coef = get_coef(dev, "efron", X, time_obs, event)
        diff = max(abs(c - s) for c, s in zip(coef, sm_coef))
        row["diffs"][dev] = diff
        print("  n=%d p=%d %s: %.2e" % (n, p, dev, diff))
    results["accuracy"].append(row)

# ── Light-ties Speed ──
print("\n[2/3] Light-ties speed...")
for n, p, seed in configs:
    X, time_obs, event = make_data(n, p, seed)
    row = {"n": n, "p": p, "events": int(event.sum()), "times": {}}
    for dev in backends:
        t = time_coxph(dev, "efron", X, time_obs, event)
        row["times"][dev] = t
        print("  n=%d p=%d %s: %.1f ms" % (n, p, dev, t))
    row["times"]["statsmodels"] = time_sm(X, time_obs, event)
    results["light_ties"].append(row)

# ── Heavy-ties Speed ──
print("\n[3/3] Heavy-ties speed...")
for n, p, seed in configs:
    X, time_obs, event = make_data(n, p, seed, n_bins=120)
    n_uft = int(len(np.unique(time_obs[event == 1])))
    row = {"n": n, "p": p, "events": int(event.sum()), "uft": n_uft, "times": {}}
    for dev in backends:
        t = time_coxph(dev, "efron", X, time_obs, event)
        row["times"][dev] = t
        print("  n=%d p=%d uft=%d %s: %.1f ms" % (n, p, n_uft, dev, t))
    row["times"]["statsmodels"] = time_sm(X, time_obs, event)
    results["heavy_ties"].append(row)

# ── Save ──
out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "results")
os.makedirs(out_dir, exist_ok=True)
suffix = "numba" if numba_enabled else "default"
out_path = os.path.join(out_dir, "coxph_efron_bench_%s.json" % suffix)
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved: %s" % out_path)
