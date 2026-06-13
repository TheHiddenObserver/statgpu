"""Three-backend performance comparison — large-scale GPU advantage test."""
import time
import numpy as np


def to_backend(arr, backend):
    if backend == "cupy":
        import cupy
        return cupy.asarray(arr)
    elif backend == "torch":
        import torch
        return torch.tensor(arr, dtype=torch.float64, device="cuda")
    return arr


def to_numpy(arr):
    if hasattr(arr, "get"):
        return arr.get()
    if hasattr(arr, "cpu"):
        return arr.cpu().numpy()
    return np.asarray(arr)


print("=" * 70)
print("Three-Backend Performance Comparison (scatter-add, no _to_numpy)")
print("Focus: large k to show GPU compute advantage")
print("=" * 70)

# --- ANOVA ---
print("\n--- ANOVA (f_oneway, 5 groups) ---")
from statgpu.anova import f_oneway

for n in [1000, 10000, 100000]:
    rng = np.random.default_rng(42)
    groups_np = [rng.standard_normal(n // 5) + i * 0.5 for i in range(5)]
    print(f"  n={n:>7}:", end="")
    for backend in ["numpy", "cupy", "torch"]:
        groups = [to_backend(g, backend) for g in groups_np]
        f_oneway(*groups, backend=backend)  # warmup
        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            f_oneway(*groups, backend=backend)
            times.append(time.perf_counter() - t0)
        mean_ms = np.mean(times) * 1000
        print(f"  {backend:>6}: {mean_ms:>8.2f}ms", end="")
    print()

# --- PanelOLS entity FE — varying k ---
print("\n--- PanelOLS (entity FE, n=50000, varying k) ---")
from statgpu.panel import PanelOLS

for k in [5, 20, 100, 500]:
    n = 50000
    n_entities = 500
    rng = np.random.default_rng(42)
    X_np = rng.standard_normal((n, k))
    eids_np = np.repeat(np.arange(n_entities), n // n_entities)
    y_np = X_np @ np.ones(k) * 0.1 + rng.standard_normal(n) * 0.1
    print(f"  k={k:>4}, n={n:>6}:", end="")
    for backend in ["numpy", "cupy", "torch"]:
        X = to_backend(X_np, backend)
        y = to_backend(y_np, backend)
        eids = to_backend(eids_np, backend)
        m = PanelOLS(entity_effects=True)
        m.fit(X, y, entity_ids=eids)  # warmup
        times = []
        for _ in range(3):
            m = PanelOLS(entity_effects=True)
            t0 = time.perf_counter()
            m.fit(X, y, entity_ids=eids)
            times.append(time.perf_counter() - t0)
        mean_ms = np.mean(times) * 1000
        print(f"  {backend:>6}: {mean_ms:>8.2f}ms", end="")
    print()

# --- PanelOLS entity FE — varying n ---
print("\n--- PanelOLS (entity FE, k=50, varying n) ---")

for n in [5000, 20000, 50000, 100000]:
    k = 50
    n_entities = min(n // 10, 5000)
    rng = np.random.default_rng(42)
    X_np = rng.standard_normal((n, k))
    eids_np = np.repeat(np.arange(n_entities), n // n_entities)
    y_np = X_np @ np.ones(k) * 0.1 + rng.standard_normal(n) * 0.1
    print(f"  n={n:>7}, k={k:>3}:", end="")
    for backend in ["numpy", "cupy", "torch"]:
        X = to_backend(X_np, backend)
        y = to_backend(y_np, backend)
        eids = to_backend(eids_np, backend)
        m = PanelOLS(entity_effects=True)
        m.fit(X, y, entity_ids=eids)  # warmup
        times = []
        for _ in range(3):
            m = PanelOLS(entity_effects=True)
            t0 = time.perf_counter()
            m.fit(X, y, entity_ids=eids)
            times.append(time.perf_counter() - t0)
        mean_ms = np.mean(times) * 1000
        print(f"  {backend:>6}: {mean_ms:>8.2f}ms", end="")
    print()

# --- PanelOLS two-way FE ---
print("\n--- PanelOLS (two-way FE, k=20) ---")

for n in [5000, 20000, 50000]:
    k = 20
    n_entities = n // 10
    n_times = 10
    rng = np.random.default_rng(42)
    X_np = rng.standard_normal((n, k))
    eids_np = np.repeat(np.arange(n_entities), n_times)
    tids_np = np.tile(np.arange(n_times), n_entities)
    y_np = X_np @ np.ones(k) * 0.1 + rng.standard_normal(n) * 0.1
    print(f"  n={n:>7}, k={k:>3}:", end="")
    for backend in ["numpy", "cupy", "torch"]:
        X = to_backend(X_np, backend)
        y = to_backend(y_np, backend)
        eids = to_backend(eids_np, backend)
        tids = to_backend(tids_np, backend)
        m = PanelOLS(entity_effects=True, time_effects=True)
        m.fit(X, y, entity_ids=eids, time_ids=tids)  # warmup
        times = []
        for _ in range(3):
            m = PanelOLS(entity_effects=True, time_effects=True)
            t0 = time.perf_counter()
            m.fit(X, y, entity_ids=eids, time_ids=tids)
            times.append(time.perf_counter() - t0)
        mean_ms = np.mean(times) * 1000
        print(f"  {backend:>6}: {mean_ms:>8.2f}ms", end="")
    print()

# --- RandomEffects ---
print("\n--- RandomEffects (k=50) ---")
from statgpu.panel import RandomEffects

for n in [5000, 20000, 50000]:
    k = 50
    n_entities = n // 10
    rng = np.random.default_rng(42)
    X_np = rng.standard_normal((n, k))
    eids_np = np.repeat(np.arange(n_entities), 10)
    y_np = X_np @ np.ones(k) * 0.1 + rng.standard_normal(n) * 0.1
    print(f"  n={n:>7}, k={k:>3}:", end="")
    for backend in ["numpy", "cupy", "torch"]:
        X = to_backend(X_np, backend)
        y = to_backend(y_np, backend)
        eids = to_backend(eids_np, backend)
        m = RandomEffects()
        m.fit(X, y, entity_ids=eids)  # warmup
        times = []
        for _ in range(3):
            m = RandomEffects()
            t0 = time.perf_counter()
            m.fit(X, y, entity_ids=eids)
            times.append(time.perf_counter() - t0)
        mean_ms = np.mean(times) * 1000
        print(f"  {backend:>6}: {mean_ms:>8.2f}ms", end="")
    print()

# --- GAM ---
print("\n--- GAM (n_splines=20) ---")
from statgpu.semiparametric import GAM

for n in [1000, 10000, 50000]:
    k = 5
    rng = np.random.default_rng(42)
    X_np = rng.standard_normal((n, k))
    y_np = np.sin(X_np[:, 0]) + 0.5 * X_np[:, 1] ** 2 + rng.standard_normal(n) * 0.1
    print(f"  n={n:>7}, k={k:>3}:", end="")
    for backend in ["numpy", "cupy", "torch"]:
        X = to_backend(X_np, backend)
        y = to_backend(y_np, backend)
        gam = GAM(n_splines=20, lam=1.0)
        gam.fit(X, y)  # warmup
        times = []
        for _ in range(3):
            gam = GAM(n_splines=20, lam=1.0)
            t0 = time.perf_counter()
            gam.fit(X, y)
            times.append(time.perf_counter() - t0)
        mean_ms = np.mean(times) * 1000
        print(f"  {backend:>6}: {mean_ms:>8.2f}ms", end="")
    print()

# --- Precision ---
print("\n--- Precision (PanelOLS, n=10000, k=50) ---")
n, k = 10000, 50
n_entities = 1000
rng = np.random.default_rng(42)
X_np = rng.standard_normal((n, k))
eids_np = np.repeat(np.arange(n_entities), 10)
beta = np.ones(k) * 0.1
y_np = X_np @ beta + rng.standard_normal(n) * 0.01

m_np = PanelOLS(entity_effects=True)
m_np.fit(X_np, y_np, entity_ids=eids_np)

for backend in ["cupy", "torch"]:
    X = to_backend(X_np, backend)
    y = to_backend(y_np, backend)
    eids = to_backend(eids_np, backend)
    m = PanelOLS(entity_effects=True)
    m.fit(X, y, entity_ids=eids)
    coef_bk = to_numpy(m.coef_)
    max_diff = np.max(np.abs(m_np.coef_ - coef_bk))
    print(f"  numpy vs {backend:>6}: max |coef diff| = {max_diff:.2e}")

print("\n" + "=" * 70)
print("Done.")
