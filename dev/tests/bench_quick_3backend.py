"""Quick 3-backend comparison at large scale."""
import time, json, os, sys
import numpy as np

def timeit(func, n=3):
    times = []
    result = None
    for _ in range(n):
        t0 = time.perf_counter()
        result = func()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return result, np.median(times)

def _to_np(x):
    if hasattr(x, 'get'): return x.get()
    if hasattr(x, 'cpu'): return x.detach().cpu().numpy()
    return np.asarray(x)

results = {}

# ========== ANOVA ==========
print("[ANOVA]")
from statgpu.anova import f_oneway
for n in [10000, 50000, 100000]:
    np.random.seed(42)
    g1 = np.random.randn(n)
    g2 = np.random.randn(n) + 1
    g3 = np.random.randn(n) + 2

    r_np, t_np = timeit(lambda: f_oneway(g1, g2, g3, backend="numpy"))

    import cupy as cp
    g1c, g2c, g3c = cp.asarray(g1), cp.asarray(g2), cp.asarray(g3)
    r_cp, t_cp = timeit(lambda: f_oneway(g1c, g2c, g3c, backend="cupy"))

    import torch
    g1t = torch.from_numpy(g1).cuda().double()
    g2t = torch.from_numpy(g2).cuda().double()
    g3t = torch.from_numpy(g3).cuda().double()
    r_t, t_t = timeit(lambda: f_oneway(g1t, g2t, g3t, backend="torch"))

    from scipy.stats import f_oneway as sp_f
    r_sp, t_sp = timeit(lambda: sp_f(g1, g2, g3))

    print(f"  n={n:>6d}: numpy={t_np:7.1f}ms  cupy={t_cp:7.1f}ms  torch={t_t:7.1f}ms  scipy={t_sp:7.1f}ms  | cupy_speedup={t_sp/t_cp:.1f}x  torch_speedup={t_sp/t_t:.1f}x")
    results[f"anova_n={n}"] = {"numpy": t_np, "cupy": t_cp, "torch": t_t, "scipy": t_sp}

    del g1c, g2c, g3c, g1t, g2t, g3t

# ========== Covariance (LedoitWolf) ==========
print("\n[Covariance - LedoitWolf]")
from statgpu.covariance import LedoitWolf
from sklearn.covariance import LedoitWolf as SkLW
for n in [10000, 50000, 100000]:
    p = 50
    np.random.seed(42)
    X = np.random.randn(n, p)

    lw_np, t_np = timeit(lambda: LedoitWolf(device='cpu').fit(X))

    X_c = cp.asarray(X)
    lw_cp, t_cp = timeit(lambda: LedoitWolf(device='cuda').fit(X_c))

    X_t = torch.from_numpy(X).cuda().double()
    lw_t, t_t = timeit(lambda: LedoitWolf(device='torch').fit(X_t))

    sk, t_sk = timeit(lambda: SkLW().fit(X))

    print(f"  n={n:>6d},p={p}: numpy={t_np:7.1f}ms  cupy={t_cp:7.1f}ms  torch={t_t:7.1f}ms  sklearn={t_sk:7.1f}ms  | cupy_speedup={t_sk/t_cp:.1f}x  torch_speedup={t_sk/t_t:.1f}x")
    results[f"lw_n={n}"] = {"numpy": t_np, "cupy": t_cp, "torch": t_t, "sklearn": t_sk}

    del X_c, X_t

# ========== Nystroem ==========
print("\n[Nystroem]")
from statgpu.nonparametric.kernel_methods import Nystroem, rbf_kernel
from sklearn.kernel_approximation import Nystroem as SkNy
for n in [10000, 50000, 100000]:
    p = 20
    np.random.seed(42)
    X = np.random.randn(n, p)
    n_comp = 100

    ny_np, t_np = timeit(lambda: Nystroem(n_components=n_comp, random_state=42, device='cpu').fit_transform(X))

    X_c = cp.asarray(X)
    ny_cp, t_cp = timeit(lambda: Nystroem(n_components=n_comp, random_state=42, device='cuda').fit_transform(X_c))

    X_t = torch.from_numpy(X).cuda().double()
    ny_t, t_t = timeit(lambda: Nystroem(n_components=n_comp, random_state=42, device='torch').fit_transform(X_t))

    sk, t_sk = timeit(lambda: SkNy(n_components=n_comp, random_state=42).fit_transform(X))

    print(f"  n={n:>6d},p={p}: numpy={t_np:7.1f}ms  cupy={t_cp:7.1f}ms  torch={t_t:7.1f}ms  sklearn={t_sk:7.1f}ms  | numpy_speedup={t_sk/t_np:.1f}x  cupy_speedup={t_sk/t_cp:.1f}x")
    results[f"nystroem_n={n}"] = {"numpy": t_np, "cupy": t_cp, "torch": t_t, "sklearn": t_sk}

    del X_c, X_t

# ========== RBF Kernel ==========
print("\n[RBF Kernel Matrix]")
for n in [10000, 20000]:
    p = 20
    np.random.seed(42)
    X = np.random.randn(n, p)

    _, t_np = timeit(lambda: rbf_kernel(X, xp=np))

    X_c = cp.asarray(X)
    _, t_cp = timeit(lambda: rbf_kernel(X_c, xp=cp))
    del X_c

    X_t = torch.from_numpy(X).cuda().double()
    _, t_t = timeit(lambda: rbf_kernel(X_t, xp=torch))
    del X_t

    from sklearn.metrics.pairwise import rbf_kernel as sk_rbf
    _, t_sk = timeit(lambda: sk_rbf(X))

    print(f"  n={n:>6d},p={p}: numpy={t_np:7.1f}ms  cupy={t_cp:7.1f}ms  torch={t_t:7.1f}ms  sklearn={t_sk:7.1f}ms  | cupy_speedup={t_sk/t_cp:.1f}x  torch_speedup={t_sk/t_t:.1f}x")
    results[f"rbf_n={n}"] = {"numpy": t_np, "cupy": t_cp, "torch": t_t, "sklearn": t_sk}

# ========== SplineTransformer ==========
print("\n[SplineTransformer]")
from statgpu.nonparametric.splines import SplineTransformer
for n in [10000, 50000, 100000]:
    p = 5
    np.random.seed(42)
    X = np.random.randn(n, p)

    _, t_np = timeit(lambda: SplineTransformer(n_knots=10, device='cpu').fit_transform(X))

    X_c = cp.asarray(X)
    _, t_cp = timeit(lambda: SplineTransformer(n_knots=10, device='cuda').fit_transform(X_c))
    del X_c

    X_t = torch.from_numpy(X).cuda().double()
    _, t_t = timeit(lambda: SplineTransformer(n_knots=10, device='torch').fit_transform(X_t))
    del X_t

    from sklearn.preprocessing import SplineTransformer as SkST
    _, t_sk = timeit(lambda: SkST(n_knots=10).fit_transform(X))

    print(f"  n={n:>6d},p={p}: numpy={t_np:7.1f}ms  cupy={t_cp:7.1f}ms  torch={t_t:7.1f}ms  sklearn={t_sk:7.1f}ms  | cupy_speedup={t_sk/t_cp:.1f}x  torch_speedup={t_sk/t_t:.1f}x")
    results[f"spline_n={n}"] = {"numpy": t_np, "cupy": t_cp, "torch": t_t, "sklearn": t_sk}

# ========== Save ==========
os.makedirs("results", exist_ok=True)
with open("results/p2_benchmark_3backend.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nResults saved to results/p2_benchmark_3backend.json")
