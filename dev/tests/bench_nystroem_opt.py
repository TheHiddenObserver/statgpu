"""Nystroem optimization benchmark."""
import numpy as np, time
from statgpu.nonparametric.kernel_methods import Nystroem

np.random.seed(42)
for n in [10000, 50000]:
    p = 20
    X = np.random.randn(n, p)
    n_comp = 100

    # numpy
    t0 = time.perf_counter()
    Nystroem(n_components=n_comp, random_state=42, device='cpu').fit_transform(X)
    t_np = (time.perf_counter() - t0) * 1000

    # cupy
    import cupy as cp
    X_c = cp.asarray(X)
    t0 = time.perf_counter()
    Nystroem(n_components=n_comp, random_state=42, device='cuda').fit_transform(X_c)
    t_cp = (time.perf_counter() - t0) * 1000
    del X_c

    # torch
    import torch
    X_t = torch.from_numpy(X).cuda().double()
    t0 = time.perf_counter()
    Nystroem(n_components=n_comp, random_state=42, device='torch').fit_transform(X_t)
    t_t = (time.perf_counter() - t0) * 1000
    del X_t

    # sklearn
    from sklearn.kernel_approximation import Nystroem as SkNy
    t0 = time.perf_counter()
    SkNy(n_components=n_comp, random_state=42).fit_transform(X)
    t_sk = (time.perf_counter() - t0) * 1000

    print(f"n={n}: numpy={t_np:.0f}ms  cupy={t_cp:.0f}ms  torch={t_t:.0f}ms  sklearn={t_sk:.0f}ms  |  cupy_speedup={t_sk/t_cp:.1f}x  torch_speedup={t_sk/t_t:.1f}x")
