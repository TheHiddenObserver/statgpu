"""Test optimized UMAP."""
import sys
sys.path.insert(0, '/root/statgpu')
import numpy as np, time, cupy as cp, torch

# Warmup
Xw = cp.random.randn(100, 10)
cp.linalg.svd(Xw, False)
cp.cuda.Stream.null.synchronize()
Xw2 = torch.randn(100, 10, device='cuda')
torch.mm(Xw2, Xw2.t())
torch.cuda.synchronize()

from statgpu.unsupervised import UMAP

for n, label in [(1000, '1K'), (10000, '10K'), (50000, '50K')]:
    X = np.random.RandomState(42).randn(n, 20)
    print(f"\n=== {label} ({n}) ===", flush=True)
    for be, dev in [('numpy', 'cpu'), ('cupy', 'cuda'), ('torch', 'torch')]:
        try:
            m = UMAP(n_components=2, device=dev)
            t0 = time.perf_counter()
            m.fit(X)
            if dev != 'cpu': cp.cuda.Stream.null.synchronize()
            t = time.perf_counter() - t0
            print(f"  {be}: {t:.2f}s", flush=True)
        except Exception as e:
            print(f"  {be}: FAIL - {e}", flush=True)
