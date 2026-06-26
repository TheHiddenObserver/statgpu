"""Test UMAP with different nn_method options."""
import numpy as np, time, sys
sys.path.insert(0, '/root/statgpu')

try:
    import cupy as cp
    Xw = cp.random.randn(100, 10)
    cp.linalg.svd(Xw, full_matrices=False)
    cp.cuda.Stream.null.synchronize()
    HAS_CUPY = True
except:
    HAS_CUPY = False

from statgpu.unsupervised import UMAP

# Small test (1K) - exact is fast here
X = np.random.randn(1000, 20)
print("=== Small (1K) ===", flush=True)
for method in ["exact", "nndescent", "auto"]:
    for be, dev in ([("numpy","cpu")] + ([("cupy","cuda"),("torch","torch")] if HAS_CUPY else [])):
        try:
            t0 = time.perf_counter()
            m = UMAP(n_components=2, nn_method=method, device=dev)
            m.fit(X)
            if dev != "cpu" and HAS_CUPY: cp.cuda.Stream.null.synchronize()
            print(f"  {method}/{be}: {time.perf_counter()-t0:.2f}s", flush=True)
        except Exception as e:
            print(f"  {method}/{be}: FAIL - {e}", flush=True)

# Medium test (10K) - only nndescent and auto
X = np.random.randn(10000, 50)
print("\n=== Medium (10K) ===", flush=True)
for method in ["nndescent", "auto"]:
    for be, dev in ([("numpy","cpu")] + ([("cupy","cuda"),("torch","torch")] if HAS_CUPY else [])):
        try:
            t0 = time.perf_counter()
            m = UMAP(n_components=2, nn_method=method, device=dev)
            m.fit(X)
            if dev != "cpu" and HAS_CUPY: cp.cuda.Stream.null.synchronize()
            print(f"  {method}/{be}: {time.perf_counter()-t0:.2f}s", flush=True)
        except Exception as e:
            print(f"  {method}/{be}: FAIL - {e}", flush=True)

print("\nDONE", flush=True)
