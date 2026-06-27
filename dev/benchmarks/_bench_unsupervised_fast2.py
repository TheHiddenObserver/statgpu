"""Fast unsupervised benchmark - skip slow UMAP numpy."""
import json, time, sys
sys.path.insert(0, '/root/statgpu')
import numpy as np
import cupy as cp
import torch

# Warmup
Xw = cp.random.randn(100, 10)
cp.linalg.svd(Xw, False)
cp.cuda.Stream.null.synchronize()
Xw2 = torch.randn(100, 10, device='cuda')
torch.mm(Xw2, Xw2.t())
torch.cuda.synchronize()

from statgpu.unsupervised import UMAP, TSNE, MiniBatchKMeans, MiniBatchNMF

results = {}

# UMAP (skip numpy medium/large - too slow)
print("=== UMAP ===", flush=True)
for s, (n, p) in [("small", (1000, 20)), ("medium", (10000, 50))]:
    X = np.random.RandomState(42).randn(n, p)
    for be, dev in [("numpy", "cpu"), ("cupy", "cuda"), ("torch", "cuda")]:
        if be == "numpy" and s == "medium":
            print(f"  {s}/{be}: SKIP (too slow)", flush=True)
            continue
        try:
            m = UMAP(n_components=2, device=dev)
            t0 = time.perf_counter()
            m.fit(X)
            if dev != "cpu": cp.cuda.Stream.null.synchronize()
            t = time.perf_counter() - t0
            results[f"umap_{s}_{be}"] = {"time": t}
            print(f"  {s}/{be}: {t:.4f}s", flush=True)
        except Exception as e:
            print(f"  {s}/{be}: FAIL - {e}", flush=True)

# TSNE (small only)
print("\n=== TSNE ===", flush=True)
X = np.random.RandomState(42).randn(1000, 20)
for be, dev in [("numpy", "cpu"), ("cupy", "cuda"), ("torch", "cuda")]:
    try:
        m = TSNE(n_components=2, max_iter=250, device=dev)
        t0 = time.perf_counter()
        m.fit(X)
        if dev != "cpu": cp.cuda.Stream.null.synchronize()
        t = time.perf_counter() - t0
        results[f"tsne_small_{be}"] = {"time": t}
        print(f"  small/{be}: {t:.4f}s", flush=True)
    except Exception as e:
        print(f"  small/{be}: FAIL - {e}", flush=True)

# MiniBatchKMeans
print("\n=== MiniBatchKMeans ===", flush=True)
SCALES = {"small": (1000, 20), "medium": (10000, 50), "large": (100000, 100)}
for s, (n, p) in SCALES.items():
    rng = np.random.RandomState(42)
    X = rng.randn(n, min(p, 50))
    for be, dev in [("numpy", "cpu"), ("cupy", "cuda"), ("torch", "cuda")]:
        try:
            if dev != "cpu":
                Xd = torch.tensor(X, device='cuda', dtype=torch.float64) if be == "torch" else cp.asarray(X)
                m = MiniBatchKMeans(n_clusters=8, random_state=42, device=dev)
                t0 = time.perf_counter()
                m.fit(Xd)
                if dev == "cuda": cp.cuda.Stream.null.synchronize()
                else: torch.cuda.synchronize()
                t = time.perf_counter() - t0
            else:
                m = MiniBatchKMeans(n_clusters=8, random_state=42, device=dev)
                t0 = time.perf_counter()
                m.fit(X)
                t = time.perf_counter() - t0
            results[f"mbkmeans_{s}_{be}"] = {"time": t}
            print(f"  {s}/{be}: {t:.4f}s", flush=True)
        except Exception as e:
            print(f"  {s}/{be}: FAIL - {e}", flush=True)

# MiniBatchNMF
print("\n=== MiniBatchNMF ===", flush=True)
for s, (n, p) in SCALES.items():
    rng = np.random.RandomState(42)
    X = np.abs(rng.randn(n, min(p, 50))) + 0.1
    nc = 10
    for be, dev in [("numpy", "cpu"), ("cupy", "cuda"), ("torch", "cuda")]:
        try:
            m = MiniBatchNMF(n_components=nc, random_state=42, device=dev)
            t0 = time.perf_counter()
            m.fit(X)
            if dev != "cpu": cp.cuda.Stream.null.synchronize()
            t = time.perf_counter() - t0
            results[f"mbnmf_{s}_{be}"] = {"time": t}
            print(f"  {s}/{be}: {t:.4f}s", flush=True)
        except Exception as e:
            print(f"  {s}/{be}: FAIL - {e}", flush=True)

with open("/root/statgpu/results_unsupervised_extra.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

print(f"\nTotal: {len(results)} results")
print("DONE")
