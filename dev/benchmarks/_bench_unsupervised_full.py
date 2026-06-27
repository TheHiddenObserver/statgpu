"""Full unsupervised benchmark with latest code."""
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

from sklearn.cluster import DBSCAN as SkDBSCAN, KMeans as SkKMeans, AgglomerativeClustering as SkAgglo
from sklearn.decomposition import PCA as SkPCA, NMF as SkNMF, TruncatedSVD as SkTSVD, IncrementalPCA as SkIPCA
from sklearn.mixture import GaussianMixture as SkGMM

from statgpu.unsupervised import (
    PCA, KMeans, NMF, TruncatedSVD, IncrementalPCA,
    MiniBatchKMeans, MiniBatchNMF, DBSCAN, GaussianMixture,
    AgglomerativeClustering, UMAP, TSNE,
)

SCALES = {"small": (1000, 20), "medium": (10000, 50), "large": (100000, 100)}
BACKENDS = [("numpy", "cpu"), ("cupy", "cuda"), ("torch", "cuda")]
N_REPEAT = 3

def make_clustering(n, p, k=5, seed=42):
    rng = np.random.RandomState(seed)
    c = rng.randn(k, p) * 3
    l = rng.randint(0, k, n)
    return c[l] + rng.randn(n, p) * 0.5

def make_reduction(n, p, seed=42):
    return np.random.RandomState(seed).randn(n, p)

def make_nmf(n, p, seed=42):
    return np.abs(np.random.RandomState(seed).randn(n, p)) + 0.1

def bench_one(cls, kw, X, dev, n_rep=N_REPEAT):
    times = []
    for _ in range(n_rep):
        try:
            m = cls(**kw, device=dev)
            t0 = time.perf_counter()
            m.fit(X)
            if dev != "cpu":
                if dev == "cuda": cp.cuda.Stream.null.synchronize()
                else: torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        except Exception as e:
            return float('nan'), str(e)[:150]
    return float(np.median(times)), None

results = {}

# PCA
print("=== PCA ===", flush=True)
for s, (n, p) in SCALES.items():
    X = make_reduction(n, min(p, 50)); nc = 10
    sk = SkPCA(n_components=nc); t0=time.perf_counter(); sk.fit(X); sk_t=time.perf_counter()-t0
    for be, dev in BACKENDS:
        t, err = bench_one(PCA, {"n_components": nc}, X, dev)
        spd = sk_t/t if t>0 else 0
        results[f"pca_{s}_{be}"] = {"time": t, "external": sk_t, "speedup": spd}
        print(f"  {s}/{be}: {t:.4f}s spd={spd:.1f}x", flush=True)

# KMeans
print("\n=== KMeans ===", flush=True)
for s, (n, p) in SCALES.items():
    X = make_clustering(n, min(p, 50), 8)
    sk = SkKMeans(n_clusters=8, n_init=1, random_state=42); t0=time.perf_counter(); sk.fit(X); sk_t=time.perf_counter()-t0
    for be, dev in BACKENDS:
        try:
            if dev != "cpu":
                Xd = torch.tensor(X, device='cuda', dtype=torch.float64) if dev == "cuda" and be == "torch" else cp.asarray(X)
                m = KMeans(n_clusters=8, n_init=1, random_state=42, device=dev)
                t0 = time.perf_counter(); m.fit(Xd); t = time.perf_counter()-t0
            else:
                t, err = bench_one(KMeans, {"n_clusters": 8, "n_init": 1, "random_state": 42}, X, dev)
            spd = sk_t/t if t>0 else 0
            results[f"kmeans_{s}_{be}"] = {"time": t, "external": sk_t, "speedup": spd}
            print(f"  {s}/{be}: {t:.4f}s spd={spd:.1f}x", flush=True)
        except Exception as e:
            print(f"  {s}/{be}: FAIL - {e}", flush=True)

# GMM
print("\n=== GMM ===", flush=True)
for s, (n, p) in SCALES.items():
    X = make_clustering(n, min(p, 50), 5)
    sk = SkGMM(n_components=5, covariance_type='diag', random_state=42); t0=time.perf_counter(); sk.fit(X); sk_t=time.perf_counter()-t0
    for be, dev in BACKENDS:
        t, err = bench_one(GaussianMixture, {"n_components": 5, "covariance_type": "diag", "random_state": 42}, X, dev)
        spd = sk_t/t if t>0 else 0
        results[f"gmm_{s}_{be}"] = {"time": t, "external": sk_t, "speedup": spd}
        print(f"  {s}/{be}: {t:.4f}s spd={spd:.1f}x", flush=True)

# NMF
print("\n=== NMF ===", flush=True)
for s, (n, p) in SCALES.items():
    X = make_nmf(n, min(p, 50)); nc = 10
    sk = SkNMF(n_components=nc, max_iter=100, random_state=42); t0=time.perf_counter(); sk.fit(X); sk_t=time.perf_counter()-t0
    for be, dev in BACKENDS:
        t, err = bench_one(NMF, {"n_components": nc, "max_iter": 100, "random_state": 42}, X, dev)
        spd = sk_t/t if t>0 else 0
        results[f"nmf_{s}_{be}"] = {"time": t, "external": sk_t, "speedup": spd}
        print(f"  {s}/{be}: {t:.4f}s spd={spd:.1f}x", flush=True)

# TruncatedSVD
print("\n=== TruncatedSVD ===", flush=True)
for s, (n, p) in SCALES.items():
    X = make_reduction(n, min(p, 50)); nc = 10
    sk = SkTSVD(n_components=nc, random_state=42); t0=time.perf_counter(); sk.fit(X); sk_t=time.perf_counter()-t0
    for be, dev in BACKENDS:
        t, err = bench_one(TruncatedSVD, {"n_components": nc, "random_state": 42}, X, dev)
        spd = sk_t/t if t>0 else 0
        results[f"tsvd_{s}_{be}"] = {"time": t, "external": sk_t, "speedup": spd}
        status = f"spd={spd:.1f}x" if not err else f"FAIL: {err}"
        print(f"  {s}/{be}: {t:.4f}s {status}", flush=True)

# IncrementalPCA
print("\n=== IncrementalPCA ===", flush=True)
for s, (n, p) in SCALES.items():
    X = make_reduction(n, min(p, 50)); nc = 10
    sk = SkIPCA(n_components=nc); t0=time.perf_counter(); sk.fit(X); sk_t=time.perf_counter()-t0
    for be, dev in BACKENDS:
        t, err = bench_one(IncrementalPCA, {"n_components": nc}, X, dev)
        spd = sk_t/t if t>0 else 0
        results[f"ipca_{s}_{be}"] = {"time": t, "external": sk_t, "speedup": spd}
        status = f"spd={spd:.1f}x" if not err else f"FAIL: {err}"
        print(f"  {s}/{be}: {t:.4f}s {status}", flush=True)

# AgglomerativeClustering
print("\n=== AgglomerativeClustering ===", flush=True)
for s, (n, p) in [("small", (1000, 20)), ("medium", (10000, 50))]:
    X = make_clustering(n, p, 5)
    sk = SkAgglo(n_clusters=5); t0=time.perf_counter(); sk.fit(X); sk_t=time.perf_counter()-t0
    for be, dev in BACKENDS:
        t, err = bench_one(AgglomerativeClustering, {"n_clusters": 5}, X, dev)
        spd = sk_t/t if t>0 else 0
        results[f"agglo_{s}_{be}"] = {"time": t, "external": sk_t, "speedup": spd}
        status = f"spd={spd:.1f}x" if not err else f"FAIL: {err}"
        print(f"  {s}/{be}: {t:.4f}s {status}", flush=True)

# DBSCAN (10d and 50d)
print("\n=== DBSCAN ===", flush=True)
for d, eps in [(10, 1.0), (50, 3.0)]:
    for s, (n, p) in SCALES.items():
        X = make_clustering(n, d, 5)
        sk = SkDBSCAN(eps=eps, min_samples=5); t0=time.perf_counter(); sk.fit(X); sk_t=time.perf_counter()-t0
        for be, dev in BACKENDS:
            t, err = bench_one(DBSCAN, {"eps": eps, "min_samples": 5}, X, dev)
            spd = sk_t/t if t>0 else 0
            results[f"dbscan_{d}d_{s}_{be}"] = {"time": t, "external": sk_t, "speedup": spd}
            status = f"spd={spd:.1f}x" if not err else f"FAIL: {err}"
            print(f"  {d}d/{s}/{be}: {t:.4f}s {status}", flush=True)

# UMAP (small/medium only)
print("\n=== UMAP ===", flush=True)
for s, (n, p) in [("small", (1000, 20)), ("medium", (10000, 50))]:
    X = make_reduction(n, p)
    for be, dev in BACKENDS:
        t, err = bench_one(UMAP, {"n_components": 2}, X, dev)
        results[f"umap_{s}_{be}"] = {"time": t}
        status = f"{t:.4f}s" if not err else f"FAIL: {err}"
        print(f"  {s}/{be}: {status}", flush=True)

# TSNE (small only)
print("\n=== TSNE ===", flush=True)
X = make_reduction(1000, 20)
for be, dev in BACKENDS:
    t, err = bench_one(TSNE, {"n_components": 2, "max_iter": 250}, X, dev)
    results[f"tsne_small_{be}"] = {"time": t}
    status = f"{t:.4f}s" if not err else f"FAIL: {err}"
    print(f"  small/{be}: {status}", flush=True)

# MiniBatchKMeans
print("\n=== MiniBatchKMeans ===", flush=True)
for s, (n, p) in SCALES.items():
    X = make_clustering(n, min(p, 50), 8)
    for be, dev in BACKENDS:
        try:
            if dev != "cpu":
                Xd = torch.tensor(X, device='cuda', dtype=torch.float64) if be == "torch" else cp.asarray(X)
                m = MiniBatchKMeans(n_clusters=8, random_state=42, device=dev)
                t0 = time.perf_counter(); m.fit(Xd); t = time.perf_counter()-t0
            else:
                t, _ = bench_one(MiniBatchKMeans, {"n_clusters": 8, "random_state": 42}, X, dev)
            results[f"mbkmeans_{s}_{be}"] = {"time": t}
            print(f"  {s}/{be}: {t:.4f}s", flush=True)
        except Exception as e:
            print(f"  {s}/{be}: FAIL - {e}", flush=True)

# MiniBatchNMF
print("\n=== MiniBatchNMF ===", flush=True)
for s, (n, p) in SCALES.items():
    X = make_nmf(n, min(p, 50)); nc = 10
    for be, dev in BACKENDS:
        t, err = bench_one(MiniBatchNMF, {"n_components": nc, "random_state": 42}, X, dev)
        results[f"mbnmf_{s}_{be}"] = {"time": t}
        status = f"{t:.4f}s" if not err else f"FAIL: {err}"
        print(f"  {s}/{be}: {status}", flush=True)

with open("/root/statgpu/results_unsupervised_full.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

print(f"\nTotal: {len(results)} results")
print("DONE")
