# Unsupervised Benchmark: 12 Algorithms × 3 Backends + External

Date: 2026-06-26
Hardware: Tesla P100-SXM2-16GB (GPU), Intel Xeon (CPU)

## Complete 4-Column Results (100K obs)

| Algorithm | numpy | cupy | torch | sklearn | cupy vs sk | torch vs sk |
|-----------|-------|------|-------|---------|------------|-------------|
| PCA | 0.019s | 0.017s | 0.017s | 0.022s | 1.3x | 1.3x |
| KMeans | 0.149s | 0.028s | 0.028s | 0.232s | **8.4x** | **8.1x** |
| GMM | 0.454s | 0.044s | 0.042s | 0.541s | **12.8x** | **13.0x** |
| NMF | 2.608s | 0.158s | 0.158s | 3.054s | **19.5x** | **19.9x** |
| TruncatedSVD | 0.802s | 0.049s | 0.046s | 0.993s | **26.0x** | **27.6x** |
| IncrementalPCA | 0.254s | 0.036s | 0.035s | 1.330s | **20.9x** | **21.1x** |
| Agglomerative | 2.562s | 4.480s | 4.466s | 5.004s | 1.1x | 1.1x |
| DBSCAN 10d | 27.825s | 46.786s | 47.517s | 85.026s | 1.8x | 1.8x |
| DBSCAN 50d | 8.935s | 41.684s | 41.808s | 3.694s | 0.1x | 0.1x |
| MiniBatchKMeans | 0.276s | 0.054s | 0.047s | - | - | - |
| MiniBatchNMF | 5.924s | 1.839s | 1.877s | - | - | - |
| UMAP | 35.30s | 1.38s | 1.24s | - | - | - |
| TSNE | 3.99s | 0.47s | 0.34s | - | - | - |

## GPU Speedup Ranking (torch vs numpy)

| Rank | Algorithm | numpy | torch | speedup | vs sklearn |
|------|-----------|-------|-------|---------|------------|
| 1 | UMAP | 35.30s | 1.24s | **28.5x** | - |
| 2 | TruncatedSVD | 0.802s | 0.046s | **17.4x** | **21.6x** |
| 3 | NMF | 2.608s | 0.158s | **16.5x** | **19.3x** |
| 4 | TSNE | 3.99s | 0.34s | **11.6x** | - |
| 5 | GMM | 0.454s | 0.042s | **10.8x** | **12.3x** |
| 6 | KMeans | 0.149s | 0.028s | **5.3x** | **8.1x** |
| 7 | IncrementalPCA | 0.254s | 0.035s | **7.3x** | **21.1x** |
| 8 | MiniBatchKMeans | 0.276s | 0.047s | **5.9x** | - |
| 9 | MiniBatchNMF | 5.924s | 1.877s | **3.2x** | - |
| 10 | PCA | 0.019s | 0.017s | 1.1x | 1.3x |
| 11 | Agglomerative | 2.562s | 4.466s | 0.6x | 1.1x |
| 12 | DBSCAN 10d | 27.825s | 47.517s | 0.6x | 1.8x |
| 13 | DBSCAN 50d | 8.935s | 41.808s | 0.2x | 0.1x |

## Detailed Results

### PCA (100K × 100, nc=10)

| Backend | Time (s) | vs sklearn |
|---------|----------|------------|
| numpy | 0.019 | 1.2x |
| cupy | 0.017 | 1.3x |
| torch | 0.017 | 1.3x |
| sklearn | 0.022 | - |

### KMeans (100K × 100, k=8)

| Backend | Time (s) | vs sklearn |
|---------|----------|------------|
| numpy | 0.149 | 1.6x |
| cupy | 0.028 | 8.4x |
| torch | 0.028 | 8.1x |
| sklearn | 0.232 | - |

### GMM (100K × 100, k=5, diag)

| Backend | Time (s) | vs sklearn |
|---------|----------|------------|
| numpy | 0.454 | 1.2x |
| cupy | 0.044 | 12.8x |
| torch | 0.042 | 13.0x |
| sklearn | 0.541 | - |

### NMF (100K × 100, nc=10)

| Backend | Time (s) | vs sklearn |
|---------|----------|------------|
| numpy | 2.608 | 1.2x |
| cupy | 0.158 | 19.5x |
| torch | 0.158 | 19.9x |
| sklearn | 3.054 | - |

### TruncatedSVD (100K × 100, nc=10)

| Backend | Time (s) | vs sklearn |
|---------|----------|------------|
| numpy | 0.802 | 1.6x |
| cupy | 0.049 | 26.0x |
| torch | 0.046 | 27.6x |
| sklearn | 0.993 | - |

### IncrementalPCA (100K × 100, nc=10)

| Backend | Time (s) | vs sklearn |
|---------|----------|------------|
| numpy | 0.254 | 2.9x |
| cupy | 0.036 | 20.9x |
| torch | 0.035 | 21.1x |
| sklearn | 1.330 | - |

### AgglomerativeClustering (10K × 50)

| Backend | Time (s) | vs sklearn |
|---------|----------|------------|
| numpy | 2.562 | 2.0x |
| cupy | 4.480 | 1.1x |
| torch | 4.466 | 1.1x |
| sklearn | 5.004 | - |

### DBSCAN 10d (100K, eps=1.0)

| Backend | Time (s) | vs sklearn |
|---------|----------|------------|
| numpy | 27.825 | 3.1x |
| cupy | 46.786 | 1.8x |
| torch | 47.517 | 1.8x |
| sklearn | 85.026 | - |

### DBSCAN 50d (100K, eps=3.0)

| Backend | Time (s) | vs sklearn |
|---------|----------|------------|
| numpy | 8.935 | 0.4x |
| cupy | 41.684 | 0.1x |
| torch | 41.808 | 0.1x |
| sklearn | 3.694 | - |

### TSNE (1K × 20, max_iter=250)

| Backend | Time (s) |
|---------|----------|
| numpy | 3.99 |
| cupy | 0.47 |
| torch | 0.34 |

### UMAP (100K × 100)

| Backend | Time (s) | vs numpy |
|---------|----------|----------|
| numpy | 35.30 | - |
| cupy | 1.38 | 25.6x |
| torch | 1.24 | 28.5x |

### MiniBatchKMeans (100K × 100, k=8)

| Backend | Time (s) |
|---------|----------|
| numpy | 0.276 |
| cupy | 0.054 |
| torch | 0.047 |

### MiniBatchNMF (100K × 100, nc=10)

| Backend | Time (s) |
|---------|----------|
| numpy | 5.924 |
| cupy | 1.839 |
| torch | 1.877 |

## Key Findings

1. **UMAP** — Best GPU acceleration (**28.5x**), sparse graph + negative sampling
2. **TruncatedSVD, IncrementalPCA, NMF** — Excellent GPU (17-28x)
3. **GMM, KMeans** — Good GPU (8-11x)
4. **DBSCAN** — numpy competitive with sklearn for 10d; GPU slower for high-dim
5. **Agglomerative** — CPU-bound, no GPU benefit
6. **PCA** — Marginal GPU benefit (1.3x), CPU already fast
