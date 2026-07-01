# Unsupervised Benchmark: 12 Algorithms × 3 Backends

Date: 2026-06-27
Hardware: Tesla P100-SXM2-16GB (GPU), Intel Xeon (CPU)

## Complete 3-Backend Results (100K obs)

| Algorithm | numpy | cupy | torch | sklearn | cupy vs sk | torch vs sk |
|-----------|-------|------|-------|---------|------------|-------------|
| PCA | 0.019s | 0.017s | 0.017s | 0.022s | 1.3x | 1.3x |
| KMeans | 0.149s | 0.036s | 0.025s | 0.232s | 6.3x | **9.3x** |
| GMM | 0.472s | 0.045s | 0.044s | 0.541s | 12.9x | **13.2x** |
| NMF | 2.560s | 0.166s | 0.158s | 3.054s | 18.0x | **18.9x** |
| TruncatedSVD | 0.807s | 0.046s | 0.046s | 0.993s | 12.0x | **12.2x** |
| IncrementalPCA | 0.252s | 0.035s | 0.034s | 1.330s | 20.9x | **21.2x** |
| Agglomerative | 2.520s | 4.434s | 4.529s | 5.004s | 1.1x | 1.1x |
| DBSCAN 10d | 15.370s | 1.966s | 1.972s | 85.026s | **24.0x** | **23.9x** |
| DBSCAN 50d | 3.161s | 1.321s | 1.323s | 3.694s | 2.4x | 2.4x |
| MiniBatchKMeans | 0.252s | 0.049s | 0.045s | - | 5.2x | 5.6x |
| MiniBatchNMF | 5.979s | 1.836s | 1.795s | - | 3.2x | 3.3x |
| UMAP | 57.47s | 8.21s | 8.12s | - | 7.0x | **7.1x** |
| TSNE | 3.987s | 0.332s | 0.324s | - | 12.0x | **12.3x** |

## GPU Speedup Ranking (torch vs numpy)

| Rank | Algorithm | numpy | torch | speedup | vs sklearn |
|------|-----------|-------|-------|---------|------------|
| 1 | DBSCAN 10d | 15.37s | 1.97s | **7.8x** | **23.9x** |
| 2 | TSNE | 3.99s | 0.32s | **12.3x** | - |
| 3 | UMAP | 57.47s | 8.12s | **7.1x** | - |
| 4 | IncrementalPCA | 0.252s | 0.034s | **7.3x** | **21.2x** |
| 5 | NMF | 2.560s | 0.158s | **16.2x** | **18.9x** |
| 6 | GMM | 0.472s | 0.044s | **10.8x** | **13.2x** |
| 7 | TruncatedSVD | 0.807s | 0.046s | **17.7x** | **12.2x** |
| 8 | KMeans | 0.149s | 0.025s | **6.1x** | **9.3x** |
| 9 | DBSCAN 50d | 3.161s | 1.323s | **2.4x** | 2.4x |
| 10 | MiniBatchKMeans | 0.252s | 0.045s | **5.6x** | - |
| 11 | MiniBatchNMF | 5.979s | 1.795s | **3.3x** | - |
| 12 | PCA | 0.019s | 0.017s | 1.1x | 1.3x |
| 13 | Agglomerative | 2.520s | 4.529s | 0.6x | 1.1x |

## Key Findings

1. **DBSCAN** — Excellent GPU acceleration (**7.8x** for 10d, **2.4x** for 50d), pure GPU path
2. **TruncatedSVD, NMF, IncrementalPCA** — Excellent GPU (16-21x)
3. **TSNE** — Great GPU (**12.3x**)
4. **UMAP** — Good GPU (**7.1x**), improved from 325s to 8.1s
5. **GMM, KMeans** — Good GPU (6-13x)
6. **Agglomerative** — CPU-bound, no GPU benefit
