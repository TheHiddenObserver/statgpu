# Unsupervised Learning

> Language: English
> Last updated: 2026-05-02
> This page: Unsupervised learning index
> Switch: [Chinese](../../unsupervised/README.md)

## Overview

`statgpu.unsupervised` contains sklearn-style unsupervised estimators with explicit CPU, CuPy/CUDA, and Torch CUDA device behavior. This directory documents each estimator separately so the loss function, estimating algorithm, backend behavior, and validation evidence are visible without compressing six models into one page.

## Estimators

- [PCA](pca.md): exact or randomized principal component analysis.
- [KMeans](kmeans.md): Lloyd clustering with random or greedy k-means++ initialization.
- [DBSCAN](dbscan.md): dense Euclidean density clustering with optional statgpu-owned Cython CPU acceleration.
- [GaussianMixture](gaussian-mixture.md): diagonal-covariance Gaussian mixture fitted by EM.
- [NMF](nmf.md): non-negative matrix factorization with multiplicative updates and Frobenius loss.
- [AgglomerativeClustering](agglomerative-clustering.md): exact CPU single-linkage clustering.

## Support Matrix

| Estimator | CPU | CuPy/CUDA | Torch CUDA | Main objective or criterion |
|---|---|---|---|---|
| `PCA` | yes | yes | yes | Maximum variance / rank-k reconstruction loss |
| `KMeans` | yes | yes | yes | Squared Euclidean inertia |
| `DBSCAN` | yes | yes | yes | Density reachability and connected components |
| `GaussianMixture` | yes | yes | yes | Diagonal Gaussian mixture log likelihood |
| `NMF` | yes | yes | yes | Frobenius reconstruction loss under non-negativity |
| `AgglomerativeClustering` | yes | no | no | Single-linkage merge criterion |

Explicit `device="cuda"` and `device="torch"` do not silently fall back to CPU. Unsupported GPU paths raise clear errors.

## Shared Validation

Unit tests:

- `dev/tests/test_unsupervised_pca.py`
- `dev/tests/test_unsupervised_kmeans.py`
- `dev/tests/test_unsupervised_dbscan.py`
- `dev/tests/test_unsupervised_gmm.py`
- `dev/tests/test_unsupervised_nmf.py`
- `dev/tests/test_unsupervised_agglomerative.py`

Benchmark scripts:

- `dev/benchmarks/benchmark_unsupervised.py`
- `dev/benchmarks/benchmark_unsupervised_phase2.py`
- `dev/benchmarks/benchmark_unsupervised_dbscan_cython.py`

Latest remote artifacts:

- `results/unsupervised_phase2_dbscan_cython_verify_20260502_210000.json`
- `results/unsupervised_phase2_verify_20260502_210000.json`
- `results/unsupervised_phase2_verify_summary_20260502_210000.md`

UMAP and t-SNE are comparison-only baselines in Phase 2; they are not public statgpu APIs.
