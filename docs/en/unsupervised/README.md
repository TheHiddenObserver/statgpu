# Unsupervised Learning

> Language: English
> Last updated: 2026-07-23
> This page: Unsupervised learning index
> Switch: [Chinese](../../cn/unsupervised/README.md)

## Overview

`statgpu.unsupervised` contains sklearn-style unsupervised estimators with explicit CPU, CuPy/CUDA, and Torch CUDA device behavior. This directory documents each estimator separately so the loss function, estimating algorithm, backend behavior, and validation evidence are visible without compressing all models into one page.

## Estimators

- [PCA](pca.md): exact or randomized principal component analysis.
- [KMeans](kmeans.md): Lloyd clustering with random or greedy k-means++ initialization.
- [DBSCAN](dbscan.md): dense Euclidean density clustering with optional statgpu-owned Cython CPU acceleration.
- [GaussianMixture](gaussian-mixture.md): Gaussian mixture fitted by EM with diagonal, spherical, tied, or full covariance.
- [NMF](nmf.md): non-negative matrix factorization with multiplicative updates and Frobenius loss.
- [AgglomerativeClustering](agglomerative-clustering.md): exact dense single, complete, average, or ward linkage clustering.
- [TruncatedSVD](truncated-svd.md): dense uncentered truncated SVD for low-rank projection.
- [MiniBatchKMeans](minibatch-kmeans.md): mini-batch Euclidean K-Means for larger dense datasets.
- [IncrementalPCA](incremental-pca.md): dense batch-wise principal component analysis.
- [MiniBatchNMF](minibatch-nmf.md): dense mini-batch non-negative matrix factorization.
- [UMAP](umap.md): dense exact Euclidean UMAP v1.
- [TSNE](tsne.md): dense exact Euclidean t-SNE v1.

## Support Matrix

| Estimator | CPU | CuPy/CUDA | Torch CUDA | Main objective or criterion |
|---|---|---|---|---|
| `PCA` | yes | yes | yes | Maximum variance / rank-k reconstruction loss |
| `KMeans` | yes | yes | yes | Squared Euclidean inertia |
| `DBSCAN` | yes | yes | yes | Density reachability and connected components |
| `GaussianMixture` | yes | yes | yes | Gaussian mixture log likelihood |
| `NMF` | yes | yes | yes | Frobenius reconstruction loss under non-negativity |
| `AgglomerativeClustering` | yes | yes | yes | Hierarchical linkage merge criterion |
| `TruncatedSVD` | yes | yes | yes | Uncentered low-rank reconstruction |
| `MiniBatchKMeans` | yes | yes | yes | Mini-batch squared Euclidean inertia |
| `IncrementalPCA` | yes | yes | yes | Batch-wise centered low-rank reconstruction |
| `MiniBatchNMF` | yes | yes | yes | Mini-batch Frobenius reconstruction loss |
| `UMAP` | yes | yes, host SciPy graph assembly | yes, host SciPy graph assembly | Fuzzy graph cross-entropy |
| `TSNE` | yes | yes | yes | KL divergence between high- and low-dimensional affinities |

Explicit `device="cuda"` and `device="torch"` do not silently fall back to CPU. Unsupported GPU paths raise clear errors.

## Shared Validation

Unit tests:

- `dev/tests/test_unsupervised_pca.py`
- `dev/tests/test_unsupervised_kmeans.py`
- `dev/tests/test_unsupervised_dbscan.py`
- `dev/tests/test_unsupervised_gmm.py`
- `dev/tests/test_unsupervised_nmf.py`
- `dev/tests/test_unsupervised_agglomerative.py`
- `dev/tests/test_unsupervised_truncated_svd.py`
- `dev/tests/test_unsupervised_minibatch_kmeans.py`
- `dev/tests/test_unsupervised_incremental_pca.py`
- `dev/tests/test_unsupervised_minibatch_nmf.py`
- `dev/tests/test_unsupervised_umap.py`
- `dev/tests/test_unsupervised_tsne.py`

Benchmark scripts:

- `dev/benchmarks/benchmark_unsupervised.py`
- `dev/benchmarks/benchmark_unsupervised_phase2.py`
- `dev/benchmarks/benchmark_unsupervised_dbscan_cython.py`
- `dev/benchmarks/benchmark_unsupervised_phase3.py`
- `dev/benchmarks/benchmark_unsupervised_phase3b.py`
- `dev/benchmarks/benchmark_unsupervised_phase3c.py`

Remote validation artifacts:

- `results/unsupervised_phase2_dbscan_cython_verify_20260502_210000.json`
- `results/unsupervised_phase2_verify_20260502_210000.json`
- `results/unsupervised_phase2_verify_summary_20260502_210000.md`
- `results/unsupervised_phase3_remote_finalopt_20260505_084444.json`
- `results/unsupervised_phase3_remote_finalopt_20260505_084444.md`
- `results/unsupervised_phase3_remote_perfopt_mediumlarge_20260505_131617.json`
- `results/unsupervised_phase3_remote_perfopt_mediumlarge_20260505_131617.md`
- `results/unsupervised_phase3_remote_perfopt2_large_tabular_20260505_132223.json`
- `results/unsupervised_phase3_remote_perfopt2_large_tabular_bs4096_20260505_132359.json`
- `results/unsupervised_phase3b_verify_20260507_003957.json`
- `results/unsupervised_phase3b_verify_summary_20260507_003957.md`
- `results/unsupervised_phase3c_opt7_20260507_185500.json`
- `results/unsupervised_phase3c_opt7_summary_20260507_185500.md`
- `results/unsupervised_phase3c_opt7_large_bs4096_20260507_185500.json`
- `results/unsupervised_phase3c_opt7_large_bs4096_summary_20260507_185500.md`
- `results/unsupervised_phase3c_opt7_xlarge_20260507_185500.json`
- `results/unsupervised_phase3c_opt7_xlarge_summary_20260507_185500.md`

External packages such as `umap-learn`, `openTSNE`, sklearn, statsmodels, R, and cuML are used only as validation or benchmark baselines; production estimator code remains statgpu-owned.
