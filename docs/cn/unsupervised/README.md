# 无监督学习

> 语言：中文
> 最后更新：2026-07-23
> 本页：无监督学习索引
> English: [English](../en/unsupervised/README.md)

## 概览

`statgpu.unsupervised` 提供 sklearn 风格的无监督学习 estimator，并遵循显式 CPU、CuPy/CUDA、Torch CUDA 设备语义。这个目录按模型拆分说明目标函数、估计过程、后端行为、输出字段、限制和外部验证方式。

## 模型列表

- [PCA](pca.md)：exact 或 randomized 主成分分析。
- [KMeans](kmeans.md)：Lloyd 聚类，支持 random 和 greedy k-means++ 初始化。
- [DBSCAN](dbscan.md)：dense Euclidean 密度聚类，支持可选的 statgpu 自有 Cython CPU 快路径。
- [GaussianMixture](gaussian-mixture.md)：支持 diagonal、spherical、tied、full covariance 的 Gaussian mixture，使用 EM 拟合。
- [NMF](nmf.md)：Frobenius loss 下的 multiplicative update 非负矩阵分解。
- [AgglomerativeClustering](agglomerative-clustering.md)：dense exact single、complete、average、ward linkage 聚类。
- [TruncatedSVD](truncated-svd.md)：不中心化输入的 dense truncated SVD。
- [MiniBatchKMeans](minibatch-kmeans.md)：面向较大 dense 数据的 mini-batch KMeans。
- [IncrementalPCA](incremental-pca.md)：dense batch-wise 主成分分析。
- [MiniBatchNMF](minibatch-nmf.md)：dense mini-batch 非负矩阵分解。
- [UMAP](umap.md)：dense exact Euclidean UMAP v1。
- [TSNE](tsne.md)：dense exact Euclidean t-SNE v1。

## 支持矩阵

| Estimator | CPU | CuPy/CUDA | Torch CUDA | 主要目标或准则 |
|---|---|---|---|---|
| `PCA` | 支持 | 支持 | 支持 | 最大方差 / rank-k 重构损失 |
| `KMeans` | 支持 | 支持 | 支持 | Squared Euclidean inertia |
| `DBSCAN` | 支持 | 支持 | 支持 | Density reachability 和 connected components |
| `GaussianMixture` | 支持 | 支持 | 支持 | Gaussian mixture log likelihood |
| `NMF` | 支持 | 支持 | 支持 | 非负约束下的 Frobenius 重构损失 |
| `AgglomerativeClustering` | 支持 | 支持 | 支持 | Hierarchical linkage merge criterion |
| `TruncatedSVD` | 支持 | 支持 | 支持 | 不中心化低秩重构 |
| `MiniBatchKMeans` | 支持 | 支持 | 支持 | Mini-batch squared Euclidean inertia |
| `IncrementalPCA` | 支持 | 支持 | 支持 | Batch-wise centered low-rank reconstruction |
| `MiniBatchNMF` | 支持 | 支持 | 支持 | Mini-batch Frobenius reconstruction loss |
| `UMAP` | 支持 | 支持，含 host SciPy graph assembly | 支持，含 host SciPy graph assembly | Fuzzy graph cross-entropy |
| `TSNE` | 支持 | 支持 | 支持 | 高低维 affinity 的 KL divergence |

显式 `device="cuda"` 和 `device="torch"` 不会静默 fallback 到 CPU；依赖不可用或模型不支持时会明确报错。

## 共享验证

单元测试：

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

benchmark：

- `dev/benchmarks/benchmark_unsupervised.py`
- `dev/benchmarks/benchmark_unsupervised_phase2.py`
- `dev/benchmarks/benchmark_unsupervised_dbscan_cython.py`
- `dev/benchmarks/benchmark_unsupervised_phase3.py`
- `dev/benchmarks/benchmark_unsupervised_phase3b.py`
- `dev/benchmarks/benchmark_unsupervised_phase3c.py`

远程验证产物：

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

`umap-learn`、`openTSNE`、sklearn、statsmodels、R 和 cuML 等外部包仅作为验证或 benchmark baseline；production estimator code 保持 statgpu 自有实现。
