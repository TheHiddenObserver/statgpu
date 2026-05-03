# 无监督学习

> 语言：中文
> 最后更新：2026-05-02
> 本页：无监督学习索引
> English: [English](../en/unsupervised/README.md)

## 概览

`statgpu.unsupervised` 提供 sklearn 风格的无监督 estimator，并遵循显式 CPU、CuPy/CUDA、Torch CUDA 设备语义。本目录按模型拆分说明，避免把六个模型的损失函数、估计方法、后端行为和验证结果压缩在一个页面里。

## Estimators

- [PCA](pca.md)：exact 或 randomized 主成分分析。
- [KMeans](kmeans.md)：Lloyd 聚类，支持 random 和 greedy k-means++ 初始化。
- [DBSCAN](dbscan.md)：dense Euclidean 密度聚类，支持可选 statgpu 自有 Cython CPU 加速。
- [GaussianMixture](gaussian-mixture.md)：diagonal covariance Gaussian mixture，使用 EM 拟合。
- [NMF](nmf.md)：Frobenius loss 下的 multiplicative update 非负矩阵分解。
- [AgglomerativeClustering](agglomerative-clustering.md)：CPU exact single-linkage 聚类。

## 支持矩阵

| Estimator | CPU | CuPy/CUDA | Torch CUDA | 主要目标或准则 |
|---|---|---|---|---|
| `PCA` | 支持 | 支持 | 支持 | 最大方差 / rank-k 重构损失 |
| `KMeans` | 支持 | 支持 | 支持 | Squared Euclidean inertia |
| `DBSCAN` | 支持 | 支持 | 支持 | Density reachability 与 connected components |
| `GaussianMixture` | 支持 | 支持 | 支持 | Diagonal Gaussian mixture log likelihood |
| `NMF` | 支持 | 支持 | 支持 | 非负约束下的 Frobenius 重构损失 |
| `AgglomerativeClustering` | 支持 | 不支持 | 不支持 | Single-linkage merge criterion |

显式 `device="cuda"` 和 `device="torch"` 不会静默 fallback 到 CPU；依赖不可用或模型不支持时会明确报错。

## 共享验证

单元测试：

- `dev/tests/test_unsupervised_pca.py`
- `dev/tests/test_unsupervised_kmeans.py`
- `dev/tests/test_unsupervised_dbscan.py`
- `dev/tests/test_unsupervised_gmm.py`
- `dev/tests/test_unsupervised_nmf.py`
- `dev/tests/test_unsupervised_agglomerative.py`

benchmark：

- `dev/benchmarks/benchmark_unsupervised.py`
- `dev/benchmarks/benchmark_unsupervised_phase2.py`
- `dev/benchmarks/benchmark_unsupervised_dbscan_cython.py`

最新远程验证产物：

- `results/unsupervised_phase2_dbscan_cython_verify_20260502_210000.json`
- `results/unsupervised_phase2_verify_20260502_210000.json`
- `results/unsupervised_phase2_verify_summary_20260502_210000.md`

UMAP 和 t-SNE 在 Phase 2 仅作为 comparison-only baseline，不是 statgpu public API。
