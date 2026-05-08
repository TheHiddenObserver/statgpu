# statgpu 无监督学习路线图：Phase 3B 封板与后续缺口矩阵

> 最后更新：2026-05-07
> 范围：`statgpu.unsupervised`
> 状态：Phase 1/2/3A 已落地；当前轮次聚焦 Phase 3B

## 1. 当前能力快照

`statgpu.unsupervised` 当前已经实现并导出以下 estimator：

- `PCA`
  - CPU / CuPy / Torch 三端。
  - 支持 `svd_solver="auto"|"full"|"covariance"|"randomized"`、`whiten`、`transform`、`inverse_transform`。
- `KMeans`
  - CPU / CuPy / Torch 三端。
  - Lloyd iterations，支持 `init="random"` 与 greedy `k-means++`。
- `DBSCAN`
  - CPU / CuPy / Torch 三端。
  - dense Euclidean only。
  - CPU 有 NumPy/SciPy fallback 和可选 statgpu 自有 Cython fast path。
- `GaussianMixture`
  - CPU / CuPy / Torch 三端。
  - 已从 `covariance_type="diag"` 扩展到 `"diag"|"spherical"|"tied"|"full"`。
  - 使用 log-domain EM，支持 `score_samples`、`score`、`aic`、`bic`。
- `NMF`
  - CPU / CuPy / Torch 三端。
  - MU 更新，Frobenius loss only。
- `AgglomerativeClustering`
  - CPU exact。
  - 已从 single linkage 扩展到 `"single"|"complete"|"average"|"ward"`。
  - 显式 `device="cuda"` / `device="torch"` 继续报 `NotImplementedError`。
- `TruncatedSVD`
  - CPU / CuPy / Torch 三端。
  - dense uncentered full/randomized SVD。
- `MiniBatchKMeans`
  - CPU / CuPy / Torch 三端。
  - mini-batch Euclidean center updates。
- `IncrementalPCA`
  - CPU / CuPy / Torch 三端。
  - dense batch-wise PCA updates。
- `MiniBatchNMF`
  - CPU / CuPy / Torch 三端。
  - dense mini-batch MU updates with Frobenius loss。
- `UMAP`
  - CPU / CuPy / Torch 三端。
  - dense exact Euclidean v1。
- `TSNE`
  - CPU / CuPy / Torch 三端。
  - dense exact Euclidean v1。

已保留的远程验证 artifact：

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

当前 follow-up validation：

- PR #28 review 修复已进入 follow-up 分支：
  - `MiniBatchKMeans.partial_fit` 首次 batch 小于 `n_clusters` 时明确 `ValueError`。
  - `MiniBatchKMeans.score` 增加 2D、feature count、sparse 输入校验。
  - `dev/benchmarks/benchmark_unsupervised_phase3.py` 使用 Python 3.8 兼容 dict merge。
  - `UMAP._fuzzy_graph` 使用 backend 向量化填图，避免 CuPy/Torch 逐行 Python loop。
- 本地环境的 `pytest.exe` wrapper 当前不可作为可靠 gate；本地只做 `python` direct smoke 与 `py_compile`。完整 pytest、GPU、外部 baseline 和性能结论必须来自 Matpool 远程环境。

## 2. 已有框架能力对比

| 方向 | 外部框架已有能力 | statgpu 当前状态 | 后续缺口 |
|---|---|---|---|
| 聚类 | sklearn `KMeans` | 已实现三端 | 大样本、大 `k` 继续优化 |
| 聚类 | sklearn `MiniBatchKMeans` | 已实现三端 | 继续优化 GPU batch scheduling 与公平 benchmark |
| 聚类 | sklearn `BisectingKMeans` | 未实现 | 后续层次式 KMeans |
| 聚类 | sklearn `DBSCAN` | 已实现三端 | CPU selector、GPU memory behavior 继续优化 |
| 聚类 | sklearn/cuML `HDBSCAN` | 未实现 | Phase 4 密度聚类高级路线 |
| 聚类 | sklearn `OPTICS` | 未实现 | Phase 4 variable-density ordering |
| 聚类 | sklearn `BIRCH` | 未实现 | 大样本增量聚类 |
| 聚类 | sklearn `MeanShift` | 未实现 | mode-seeking clustering |
| 聚类 | sklearn `AffinityPropagation` | 未实现 | message-passing clustering |
| 聚类 | sklearn/cuML `SpectralClustering` | 未实现 | 需要 graph/eigensolver 基础 |
| 聚类 | sklearn biclustering | 未实现 | matrix biclustering 后续单列 |
| 聚类 | sklearn `FeatureAgglomeration` | 未实现 | feature-side 层次聚类 |
| 分解 | sklearn/statsmodels `PCA` | 已实现三端 | missing-data / normalization variants 非当前重点 |
| 分解 | sklearn/cuML `TruncatedSVD` | 已实现三端 | 后续补 sparse/LSA 路线 |
| 分解 | sklearn `IncrementalPCA` | 已实现 dense v1 | 后续补 out-of-core 文件流和更大规模验证 |
| 分解 | sklearn `KernelPCA` | 未实现 | 需要 kernel matrix + eigensolver |
| 分解 | sklearn `SparsePCA` / `MiniBatchSparsePCA` | 未实现 | sparse loading / dictionary-style PCA |
| 分解 | sklearn `FastICA` | 未实现 | whitening + fixed-point iteration |
| 分解 | sklearn `FactorAnalysis` | 未实现 | latent Gaussian factor model |
| 分解 | sklearn `DictionaryLearning` | 未实现 | sparse coding / dictionary learning |
| 分解 | sklearn `LatentDirichletAllocation` | 未实现 | topic model，和当前 dense numeric backend 关系较远 |
| 分解 | sklearn `NMF` | 已实现 MU/Frobenius；`MiniBatchNMF` dense v1 已实现 | 缺更多 beta loss / init / solver |
| 流形 | sklearn/openTSNE `TSNE` | 已实现 dense exact v1 | 缺 Barnes-Hut、FFT/FIt-SNE、approximate path |
| 流形 | umap-learn `UMAP` | 已实现 dense exact v1 | 缺 approximate neighbor search 和 new-data transform |
| 流形 | sklearn `Isomap` | 未实现 | graph shortest path + embedding |
| 流形 | sklearn `LocallyLinearEmbedding` | 未实现 | local reconstruction weights + eigensolver |
| 流形 | sklearn `MDS` / `ClassicalMDS` | 未实现 | distance embedding |
| 流形 | sklearn `SpectralEmbedding` | 未实现 | graph Laplacian embedding |
| 混合模型 | sklearn `GaussianMixture(full/tied/diag/spherical)` | 已实现三端 | benchmark 继续扩大规模和 covariance parity |
| 混合模型 | sklearn `BayesianGaussianMixture` | 未实现 | variational Bayesian mixture |
| 层次聚类 | SciPy `single/complete/average/ward` | 已实现 CPU exact | `weighted/centroid/median` 待评估 |
| 层次聚类 | R `cluster::agnes` | 用作 baseline | 更多 linkage alignment 与 R validation |
| R cluster | `pam/clara/fanny/diana/mona/daisy` | 未实现 | medoid/fuzzy/divisive/mixed-type clustering 后续方向 |
| RAPIDS cuML | GPU `KMeans/DBSCAN/PCA/TruncatedSVD` | statgpu 已有对应子集 | cuML 继续作为 GPU runtime baseline |
| RAPIDS cuML | GPU `Agglomerative/Spectral/HDBSCAN` | 未实现 | graph/hierarchy GPU 后续单列 |

## 3. statgpu 缺口矩阵

### P0 / P1：当前与下一轮优先

| 能力 | 优先原因 | 实施策略 | 外部 baseline |
|---|---|---|---|
| GMM covariance parity | sklearn GMM 主 API 已支持四种 covariance | 已补 `"spherical"|"tied"|"full"`；继续扩大三端 benchmark | sklearn `GaussianMixture` |
| Agglomerative common linkages | sklearn/SciPy/R 常用层次聚类能力 | 已补 CPU exact `"complete"|"average"|"ward"`；GPU 继续显式不支持 | sklearn、SciPy、R `cluster::agnes` |
| MiniBatchKMeans GPU 性能 | 直接影响大样本 KMeans 可用性 | 优化 batch order、center update 和 host-device 边界 | sklearn `MiniBatchKMeans`、cuML |
| DBSCAN selector / memory | medium 数据下 GPU/CPU 路径仍需调优 | 保持自有实现，优化 dense selector 与 chunk memory | sklearn、cuML |

### P2：中期补齐

| 能力 | 为什么重要 | 初始实现策略 | 外部 baseline |
|---|---|---|---|
| `IncrementalPCA` | out-of-core 和大样本 PCA | dense v1 已实现；后续补文件流和更大规模 GPU 验证 | sklearn |
| `MiniBatchNMF` | 大样本 NMF | dense v1 已实现；后续补更多 init/beta loss | sklearn |
| `KernelPCA` | nonlinear dimensionality reduction | dense kernel matrix + eigensolver | sklearn |
| `SpectralClustering` | graph clustering 常用方法 | neighbor graph + Laplacian + KMeans | sklearn、cuML |
| `SpectralEmbedding` | graph embedding 基础能力 | 与 SpectralClustering 共享 graph/eigensolver | sklearn |
| `OPTICS` | variable density DBSCAN alternative | CPU first，GPU 后续研究 | sklearn |
| `HDBSCAN` | 高级 density clustering | 先 benchmark-only，再决定 statgpu API | sklearn/cuML/hdbscan |

### P3：研究型 / 后续

| 能力 | 备注 |
|---|---|
| `FastICA` | 可复用 PCA whitening，需 fixed-point iteration |
| `FactorAnalysis` | 可复用部分 EM/log-domain 工具 |
| `SparsePCA` / `DictionaryLearning` | 需要 sparse coding / coordinate descent 路线 |
| `LatentDirichletAllocation` | topic model，与当前 dense numeric backend 关系较远 |
| `Isomap` / `LLE` / `MDS` | 依赖 graph shortest path、local weights 或 distance embedding |
| R `pam/clara/fanny/diana/mona` | medoid、fuzzy、divisive clustering 后续方向 |

## 4. Phase 3B 执行计划

本轮封板目标：

1. 合入 PR #28 follow-up 修复。
2. 完成 `GaussianMixture` covariance type parity。
3. 完成 `AgglomerativeClustering` 常用 CPU exact linkage。
4. 更新 EN/CN 文档和 benchmark 入口。
5. 在 Matpool 远程跑完整三端和外部 baseline validation。

实现约束：

- production code 不调用 sklearn、statsmodels、R、umap-learn、openTSNE 或 cuML。
- 外部框架只用于 tests/benchmarks baseline。
- 显式 `device="cuda"` / `device="torch"` 不允许静默回退。
- 远程凭据不得写入代码、文档、测试、benchmark 或结果 artifact。

## 5. External Validation / 外部验证

Phase 3B 新增远程矩阵：

- GMM：
  - statgpu CPU/CuPy/Torch vs sklearn CPU。
  - 覆盖 `"diag"|"spherical"|"tied"|"full"`。
  - 指标：average log likelihood、AIC/BIC、component-matched means/covariances、`predict_proba` 差异、运行时间。
- Agglomerative：
  - statgpu CPU vs sklearn/SciPy/R `cluster::agnes` where aligned。
  - 覆盖 `"single"|"complete"|"average"|"ward"`。
  - 指标：ARI、cluster count、children/distances small-sample consistency、运行时间。
- 结果输出：
  - `results/unsupervised_phase3b_verify_*.json`
  - `results/unsupervised_phase3b_verify_summary_*.md`

## 6. 文档 Gate

用户可见能力变化必须同步：

- `README.md`
- `USAGE.md`
- `USAGE_CN.md`
- `docs/en/unsupervised/`
- `docs/unsupervised/`
- `docs/en/models/README.md`
- `docs/models/README.md`
- `docs/en/benchmarks.md`
- `docs/benchmarks.md`
- `docs/changelog.md`

逐模型文档至少包含：

- Overview
- Path
- Objective Function / Loss Function
- Estimating Equation
- Parameters
- CPU+GPU Examples
- Strict/Approx Difference
- Outputs
- FAQ
- External Validation
- References

## 7. 参考来源

- sklearn clustering API: https://scikit-learn.org/stable/api/sklearn.cluster.html
- sklearn decomposition API: https://scikit-learn.org/stable/api/sklearn.decomposition.html
- sklearn manifold API: https://scikit-learn.org/stable/api/sklearn.manifold.html
- sklearn mixture API: https://scikit-learn.org/stable/api/sklearn.mixture.html
- SciPy hierarchical clustering: https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html
- statsmodels PCA: https://www.statsmodels.org/stable/generated/statsmodels.multivariate.pca.PCA.html
- RAPIDS cuML cluster: https://docs.rapids.ai/api/cuml/nightly/api/cuml.cluster/
- RAPIDS cuML PCA: https://docs.rapids.ai/api/cuml/nightly/api/generated/cuml.decomposition.PCA/
- RAPIDS cuML TruncatedSVD: https://docs.rapids.ai/api/cuml/nightly/api/generated/cuml.decomposition.TruncatedSVD/
- R cluster package: https://www.rdocumentation.org/packages/cluster
- R `agnes`: https://www.rdocumentation.org/packages/cluster/versions/2.1.8.1/topics/agnes
