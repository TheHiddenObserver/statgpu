# statgpu 无监督学习路线图：外部框架差距矩阵与 Phase 3/4 计划

> 最后更新：2026-05-02
> 状态：Phase 1/2 已落地；本文档用于规划 Phase 3/4
> 范围：`statgpu.unsupervised`

## 1. 当前能力快照

`statgpu.unsupervised` 当前已经不是空模块。已实现并导出：

- `PCA`
  - CPU / CuPy / Torch
  - `svd_solver="auto"|"full"|"covariance"|"randomized"`
  - `whiten`、`transform`、`inverse_transform`
- `KMeans`
  - CPU / CuPy / Torch
  - Lloyd iterations
  - `init="random"|"k-means++"`，其中 k-means++ 使用 greedy local trials
- `DBSCAN`
  - CPU / CuPy / Torch
  - dense Euclidean only
  - CPU exact NumPy/SciPy fallback
  - optional statgpu-owned Cython fast path `_dbscan_cpu.pyx`
- `GaussianMixture`
  - CPU / CuPy / Torch
  - `covariance_type="diag"` only
  - log-domain EM
  - `score_samples`、`score`、`aic`、`bic`
- `NMF`
  - CPU / CuPy / Torch
  - multiplicative update
  - Frobenius loss only
- `AgglomerativeClustering`
  - CPU only
  - exact single-linkage
  - explicit `device="cuda"` / `device="torch"` raises `NotImplementedError`

当前验证与结果 artifact：

- `results/unsupervised_phase2_dbscan_cython_verify_20260502_210000.json`
- `results/unsupervised_phase2_verify_20260502_210000.json`
- `results/unsupervised_phase2_verify_summary_20260502_210000.md`
- `results/unsupervised_external_matrix_10k_200.json`
- `results/unsupervised_external_matrix_summary_2026-04-30.md`
- `results/unsupervised_kmeans_greedy_full_framework_comparison_2026-04-30.json`
- `results/unsupervised_kmeans_greedy_full_framework_comparison_2026-04-30.md`

最新 Matpool 验证结论：

- targeted tests：`50 passed, 1 warning in 5.97s`
- DBSCAN compact `n=5000`：
  - Cython CPU `219.62ms`
  - fallback CPU `379.80ms`
  - sklearn CPU `178.94ms`
  - CuPy `21.56ms`
  - Torch `21.07ms`
  - labels ARI `1.0`，noise mask match
  - Cython CPU 约 `1.23x` sklearn CPU，接近但未严格达到 `<=1.2x` 目标
- GaussianMixture / NMF GPU 与 CPU 精度差异在浮点噪声量级
- UMAP / openTSNE 当前只作为 comparison-only baseline，不是 statgpu public API

## 2. 已有框架能力对比

### 2.1 scikit-learn

| 方向 | 外部框架已有能力 | statgpu 当前状态 | 差距 |
|---|---|---|---|
| 聚类 | `KMeans` | 已实现 CPU/CuPy/Torch | 需继续优化大 `k` / 大样本性能 |
| 聚类 | `MiniBatchKMeans` | 未实现 | 大样本在线/分批 KMeans 缺口 |
| 聚类 | `BisectingKMeans` | 未实现 | 层次式 KMeans 缺口 |
| 聚类 | `DBSCAN` | 已实现 CPU/CuPy/Torch | 仅 Euclidean dense；CPU Cython 仍略慢于 sklearn compact case |
| 聚类 | `HDBSCAN` | 未实现 | 高级密度聚类缺口 |
| 聚类 | `OPTICS` | 未实现 | 不同密度簇的 density ordering 缺口 |
| 聚类 | `BIRCH` | 未实现 | 大样本增量聚类缺口 |
| 聚类 | `MeanShift` | 未实现 | mode-seeking 聚类缺口 |
| 聚类 | `AffinityPropagation` | 未实现 | message-passing 聚类缺口 |
| 聚类 | `SpectralClustering` | 未实现 | graph Laplacian + eigensolver 路线缺口 |
| 聚类 | biclustering | 未实现 | matrix biclustering 不在当前 API |
| 聚类 | `FeatureAgglomeration` | 未实现 | feature-side hierarchical clustering 缺口 |
| 分解 | `PCA` | 已实现 CPU/CuPy/Torch | 已覆盖主要 dense PCA；缺 `IncrementalPCA` |
| 分解 | `TruncatedSVD` | 未实现 | 稀疏/LSA 和 dense truncated SVD 缺口 |
| 分解 | `IncrementalPCA` | 未实现 | out-of-core PCA 缺口 |
| 分解 | `KernelPCA` | 未实现 | kernel matrix + eigensolver 缺口 |
| 分解 | `SparsePCA` / `MiniBatchSparsePCA` | 未实现 | sparse loading / dictionary-style PCA 缺口 |
| 分解 | `FastICA` | 未实现 | independent component analysis 缺口 |
| 分解 | `FactorAnalysis` | 未实现 | latent Gaussian factor model 缺口 |
| 分解 | `DictionaryLearning` / `MiniBatchDictionaryLearning` | 未实现 | sparse coding / dictionary learning 缺口 |
| 分解 | `LatentDirichletAllocation` | 未实现 | topic model 缺口 |
| 分解 | `NMF` | 已实现 MU/Frobenius | 缺 `MiniBatchNMF`、其他 beta loss / init / solver |
| 流形 | `TSNE` | 未实现 | 当前仅 openTSNE baseline |
| 流形 | `Isomap` | 未实现 | graph shortest path + embedding 缺口 |
| 流形 | `LocallyLinearEmbedding` | 未实现 | local reconstruction weights + eigensolver 缺口 |
| 流形 | `MDS` / `ClassicalMDS` | 未实现 | distance embedding 缺口 |
| 流形 | `SpectralEmbedding` | 未实现 | graph Laplacian embedding 缺口 |
| 流形 | `trustworthiness` | benchmark 中可用 | 可考虑纳入 `statgpu.metrics`，不一定属于 estimator |
| 混合模型 | `GaussianMixture(full/tied/diag/spherical)` | 仅 `diag` | covariance type parity 缺口 |
| 混合模型 | `BayesianGaussianMixture` | 未实现 | variational Bayesian mixture 缺口 |
| 混合模型 | sampling / warm-start / more init variants | 未实现或有限 | API parity 缺口 |

### 2.2 SciPy

| 外部框架已有能力 | statgpu 当前状态 | 差距 |
|---|---|---|
| `scipy.cluster.hierarchy.linkage(method="single")` | 已通过 CPU single linkage 对齐 | 已覆盖核心 single linkage |
| `complete` / `average` / `weighted` linkage | 未实现 | Phase 3 可先做 CPU exact |
| `centroid` / `median` linkage | 未实现 | 数值和行为兼容性需单独验证 |
| `ward` linkage | 未实现 | 需要 Ward variance objective 与 sklearn/SciPy 对齐 |
| `fcluster` / dendrogram utilities | 仅 `n_clusters` labels | tree cutting 和 dendrogram 工具缺口 |

### 2.3 statsmodels

| 外部框架已有能力 | statgpu 当前状态 | 差距 |
|---|---|---|
| `statsmodels.multivariate.pca.PCA` `method="svd"` | 已纳入 PCA 外部对比 | statgpu runtime 更快；需保留 benchmark baseline |
| `method="eig"` | 已做过补充对比 | 可继续作为 covariance/eigh PCA parity baseline |
| PCA missing-data / normalization variants | 未实现 | 非 Phase 3 首要目标 |

### 2.4 R `cluster`

| 外部框架已有能力 | statgpu 当前状态 | 差距 |
|---|---|---|
| `agnes` | 用作 agglomerative baseline | statgpu 仅 single linkage |
| `pam` | 未实现 | medoid clustering 缺口 |
| `clara` | 未实现 | large-sample medoid clustering 缺口 |
| `fanny` | 未实现 | fuzzy clustering 缺口 |
| `diana` | 未实现 | divisive hierarchical clustering 缺口 |
| `mona` | 未实现 | binary data divisive clustering 缺口 |
| `daisy` | 未实现 | mixed-type dissimilarity 缺口 |

### 2.5 RAPIDS cuML

| 外部框架已有能力 | statgpu 当前状态 | 差距 |
|---|---|---|
| GPU `KMeans` | 已有 statgpu GPU KMeans | cuML 是性能 baseline；statgpu 大 `k` 仍需优化 |
| GPU `DBSCAN` | 已有 statgpu GPU DBSCAN | statgpu dense graph 适用范围较窄；需研究 neighbor search / union-find |
| GPU `AgglomerativeClustering` | statgpu 仅 CPU single | GPU 层次聚类缺口 |
| GPU `SpectralClustering` | 未实现 | graph/eigensolver 缺口 |
| GPU `HDBSCAN` | 未实现 | 高级 density clustering 缺口 |
| GPU `PCA` | 已有 statgpu GPU PCA | 继续作为 runtime baseline |
| GPU `TruncatedSVD` | 未实现 | Phase 3 优先补齐 |

### 2.6 umap-learn / openTSNE

| 外部框架已有能力 | statgpu 当前状态 | 差距 |
|---|---|---|
| `umap-learn` UMAP | 远程 benchmark baseline only | 暂不作为 statgpu API |
| `openTSNE` t-SNE | 远程 benchmark baseline only | 暂不作为 statgpu API |

## 3. statgpu 缺口矩阵

### P0 / P1：优先补齐

| 能力 | 为什么优先 | 初始实现策略 | 外部 baseline |
|---|---|---|---|
| `TruncatedSVD` | 和 PCA 共享大量后端能力，适合三端实现；也是 sparse/LSA 路线前置能力 | 先 dense randomized SVD，后续再加 sparse | sklearn `TruncatedSVD`、cuML `TruncatedSVD` |
| `MiniBatchKMeans` | 直接解决当前 KMeans 大样本/大 `k` 性能短板 | batch update + running counts；CPU/CuPy/Torch 一致 | sklearn `MiniBatchKMeans`、cuML KMeans |
| `GaussianMixture(full/tied/spherical)` | sklearn GMM 主要 parity 缺口 | 先 spherical/tied，再 full；保持 diag 现有路径 | sklearn `GaussianMixture` |
| `AgglomerativeClustering(complete/average/ward)` | SciPy/sklearn/R 层次聚类常用功能 | 先 CPU exact；GPU 显式不支持，直到有清晰算法 | sklearn、SciPy、R `cluster::agnes` |

### P2：中期补齐

| 能力 | 为什么重要 | 初始实现策略 | 外部 baseline |
|---|---|---|---|
| `IncrementalPCA` | out-of-core 和大样本 PCA | CPU first，后续 GPU batch accumulation | sklearn `IncrementalPCA` |
| `MiniBatchNMF` | 大样本 NMF | batch MU / online update；先 CPU+GPU dense | sklearn `MiniBatchNMF` |
| `KernelPCA` | nonlinear dimensionality reduction | dense kernel matrix + eigensolver；先 CPU/CuPy/Torch small/medium | sklearn `KernelPCA` |
| `SpectralClustering` | graph clustering 常用方法 | neighbor graph + Laplacian eigenvectors + KMeans | sklearn、cuML |
| `SpectralEmbedding` | graph embedding 基础能力 | 和 SpectralClustering 共享 Laplacian/eigensolver | sklearn |
| `OPTICS` | variable density DBSCAN alternative | CPU first；GPU 后续研究 | sklearn `OPTICS` |
| `HDBSCAN` | cuML/sklearn 已有高级 density clustering | 先 benchmark-only，再决定 statgpu API | sklearn/cuML/hdbscan |

### P3：研究型 / 后续

| 能力 | 备注 |
|---|---|
| `FastICA` | 需要 whitening + fixed-point iteration；可复用 PCA |
| `FactorAnalysis` | 需要 EM / likelihood tracking；可复用 GMM 部分 log-domain 技术 |
| `SparsePCA` / `MiniBatchSparsePCA` | 需要 sparse coding / coordinate descent 路线 |
| `DictionaryLearning` / `MiniBatchDictionaryLearning` | 和 sparse coding 共享优化器 |
| `LatentDirichletAllocation` | topic model，和当前 dense numeric backend 关系较远 |
| `UMAP` | 先保持 external benchmark；statgpu API 需单独立项 |
| `TSNE` | 先保持 openTSNE baseline；statgpu API 需单独立项 |
| `Isomap` / `LLE` / `MDS` | graph shortest path、local reconstruction、distance embedding 后续规划 |
| R `pam` / `clara` / `fanny` / `diana` / `mona` | robust clustering、medoid、fuzzy、divisive clustering 后续方向 |

## 4. Phase 3 实施计划

### 4.1 Phase 3A：性能与 scalable variants

目标：

- 稳定现有 PCA/KMeans/DBSCAN/GMM/NMF/Agglomerative 的 benchmark gate。
- 新增 `TruncatedSVD` 和 `MiniBatchKMeans`。

实现原则：

- production code 不调用 sklearn。
- CPU/CuPy/Torch 三端同步实现；不可行时显式报错并记录限制。
- benchmark 默认排除 host-to-device 搬运；另设单独端到端计时。

验收：

- `TruncatedSVD` 对齐 sklearn explained variance / projection / reconstruction metrics。
- `MiniBatchKMeans` 对齐 sklearn inertia 和 ARI，记录同 seed 与多 seed 稳定性。
- GPU benchmark 在中大样本上应明显快于 statgpu CPU；若未达到，记录原因。

### 4.2 Phase 3B：GMM covariance parity 与层次聚类扩展

目标：

- 扩展 `GaussianMixture` 支持 `spherical`、`tied`、`full` covariance。
- 扩展 `AgglomerativeClustering` 支持 `complete`、`average`、`ward` CPU exact。

实现原则：

- GMM 使用 log-domain EM，保持 `reg_covar` 和 lower bound 收敛语义。
- Agglomerative 先 CPU exact；GPU 路径继续显式报错，直到有独立实现。

验收：

- GMM 对齐 sklearn `GaussianMixture` 的 average log likelihood、AIC/BIC、predict_proba component-matched diff。
- Agglomerative 对齐 sklearn/SciPy/R labels、children/distances where comparable、ARI。

### 4.3 Phase 3C：graph/eigensolver 基础能力

目标：

- 为 `KernelPCA`、`SpectralClustering`、`SpectralEmbedding` 建立共用 graph/eigensolver 基础。

实现原则：

- 先 dense small/medium；大规模 sparse graph 后续单独规划。
- metric/kernel API 保持窄范围，避免过早复制 sklearn 全量参数。

验收：

- `KernelPCA` 对齐 sklearn transform / inverse_transform where supported。
- `SpectralClustering` 对齐 sklearn labels up to permutation / ARI。
- `SpectralEmbedding` 对齐 sklearn embedding subspace / trustworthiness。

## 5. Phase 4 研究方向

- `OPTICS` / `HDBSCAN`：密度聚类高级路线，先扩展 benchmark matrix，再决定 API。
- `UMAP` / `TSNE`：继续使用 umap-learn/openTSNE 做 comparison-only baseline；statgpu API 需要单独设计 neighbor graph、negative sampling、optimization 和 trustworthiness metrics。
- `FastICA` / `FactorAnalysis` / `SparsePCA` / `DictionaryLearning`：作为矩阵分解和 latent model 的后续批次。
- R `cluster` 系列：`pam`、`clara`、`fanny`、`diana`、`mona`、`daisy` 用作 robust / medoid / fuzzy / mixed-type clustering 规划来源。

## 6. External Validation

每个新增 estimator 的准入要求：

- 单元测试覆盖 shape、属性、错误路径、determinism、CPU/GPU consistency。
- 外部 baseline 至少包含 sklearn；涉及层次聚类补 SciPy/R；涉及 GPU 性能补 cuML where available。
- benchmark 输出 JSON 和 Markdown summary 到 `results/`。
- 显式记录 unavailable external packages，不伪造跳过项或性能结论。
- 远程 GPU 结论必须来自 Matpool 或等价 CUDA 环境，本地不作为 GPU 结论来源。

当前外部验证脚本：

- `dev/benchmarks/benchmark_unsupervised.py`
- `dev/benchmarks/benchmark_unsupervised_external_matrix.py`
- `dev/benchmarks/benchmark_unsupervised_phase2.py`
- `dev/benchmarks/benchmark_unsupervised_dbscan_cython.py`

新增 Phase 3 benchmark 建议：

- `dev/benchmarks/benchmark_unsupervised_phase3.py`
- 覆盖 `TruncatedSVD`、`MiniBatchKMeans`、GMM covariance variants、Agglomerative linkage variants。
- 输出字段统一包含：`method`、`framework`、`backend`、`status`、`fit_ms`、`fit_ms_all`、precision metrics、runtime notes、package availability。

## 7. 文档 Gate

新增或扩展无监督 estimator 时必须同步：

- `README.md`
- `USAGE.md`
- `USAGE_CN.md`
- `docs/en/unsupervised/`
- `docs/unsupervised/`
- `docs/en/models/README.md`
- `docs/models/README.md`
- `docs/en/benchmarks.md`
- `docs/benchmarks.md`
- `docs/en/changelog.md`
- `docs/changelog.md`

逐模型文档必须包含：

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

## 8. 参考来源

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
- R `clara`: https://www.rdocumentation.org/packages/cluster/versions/2.1.8.2/topics/clara
- R `agnes`: https://www.rdocumentation.org/packages/cluster/versions/2.1.8.1/topics/agnes

## 9. 下一步

建议下一轮实施顺序：

1. `TruncatedSVD`
2. `MiniBatchKMeans`
3. `GaussianMixture(spherical/tied/full)`
4. `AgglomerativeClustering(complete/average/ward)`
5. `KernelPCA` / `SpectralClustering` / `SpectralEmbedding`

完成上述前四项后，statgpu 将覆盖 sklearn 常用无监督能力的更大核心子集，同时保持 CPU/CuPy/Torch 三端一致性和外部 baseline 可审计性。
