# 模型总览

> 语言：中文
> 最后更新：2026-05-08
> 页面定位：模型索引
> English: [English](../en/models/README.md)

本节按方法类别组织文档，便于后续新增统计方法时持续扩展。

## 线性与 GLM 模型

- [LinearRegression](linear-regression.md)
- [GeneralizedLinearModel 与 Penalized GLM](generalized-linear-model.md)
- [PoissonRegression](poisson-regression.md)
- [Ridge](ridge.md)
- [Lasso](lasso.md)
- [ElasticNet](elastic-net.md)
- [LogisticRegression](logistic-regression.md)
- [Ordered Generalized Linear Models (Logit/Probit)](ordered.md)

## 生存分析

- [CoxPH](coxph.md)

## 特征选择

- [Knockoff](knockoff.md)

## 非参数方法

- [Nonparametric](nonparametric.md)

## 无监督学习

- [无监督学习兼容入口](unsupervised.md)
- [无监督学习详细文档](../unsupervised/README.md)
- 降维与矩阵分解：[PCA](../unsupervised/pca.md)、[TruncatedSVD](../unsupervised/truncated-svd.md)、[IncrementalPCA](../unsupervised/incremental-pca.md)、[NMF](../unsupervised/nmf.md)、[MiniBatchNMF](../unsupervised/minibatch-nmf.md)
- 聚类与混合模型：[KMeans](../unsupervised/kmeans.md)、[MiniBatchKMeans](../unsupervised/minibatch-kmeans.md)、[DBSCAN](../unsupervised/dbscan.md)、[GaussianMixture](../unsupervised/gaussian-mixture.md)、[AgglomerativeClustering](../unsupervised/agglomerative-clustering.md)
- Manifold embedding：[UMAP](../unsupervised/umap.md)、[TSNE](../unsupervised/tsne.md)

## 新增模型文档流程

新增一个 estimator 时：

1. 创建 `docs/models/<model-name>.md` 和对应 `docs/en/models/<model-name>.md`。
2. 将新页面加入本索引和英文索引。
3. 将入口链接同步加入 `USAGE_CN.md` 和 `USAGE.md`。
4. 若包含外部验证或 benchmark，同步在 `docs/benchmarks.md` 与 `docs/en/benchmarks.md` 添加脚本和结果 artifact。
5. 模型页应符合 `PLAN_UNIFIED.md` 模板：Overview、Path、Objective Function、Estimating Equation、Covariance/Inference、Parameters、CPU+GPU Examples、strict/approx difference、Outputs、FAQ、External Validation、References。

## 当前覆盖说明

- 设备支持是模型级能力，不是所有模型都支持所有后端。显式 `device="cuda"` 或 `device="torch"` 在依赖不可用或模型不支持时应报错，不应静默 fallback 到 CPU。
- `GeneralizedLinearModel` 与 typed penalized GLM 见 [GeneralizedLinearModel 与 Penalized GLM](generalized-linear-model.md)。
- `PoissonRegression` 作为普通 Poisson GLM estimator 单独记录。
- `statgpu.unsupervised` 包含降维、聚类、混合模型、矩阵分解和 manifold estimator。当前 public estimators 提供 CPU/CuPy/Torch 路径，具体 dense/exact 或 iterative 限制见逐模型页面。无监督学习详细文档包含逐模型目标函数、估计过程、设备说明、示例和外部验证 artifact。
- 推断能力较完整的模型：
  - `LinearRegression`：classical + `HC0/HC1/HC2/HC3/HAC`
  - `Ridge`：classical + `HC0/HC1/HC2/HC3/HAC`
  - `Lasso`：CPU/GPU OLS 风格推断 + bootstrap
  - `LogisticRegression`：classical + `HC0/HC1/HC2/HC3/HAC`
- `CoxPH` 支持 Breslow/Efron ties，并提供 CPU/GPU 拟合路径。
- `OrderedLogitRegression` / `OrderedProbitRegression` 支持 CPU/CuPy/Torch 后端，跨后端精度已修复。
- `CoxPH` 的 delayed entry 路径：
  - `entry + breslow`：CPU/CUDA/Torch
  - `entry + efron`：CPU/CUDA/Torch
- 特征选择：
  - `Knockoff`：fixed-X / model-X 统一 API + selector 封装。
- `LassoCV` 已实现并可直接训练。
- 已导出的 CV 类：
  - `RidgeCV`、`LogisticRegressionCV`、`CoxPHCV` 均已可直接训练。
  - `CoxPHCV` 当前边界：GPU 下 `entry` 目前仅支持 `ties='breslow'`；`cluster` robust CV 暂未支持，会抛出 `NotImplementedError`。
