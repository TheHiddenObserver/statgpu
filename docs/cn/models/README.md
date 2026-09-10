# 模型总览

> 语言：中文
> 最后更新：2026-09-03
> 当前软件包版本：**0.2.5**
> 切换：[English](../../en/models/README.md)

本目录反映当前公开的模型家族。左侧“模型参考”目录可直接进入每个模型页面；
后端、求解器与惩罚项覆盖范围请同时参阅[已实现方法](../guides/implemented-methods.md)。

## 回归与广义线性模型

| 模型家族 | 公开 API | 文档 |
|---|---|---|
| 普通最小二乘 | `LinearRegression` | [线性回归](linear-regression.md) |
| 高斯正则化 | `Ridge`、`Lasso`、`ElasticNet` 及其 CV 版本 | [Ridge](ridge.md)、[Lasso](lasso.md)、[Elastic Net](elastic-net.md) |
| 结构化与非凸正则化 | `AdaptiveLasso`、`SCADRegression`、`MCPRegression` | [Adaptive Lasso](adaptive-lasso.md)、[SCAD](scad.md)、[MCP](mcp.md) |
| 广义线性模型 | `GeneralizedLinearModel`、`GammaRegression`、`InverseGaussianRegression`、`NegativeBinomialRegression`、`TweedieRegression` | [GLM](generalized-linear-model.md) |
| 分类与计数 | `LogisticRegression`、`LogisticRegressionCV`、`PoissonRegression` | [Logistic](logistic-regression.md)、[Poisson](poisson-regression.md) |
| 有序响应 | `OrderedLogitRegression`、`OrderedProbitRegression` | [有序模型](ordered.md) |
| 分位数与稳健拟合 | `QuantileRegression`、`PenalizedQuantileRegression`、`PenalizedRobustRegression` | [分位数回归](quantile.md)、[稳健回归](robust.md) |

统一惩罚模型 API 还通过 `statgpu.linear_model` 提供 Gaussian、Logistic、
Poisson、Gamma、Inverse-Gaussian、Negative-Binomial、Tweedie、稳健、
分位数与 Cox 等家族。

## 生存分析

| 需求 | Estimator | 契约 |
|---|---|---|
| 完整 Cox 拟合、预测、formula 与统计推断 | `CoxPH` | Breslow/Efron/Exact ties、delayed entry、分层与 robust/cluster 协方差 |
| 通过 held-out partial likelihood 选择 L2 | `CoxPHCV` | 保持受试者完整性的 CV，并最终重拟合 `CoxPH` |
| 更广的凸/非凸惩罚项 | `PenalizedCoxPHModel` | 面向估计的通用惩罚路径 |

精确功能与后端矩阵见 [Cox 比例风险模型](coxph.md)。

## 专业统计模块

| 领域 | 文档 |
|---|---|
| 方差分析与事后检验 | [ANOVA](anova.md) |
| 协方差估计 | [协方差](covariance.md) |
| 面板与纵向模型 | [面板数据](panel.md) |
| 非参数估计 | [非参数方法](nonparametric.md) |
| 核模型 | [核方法](kernel-methods.md) |
| 样条基函数 | [样条](splines.md) |
| GAM 与半参数模型 | [半参数模型](semiparametric.md) |
| 特征选择与诊断 | [特征选择](feature-selection.md) |
| Knockoff 筛选 | [Knockoff](knockoff.md) |
| 多重检验 | [多重检验](multiple-testing.md) |
| 损失函数定义 | [损失函数](losses.md) |

## 无监督学习

| 家族 | 模型 |
|---|---|
| 矩阵分解 | [PCA](../unsupervised/pca.md)、[Incremental PCA](../unsupervised/incremental-pca.md)、[Truncated SVD](../unsupervised/truncated-svd.md)、[NMF](../unsupervised/nmf.md)、[MiniBatch NMF](../unsupervised/minibatch-nmf.md) |
| 聚类 | [K-Means](../unsupervised/kmeans.md)、[MiniBatch K-Means](../unsupervised/minibatch-kmeans.md)、[层次聚类](../unsupervised/agglomerative-clustering.md)、[DBSCAN](../unsupervised/dbscan.md)、[高斯混合](../unsupervised/gaussian-mixture.md) |
| 流形学习 | [t-SNE](../unsupervised/tsne.md)、[UMAP](../unsupervised/umap.md) |

共用 API 约定见[无监督学习总览](../unsupervised/)。

## 面板数据

面板 API 现已提供完整的 estimator、协方差、拟合统计量与诊断文档：

- [Pooled OLS](../panel/pooled-ols.md)、[固定效应 Panel OLS](../panel/panel-ols.md)、
  [Between OLS](../panel/between-ols.md)、[随机效应](../panel/random-effects.md)、
  [一阶差分 OLS](../panel/first-difference-ols.md)与
  [Fama-MacBeth](../panel/fama-macbeth.md)
- [协方差估计](../panel/covariance.md)、
  [拟合统计量](../panel/fit-statistics.md)与
  [模型诊断](../panel/diagnostics.md)

## 核心框架

- [求解器算法](../guides/solver-algorithms.md)
- [Loss × Penalty × Solver 框架](../guides/loss-penalty-solver-framework.md)
- [Solver × Penalty 矩阵](../guides/solver-penalty-matrix.md)
- [推断 API](../guides/inference-api.md)

后端支持可能因模型、求解器、惩罚项、推断方法和可选依赖而异。性能与验证结论
始终限定到对应证据记录的 commit、硬件、后端与数据集。
