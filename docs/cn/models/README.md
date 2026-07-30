# 模型总览

> 语言：中文  
> 最后更新：2026-07-30  
> 切换：[English](../../en/models/README.md)

本页仅作为导航。当前 solver、penalty、后端与推断覆盖以
[已实现方法](../guides/implemented-methods.md)和对应模型页为准。

## 核心框架

| 页面 | 内容 |
|---|---|
| [损失函数](losses.md) | Loss 定义与逐样本公式 |
| [求解器算法](../guides/solver-algorithms.md) | 公开与内部 solver 实现 |
| [Loss × Penalty × Solver 框架](../guides/loss-penalty-solver-framework.md) | 调度逻辑与兼容范围 |
| [Solver × Penalty 矩阵](../guides/solver-penalty-matrix.md) | 显式 solver 路由与限制 |
| [推断 API](../guides/inference-api.md) | 协方差、重抽样与推断接口 |

## 回归与 GLM

- [线性回归](linear-regression.md)
- [Ridge](ridge.md)
- [Lasso](lasso.md)
- [Elastic Net](elastic-net.md)
- [Adaptive Lasso](adaptive-lasso.md)
- [SCAD](scad.md)
- [MCP](mcp.md)
- [Logistic 回归](logistic-regression.md)
- [Poisson 回归](poisson-regression.md)
- [广义线性模型](generalized-linear-model.md)
- [Ordered Logit/Probit](ordered.md)
- [分位数回归](quantile.md)
- [稳健回归](robust.md)

## 生存分析

- [Cox 比例风险模型](coxph.md)

### 如何选择 Cox estimator

| 需求 | Estimator | 导入路径 | 当前契约 |
|---|---|---|---|
| 完整 Cox 拟合、baseline hazard、生存预测、formula 与统计推断 | `CoxPH` | `from statgpu.survival import CoxPH` | Breslow/Efron/Exact ties、delayed entry、`(start, stop]`、strata、robust/cluster 协方差、NumPy/CuPy/Torch |
| 通过 held-out partial likelihood 选择非负 L2 penalty | `CoxPHCV` | `from statgpu.survival import CoxPHCV` | CV 期间沿用 canonical Cox 语义，并使用最佳 penalty 最终重拟合 `CoxPH` |
| 使用 L1、L2、ElasticNet、SCAD 或 MCP 进行估计 | `PenalizedCoxPHModel` | `from statgpu.linear_model import PenalizedCoxPHModel` | 广义 penalty 与通用 solver 路径；当前仅支持 estimation，`compute_inference=True` 会被拒绝 |

`CoxPH(penalty=...)` 与 `PenalizedCoxPHModel` 不是可互换的别名。需要
counting-process、strata、baseline prediction 或统计推断时，应使用 canonical
`CoxPH`/`CoxPHCV` 路径；主要需求是更广的 penalty family 且 estimation-only
结果足够时，使用 `PenalizedCoxPHModel`。

[Cox 模型页](coxph.md)是 Breslow/Efron/Exact ties、delayed-entry 与
`(start, stop]` 数据、strata、robust/cluster 推断、subject-grouped CV、预测数值边界
以及 NumPy/CuPy/Torch 支持矩阵的用户侧权威来源。内部模块 ownership、调用图与扩展
规则见 [`dev/design/ARCHITECTURE.md`](../../../dev/design/ARCHITECTURE.md#5-survival--cox-architecture)。

## 专业统计模块

- [ANOVA](anova.md)
- [协方差估计](covariance.md)
- [面板数据](panel.md)
- [非参数方法](nonparametric.md)
- [核方法](kernel-methods.md)
- [样条基函数](splines.md)
- [GAM / 半参数模型](semiparametric.md)
- [特征选择](feature-selection.md)
- [Knockoff](knockoff.md)
- [多重检验](multiple-testing.md)

## 无监督学习

- [无监督学习总览](unsupervised.md)
- [PCA](../unsupervised/pca.md)
- [Truncated SVD](../unsupervised/truncated-svd.md)
- [Incremental PCA](../unsupervised/incremental-pca.md)
- [NMF](../unsupervised/nmf.md)
- [MiniBatch NMF](../unsupervised/minibatch-nmf.md)
- [DBSCAN](../unsupervised/dbscan.md)
- [UMAP](../unsupervised/umap.md)
- [t-SNE](../unsupervised/tsne.md)

## 当前覆盖原则

- NumPy、CuPy 与 Torch 是不同执行后端；显式 device 请求不得静默选择其他后端。
- 后端支持可能因 solver、penalty、推断方法及可选依赖而不同，应查看详细兼容矩阵，
  而不是依赖单一固定数量。
- 验证结论应限定到实际测试的模型、后端、硬件与 commit。
- 历史 release 与 benchmark 记录是证据快照，不是当前支持矩阵。
