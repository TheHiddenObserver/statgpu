# 模型总览

> 语言：中文  
> 最后更新：2026-09-17  
> 切换：[English](../../en/models/README.md)

本页只作为模型文档导航。当前求解器、惩罚项、后端与推断支持范围，以 [已实现方法](../guides/implemented-methods.md)、[求解器 × 惩罚项兼容性矩阵](../guides/solver-penalty-matrix.md) 和对应模型页为准。

## 核心参考

| 页面 | 内容 |
|---|---|
| [损失函数](losses.md) | 损失函数定义与数值性质 |
| [求解器算法](../guides/solver-algorithms.md) | 求解器更新公式与适用前提 |
| [损失函数 × 惩罚项 × 求解器框架](../guides/loss-penalty-solver-framework.md) | 组件分工与组合方式 |
| [求解器 × 惩罚项兼容性矩阵](../guides/solver-penalty-matrix.md) | 显式求解器与自动分发范围 |
| [推断 API](../guides/inference-api.md) | 分布、多重检验、重抽样与推断工具入口 |

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

### 如何选择 Cox 估计器

| 需求 | 估计器 | 导入路径 | 公开行为 |
|---|---|---|---|
| 完整 Cox 拟合、基准风险、生存预测、公式接口与统计推断 | `CoxPH` | `from statgpu.survival import CoxPH` | 支持 Breslow/Efron/Exact 并列事件、延迟进入、`(start, stop]` 数据、`strata`、稳健/聚类协方差以及 NumPy/CuPy/Torch 路径 |
| 通过留出部分似然选择非负 L2 惩罚 | `CoxPHCV` | `from statgpu.survival import CoxPHCV` | CV 期间保持 Cox 风险集语义，并使用选出的惩罚强度最终重拟合 `CoxPH` |
| 使用 L1、L2、ElasticNet、SCAD 或 MCP 进行参数估计 | `PenalizedCoxPHModel` | `from statgpu.linear_model import PenalizedCoxPHModel` | 提供更广的惩罚项与通用求解路径；当前只提供参数估计，`compute_inference=True` 会被拒绝 |

`CoxPH(penalty=...)` 与 `PenalizedCoxPHModel` 不是可以互换的别名。需要计数过程数据、`strata`、基准风险预测或统计推断时，应使用 `CoxPH` / `CoxPHCV`；如果主要需求是更广的惩罚项，并且只需要参数估计，可以使用 `PenalizedCoxPHModel`。

[Cox 模型页](coxph.md)统一说明 Breslow/Efron/Exact 并列事件、延迟进入与 `(start, stop]` 数据、`strata`、稳健/聚类推断、保持受试者完整的 CV、预测数值边界以及 NumPy/CuPy/Torch 支持范围。

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

## 支持范围的阅读方式

- NumPy、CuPy 与 Torch 是不同的执行后端；显式 `device` 请求不会静默切换到其他后端。
- 后端支持可能因求解器、惩罚项、推断方法和可选依赖而不同，应查看详细兼容性文档，而不要从某个模型族的概览自行推断。
- 具体模型的统计定义、输入限制、预测、推断和 CV 行为，以对应模型页为准。
