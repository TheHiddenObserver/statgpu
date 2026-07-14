# 模型总览

> 语言：中文
>
> 最后更新：2026-07-12
>
> 切换：[English](../../en/models/README.md)

---

## 核心框架

| 页面 | 内容 |
|------|------|
| [损失函数 (LossBase)](losses.md) | 架构概述：12 种损失类型，逐样本公式 |
| [求解器算法](../guides/solver-algorithms.md) | 10 种求解器：算法步骤、收敛条件、后端支持 |
| [Loss × Penalty × Solver 框架](../guides/loss-penalty-solver-framework.md) | 完整调度逻辑与覆盖矩阵 |
| [Solver × Penalty 矩阵](../guides/solver-penalty-matrix.md) | 求解器路由与惩罚约束 |

---

## 损失函数

| 损失 | 页面 | 惩罚模型 | 核心求解器 |
|------|------|-----------------|------------|
| Quantile | [quantile.md](quantile.md) | `PenalizedQuantileRegression` | Proximal IRLS-CD |
| Huber | [robust.md](robust.md) | `PenalizedRobustRegression` | Proximal Newton |
| Bisquare | [robust.md](robust.md) | `PenalizedRobustRegression` | Proximal Newton |
| Fair | [robust.md](robust.md) | `PenalizedRobustRegression` | Proximal Newton |
| Cox PH | [coxph.md](coxph.md) | `PenalizedCoxPHModel` | FISTA / FISTA-LLA |
| GLM (7 家族) | [losses.md](losses.md) | `PenalizedGeneralizedLinearModel` | IRLS / Newton / FISTA |

---

## 回归与 GLM

| 模型 | 页面 | 惩罚 |
|-------|------|---------|
| LinearRegression | [linear-regression.md](linear-regression.md) | — |
| Ridge | [ridge.md](ridge.md) | L2 |
| Lasso | [lasso.md](lasso.md) | L1 |
| ElasticNet | [elastic-net.md](elastic-net.md) | L1 + L2 |
| SCAD | [scad.md](scad.md) | SCAD（非凸） |
| MCP | [mcp.md](mcp.md) | MCP（非凸） |
| AdaptiveLasso | [adaptive-lasso.md](adaptive-lasso.md) | 加权 L1 |
| LogisticRegression | [logistic-regression.md](logistic-regression.md) | L2 |
| PoissonRegression | [poisson-regression.md](poisson-regression.md) | — |
| GeneralizedLinearModel | [generalized-linear-model.md](generalized-linear-model.md) | 全部惩罚 |
| Ordered (Logit/Probit) | [ordered.md](ordered.md) | — | Newton-Raphson + 解析 Hessian 推断 |

---

## 生存分析

| 模型 | 页面 | 特性 |
|-------|------|----------|
| `CoxPH` | [coxph.md](coxph.md) | Breslow/Efron/Exact、start-stop、strata、subject、稳健推断与 Breslow baseline |
| `CoxPHCV` | [coxph.md](coxph.md) | 三后端 L2 部分似然 CV，支持 Exact 与计数过程轴 |
| `PenalizedCoxPHModel` | [coxph.md](coxph.md) | L1/L2/Elastic Net/SCAD/MCP；无截距、仅估计 |

---

## 无监督学习

| 模型 | 页面 | 备注 |
|-------|------|-------|
| PCA | [unsupervised.md](unsupervised.md) | 线性降维 |
| KMeans | [unsupervised.md](unsupervised.md) | Lloyd k-means++ |
| DBSCAN | [unsupervised.md](unsupervised.md) | Torch CUDA on-device, CuPy + host syncs |
| GaussianMixture | [unsupervised.md](unsupervised.md) | Log-domain EM |
| UMAP | [unsupervised.md](unsupervised.md) | 稀疏 COO 图, backend-aware 负采样 |
| NNDescent | [unsupervised.md](unsupervised.md) | 近似最近邻, 逐点候选集 |
| TSNE | [unsupervised.md](unsupervised.md) | KL divergence |
| 其他 | [unsupervised.md](unsupervised.md) | NMF, IncrementalPCA, TruncatedSVD, Agglomerative |

---

## 专业模块

| 领域 | 页面 |
|--------|------|
| ANOVA | [anova.md](anova.md) |
| 协方差估计 | [covariance.md](covariance.md) |
| 面板数据 | [panel.md](panel.md) |
| 非参数 (KDE, 核回归) | [nonparametric.md](nonparametric.md) |
| 核岭回归 | [kernel-methods.md](kernel-methods.md) |
| 样条基函数 | [splines.md](splines.md) |
| GAM (半参数) | [semiparametric.md](semiparametric.md) |
| Knockoff (特征选择) | [knockoff.md](knockoff.md) |
| 多重检验 | [multiple-testing.md](multiple-testing.md) |

---

## v0.2.1 覆盖摘要

| 类别 | 详情 |
|----------|---------|
| 损失类型 | 12 种：7 GLM + quantile + huber + bisquare + fair + cox_ph |
| 惩罚 | 10 种：l1, l2, elasticnet, scad, mcp, adaptive_l1, group_lasso, group_mcp, group_scad |
| 求解器 | 10 种：exact, irls, newton, lbfgs, fista, fista_bb, fista_lla, proximal_irls_cd, proximal_newton, admm |
| 后端 | numpy, cupy, torch — 核心求解器均三端支持 |
| GPU 回退 | 显式 GPU 设备不静默回退 CPU |
| sample_weight | IRLS/FISTA 路径支持；有序模型、CoxPH 和 GLM Newton/LBFGS 不支持 |
| CV | LassoCV, RidgeCV, LogisticRegressionCV, CoxPHCV, PenalizedGLM_CV；CoxPHCV 支持 NumPy/CuPy/Torch |
| 推断 | nonrobust/HC0/HC1 (sandwich), HC2/HC3/HAC (仅 Gaussian), bootstrap, debiased Lasso, analytical Hessian (ordered)；CoxPH 支持 nonrobust/HC0/HC1/cluster，Exact 仅 nonrobust |
