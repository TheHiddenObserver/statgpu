# StatGPU 文档

> 语言：中文  
> 切换：[English](../en/README.md)

## 快速开始

- [快速入门](getting-started/quickstart.md) — 安装、第一个模型、设备选择

## 指南

- [推断 API](guides/inference-api.md) — 分布函数、多重检验、排列检验、自助法
- [分布 API](guides/distribution-api.md) — 详细分布后端参考
- [已实现方法](guides/implemented-methods.md) — 完整方法列表（solver、penalty、link function）
- [交叉验证](guides/cross-validation.md) — CV API、架构、GPU 加速、缓存
- [求解器算法](guides/solver-algorithms.md) — 10 种求解器：算法、收敛条件、后端支持
- [Loss × Penalty × Solver 框架](guides/loss-penalty-solver-framework.md) — 完整架构、损失/惩罚/求解器覆盖矩阵
- [Solver × Penalty 矩阵](guides/solver-penalty-matrix.md) — loss × penalty × solver 兼容性
- [设备与 GPU 内存](guides/device-and-memory.md) — 设备选择、内存清理
- [PyTorch 后端](guides/pytorch-backend.md) — torch 后端指南、torch.compile
- [推断模式](guides/inference-modes.md) — Lasso 推断（debiased、bootstrap）
- [多重检验](guides/multiple-testing-combine-pvalues.md) — p 值校正与合并
- [基准测试](guides/benchmarks.md) — 性能基准与对比

## 模型

### 线性族
- [LinearRegression](models/linear-regression.md) — OLS + 推断
- [Ridge](models/ridge.md) — Ridge 回归 + RidgeCV
- [Lasso](models/lasso.md) — Lasso + LassoCV + debiased 推断
- [ElasticNet](models/elastic-net.md) — ElasticNet + ElasticNetCV
- [SCAD](models/scad.md) — 非凸惩罚，oracle property
- [MCP](models/mcp.md) — 非凸惩罚，oracle property
- [AdaptiveLasso](models/adaptive-lasso.md) — 自适应 L1 惩罚

### 损失函数 (v0.2.1)
- [损失函数概览](models/losses.md) — 12 种损失类型的架构
- [分位数回归](models/quantile.md) — pinball 损失 + PenalizedQuantileRegression
- [稳健回归](models/robust.md) — Huber、Bisquare、Fair + PenalizedRobustRegression

### 广义线性模型
- [GeneralizedLinearModel](models/generalized-linear-model.md) — GLM + PenalizedGLM 基类
- [LogisticRegression](models/logistic-regression.md) — logistic 分类
- [PoissonRegression](models/poisson-regression.md) — 计数回归
- [有序模型](models/ordered.md) — ordered logit/probit

### 生存分析
- [CoxPH](models/coxph.md) — Cox 比例风险 + 惩罚

### 无监督学习
- [无监督概览](models/unsupervised.md) — 13 种算法：PCA、KMeans、DBSCAN、GMM、UMAP、NNDescent、t-SNE、NMF、Agglomerative、TruncatedSVD、IncrementalPCA、MiniBatchKMeans、MiniBatchNMF

### 面板数据
- [Panel](models/panel.md) — 六类面板估计器，含 pooled、between、first-difference 与 Fama–MacBeth

### 非参数
- [非参数概述](models/nonparametric.md) — 核方法与样条
- [核方法](models/kernel-methods.md) — KDE、核回归、KRR
- [样条](models/splines.md) — B/自然/周期/薄板样条与 SplineTransformer
- [半参数 (GAM)](models/semiparametric.md) — 广义可加模型

### 推断
- [ANOVA](models/anova.md) — 单/双因素、Welch、事后检验与效应量
- [Covariance](models/covariance.md) — 经验/收缩、稳健 MCD 与稀疏精度矩阵
- [多重检验](models/multiple-testing.md) — P 值校正（BH、Holm、Bonferroni）和合并（Fisher、Cauchy、Stouffer）
- [Knockoff](models/knockoff.md) — knockoff 特征选择
- [特征选择](models/feature-selection.md) — stepwise 与 knockoff 总览
- [回归诊断](guides/regression-diagnostics.md) — 残差、杠杆值、Cook 距离与 VIF

## 参考

- [求解器算法](guides/solver-algorithms.md) — 10 种求解器：算法详解
- [Loss × Penalty × Solver 框架](guides/loss-penalty-solver-framework.md) — 调度逻辑
- [变更记录](changelog.md) — 版本历史
