# StatGPU 文档

> 语言：中文  
> 最后更新：2026-09-17  
> 切换：[English](../en/README.md)

## 快速开始

- [快速入门](getting-started/quickstart.md) — 安装、第一个模型、设备选择

## 指南

- [推断 API](guides/inference-api.md) — 分布函数、多重检验、排列检验、自助法
- [分布 API](guides/distribution-api.md) — 详细的分布函数与后端使用方法
- [已实现方法](guides/implemented-methods.md) — 公开模型与方法清单
- [交叉验证](guides/cross-validation.md) — 数据折、调参网格、参数选择与最终重拟合
- [statgpu 的交叉验证如何工作](guides/cross-validation-design.md) — 公开执行模型、加速思想与统计不变量
- [CoxPHCV 实验性筛选安全策略](guides/cox-cv-staged-safety.md) — 实验性筛选控制的用户可见行为
- [求解器 × 惩罚项兼容性矩阵](guides/solver-penalty-matrix.md) — 损失函数 × 惩罚项 × 求解器兼容性
- [设备与 GPU 内存](guides/device-and-memory.md) — 设备选择与内存清理
- [PyTorch 后端](guides/pytorch-backend.md) — PyTorch 后端使用说明
- [推断模式](guides/inference-modes.md) — 选择并解释系数推断方法
- [多重检验](guides/multiple-testing-combine-pvalues.md) — p 值校正与合并
- [基准测试](guides/benchmarks.md) — 基准测试方法与对比说明
- [交互式基准面板](../assets/benchmarks/index.html) — 筛选、图表、指标与数据来源

## 模型

### 线性模型族
- [LinearRegression](models/linear-regression.md) — OLS 与统计推断
- [Ridge](models/ridge.md) — Ridge 回归 + RidgeCV
- [Lasso](models/lasso.md) — Lasso + LassoCV + 去偏推断
- [ElasticNet](models/elastic-net.md) — ElasticNet + ElasticNetCV
- [SCAD](models/scad.md) — 非凸惩罚与 oracle 性质
- [MCP](models/mcp.md) — 非凸惩罚与 oracle 性质
- [AdaptiveLasso](models/adaptive-lasso.md) — 自适应 L1 惩罚

### 损失函数
- [损失函数概览](models/losses.md) — 底层损失函数定义与数值性质
- [分位数回归](models/quantile.md) — pinball 损失 + `PenalizedQuantileRegression`
- [稳健回归](models/robust.md) — Huber、Bisquare、Fair + `PenalizedRobustRegression`

### 广义线性模型
- [GeneralizedLinearModel](models/generalized-linear-model.md) — GLM + PenalizedGLM 基类
- [LogisticRegression](models/logistic-regression.md) — logistic 分类
- [PoissonRegression](models/poisson-regression.md) — 计数回归
- [有序模型](models/ordered.md) — ordered logit/probit

### 生存分析
- [CoxPH](models/coxph.md) — Breslow/Efron/Exact、延迟进入/起止时间数据、分层、稳健协方差与 NumPy/CuPy/Torch 路径
- [CoxPHCV](models/coxph.md) — L2 网格选择、最终重拟合与保持受试者完整的数据折
- [PenalizedCoxPHModel](models/coxph.md) — L1/L2/ElasticNet/SCAD/MCP Cox 估计；无截距

### 无监督学习
- [无监督概览](models/unsupervised.md) — PCA、聚类、混合模型、流形学习与矩阵分解方法

### 面板数据
- [Panel](models/panel.md) — 六类面板估计器，包括 pooled、between、first-difference 与 Fama–MacBeth

### 非参数
- [非参数概述](models/nonparametric.md) — 核方法与样条
- [核方法](models/kernel-methods.md) — KDE、核回归、KRR
- [样条](models/splines.md) — B 样条、自然样条、周期样条、薄板样条与 `SplineTransformer`
- [半参数（GAM）](models/semiparametric.md) — 广义可加模型

### 推断
- [ANOVA](models/anova.md) — 单/双因素、Welch、事后检验与效应量
- [协方差](models/covariance.md) — 经验/收缩协方差、稳健 MCD 与稀疏精度矩阵
- [多重检验](models/multiple-testing.md) — p 值校正与合并
- [Knockoff](models/knockoff.md) — knockoff 特征选择
- [特征选择](models/feature-selection.md) — 逐步选择与 knockoff 概览
- [回归诊断](guides/regression-diagnostics.md) — 残差、杠杆值、Cook 距离与 VIF

## 参考

- [求解器算法](guides/solver-algorithms.md) — 优化算法详解
- [损失函数 × 惩罚项 × 求解器框架](guides/loss-penalty-solver-framework.md) — 各组件如何组合与分发
- [L-BFGS Float32 数值行为](guides/lbfgs-float32-precision-contract.md) — 如何解释原生 float32 的跨后端差异
- [变更记录](changelog.md) — 版本历史
