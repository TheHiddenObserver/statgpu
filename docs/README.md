# StatGPU 文档

> 语言：中文  
> 切换：[English](en/README.md)

## 快速开始

- [快速入门](getting-started/quickstart.md) — 安装、第一个模型、设备选择

## 指南

- [交叉验证](guides/cross-validation.md) — CV API、架构、GPU 加速、缓存
- [Solver × Penalty 矩阵](guides/solver-penalty-matrix.md) — loss × penalty × solver 兼容性
- [设备与 GPU 内存](guides/device-and-memory.md) — 设备选择、内存清理
- [PyTorch 后端](guides/pytorch-backend.md) — torch 后端指南、torch.compile
- [分布 API](guides/distribution-api.md) — 统计分布函数
- [推断模式](guides/inference-modes.md) — Lasso 推断（debiased、bootstrap）
- [多重检验](guides/multiple-testing-combine-pvalues.md) — p 值校正与合并
- [基准测试](guides/benchmarks.md) — 性能基准与对比

## 模型

### 线性族
- [LinearRegression](models/linear-regression.md) — OLS + 推断
- [Ridge](models/ridge.md) — Ridge 回归 + RidgeCV
- [Lasso](models/lasso.md) — Lasso + LassoCV + debiased 推断
- [ElasticNet](models/elastic-net.md) — ElasticNet + ElasticNetCV

### 广义线性模型
- [GeneralizedLinearModel](models/generalized-linear-model.md) — GLM + PenalizedGLM 基类
- [LogisticRegression](models/logistic-regression.md) — logistic 分类
- [PoissonRegression](models/poisson-regression.md) — 计数回归
- [有序模型](models/ordered.md) — ordered logit/probit

### 生存分析
- [CoxPH](models/coxph.md) — Cox 比例风险

### 面板数据
- [Panel](models/panel.md) — 固定/随机效应面板模型

### 非参数
- [非参数概述](models/nonparametric.md) — 核方法与样条
- [核方法](models/kernel-methods.md) — KDE、核回归、KRR
- [样条](models/splines.md) — B 样条基、惩罚样条
- [半参数 (GAM)](models/semiparametric.md) — 广义可加模型

### 推断
- [ANOVA](models/anova.md) — 方差分析
- [Covariance](models/covariance.md) — 协方差估计、收缩
- [Knockoff](models/knockoff.md) — knockoff 特征选择

## 参考

- [变更记录](changelog.md) — 版本历史
