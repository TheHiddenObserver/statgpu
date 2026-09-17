# statgpu 文档入口（中文）

> 语言：中文  
> 最后更新：2026-09-17  
> 切换：[English](../en/usage.md)

本页只作为用户文档入口。容易变化的支持矩阵放在对应模型页与 reference 页中，避免入口页重复保存一套实现状态。

## 快速开始

- [快速入门](getting-started/quickstart.md) — 安装、第一次拟合、预测与设备选择
- [已实现方法](guides/implemented-methods.md) — public estimator 与方法清单
- [设备与 GPU 内存](guides/device-and-memory.md) — CPU/CUDA/Torch 行为与内存控制
- [交叉验证](guides/cross-validation.md) — fold、tuning grid、选择与最终重拟合
- [推断模式](guides/inference-modes.md) — 选择并解释系数推断方法
- [推断 API](guides/inference-api.md) — 分布函数、多重检验、排列检验与 bootstrap 工具
- [变更记录](changelog.md) — 版本历史

CuPy 请按 CUDA 主版本安装 `statgpu[gpu11]` 或 `statgpu[gpu12]`；PyTorch 后端使用 `statgpu[torch]`。基础安装已经包含 Cox 实现，`statgpu[survival]` 增加的是可选的外部比较依赖。

## 模型族

- [模型总览](models/README.md)
- [线性回归](models/linear-regression.md)
- [Ridge](models/ridge.md)
- [Lasso](models/lasso.md)
- [ElasticNet](models/elastic-net.md)
- [广义线性模型](models/generalized-linear-model.md)
- [分位数回归](models/quantile.md)
- [稳健回归](models/robust.md)
- [Cox 比例风险模型](models/coxph.md)
- [面板模型](models/panel.md)
- [ANOVA](models/anova.md)
- [协方差估计](models/covariance.md)
- [非参数方法](models/nonparametric.md)
- [无监督学习](models/unsupervised.md)
- [特征选择](models/feature-selection.md)
- [回归诊断](guides/regression-diagnostics.md)

## 优化与兼容性参考

- [损失函数](models/losses.md) — 底层 loss 定义与数值性质
- [Loss × Penalty × Solver 框架](guides/loss-penalty-solver-framework.md) — 各组件如何组合
- [Solver × Penalty 矩阵](guides/solver-penalty-matrix.md) — 支持与不支持的组合
- [求解器算法](guides/solver-algorithms.md) — 算法定义
- [Penalized Solver API 迁移](guides/penalized-solver-api-migration.md) — 从旧 solver 控制迁移

## 统计工具

- [分布 API](guides/distribution-api.md)
- [多重检验](guides/multiple-testing-combine-pvalues.md)
- [ANOVA](models/anova.md)
- [协方差估计](models/covariance.md)

开发、贡献、测试与仓库内部 architecture 请查看根目录 `CONTRIBUTING.md` 与 `dev/` 文档，不再混入用户使用指南。
