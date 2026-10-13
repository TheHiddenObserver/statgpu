# StatGPU 文档

> 语言：中文  
> 最后更新：2026-10-13  
> 切换：[English](../en/README.md)

## 快速开始

- [入门路径](usage.md) — 从安装到选择方法、检查结果的简明路线
- [快速入门](getting-started/quickstart.md) — 安装、第一次拟合、预测与设备选择

<a id="模型"></a>
<a id="线性模型族"></a>
<a id="广义线性模型"></a>
<a id="生存分析"></a>
<a id="无监督学习"></a>
<a id="面板数据"></a>
<a id="非参数"></a>
<a id="推断"></a>

## 模型与统计方法

通过[方法族选择入口](models/README.md)匹配要回答的问题。
各方法族入口链接到对应教程与完整 API 参考。

- 响应模型：[回归](models/regression/README.md)按响应类型与惩罚项选择；[生存分析](models/coxph.md)处理事件时间数据
- 统计问题：[检验](models/testing/README.md)处理组间比较与多个检验；[协方差](models/covariance/README.md)描述共同变化与精度矩阵
- 灵活拟合与变量选择：[非参数方法](models/nonparametric/README.md)涵盖密度、曲线与核特征；[特征选择](models/feature-selection/README.md)比较逐步与 knockoff 方法
- 数据结构：[面板数据](panel/README.md)处理个体/时间观测；[无监督学习](unsupervised/README.md)在没有响应变量时探索结构

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
- [回归诊断](guides/regression-diagnostics.md) — 残差、杠杆值、Cook 距离与 VIF
- [推断模式](guides/inference-modes.md) — 选择并解释系数推断方法
- [多重检验](guides/multiple-testing-combine-pvalues.md) — p 值校正与合并
- [基准测试](guides/benchmarks.md) — 基准测试方法与对比说明
- [交互式基准面板](../assets/benchmarks/index.html) — 筛选、图表、指标与数据来源

## 参考

### 完整 API 参考

- [完整协方差 API](reference/covariance-api.md) — 估计器、拟合矩阵、评分与后端限制
- [ANOVA API](reference/anova-api.md) — 七个函数、五种结果类型、默认值与输入及设备限制
- [完整核方法 API](reference/kernel-methods-api.md) — 四类估计器、成对核函数、形状及数值与后端限制
- [估计器通用 API](reference/estimator-api.md) — 参数管理与继承的推断辅助方法
- [线性模型 API](reference/linear-model-api.md) — 构造参数、拟合与预测方法、交叉验证及拟合结果
- [特征选择 API](reference/feature-selection-api.md) — 逐步选择与 knockoff 接口
- [生存分析与平滑 API](reference/survival-smoothing-api.md) — Cox、核方法、样条、带宽选择及 GAM
- [无监督学习 API](unsupervised/api-reference.md) — 十二类估计器的接口与拟合属性
- [分布 API](guides/distribution-api.md) — 支持的分布族、参数与数组语义
- [推断 API](guides/inference-api.md) — 结果对象、重采样与多重检验函数

<a id="损失函数"></a>

### 算法与版本记录

- [损失函数](reference/losses.md) — 底层损失函数定义与数值性质
- [惩罚模型求解器 API 迁移](guides/penalized-solver-api-migration.md) — 从旧求解器控制方式迁移
- [求解器算法](guides/solver-algorithms.md) — 优化算法详解
- [损失函数 × 惩罚项 × 求解器框架](guides/loss-penalty-solver-framework.md) — 各组件如何组合与分发
- [L-BFGS Float32 数值行为](guides/lbfgs-float32-precision-contract.md) — 如何解释原生 float32 的跨后端差异
- [变更记录](changelog.md) — 版本历史
