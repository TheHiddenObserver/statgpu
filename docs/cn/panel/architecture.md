# 面板模型架构

> 语言：中文  
> 最后更新：2026-09-17  
> 页面定位：Panel 模型族的公开 architecture  
> 切换：[English](../../en/panel/architecture.md)

本页说明 statgpu 的 Panel estimator 如何组织：哪些职责由共享层承担，各 estimator 的数据变换在哪里进入，以及估计如何连接 covariance、inference、diagnostics、prediction 与 summary。

模型选择、识别假设、公式与统计解释应从 [面板模型总览](../models/panel.md)及各模型专页开始。内部 module ownership 与数值实现细节属于仓库 `dev/` architecture 文档。

## 1. 总体结构

用户通过以下 public Panel estimator 进行拟合：

- `PanelOLS`
- `RandomEffects`
- `PooledOLS`
- `BetweenOLS`
- `FirstDifferenceOLS`
- `FamaMacBeth`

这一模型族共享 `BasePanelModel` 基础设施，用来统一输入、状态、预测与 reporting 行为；具体 estimator 则负责定义该模型真正使用的统计变换或辅助估计。

高层流程可以概括为：

```text
输入 / formula / panel indices
        |
        v
共享 Panel 输入与 metadata 处理
        |
        v
estimator-specific transformation / regression construction
        |
        v
OLS / GLS / period-wise numerical estimation
        |
        v
covariance + inference + diagnostics
        |
        v
fitted state + predict() + summary()
```

关键边界是：共享基础设施并没有定义一个“统一 Panel estimator”；具体 model class 仍决定数据如何变换，以及最终求解哪一个统计估计问题。

## 2. 共享职责

`BasePanelModel` 与共享 Panel layer 提供可复用能力，包括：

- formula/input alignment；
- entity/time index metadata；
- backend/device preparation；
- fitted-state lifecycle；
- 共享 prediction 行为；
- result/summary plumbing；
- 在 estimator 可以写成 residual-OLS 形式时，对 covariance 与 coefficient inference 进行统一衔接。

共享 result object 还用于表示 panel index metadata、fit statistics 与 diagnostic-test result。

## 3. Estimator-specific construction

六类 estimator 复用相同基础设施，但构造不同的 estimation problem：

| Estimator | 主要数据构造 | 数值形式 |
|---|---|---|
| `PooledOLS` | 堆叠后的 level data | pooled OLS |
| `PanelOLS` | level data 或 entity/time/two-way within transformation | transformed OLS |
| `BetweenOLS` | entity means | 在 entity-level mean 上做 OLS |
| `FirstDifferenceOLS` | entity 内的一阶差分 | differenced OLS |
| `RandomEffects` | auxiliary regressions、variance components、quasi-demeaning | 通过 transformed regression 表示的 feasible GLS |
| `FamaMacBeth` | 每个 period 单独构造 cross-sectional regression | period-wise OLS 后聚合 coefficient |

这张表描述的是 estimation pipeline 的 architecture。各变换的统计推导与假设应放在对应模型文档，而不是继续扩展本页。

## 4. Estimation 与 inference layer

多数 Panel estimator 最终会形成 regression design、response、coefficient 与 residual。拟合后的统计层再按 estimator 需要组合：

- covariance estimation；
- coefficient standard error、statistic、p-value 与 confidence interval；
- degrees of freedom 与 fit statistics；
- model-specific diagnostics；
- 适用时的 effect recovery 或其他 model-specific state。

`FamaMacBeth` 与普通 residual-OLS model 在结构上不同，因为其 covariance 基于 period coefficient series，而不是只依赖一个堆叠 residual regression。即使共享外围基础设施，这一统计差别仍保持独立。

covariance 定义与 diagnostic 的统计解释应查看 Panel covariance/diagnostic 文档，本 architecture 页不再重复承担统计 reference 的职责。

## 5. Backend 边界

支持 NumPy、CuPy 与 Torch 的 Panel estimator 使用 [设备与 GPU 内存](../guides/device-and-memory.md)中统一的 `device` vocabulary。

显式 accelerator 请求保持显式：请求的 backend 不可用时直接报错，不会静默替换成 CPU。Formula parsing 或 metadata preparation 仍可以先在 CPU 上完成，然后再把数值数组准备到所选 backend。

具体 linear-algebra stabilization、rank detection、grouped reduction 与 implementation-specific numerical check 属于内部 numerical policy。用户应依赖公开 failure behavior 与模型输出，而不是 private helper 的组织方式。

## 6. Fit lifecycle

从用户角度看，Panel `fit()` 具有事务式语义：失败的 fit 不会留下看起来像成功模型的部分结果。成功拟合后，prediction、summary 与 inference property 都对应这一次成功发布的 fitted state。

Formula-based prediction 同样保持输入行对齐。如果 formula processing 会删除或使某些 prediction row 无效，statgpu 会报错，而不是返回与调用者输入行错位的输出。

## 7. 与通用 loss/penalty/solver framework 的关系

Panel model 以**panel-data construction + OLS/GLS/period-wise regression + Panel-specific post-fit statistics**组织计算。

这与 penalized objective-based estimator 使用的通用 `LossBase + Penalty + Solver` 组合方式不同。后者见 [Loss × Penalty × Solver 框架](../guides/loss-penalty-solver-framework.md)。

因此，没有额外的 generic `PanelLoss` layer 并不代表缺少某个 public capability；Panel estimator 通过各自的数据变换与 regression construction 表达自己的统计定义。

## 8. 下一步阅读

- [面板模型总览](../models/panel.md) — 模型选择与模型族概览
- 各 Panel 模型专页 — 公式、假设、参数、示例与结果解释
- [设备与 GPU 内存](../guides/device-and-memory.md) — device 语义
- Panel covariance/diagnostic 页面 — covariance estimator 与 specification test
- [Loss × Penalty × Solver 框架](../guides/loss-penalty-solver-framework.md) — objective-composed penalized model 的 architecture
