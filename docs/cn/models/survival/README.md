# 生存分析

> 语言：中文  
> 最后更新：2026-10-17  
> 切换：[English](../../../en/models/survival/README.md)

[模型与统计方法](../README.md) / 生存分析

当响应是**事件发生前的时间**，而部分样本在观察结束时仍未发生事件，可以使用生存分析。
输入同时包含观察时长和事件指示，因此尚未发生事件的删失样本也能提供信息。
Cox 回归研究预测变量与事件风险之间的关系，不把删失时间当作普通的连续响应。

当前教程围绕 Cox 比例风险模型展开。先阅读 [Cox 教程](../coxph.md)，理解假设、输入含义并完成首次 CPU 拟合，
再根据所需结果和惩罚项选择估计器。

## 如何选择 Cox 估计器

| 需求 | 估计器 | 导入路径 | 公开行为 |
|---|---|---|---|
| 完整 Cox 拟合、基准风险、生存预测、公式接口与统计推断 | `CoxPH` | `from statgpu.survival import CoxPH` | 支持 Breslow/Efron/Exact 并列事件、延迟进入、`(start, stop]` 数据、`strata`、稳健/聚类协方差以及 NumPy/CuPy/Torch 路径 |
| 通过留出部分似然选择非负 L2 惩罚 | `CoxPHCV` | `from statgpu.survival import CoxPHCV` | CV 期间保持 Cox 风险集语义，并使用选出的惩罚强度最终重拟合 `CoxPH` |
| 使用 L1、L2、ElasticNet、SCAD 或 MCP 进行参数估计 | `PenalizedCoxPHModel` | `from statgpu.linear_model import PenalizedCoxPHModel` | 提供更广的惩罚项与通用求解路径；当前只提供参数估计，`compute_inference=True` 会被拒绝 |

`CoxPH(penalty=...)` 与 `PenalizedCoxPHModel` 不是可以互换的别名。需要计数过程数据、`strata`、基准风险预测或统计推断时，应使用 `CoxPH` / `CoxPHCV`；如果主要需求是更广的惩罚项，并且只需要参数估计，可以使用 `PenalizedCoxPHModel`。

[Cox 模型页](../coxph.md)统一说明 Breslow/Efron/Exact 并列事件、延迟进入与 `(start, stop]` 数据、`strata`、稳健/聚类推断、保持受试者完整的 CV、预测数值边界以及 NumPy/CuPy/Torch 支持范围。

## 继续分析

- [Cox 比例风险模型](../coxph.md)：完整示例、预测、推断与交叉验证
- [CoxPH 与 CoxPHCV API](../../reference/survival-smoothing-api.md#coxph-与-coxphcv)：构造参数、方法和拟合属性
- [其他 Cox 惩罚项及交叉验证](../coxph.md#使用其他惩罚l1l2elasticnetscad-与-mcp)：带惩罚拟合与调参
- [CoxPH 设备与 CV 诊断](../../reference/coxph-diagnostics.md)：设备控制与调参诊断
- [回归概览](../regression/README.md)：其他响应类型的建模方法
