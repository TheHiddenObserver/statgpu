# 面板模型

> 语言：中文  
> 最后更新：2026-08-13  
> 切换：[English](../../en/models/panel.md)

`statgpu.panel` 提供六类面板估计器。模型公式与 API 说明放在独立页面。

| 估计器 | 模型 |
|---|---|
| [PanelOLS](../panel/panel-ols.md) | 固定效应 |
| [RandomEffects](../panel/random-effects.md) | Swamy-Arora 随机效应 |
| [PooledOLS](../panel/pooled-ols.md) | Pooled OLS |
| [BetweenOLS](../panel/between-ols.md) | Between 回归 |
| [FirstDifferenceOLS](../panel/first-difference-ols.md) | 一阶差分 |
| [FamaMacBeth](../panel/fama-macbeth.md) | 分期横截面回归 |

共享定义见 [covariance](../panel/covariance.md) 与 [diagnostics](../panel/diagnostics.md)。

数值路径使用 NumPy、CuPy CUDA 或 Torch CUDA；formula 与 categorical label 属于 CPU 元数据边界。
