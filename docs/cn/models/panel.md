# 面板模型

> 语言：中文  
> 最后更新：2026-08-15  
> 切换：[English](../../en/models/panel.md)

`statgpu.panel` 提供六类面板估计器。可以先根据下表选择模型，再进入对应页面查看模型定义、参数、formula 示例、CPU/GPU 用法、统计推断和 diagnostics。

| 估计器 | 适用场景 |
|---|---|
| [PanelOLS](../panel/panel-ols.md) | 带 entity 和/或 time fixed effects 的线性面板回归。 |
| [RandomEffects](../panel/random-effects.md) | 使用 Swamy-Arora 方法的单向 random-intercept model。 |
| [PooledOLS](../panel/pooled-ols.md) | 所有 panel observations 共享同一组 OLS coefficient。 |
| [BetweenOLS](../panel/between-ols.md) | 对 entity 均值做回归，关注 entity 之间的长期平均差异。 |
| [FirstDifferenceOLS](../panel/first-difference-ols.md) | 对同一 entity 相邻已观测时期的变化做回归。 |
| [FamaMacBeth](../panel/fama-macbeth.md) | 每个时期分别做横截面回归，再对 coefficient 取平均。 |

共享统计定义见 [covariance](../panel/covariance.md)、[fit statistics](../panel/fit-statistics.md) 与 [diagnostics](../panel/diagnostics.md)。

六类 estimator 都可通过 `device` 使用 NumPy CPU、CuPy CUDA 或 Torch CUDA。每个模型页面都给出了 CPU/GPU 与 formula 示例。若显式指定 `device="cuda"` 或 `device="torch"`，但对应 backend 不可用，statgpu 会直接报错，而不是静默切换到 CPU。
