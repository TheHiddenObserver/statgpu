# Panel Models

> Language: English  
> Last updated: 2026-08-15  
> Switch: [Chinese](../../cn/models/panel.md)

`statgpu.panel` provides six panel-data estimators. Use this page to choose a model, then follow the linked page for the model definition, parameters, formula examples, CPU/GPU usage, inference, and diagnostics.

| Estimator | When to use it |
|---|---|
| [PanelOLS](../panel/panel-ols.md) | Linear panel regression with entity and/or time fixed effects. |
| [RandomEffects](../panel/random-effects.md) | One-way random-intercept model using the Swamy-Arora estimator. |
| [PooledOLS](../panel/pooled-ols.md) | One common OLS relationship for all stacked panel observations. |
| [BetweenOLS](../panel/between-ols.md) | Regression of entity averages; focuses on differences between entities. |
| [FirstDifferenceOLS](../panel/first-difference-ols.md) | Regression of within-entity changes between consecutive observed periods. |
| [FamaMacBeth](../panel/fama-macbeth.md) | Separate cross-sectional regressions by period followed by coefficient averaging. |

Shared statistical definitions are collected in [covariance](../panel/covariance.md), [fit statistics](../panel/fit-statistics.md), and [diagnostics](../panel/diagnostics.md).

All six estimators support NumPy CPU, CuPy CUDA, and Torch CUDA through the `device` parameter. Each estimator page includes CPU/GPU and formula examples. If `device="cuda"` or `device="torch"` is requested explicitly but that backend is unavailable, statgpu raises an error instead of silently switching to CPU.
