# Panel Models

> Language: English  
> Last updated: 2026-08-13  
> Switch: [Chinese](../../cn/models/panel.md)

`statgpu.panel` provides six panel-data estimators. This page is intentionally navigation-only; equations and API details live on the linked pages.

| Estimator | Model |
|---|---|
| [PanelOLS](../panel/panel-ols.md) | Fixed effects |
| [RandomEffects](../panel/random-effects.md) | Swamy-Arora random effects |
| [PooledOLS](../panel/pooled-ols.md) | Pooled OLS |
| [BetweenOLS](../panel/between-ols.md) | Between regression |
| [FirstDifferenceOLS](../panel/first-difference-ols.md) | First differences |
| [FamaMacBeth](../panel/fama-macbeth.md) | Period-wise cross-sectional regression |

Shared references: [covariance](../panel/covariance.md), [fit statistics](../panel/fit-statistics.md), and [diagnostics](../panel/diagnostics.md).

Numerical paths use NumPy, CuPy CUDA, or Torch CUDA. Formula evaluation and categorical labels are CPU metadata boundaries; explicit GPU requests do not silently fall back to CPU.
