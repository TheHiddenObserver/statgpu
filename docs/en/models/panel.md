# Panel Models

> Language: English  
> Last updated: 2026-08-16  
> Switch: [Chinese](../../cn/models/panel.md)

`statgpu.panel` provides six panel-data estimators. These estimators should not be read as six unrelated data-generating processes. Several of them can be applied to the same underlying panel model but use different assumptions or different sources of variation to identify the coefficient of interest.

A useful way to distinguish them is:

| Estimator | Statistical view | Main source of identification / extra condition |
|---|---|---|
| [PanelOLS](../panel/panel-ols.md) | Entity and/or time effects are fixed but unknown nuisance parameters. | Uses variation remaining after the selected fixed effects are removed; no orthogonality between the fixed effects and regressor history is required. |
| [FirstDifferenceOLS](../panel/first-difference-ols.md) | The same fixed-parameter entity-effect model can be differenced to eliminate a time-invariant entity effect. | Uses within-entity changes between consecutive observed periods. |
| [RandomEffects](../panel/random-effects.md) | The entity effect is modeled as a random component. | Classical RE interpretation requires the random effect to be orthogonal to the regressor history, for example $E(a_i\mid X_i)=0$. |
| [BetweenOLS](../panel/between-ols.md) | Averages a panel model to one observation per entity. | Uses between-entity variation; recovering the same structural slope requires the averaged composite error to be orthogonal to the averaged regressors. |
| [PooledOLS](../panel/pooled-ols.md) | One common conditional-mean relationship is fitted to all stacked observations. | Uses all stacked variation; the combined regression error must be exogenous with respect to the regressors. |
| [FamaMacBeth](../panel/fama-macbeth.md) | Each time period has its own cross-sectional regression. | Targets the average of the retained period-specific slopes and bases uncertainty on their time-series variation. |

Each estimator page separates the **statistical model and identification assumptions** from the **numerical estimator**. The assumptions describe when the reported coefficient has the usual panel-econometric interpretation; the software can evaluate an estimator mechanically even when those substantive assumptions are not credible in a particular application.

Shared statistical definitions are collected in [covariance](../panel/covariance.md), [fit statistics](../panel/fit-statistics.md), and [diagnostics](../panel/diagnostics.md).

All six estimators support NumPy CPU, CuPy CUDA, and Torch CUDA through the `device` parameter. Each estimator page includes CPU/GPU and formula examples. If `device="cuda"` or `device="torch"` is requested explicitly but that backend is unavailable, statgpu raises an error instead of silently switching to CPU.

Panel fits are transactional. A new `fit()` attempt invalidates the previous fitted and inference state before work begins, and any exception during the new fit clears partially written outputs before it is re-raised. After a failed refit, `predict()` and `summary()` therefore report the estimator as unfitted rather than exposing coefficients or inference from either the previous data or an incomplete new fit. Formula-based prediction is also row-preserving: if Patsy would drop a prediction row because a modeled value is missing, or if a formula transformation produces NaN/Inf, prediction fails clearly instead of returning a shorter or non-finite result.
