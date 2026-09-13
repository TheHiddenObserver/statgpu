# Panel Models

> Language: English  
> Last updated: 2026-09-13  
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

## Documentation map

- [Panel Architecture](../panel/architecture.md) — current `BasePanelModel`, estimator-specific data transformations and regression problems, shared numerical linear algebra, covariance/inference, diagnostics, and fit-lifecycle boundaries.
- [Covariance](../panel/covariance.md) — nonrobust, HC, cluster, HAC, and Driscoll-Kraay covariance definitions.
- [Fit statistics](../panel/fit-statistics.md) — within/between/overall $R^2$, adjusted $R^2$, model F, and related statistics.
- [Diagnostics](../panel/diagnostics.md) — Hausman, pooling F, Breusch-Pagan LM, and related model diagnostics.

Each estimator page separates the **statistical model and identification assumptions** from the **numerical estimation method**. The assumptions describe when the reported coefficient has the usual panel-econometric interpretation; the software can compute an estimator mechanically even when those substantive assumptions are not credible in a particular application.

All six model classes support NumPy CPU, CuPy CUDA, and Torch CUDA through the `device` parameter. Each model page includes CPU/GPU and formula examples. If `device="cuda"` or `device="torch"` is requested explicitly but that backend is unavailable, statgpu raises an error instead of silently switching to CPU.

For the implementation-level responsibility map—what is shared across Panel models, which statistical steps remain model-specific, and why current Panel fitting is outside the generic `LossBase + Penalty + Solver` path—see [Panel Architecture](../panel/architecture.md).
