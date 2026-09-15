# Panel Models

> Language: English  
> Last updated: 2026-09-13  
> Switch: [Chinese](../../cn/models/panel.md)

`statgpu.panel` provides six commonly used panel-data estimators covering pooled regression, fixed effects, between estimation, first differences, random effects, and Fama–MacBeth cross-sectional regressions. These estimators can be applied to the same or related underlying panel models, with different data transformations, identifying variation, and additional assumptions determining the resulting coefficient estimates.

A useful way to distinguish them is:

| Estimator | Statistical view | Main source of identification / extra condition |
|---|---|---|
| [PanelOLS](../panel/panel-ols.md) | Entity and/or time effects are fixed but unknown nuisance parameters. | Uses variation remaining after the selected fixed effects are removed; no orthogonality between the fixed effects and regressor history is required. |
| [FirstDifferenceOLS](../panel/first-difference-ols.md) | Starts from a model with fixed entity effects and removes time-invariant entity effects by differencing. | Uses within-entity changes between consecutive observed periods. |
| [RandomEffects](../panel/random-effects.md) | The entity effect is modeled as a random component. | Classical RE interpretation requires the random effect to be orthogonal to the regressor history, for example $E(a_i\mid X_i)=0$. |
| [BetweenOLS](../panel/between-ols.md) | Averages $X$ and $y$ within each entity, producing one mean observation per entity. | Uses between-entity variation; recovering the same structural slope requires the averaged composite error to be orthogonal to the averaged regressors. |
| [PooledOLS](../panel/pooled-ols.md) | One common conditional-mean relationship is fitted to all stacked observations. | Uses all stacked variation; the combined regression error must be exogenous with respect to the regressors. |
| [FamaMacBeth](../panel/fama-macbeth.md) | Each time period has its own cross-sectional regression. | Averages the period-specific slope estimates and bases inference on their time-series variation. |

## Documentation map

- [Panel Architecture](../panel/architecture.md) — `BasePanelModel`, estimator-specific data transformations and regression problems, shared numerical linear algebra, covariance/inference, diagnostics, and fit lifecycle.
- [Covariance](../panel/covariance.md) — nonrobust, HC, cluster, HAC, and Driscoll-Kraay covariance definitions.
- [Fit statistics](../panel/fit-statistics.md) — within/between/overall $R^2$, adjusted $R^2$, model F, and related statistics.
- [Diagnostics](../panel/diagnostics.md) — Hausman, pooling F, Breusch-Pagan LM, and related model diagnostics.

Each model page explains both the **statistical model and identification assumptions** and the **numerical estimation method**. The assumptions determine the econometric interpretation of the reported coefficients, while the numerical sections describe how statgpu computes the corresponding estimator.

All six model classes support NumPy CPU, CuPy CUDA, and Torch CUDA through the `device` parameter. Each model page includes CPU/GPU and formula examples; an explicit `device="cuda"` or `device="torch"` request uses the corresponding backend.

For the shared implementation structure, data transformations, numerical components, and inference layer, see [Panel Architecture](../panel/architecture.md).