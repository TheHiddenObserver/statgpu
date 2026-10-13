# Panel Models

> Language: English  
> Last updated: 2026-10-13  
> Switch: [Chinese](../../cn/panel/README.md)

Use panel models when the same entities, such as firms, people, or regions, are observed over several periods. Start by deciding which variation answers your question: changes within an entity, differences between entities, or cross-sectional relationships that vary over time. `statgpu.panel` provides six estimators for these different targets.

## Choose the model for your question

If persistent entity differences may be related to your predictors, start with [fixed effects](panel-ols.md) or [first differences](first-difference-ols.md). If you want to use both within- and between-entity variation, [random effects](random-effects.md) requires the stronger orthogonality assumption below. Use [pooled OLS](pooled-ols.md) only when a common level regression is the intended mean model; supplying entity labels does not remove omitted entity effects.

For long-run differences in entity averages, use [BetweenOLS](between-ols.md). For the average of period-by-period cross-sectional coefficients, use [FamaMacBeth](fama-macbeth.md). These targets are not interchangeable, and a covariance option cannot repair a failed identification assumption.

| Estimator | Statistical view | Main source of identification / extra condition |
|---|---|---|
| [PanelOLS](panel-ols.md) | Entity and/or time effects are fixed but unknown nuisance parameters. | Uses variation remaining after the selected fixed effects are removed; no orthogonality between the fixed effects and regressor history is required. |
| [FirstDifferenceOLS](first-difference-ols.md) | Starts from a model with fixed entity effects and removes time-invariant entity effects by differencing. | Uses within-entity changes between consecutive observed periods. |
| [RandomEffects](random-effects.md) | The entity effect is modeled as a random component. | Classical RE interpretation requires the random effect to be orthogonal to the regressor history, for example $E(a_i\mid X_i)=0$. |
| [BetweenOLS](between-ols.md) | Averages $X$ and $y$ within each entity, producing one mean observation per entity. | Uses between-entity variation; recovering the same structural slope requires the averaged composite error to be orthogonal to the averaged regressors. |
| [PooledOLS](pooled-ols.md) | One common conditional-mean relationship is fitted to all stacked observations. | Uses all stacked variation; the combined regression error must be exogenous with respect to the regressors. |
| [FamaMacBeth](fama-macbeth.md) | Each time period has its own cross-sectional regression. | Averages the period-specific slope estimates and bases inference on their time-series variation. |

## Prepare one aligned panel sample

Use long-form data: one row per entity-period observation, `X` with shape `(n_observations, n_features)`, and `y` with shape `(n_observations,)`. Keep each entity label, time label, and any cluster label aligned with that row when filtering or sorting. Labels describe the panel structure; they are not predictor columns unless you explicitly put them in `X`.

- `PanelOLS`: supply `entity_ids` and/or `time_ids` for the effects you enable. Neither effect is enabled by default.
- `RandomEffects`, `BetweenOLS`, and `FirstDifferenceOLS`: supply `entity_ids`. For first differences, also supply correctly ordered `time_ids` unless input rows already have the intended within-entity order; differences join consecutive observed periods and do not fill gaps.
- `FamaMacBeth`: supply `time_ids`; each retained period needs a full-column-rank cross-sectional design, including its intercept.
- `PooledOLS`: entity labels are optional for fitting, but enable panel fit statistics and the LM diagnostic.

Do not add an intercept column uniformly across models. The array interfaces of `PooledOLS`, `BetweenOLS`, and `FamaMacBeth` add one; `PanelOLS` and `RandomEffects` use the columns you supply, while first differences do not estimate a level intercept. Each model page explains its formula interface and provides a self-contained CPU example.

## Fit, then read the result

Open the model you selected and run its self-contained CPU example before consulting the full parameter table. The examples build their own panel data and explain coefficient order, observation counts, and degrees of freedom. The result section then explains standard errors and prediction in that model's scale.

Read `coef_` together with `bse_` and `conf_int_`, after checking that coefficient inference is available. A small p-value does not establish exogeneity. Use [fit statistics](fit-statistics.md) to distinguish within, between, and overall fit; use [diagnostics](diagnostics.md) for a particular model-comparison question rather than an automatic model-selection sequence.

For repeated observations, choose uncertainty estimates to match the error dependence you expect. The [covariance guide](covariance.md) explains when HC, clustering, or time-dependent covariance is relevant and which models support each choice. `FamaMacBeth` instead uses the time series of period coefficients. Robust standard errors do not change the identifying assumptions or turn the classical model F into a robust Wald test.

## Documentation map

- Model pages above: first fit, result interpretation, prediction, assumptions, and complete constructor/fit parameter reference.
- [Covariance](covariance.md): supported uncertainty choices, required labels, formulas, and comparison conventions.
- [Fit statistics](fit-statistics.md): which reported $R^2$ or F statistic answers your question.
- [Diagnostics](diagnostics.md): pooling F, Breusch-Pagan LM, and the classical FE-versus-RE Hausman comparison.
- [Panel Architecture](architecture.md): shared transformations, fit lifecycle, backend boundaries, and implementation organization; optional for a first fit.

Start with `device="cpu"` as in the examples. These six model classes also support CuPy CUDA and Torch CUDA through `device`; an unavailable explicitly requested `"cuda"` or `"torch"` backend raises instead of silently falling back. This does not imply that every preprocessing step runs on the GPU or that every model supports the same covariance options. See [device guidance](../guides/device-and-memory.md) for setup and backend semantics.
