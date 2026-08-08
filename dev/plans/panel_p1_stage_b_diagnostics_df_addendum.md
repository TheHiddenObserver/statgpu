# Panel Stage B diagnostic-df addendum

This addendum is normative for `panel_p1_stage_b_diagnostics_plan.md` and closes a second-round definition issue discovered immediately before estimator integration.

## Why a separate diagnostic df is required

Stage A intentionally preserved the historical `PanelOLS.df_resid` convention:

```text
df_resid_legacy = n - k - [(N - 1) entity effects] - [(T - 1) time effects]
```

for the corresponding included effects. This quantity continues to drive the existing nonrobust/robust covariance, t statistics, p-values, and confidence intervals and **must not change in Stage B**.

The standard poolability/model-F definitions in `linearmodels`/`plm`, however, count the rank of the full fixed-effect nuisance space. With no explicit exogenous constant, a one-way entity FE model has nuisance-effect rank `N`; a two-way entity+time model has effect rank `N + T - 1`. In the ordinary full-rank case this makes the standard diagnostic residual df one lower than statgpu's legacy Stage-A inference df.

Therefore Stage B introduces an internal **standard diagnostic residual df** rather than changing `model.df_resid`.

## Standard diagnostic rank/df

Let `r_x` be the numerical rank of the transformed slope design used for FE estimation.

For current statgpu `PanelOLS`, formula intercepts are stripped before fitting and the estimator does not add a separate constant. Consequently the standard effect-space rank is:

- no effects: `0`;
- entity only: `N`;
- time only: `T`;
- entity + time: `N + T - 1`.

Then

```text
df_model_diag = r_x + effect_rank_standard
df_resid_diag = n - df_model_diag
```

The implementation stores these values only in Stage-B diagnostic metadata/internal context. It does not overwrite the Stage-A `df_resid` attribute.

If a future PanelOLS path contains a retained identified exogenous constant, the effect-rank accounting must switch to the equivalent constant-present parameterization rather than double counting the common mean.

## Which Stage-B quantities use which df

Use **standard diagnostic df** for new standardized diagnostics whose external definitions depend on model rank:

- pooling F denominator df;
- pooling F numerator df through nested-model rank difference;
- `PanelFitStatistics.f_statistic` denominator df for `PanelOLS`;
- `PanelFitStatistics.rsquared_adj` residual df for `PanelOLS`.

Use the existing **legacy Stage-A df** unchanged for:

- existing covariance calculations;
- existing `bse_`, `tvalues_`, `pvalues_`, `conf_int_`;
- existing public `PanelOLS.df_resid`;
- any legacy summary field that already reports it.

For PooledOLS, BetweenOLS, FirstDifferenceOLS, and RandomEffects, Stage B should use effective numerical rank for the new model-F/adjusted-R² metadata. When the existing fit is full rank this agrees with the current residual df. Stage B does not modify legacy inference df if a rank-deficient corner case exposes a difference.

## Pooling F after this correction

Construct the pooled null on the exact aligned level sample. Since current statgpu FE does not retain an exogenous constant, apply the same common-constant projection used by `linearmodels` before the pooled slope regression. Let

```text
df_pool_diag = n - rank(X_centered) - 1
```

where the `-1` is the projected common constant. Then

```text
df_num = df_pool_diag - df_resid_diag_FE
F = ((RSS_pool - RSS_FE) / df_num) / (RSS_FE / df_resid_diag_FE)
```

This is equivalent to the external effect-rank formulation in the ordinary full-rank case and remains auditable under rank deficiency.

Roundoff/material nesting-failure behavior from the main plan remains unchanged.

## R² constant convention clarification

For standardized parameter-based **overall** and **between** R², centering depends on an actual identified constant in the level exogenous design. Fixed effects alone do not set `has_constant=True` for these two parameter-based measures.

The common-constant projection used by the **pooling F test** is a separate nested-test construction and must not be reused as a general R² centering rule.

## Review status

- **[HIGH][INFER] fixed in specification** — Stage-B poolability/model-F inference will no longer reuse a legacy FE residual df that differs from the standard nuisance-effect rank convention.
- **[MEDIUM][INFER] fixed in specification** — overall/between R² centering is explicitly separated from the pooling-F common-constant correction.
