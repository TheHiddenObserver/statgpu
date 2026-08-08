# Panel Stage B diagnostic-df addendum

This addendum is normative for `panel_p1_stage_b_diagnostics_plan.md` and closes second-round definition issues discovered immediately before estimator integration. Where this addendum explicitly overrides a conflicting clause in the main plan, the addendum is the final Stage-B contract.

## Why a separate diagnostic df is required

Stage A intentionally preserved the historical `PanelOLS.df_resid` convention:

```text
df_resid_legacy = n - k - [(N - 1) entity effects] - [(T - 1) time effects]
```

for the corresponding included effects. This quantity continues to drive the existing nonrobust/robust covariance, t statistics, p-values, and confidence intervals and **must not change in Stage B**.

The standard poolability/model-F definitions in `linearmodels`/`plm`, however, count the rank of the full fixed-effect nuisance space. With no explicit exogenous constant, a one-way entity FE model has nuisance-effect rank `N`. For two-way entity+time effects, let `C` denote the number of connected components in the observed bipartite entity-time incidence graph; the full dummy-space rank is `N + T - C`. The familiar `N + T - 1` expression is therefore the connected-panel special case `C = 1`, not a valid formula for every incomplete panel. In the ordinary connected full-rank case this makes the standard diagnostic residual df one lower than statgpu's legacy Stage-A inference df.

Therefore Stage B introduces an internal **standard diagnostic residual df** rather than changing `model.df_resid`.

## Standard diagnostic rank/df

Let `r_x` be the numerical rank of the transformed slope design used for FE estimation.

For current statgpu `PanelOLS`, formula intercepts are stripped before fitting and the estimator does not add a separate constant. Consequently the standard effect-space rank is:

- no effects: `0`;
- entity only: `N`;
- time only: `T`;
- entity + time: `N + T - C`, where `C` is the connected-component count of the observed entity-time incidence graph.

Then

```text
df_model_diag = r_x + effect_rank_standard
df_resid_diag = n - df_model_diag
```

The implementation stores these values only in Stage-B diagnostic metadata/internal context. It does not overwrite the Stage-A `df_resid` attribute. For two-way FE it also records the incidence-component count so the rank decision is auditable.

The primary transformed FE response lives in the orthogonal complement of the nuisance-effect space, so the corresponding total-variation degrees of freedom for Stage-B adjusted R² are

```text
df_total_diag = n - effect_rank_standard
```

and hence

```text
R2_adj = 1 - (RSS / df_resid_diag) / (TSS_transformed / df_total_diag).
```

This replaces the earlier provisional `n-1` wording in the main plan for FE adjusted R². It is the rank-consistent definition: the restricted zero-slope model has exactly the nuisance effects removed before the transformed total sum of squares is formed.

If a future PanelOLS path contains a retained identified exogenous constant, the effect-rank accounting must switch to the equivalent constant-present parameterization rather than double counting the common mean.

## Which Stage-B quantities use which df

Use **standard diagnostic df** for new standardized diagnostics whose external definitions depend on model rank:

- pooling F denominator df;
- pooling F numerator df through nested-model rank difference;
- `PanelFitStatistics.f_statistic` denominator df for `PanelOLS`;
- `PanelFitStatistics.rsquared_adj` residual and total df for `PanelOLS`.

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

This is equivalent to the external effect-rank formulation in the ordinary full-rank case and remains auditable under rank deficiency and disconnected two-way incidence graphs.

The exact-fit boundary matches the classical model-F rule:

- if `RSS_FE` is numerically zero while `RSS_pool - RSS_FE` is materially positive, report the limiting result `F = inf`, `p = 0` with the ordinary numerator/denominator df;
- if both pooled and FE RSS are numerically zero, the ratio is indeterminate and the structured result remains inapplicable;
- a materially negative nested-model RSS difference remains inapplicable rather than being clipped.

## R² constant convention clarification

For standardized parameter-based **overall** and **between** R², centering depends on an actual identified constant in the level exogenous design. Fixed effects alone do not set `has_constant=True` for these two parameter-based measures.

The common-constant projection used by the **pooling F test** is a separate nested-test construction and must not be reused as a general R² centering rule.

For `RandomEffects`, an explicit nonzero constant column in the supplied level design is detected directly. The quasi-demeaned transformed version of that same column is retained as the restricted intercept design for adjusted R² and classical model F, including on unbalanced panels where the transformed intercept is not a vector of ones.

Constant detection is **scale equivariant**: multiplying an identified nonzero constant column by any nonzero unit-conversion factor must not change whether it is classified as a constant. The tolerance is therefore relative to that column's own magnitude; there is no `max(1, scale)` absolute floor. An exactly zero column is not treated as an intercept.

## Normative post-review overrides

Final PR review exposed additional correctness boundaries that supersede conflicting early-plan wording in Sections 5.1--5.3, 5.5, 6, 7, and 12 of `panel_p1_stage_b_diagnostics_plan.md`.

### Hausman sample/design identity

The earlier O(k) low-order-moment fingerprint is retained only as optional audit metadata; it is **not** sufficient proof that FE and RE used the same aligned numerical sample. Distinct row sequences can share sums, sums of squares, and index-weighted first moments.

The authoritative Stage-B identity contract is:

- compute a versioned SHA-256 digest over **every aligned float64 X/y value in row order**, including shape framing;
- compare the digest together with feature-name/intercept metadata and a separate aligned entity-code signature;
- require exact digest equality rather than an `allclose` tolerance for identity;
- on CuPy/Torch fits, transfer X/y to host in bounded chunks solely for hashing; no single extra full-design host allocation is required;
- persist only the digest and compact metadata after hashing, never a second retained CPU copy of the full design.

This bounded hashing transfer is an explicit exception to the early-plan statement that Hausman may transfer only O(k) fingerprint scalars. The statistical estimation, covariance construction, and fit-statistic reductions remain backend-native; the exception exists only to make the identity check collision-resistant.

The full-content digest is constructed only when the fitted FE model is actually in the Stage-B Hausman domain: one-way entity FE with nonrobust covariance. Robust/clustered, time-only, and two-way FE are rejected before identity comparison and must not pay the full X/y host-hash cost. `RandomEffects` remains a potential Hausman input and therefore retains the identity contract.

### Scale-equivariant numerical tolerances

RSS, covariance matrices, and coefficient differences carry units. Numerical applicability/rank decisions must therefore be invariant to a change of units. In particular:

- model-F and pooling-F RSS tolerances scale with the compared RSS values and do not use an absolute `max(1, RSS)` floor;
- Hausman eigenvalue/range tolerances scale with the covariance/difference norms and do not impose a unit-sized absolute floor;
- multiplying `y` and fitted coefficients by `c`, or multiplying a Hausman coefficient difference by `c` and its covariance difference by `c^2`, must leave the dimensionless test statistic/applicability unchanged up to floating-point roundoff.

The ordinary dimensionless post-computation guard on a near-zero negative test statistic may retain a unit floor because the statistic itself is dimensionless.

### Classical model-F exact-fit boundary

The early generic “unavailable when unrestricted RSS is zero” wording is too coarse. The final Stage-B contract is:

- if unrestricted RSS is numerically zero and restricted RSS is materially positive, report the limiting classical result `F = inf`, `p = 0`, with the ordinary numerator/denominator df;
- if both restricted and unrestricted RSS are numerically zero, the joint-slope statistic is indeterminate and remains unavailable with an explicit reason;
- material nesting violations remain inapplicable rather than being silently clipped.

### RandomEffects explicit constant

`RandomEffects` must detect an actual nonzero constant column in the supplied level design. Because Swamy-Arora quasi-demeaning transforms that column and an unbalanced panel generally does not leave a vector of ones, the transformed constant column itself is the restricted design for adjusted R²/model-F accounting. No implicit intercept is invented when the level design has none.

Physical CUDA validation must include balanced and unbalanced explicit-constant RandomEffects cases, in addition to the ordinary no-explicit-constant cases, and must compare the constant/restricted-design metadata as well as numerical fit statistics.

## Review status

- **[HIGH][INFER] fixed in specification** — Stage-B poolability/model-F inference no longer reuses a legacy FE residual df that differs from the standard nuisance-effect rank convention.
- **[HIGH][INFER] fixed after review** — two-way FE nuisance rank uses the observed incidence-graph component count (`N + T - C`) instead of assuming every incomplete panel is connected.
- **[HIGH][API/INFER] fixed after review** — Hausman identity uses a collision-resistant full-content digest; low-order moments alone are no longer accepted as proof of sample identity.
- **[CRITICAL][INFER] fixed after fresh review** — F/Hausman applicability tolerances and explicit-constant detection are scale equivariant instead of imposing a unit-sized absolute floor.
- **[HIGH][INFER] fixed after fresh review** — an exact FE fit with positive pooled RSS reports the limiting pooling `F=inf, p=0`; the both-zero case remains explicitly inapplicable.
- **[HIGH][TEST/BACKEND] fixed locally after fresh review** — the physical runner includes balanced/unbalanced explicit-constant RandomEffects cases and checks the restricted-design contract; exact-head P100 execution is still required after the final code head is fixed.
- **[MEDIUM][PERF] fixed/measurement pending** — FE fits outside the Hausman domain no longer build the full-content digest; a dedicated physical benchmark measures the remaining digest overhead for Hausman-compatible one-way FE/RE fits.
- **[MEDIUM][INFER] fixed in specification** — FE adjusted R² uses nuisance-rank-consistent total df rather than provisional `n-1`.
- **[MEDIUM][INFER] fixed in specification** — overall/between R² centering is explicitly separated from the pooling-F common-constant correction.
- **[MEDIUM][INFER] fixed after review** — RandomEffects explicit-constant diagnostics retain the transformed intercept in the restricted fit-space definition.
- **[MEDIUM][INFER] fixed after review** — exact unrestricted fits with a nonzero restricted RSS report `F=inf, p=0` instead of being discarded as unavailable.
