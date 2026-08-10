# Panel Models

> Language: English  
> Last updated: 2026-08-10
> This page: Model documentation  
> Switch: [Chinese](../../cn/models/panel.md)

## Overview

The `statgpu.panel` module provides six panel-data estimators:

- `PanelOLS`: entity and/or time fixed effects;
- `RandomEffects`: Swamy-Arora feasible GLS random effects;
- `PooledOLS`: stacked OLS without demeaning;
- `BetweenOLS`: regression on entity means;
- `FirstDifferenceOLS`: within-entity first differences;
- `FamaMacBeth`: period-by-period cross-sectional regressions with coefficient averaging.

Array-input numerical paths support NumPy, CuPy CUDA, and Torch CUDA. Formula construction and categorical entity/time/cluster labels are intentional CPU metadata boundaries; compact aligned codes are transferred to the selected numerical backend. Explicit GPU devices do not silently fall back to CPU.

Stage C of the Tier-1 panel roadmap completes the residual-sandwich covariance layer on top of the Stage-B diagnostics: historical defaults remain unchanged, while HC0/HC2/HC3, robust RandomEffects inference, explicit cluster group debiasing, and Driscoll-Kraay covariance are added with NumPy/CuPy/Torch-native accumulation. The repaired covariance/provenance implementation was revalidated on exact-clean head `aad53587...` using Tesla P100: CuPy and Torch each pass all 26 estimator covariance cases plus six direct public covariance primitives (32/32 per backend), including full-rank ill-conditioned HC0/HC2/HC3/DK, and the synchronized performance run includes the bounded `N=10,000`, `k=2`, `T=200` QS scenario. Earlier `c151550a...` and `9c0b3050...` artifacts remain immutable historical evidence.

## Paths

```python
from statgpu.panel import (
    PanelOLS,
    RandomEffects,
    PooledOLS,
    BetweenOLS,
    FirstDifferenceOLS,
    FamaMacBeth,
    PanelTestResult,
    PanelFitStatistics,
    hausman_test,
    pooling_f_test,
    breusch_pagan_lm_test,
    clustered_covariance,
    two_way_clustered_covariance,
    hac_covariance,
    driscoll_kraay_covariance,
)
```

The diagnostic result classes and functions are also exported from top-level `statgpu`.

## Model Summary

| Model | Transformation | Main inference choices | Standardized fit statistics |
|---|---|---|---|
| `PanelOLS` | Entity/time within transformation | nonrobust; HC0/HC1/HC2/HC3; one-/two-way clustered; Driscoll-Kraay | within/between/overall R², adjusted R², classical model F, pooling F |
| `RandomEffects` | Swamy-Arora feasible GLS | nonrobust; HC0/HC1/HC2/HC3; one-/two-way clustered; Driscoll-Kraay | within/between/overall R², adjusted R², classical model F, Hausman input when nonrobust |
| `PooledOLS` | Stacked OLS | nonrobust; HC0/HC1/HC2/HC3; clustered; legacy row-HAC; Driscoll-Kraay | overall R² always; within/between R² and BP-LM with `entity_ids`; adjusted R² and classical model F |
| `BetweenOLS` | Entity means | nonrobust; HC0/HC1/HC2/HC3 | within/between/overall R², adjusted R², classical model F |
| `FirstDifferenceOLS` | Within-entity first differences | nonrobust; HC0/HC1/HC2/HC3 | within/between/overall R², adjusted R² on the differenced fit space, classical model F |
| `FamaMacBeth` | Cross-sectional regressions by period | nonrobust, Newey-West | parameter-based within/between/overall R²; no residual-OLS adjusted R² or model F |

## Core Estimating Equations

`PanelOLS` fits OLS after removing requested fixed effects. With entity effects,

$$
y_{it}^{\mathrm{within}} = y_{it} - \bar y_{i\cdot},
\qquad
X_{it}^{\mathrm{within}} = X_{it} - \bar X_{i\cdot}.
$$

With entity and time effects, the two-way transformation also removes time means and adds back the grand mean.

`PooledOLS` fits

$$
\hat\beta = X^+ y,
$$

where \(X^+\) denotes the inverse or Moore-Penrose pseudoinverse as required. `BetweenOLS` applies OLS to entity means, `FirstDifferenceOLS` applies OLS to Δ\(X\) and Δ\(y\), and `FamaMacBeth` averages period-specific coefficient vectors.

## Stage-C Covariance and Inference

Stage C is additive: coefficient estimation, Stage-B fit statistics, and the historical default inference remain unchanged. Covariance names are normalized as follows.

| `cov_type` | Behavior |
|---|---|
| `"nonrobust"` | Classical fit-space OLS covariance with Student-t inference |
| `"robust"`, `"hc1"` | The historical statgpu HC1 sandwich with asymptotic normal inference; `hc1` normalizes to canonical `robust` |
| `"hc0"` | Unscaled Eicker-White sandwich on the estimator's actual fit-space regression |
| `"hc2"`, `"hc3"` | Leverage-adjusted sandwich on the estimator's actual fit-space regression |
| `"clustered"` | One- or two-way clustered sandwich; `group_debias=True` opt-in correction where supported |
| `"driscoll-kraay"`, `"dk"`, `"kernel"` | Time-aggregated Driscoll-Kraay covariance with Bartlett, Parzen, or quadratic-spectral kernels |
| `"hac"` | Historical row-order Bartlett/Newey-West covariance for `PooledOLS`; deliberately distinct from Driscoll-Kraay |
| `"newey-west"` | Existing HAC on the `FamaMacBeth` coefficient path; not routed through the residual-OLS Stage-C layer |

### HC0/HC2/HC3 fit-space definition

For the numerical regression actually used by an estimator, let `Z` be the fit-space design, `e` the fit-space residual vector, and `B=(Z'Z)^+`. The leverage is

$$
h_i=z_i^\top Bz_i.
$$

HC0 uses the meat $\sum_i z_i z_i^\top e_i^2$; HC2 divides each squared residual by $1-h_i$; HC3 divides by $(1-h_i)^2$. The implementation computes leverage rowwise and never materializes an `n x n` hat matrix. A numerically unit leverage makes HC2/HC3 undefined and raises rather than being clipped into a valid-looking covariance.

The fit space is model-specific: pooled level design for `PooledOLS`, fixed-effect transformed slopes for `PanelOLS`, quasi-demeaned `X_star` for `RandomEffects`, entity means for `BetweenOLS`, and retained first differences for `FirstDifferenceOLS`. Consequently Panel HC2/HC3 is documented as **transformed-fit-space HC2/HC3**; it is not silently redefined as HC2/HC3 from a literal full dummy regression.

### Cluster covariance and `group_debias`

One-way clustering aggregates score vectors within a cluster. Two-way clustering uses inclusion-exclusion of cluster 1, cluster 2, and the exact paired-label intersection. The default `group_debias=False` preserves the historical statgpu clustered covariance. With `group_debias=True`, each component's meat is multiplied by

$$
\frac{G}{G-1}\frac{n-1}{n},
$$

using that component's own group count before two-way inclusion-exclusion. This changes covariance magnitude only; Stage C does not silently switch coefficient tests to a finite-group t reference. String/categorical cluster labels are metadata and are factorized without moving the numerical score matrix to CPU.

### Driscoll-Kraay covariance

Driscoll-Kraay first aggregates fit-space scores by observed time,

$$
g_t=\sum_i z_{it}e_{it},
$$

then applies a kernel HAC to the ordered `g_t` series. `PooledOLS` uses `time_index=`, while `PanelOLS` and `RandomEffects` use aligned `time_ids=`. Unbalanced panels are supported because each time aggregate contains only observed rows.

For a full-rank fit-space design with `k` columns, Stage C uses the `linearmodels==7.0`-compatible debiased scale

$$
\mathrm{scale}_{DK}=\frac{n}{n-\mathrm{extra\_df}-k}.
$$

`PooledOLS` and `RandomEffects` use `extra_df=0`. `PanelOLS` uses the Stage-B standard fixed-effect nuisance rank (`N`, `T`, or `N+T-C`). If statgpu validly reaches a rank-deficient fit, the documented extension replaces `k` by the numerical rank and uses a pseudoinverse; this corner is not claimed to be a `linearmodels` equality case.

`bandwidth=None` uses `floor(4*(T/100)^(2/9))`, where `T` is the number of distinct observed periods. Bartlett/Newey-West and Parzen/Gallant are truncated at the bandwidth. Quadratic Spectral (`qs`, Andrews) treats bandwidth as a smoothing scale and applies weights to **all observed lags** when bandwidth is positive; it is not truncated at `bw`. Numeric and datetime time keys use their natural sorted order. An ordered pandas categorical preserves its declared category chronology, restricted to observed categories. Plain string/object labels retain deterministic sorted-label ordering; when chronological order differs from lexical order, pass an ordered categorical or an explicit numeric/datetime time key.

### RandomEffects covariance

Stage C does not alter Swamy-Arora variance-component or coefficient estimation. Robust, HC, cluster, and Driscoll-Kraay covariance are computed from the quasi-demeaned GLS design `X_star` and residuals. Therefore changing `cov_type` changes only inference. The classical Stage-B Hausman test requires **both** the FE and RE fits to use nonrobust covariance; robust auxiliary Hausman remains out of scope and returns a structured inapplicable result.

### Backend and validation status

HC leverage, row scores, grouped cluster/time scores, lag products, bread/meat matrices, and covariance accumulation remain on NumPy/CuPy/Torch. CPU transfers are restricted to labels/group codes, small configuration, and scalar audit reductions. Explicit GPU devices never silently fall back to CPU.

Hosted Stage-C tests pin HC2/HC3 against analytic/statsmodels fit-space calculations and cluster/Driscoll-Kraay definitions against `linearmodels==7.0`. Fresh exact-clean-head Tesla P100 acceptance is complete on `aad53587...`: CuPy and Torch each pass 26/26 estimator covariance cases plus 6/6 direct public covariance primitives (32/32 per backend), including the full-rank ill-conditioned HC0/HC2/HC3/DK regressions, with requested/executed backend identity and no CPU fallback. The synchronized performance rerun covers the three base scales and explicit `N=10,000`, `k=2`, `T=200` QS all-lag scenario; it records timing only and makes no speedup claim. The earlier `c151550a...` and `9c0b3050...` artifacts remain historical audit evidence.

### PooledOLS HAC ordering

For `PooledOLS(cov_type="hac")`, pass `time_index=` to `fit`. The implementation validates the side array and uses a stable time ordering while keeping X, y, and Stage-B entity diagnostic metadata under the identical permutation. Consequently, BP-LM and parameter-based R² cannot accidentally group residuals using pre-sort entity metadata.

### Rank-deficient PooledOLS

A rank-deficient design separates fitted-space validity from coefficient-space identifiability:

- fitting, prediction, residuals, RSS, rank, and fitted-space comparisons remain valid;
- `df_resid` is computed as `nobs - rank(X)`, not `nobs - n_columns`;
- individual coefficients are not unique under exact collinearity;
- coefficient-level covariance, BSE, test statistics, p-values, and confidence intervals are therefore non-identifiable and should not be interpreted as unique coefficient inference.

Stage-B model-F restrictions use effective numerical rank rather than blindly using the raw column count.

## Standardized `fit_statistics_`

Supported fits expose a structured `PanelFitStatistics` object:

```python
stats = model.fit_statistics_

stats.rsquared_within
stats.rsquared_between
stats.rsquared_overall
stats.rsquared_adj
stats.f_statistic
stats.f_pvalue
stats.f_df
stats.metadata
```

### Parameter-based R²

The within/between/overall R² family follows the parameter-based convention used by `linearmodels`: each statistic evaluates the fitted coefficient vector rather than squaring a fitted-value correlation.

For coefficient vector \(\hat\beta\):

- **overall R²** evaluates \(y-X\hat\beta\) on the level panel;
- **between R²** evaluates entity means \(\bar y_i-\bar X_i\hat\beta\);
- **within R²** evaluates entity-demeaned \(y\) and \(X\).

Overall and between total sums of squares are centered only when the actual level regressor design contains an identified constant. Fixed effects by themselves do not change this centering rule. `RandomEffects` detects an explicit nonzero constant column in the supplied level design and retains its quasi-demeaned transformed column when defining adjusted R² and the restricted model-F regression. Constant detection uses a tolerance relative to the column's own magnitude, so changing units does not turn a nonzero constant into a slope or vice versa. A zero total sum of squares is reported as `0.0` in the standardized Stage-B field and marked in `metadata["degenerate_total_ss"]`.

### Legacy `PanelOLS.rsquared_within`

`PanelOLS.rsquared_within` is retained exactly for Stage-A compatibility. In a two-way FE model it describes the historical full entity+time transformed fit and can differ from the standardized entity-within `fit_statistics_.rsquared_within`. Stage B does not silently overwrite the legacy attribute; the compatibility value is recorded in fit-statistics metadata.

### Adjusted R² and diagnostic degrees of freedom

For new standardized diagnostics, `PanelOLS` uses the rank of the complete fixed-effect nuisance space. Current `PanelOLS` does not retain an exogenous intercept, so the standard nuisance-effect rank is:

- entity effects only: \(N\);
- time effects only: \(T\);
- entity and time effects: \(N+T-C\), where \(C\) is the number of connected components in the observed entity-time incidence graph.

Thus the familiar \(N+T-1\) formula is the connected-panel special case \(C=1\); incomplete panels with disconnected incidence components receive their actual dummy-space rank rather than a hard-coded connected-panel rank.

If \(r_X\) is the numerical rank of the transformed slope design, Stage B uses

$$
\mathrm{df}_{\mathrm{resid,diag}}
= n-r_X-r_{\mathrm{effects}},
$$

and

$$
\mathrm{df}_{\mathrm{total,diag}}
= n-r_{\mathrm{effects}}
$$

for the standardized FE model F and adjusted R².

This is intentionally separate from the historical public `PanelOLS.df_resid`, which remains unchanged because existing covariance, t statistics, p-values, confidence intervals, and summaries depend on that compatibility convention. Hausman uses a diagnostic-only small FE covariance matrix rescaled to the standard nuisance-rank denominator; the public Stage-A BSE/CI remain unchanged.

### Classical model F

For OLS-style panel estimators, Stage B reports the classical homoskedastic joint-slope F statistic on the estimator's primary fit space:

$$
F=
\frac{(RSS_R-RSS_U)/q}
     {RSS_U/\mathrm{df}_{\mathrm{resid}}},
$$

where \(q\) is the effective restriction rank. A robust or clustered covariance choice does not silently convert this field into a robust Wald test. When the unrestricted regression fits exactly while the restricted regression has positive RSS, the standardized result is the limiting classical value `F=inf`, `p=0` rather than an unavailable statistic. RSS zero/nesting tolerances are relative to the RSS scale, so multiplying the response and fitted coefficients by a common unit-conversion factor does not change the dimensionless F statistic or its applicability.

`FamaMacBeth` does not receive a residual-OLS model F or adjusted R². Its covariance is based on the time series of cross-sectional coefficient estimates, and Stage B does not relabel that beta-series inference as residual OLS.

## Specification Tests

All specification tests return `PanelTestResult`. Econometrically inapplicable cases are structured results with `applicable=False` and a reason; malformed API calls still raise normal programming errors.

### Pooling F test

For a fitted `PanelOLS` with at least one effect:

```python
result = fe.pooling_f_test()
# or
result = pooling_f_test(fe)
```

The classical poolability null is that all included fixed effects are jointly zero. The statistic compares the effect model against a nested pooled regression on the exact aligned estimation sample.

When the FE level design has no explicit constant, the pooled null projects both y and X off the common constant before fitting the slopes. This prevents the common mean from being counted as a tested fixed effect. The numerator and denominator degrees of freedom are derived from effective nested-model ranks, not hard-coded effect counts.

If the fixed-effects fit is exact (`RSS_FE` numerically zero) while the restricted pooled RSS is materially positive, the classical limiting result is `F=inf`, `p=0`. If both pooled and FE RSS are numerically zero, the ratio is indeterminate and the structured result is inapplicable. These zero/nesting decisions use RSS-relative tolerances rather than a unit-sized absolute floor.

The test is classical/homoskedastic even when the fitted FE object uses a robust or clustered covariance for coefficient inference.

### Breusch-Pagan LM for entity random effects

Supply `entity_ids` to `PooledOLS.fit()`:

```python
pooled = PooledOLS().fit(X, y, entity_ids=entity_ids)
result = pooled.breusch_pagan_lm_test()
# or
result = breusch_pagan_lm_test(pooled)
```

This is the **panel error-components Breusch-Pagan LM test**, not the cross-sectional heteroskedasticity Breusch-Pagan test. Stage B implements the one-way entity version, including the Baltagi-Li incomplete/unbalanced-panel formula used by `plm::plmtest(type="bp", effect="individual")`.

The null is zero entity random-effect variance. At least two entities, positive pooled RSS, and at least one repeated observation within an entity are required. Without `entity_ids`, the result is structured as inapplicable rather than guessed from row order.

### Classical Hausman FE versus RE

```python
fe = PanelOLS(entity_effects=True, cov_type="nonrobust").fit(
    X, y, entity_ids=entity_ids
)
re = RandomEffects().fit(X, y, entity_ids=entity_ids)

result = fe.hausman_test(re)
# or
result = hausman_test(fe, re)
```

Stage B implements the original quadratic-form one-way entity FE-versus-RE Hausman test:

$$
H=(\hat\beta_{FE}-\hat\beta_{RE})^\top
  (V_{FE}-V_{RE})^{-1}
  (\hat\beta_{FE}-\hat\beta_{RE}).
$$

Applicability rules are explicit:

- FE must be one-way entity effects only;
- the FE coefficient covariance must be classical/nonrobust;
- the RE coefficient covariance must also be classical/nonrobust; Stage-C robust/HC/cluster/DK RE fits are not inputs to this classical test;
- FE and RE must be fitted to the same aligned y/entity sample and the same canonical slope design;
- an RE-only explicit constant is allowed because entity FE absorbs the common intercept; that constant is excluded from the Hausman coefficient vector;
- row/sample compatibility uses a collision-resistant SHA-256 digest of every aligned float64 **slope-X/y** value plus the entity-code signature and canonical feature metadata, not only matching shapes or low-order moments;
- canonical slope positions retain a map to each model's original coefficient/covariance positions, so an RE intercept at position 0 does not shift `x1`, `x2`, ... into the wrong coefficients;
- a materially indefinite covariance difference is reported as inapplicable instead of being eigenvalue-clipped into a statistic.

For array input, slopes are canonically renumbered after an RE-only constant is removed. For named/formula designs, slope names are preserved. Hausman covariance-rank and identified-range tolerances are relative to the covariance/coefficient scale, so a change of outcome units does not change applicability for the same mathematical problem.

For GPU fits, the canonical slope full-content digest is computed through bounded chunks copied to host solely for hashing. One-way entity FE with nonrobust covariance and `RandomEffects` retain this identity because they can participate in Stage-B Hausman. Robust/clustered, time-only, and two-way FE are rejected before identity comparison and do not pay the full X/y hashing cost. Fitted models retain only the digest/index metadata, not a second CPU copy of the design. Statistical estimation, covariance construction, and fit-statistic reductions remain on the selected numerical backend.

If the covariance difference is positive semidefinite but rank-deficient, statgpu provides a documented generalized-inverse extension: the test uses the identified range and chi-square degrees of freedom equal to the numerical rank, but only when the coefficient difference lies in that range. Metadata records `used_pinv=True` and labels this as the `singular PSD generalized-inverse Hausman` extension. Robust auxiliary-regression Hausman is not part of Stage B.

## Parameters and Fit Signatures

### `PanelOLS`

```python
PanelOLS(
    entity_effects=False,
    time_effects=False,
    cov_type="nonrobust",  # robust/hc0/hc1/hc2/hc3/clustered/dk also supported
    bandwidth=None,
    kernel="bartlett",
    group_debias=False,
    device="auto",
)
```

```python
model.fit(X, y, entity_ids=entity_ids, time_ids=time_ids, cluster=cluster)
```

Formula input can also request effects through the existing pipe syntax, for example `"y ~ x1 + x2 | entity"`. Formula row filtering aligns side arrays to the retained estimation sample.

### `PooledOLS`

```python
PooledOLS(
    cov_type="nonrobust",  # includes HC, clustered, legacy hac, and dk
    alpha=0.05,
    bandwidth=None,
    kernel="bartlett",
    group_debias=False,
    device="auto",
)
```

```python
model.fit(
    X,
    y,
    cluster=None,
    time_index=None,
    entity_ids=None,
)
```

`cluster` is required for clustered inference. `time_index` defines stable temporal ordering for HAC. `entity_ids` is optional and does not change coefficients; it enables standardized within/between R² and the panel BP-LM diagnostic. `PooledOLS` and `BetweenOLS` always include an intercept, so formulas that explicitly remove it (for example, `0 +` or `-1`) are rejected rather than silently ignored.

### Other models

```python
RandomEffects(
    device="auto",
    cov_type="nonrobust",  # robust/hc0/hc1/hc2/hc3/clustered/dk
    bandwidth=None,
    kernel="bartlett",
    group_debias=False,
)
BetweenOLS(cov_type="nonrobust", alpha=0.05, device="auto")  # HC0/1/2/3 supported
FirstDifferenceOLS(cov_type="nonrobust", alpha=0.05, device="auto")  # HC0/1/2/3 supported
FamaMacBeth(
    cov_type="newey-west",
    bandwidth=None,
    alpha=0.05,
    min_obs_per_period=1,
    device="auto",
)
```

`FamaMacBeth.fit(..., entity_ids=None)` accepts optional entity IDs only for Stage-B within/between R². The beta-series estimation and covariance path are unchanged.

## CPU and GPU Example

```python
import numpy as np
from statgpu.panel import PanelOLS, PooledOLS, RandomEffects

n_entities, n_times = 50, 10
n = n_entities * n_times
entity_ids = np.repeat(np.arange(n_entities), n_times)
time_ids = np.tile(np.arange(n_times), n_entities)
X = np.random.default_rng(0).normal(size=(n, 3))
y = X @ np.array([1.0, -0.5, 0.3]) + np.random.default_rng(1).normal(size=n) * 0.1

fe = PanelOLS(entity_effects=True, device="cpu").fit(
    X, y, entity_ids=entity_ids
)
print(fe.fit_statistics_.rsquared_within)
print(fe.pooling_f_test())

pooled = PooledOLS(device="cpu").fit(X, y, entity_ids=entity_ids)
print(pooled.breusch_pagan_lm_test())

re = RandomEffects(device="cpu").fit(X, y, entity_ids=entity_ids)
print(fe.hausman_test(re))
```

For CuPy CUDA use `device="cuda"`; for Torch CUDA use CUDA tensors and `device="torch"`. Stage-B statistical transforms and sufficient-statistic accumulation follow the selected numerical backend. Formula/label metadata, final scalars, and small covariance matrices use the CPU metadata boundary; Hausman-compatible one-way FE/RE fits additionally perform bounded chunked host copies of canonical slope X/y solely for collision-resistant identity hashing.

## Outputs

Common fitted attributes include:

- `coef_`;
- `bse_`, `tvalues_`, `pvalues_`, `conf_int_` when coefficient-space inference is identifiable;
- legacy `rsquared` or `rsquared_within` where historically exposed;
- standardized `fit_statistics_`;
- `nobs`, `df_resid`, and effective rank where exposed;
- `betas_`, `cov_params_`, and `n_periods` for `FamaMacBeth`.

`PanelTestResult` contains `statistic`, `pvalue`, `distribution`, `df`, `null`, `alternative`, `applicable`, `reason`, and `metadata`.

## Formula and Metadata Boundaries

Formula evaluation may drop rows with missing values. Entity, time, cluster, and other side arrays are aligned to the retained rows. String and categorical labels are factorized on CPU; numerical transforms and sufficient-statistic calculations remain on the selected backend. For Hausman-compatible fits, canonical slope X/y are copied to host in bounded chunks for SHA-256 identity, after excluding any RE-only explicit constant; the fitted model stores only the digest, original coefficient-index map, feature/entity metadata, and a small covariance matrix rather than a second CPU copy of the design.

## Validation

Stage A / PR #119 established the shared panel framework and passed exact-head physical validation on Tesla P100 across 10 CuPy and 10 Torch cases.

Stage B adds maintained analytic and fitted-model regression tests, formula/missing-row alignment tests, Python 3.9 + Torch 2.0 CPU parity coverage, and an executable `linearmodels==7.0` external-definition gate. The physical runner contains 17 estimator cases per backend and four Hausman diagnostic cases per backend, including balanced/unbalanced RandomEffects with an explicit constant and FE-versus-RE Hausman where FE absorbs that intercept. Final promotion requires `dev/benchmarks/validate_panel_stage_b_gpu.py` to pass on an exact clean commit for both CuPy and Torch CUDA; this runner is a correctness/provenance gate rather than a performance benchmark. A separate physical benchmark measures the remaining full-content identity overhead on Hausman-compatible FE/RE fits.

## References

- Hausman, J. A. (1978). Specification tests in econometrics.
- Breusch, T. S., & Pagan, A. R. (1980). The Lagrange multiplier test and its applications to model specification in econometrics.
- Baltagi, B. H., & Li, Q. (1990). A Lagrange multiplier test for the error components model with incomplete panels.
- White, H. (1980). A heteroskedasticity-consistent covariance matrix estimator.
- Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix.
- Fama, E. F., & MacBeth, J. D. (1973). Risk, return, and equilibrium.
- Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011). Robust inference with multiway clustering.
