# Panel Models

> Language: English  
> Last updated: 2026-08-08  
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

Stage B of the Tier-1 panel roadmap adds parameter-based fit statistics and three structured specification tests without changing the Stage-A coefficient, prediction, covariance-normalization, or legacy inference contracts.

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
)
```

The diagnostic result classes and functions are also exported from top-level `statgpu`.

## Model Summary

| Model | Transformation | Main inference choices | Stage-B fit statistics |
|---|---|---|---|
| `PanelOLS` | Entity/time within transformation | nonrobust, HC1 robust, clustered | within/between/overall R², adjusted R², classical model F, pooling F |
| `RandomEffects` | Swamy-Arora feasible GLS | nonrobust | within/between/overall R², adjusted R², classical model F, Hausman input |
| `PooledOLS` | Stacked OLS | nonrobust, robust, clustered, HAC | overall R² always; within/between R² and BP-LM when `entity_ids` is supplied; adjusted R² and classical model F |
| `BetweenOLS` | Entity means | nonrobust, robust | within/between/overall R², adjusted R², classical model F |
| `FirstDifferenceOLS` | Within-entity first differences | nonrobust, robust | within/between/overall R², adjusted R² on the differenced fit space, classical model F |
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

## Covariance and Existing Inference

| `cov_type` | Behavior |
|---|---|
| `"nonrobust"` | Classical OLS covariance and t-based inference |
| `"robust"` | HC1 sandwich covariance and asymptotic normal inference |
| `"clustered"` | Cluster-robust covariance where supported by the estimator |
| `"hac"` | Bartlett/Newey-West HAC for `PooledOLS` |
| `"newey-west"` | HAC applied to the `FamaMacBeth` coefficient path |

Stage B does not change existing `bse_`, `tvalues_`, `pvalues_`, `conf_int_`, or estimator-specific covariance definitions. RandomEffects robust covariance, HC0/HC2/HC3, Driscoll-Kraay, and expanded cluster corrections remain later Stage-C work.

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

Overall and between total sums of squares are centered only when the actual level regressor design contains an identified constant. Fixed effects by themselves do not change this centering rule. A zero total sum of squares is reported as `0.0` in the standardized Stage-B field and marked in `metadata["degenerate_total_ss"]`.

### Legacy `PanelOLS.rsquared_within`

`PanelOLS.rsquared_within` is retained exactly for Stage-A compatibility. In a two-way FE model it describes the historical full entity+time transformed fit and can differ from the standardized entity-within `fit_statistics_.rsquared_within`. Stage B does not silently overwrite the legacy attribute; the compatibility value is recorded in fit-statistics metadata.

### Adjusted R² and diagnostic degrees of freedom

For new standardized diagnostics, `PanelOLS` uses the rank of the complete fixed-effect nuisance space. Current `PanelOLS` does not retain an exogenous intercept, so the standard nuisance-effect rank is:

- entity effects only: \(N\);
- time effects only: \(T\);
- entity and time effects: \(N+T-1\).

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

where \(q\) is the effective restriction rank. A robust or clustered covariance choice does not silently convert this field into a robust Wald test.

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
- FE and RE must be fitted to the same aligned X/y/entity sample and common slope design;
- row/sample compatibility is checked using compact backend-native numerical fingerprints, not just matching shapes;
- intercepts are excluded from the comparison;
- a materially indefinite covariance difference is reported as inapplicable instead of being eigenvalue-clipped into a statistic.

If the covariance difference is positive semidefinite but rank-deficient, statgpu provides a documented generalized-inverse extension: the test uses the identified range and chi-square degrees of freedom equal to the numerical rank, but only when the coefficient difference lies in that range. Metadata records `used_pinv=True` and labels this as the `singular PSD generalized-inverse Hausman` extension. Robust auxiliary-regression Hausman is not part of Stage B.

## Parameters and Fit Signatures

### `PanelOLS`

```python
PanelOLS(
    entity_effects=False,
    time_effects=False,
    cov_type="nonrobust",
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
    cov_type="nonrobust",
    alpha=0.05,
    bandwidth=None,
    kernel="bartlett",
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

`cluster` is required for clustered inference. `time_index` defines stable temporal ordering for HAC. `entity_ids` is optional and does not change coefficients; it enables standardized within/between R² and the panel BP-LM diagnostic.

### Other models

```python
RandomEffects(device="auto")
BetweenOLS(cov_type="nonrobust", alpha=0.05, device="auto")
FirstDifferenceOLS(cov_type="nonrobust", alpha=0.05, device="auto")
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

For CuPy CUDA use `device="cuda"`; for Torch CUDA use CUDA tensors and `device="torch"`. Stage-B sufficient-statistic accumulation follows the selected numerical backend. Only compact metadata, final scalars, and small covariance matrices cross the CPU metadata boundary.

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

Formula evaluation may drop rows with missing values. Entity, time, cluster, and other side arrays are aligned to the retained rows. String and categorical labels are factorized on CPU; numerical transforms and sufficient-statistic calculations remain on the selected backend. Hausman stores only compact sample/design fingerprints and a small covariance matrix rather than a second CPU copy of the full design.

## Validation

Stage A / PR #119 established the shared panel framework and passed exact-head physical validation on Tesla P100 across 10 CuPy and 10 Torch cases.

Stage B adds maintained analytic and fitted-model regression tests, formula/missing-row alignment tests, Python 3.9 + Torch 2.0 CPU parity coverage, and an executable `linearmodels==7.0` external-definition gate. Final promotion also requires `dev/benchmarks/validate_panel_stage_b_gpu.py` to pass on an exact clean commit for both CuPy and Torch CUDA; this runner is a correctness/provenance gate rather than a performance benchmark.

## References

- Hausman, J. A. (1978). Specification tests in econometrics.
- Breusch, T. S., & Pagan, A. R. (1980). The Lagrange multiplier test and its applications to model specification in econometrics.
- Baltagi, B. H., & Li, Q. (1990). A Lagrange multiplier test for the error components model with incomplete panels.
- White, H. (1980). A heteroskedasticity-consistent covariance matrix estimator.
- Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix.
- Fama, E. F., & MacBeth, J. D. (1973). Risk, return, and equilibrium.
- Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011). Robust inference with multiway clustering.
