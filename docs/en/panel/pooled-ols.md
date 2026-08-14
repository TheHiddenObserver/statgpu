# PooledOLS

> Language: English  
> Last updated: 2026-08-14  
> Switch: [Chinese](../../cn/panel/pooled-ols.md)

## Overview

`PooledOLS` fits one common linear regression to the stacked panel and always includes an intercept. Entity identifiers are optional metadata for panel fit statistics and diagnostics, not coefficient estimation.

## Path

Implementation: `statgpu/panel/_pooled.py`.

## Objective and Estimator

With $Z=[\mathbf 1,X]$,

$$
\widehat\beta_{\mathrm{pooled}}
=\arg\min_\beta\|y-Z\beta\|_2^2
=(Z^\top Z)^+Z^\top y.
$$

## Covariance and Inference

Covariance uses the level design $Z$; see [Panel covariance](covariance.md). `cov_type="hac"` is the legacy row-order Bartlett/Newey-West HAC, while Driscoll-Kraay first groups influence scores by `time_index`.

## Parameters

| Parameter | Meaning |
|---|---|
| `cov_type` | Nonrobust, HC, clustered, legacy HAC, or Driscoll-Kraay. |
| `bandwidth`, `kernel` | HAC/DK controls where applicable. |
| `group_debias` | Small-group cluster correction. |
| `alpha` | Confidence-interval significance level. |
| `device` | `auto`, `cpu`, `cuda`, or `torch`. |
| `n_jobs` | Shared parallelism hint. |

```python
model.fit(X, y, cluster=None, time_index=None, entity_ids=None)
```

`cluster` is required for clustered covariance and `time_index` for Driscoll-Kraay. `entity_ids` enables standardized within/between $R^2$ and the Breusch-Pagan LM test. Formula input must retain the model intercept.

## CPU and GPU Example

```python
from statgpu.panel import PooledOLS

cpu = PooledOLS(device="cpu").fit(X, y)
cuda = PooledOLS(device="cuda").fit(X, y)
torch = PooledOLS(device="torch").fit(X, y)
```

Explicit GPU requests do not silently fall back to CPU.

## Outputs

Public results include `coef_`, `bse_`, `tvalues_`, `pvalues_`, `conf_int_`, `rsquared`, `fit_statistics_`, `nobs`, and `df_resid`. With `entity_ids`, `breusch_pagan_lm_test()` is available; see [Panel diagnostics](diagnostics.md).

## Numerical and Strict Behavior

There is no approximate fallback for unsupported covariance inputs: missing required `cluster`/`time_index`, invalid cluster shape, or unavailable explicit GPU backends raise. Exact rank deficiency follows the shared panel inference contract in [Panel covariance](covariance.md).

## FAQ

**Does supplying `entity_ids` change coefficients?**  No; it only enables panel-aware statistics and diagnostics.

**Is `hac` the same as Driscoll-Kraay?**  No. Legacy HAC is row-order Newey-West; DK aggregates scores by time labels.

## External Validation

Estimator-level DK and clustered covariance are compared with `linearmodels==7.0` in `dev/tests/test_panel_stage_c_linearmodels_estimators.py`; overlapping OLS definitions also use `statsmodels==0.14.6`. R `plm==2.6-7` / `sandwich==3.1-3` checks are maintained in `dev/tests/test_panel_stage_c_r_external.py`. Physical GPU acceptance is in `results/pr126_p100_fresh/validation_summary.txt`.

## References

Wooldridge (2010), *Econometric Analysis of Cross Section and Panel Data*; covariance references are collected in [Panel covariance](covariance.md).
