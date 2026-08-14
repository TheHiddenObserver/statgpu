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

| Parameter | Default | Allowed / Constraint | Meaning |
|---|---:|---|---|
| `cov_type` | `"nonrobust"` | `nonrobust`, `robust`/`hc1`, `hc0`, `hc2`, `hc3`, `clustered`, `hac`, `driscoll-kraay`/`dk`/`kernel` | Covariance estimator. |
| `alpha` | `0.05` | Significance level; `0.05` gives 95% confidence intervals. | Confidence-interval level control. |
| `bandwidth` | `None` | `None` or a non-negative integer | HAC/DK bandwidth. Legacy HAC caps the effective lag at $n-1$; DK uses the observed-period rule in [Panel covariance](covariance.md). |
| `kernel` | `"bartlett"` | `hac` requires Bartlett; DK also accepts Parzen and QS aliases | HAC/DK kernel control. |
| `device` | `"auto"` | `auto`, `cpu`, `cuda`, `torch` | Numerical backend/device. |
| `n_jobs` | `None` | integer or `None` | Shared parallelism hint. |
| `group_debias` | `False` | boolean; clustered covariance only | Apply the small-group cluster correction. |

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

`linearmodels==7.0` checks the public estimator's Driscoll-Kraay coefficients, covariance, and BSE plus group-debiased clustered covariance. Coefficients use `rtol=2e-10, atol=2e-11`; covariance/BSE use `rtol=5e-9, atol=5e-11`. Definition-level HC, cluster, DK, default-bandwidth, and R `sandwich` checks are summarized in the [validation matrix](covariance.md#validation-matrix).

The Stage-C physical runner separately compares PooledOLS CuPy/Torch cases with NumPy at default `rtol=5e-6, atol=5e-7`; actual maximum absolute differences are stored in `results/pr126_p100_fresh/panel_stage_c_correctness_p100.json`.

## References

Wooldridge (2010), *Econometric Analysis of Cross Section and Panel Data*; covariance references are collected in [Panel covariance](covariance.md).
