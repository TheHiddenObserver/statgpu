# FirstDifferenceOLS

> Language: English  
> Last updated: 2026-08-14  
> Switch: [Chinese](../../cn/panel/first-difference-ols.md)

## Overview

`FirstDifferenceOLS` removes time-invariant entity effects by differencing consecutive observed periods within each entity and then fitting OLS without an intercept.

## Path

Implementation: `statgpu/panel/_first_diff.py`.

## Objective and Estimator

For consecutive observed periods within entity $i$,

$$
\Delta y_{it}=y_{it}-y_{i,t^-},
\qquad
\Delta x_{it}=x_{it}-x_{i,t^-},
$$

and

$$
\widehat\beta_{\mathrm{FD}}
=\arg\min_\beta\|\Delta y-\Delta X\beta\|_2^2
=(\Delta X^\top\Delta X)^+\Delta X^\top\Delta y.
$$

## Covariance and Inference

Covariance uses $Z=\Delta X$; see [Panel covariance](covariance.md). Supported choices are nonrobust and HC0/1/2/3.

## Parameters

| Parameter | Default | Allowed / Constraint | Meaning |
|---|---:|---|---|
| `cov_type` | `"nonrobust"` | `nonrobust`, `robust`/`hc1`, `hc0`, `hc2`, `hc3` | Covariance estimator on the differenced fit space. |
| `alpha` | `0.05` | Significance level; `0.05` gives 95% confidence intervals. | Confidence-interval level control. |
| `device` | `"auto"` | `auto`, `cpu`, `cuda`, `torch` | Numerical backend/device. |
| `n_jobs` | `None` | integer or `None` | Shared parallelism hint. |

```python
model.fit(X, y, entity_ids=entity_ids, time_ids=None)
```

`entity_ids` is required. When `time_ids` is supplied, `(entity_id, time_id)` pairs must be unique. Differences use consecutive **observed** times; calendar gaps are neither filled nor divided out. Ordered categorical time labels preserve declared chronology.

## CPU and GPU Example

```python
from statgpu.panel import FirstDifferenceOLS

cpu = FirstDifferenceOLS(device="cpu").fit(X, y, entity_ids=entity_ids, time_ids=time_ids)
cuda = FirstDifferenceOLS(device="cuda").fit(X, y, entity_ids=entity_ids, time_ids=time_ids)
torch = FirstDifferenceOLS(device="torch").fit(X, y, entity_ids=entity_ids, time_ids=time_ids)
```

## Outputs

Public results include `coef_`, `bse_`, `tvalues_`, `pvalues_`, `conf_int_`, `rsquared`, `fit_statistics_`, `nobs`, and `df_resid`.

## Numerical and Strict Behavior

The maintained transformation never invents missing calendar periods. Duplicate entity-time observations fail closed when `time_ids` is supplied. There is no silent approximate-inference or backend fallback.

## FAQ

**Does a two-period calendar gap create a two-step difference?**  No. The method differences consecutive observed rows after chronological ordering.

**Is an intercept estimated after differencing?**  No.

## External Validation

`statsmodels==0.14.6` is fit on the identical differenced sample for HC0/HC2/HC3. Coefficients are asserted with `rtol=5e-10, atol=5e-12`; covariance and BSE use `rtol=5e-9, atol=5e-11`. Shared covariance-definition checks are listed in the [validation matrix](covariance.md#validation-matrix).

The Stage-C physical runner separately compares FirstDifferenceOLS CuPy/Torch cases with NumPy at default `rtol=5e-6, atol=5e-7`.

## References

- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). The MIT Press.
