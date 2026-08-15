# FirstDifferenceOLS

> Language: English  
> Last updated: 2026-08-15  
> Switch: [Chinese](../../cn/panel/first-difference-ols.md)

## Overview

`FirstDifferenceOLS` removes time-invariant entity effects by subtracting each entity's previous **observed** value from its current observed value. It then regresses the differenced outcome on the differenced predictors without an intercept.

This model is useful when identification comes from changes within an entity over time rather than differences in levels across entities.

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

Here $t^-$ means the previous **observed** time for that entity. Missing calendar periods are not created, and the difference is not divided by the length of a calendar gap.

## Covariance and Inference

Standard errors are computed from the differenced regression $(\Delta y,\Delta X)$. Supported choices are nonrobust and HC0/HC1/HC2/HC3; see [Panel covariance](covariance.md).

## Parameters

| Parameter | Default | Allowed / Constraint | Meaning |
|---|---:|---|---|
| `cov_type` | `"nonrobust"` | `nonrobust`, `robust`/`hc1`, `hc0`, `hc2`, `hc3` | How standard errors are computed after differencing. |
| `alpha` | `0.05` | finite and strictly between 0 and 1 | Confidence-interval significance level; `0.05` gives 95% intervals. |
| `device` | `"auto"` | `auto`, `cpu`, `cuda`, `torch` | Where numerical computation runs. |
| `n_jobs` | `None` | integer or `None` | Shared parallelism hint. |

```python
model.fit(X, y, entity_ids=entity_ids, time_ids=None)
```

`entity_ids` is required. Supplying `time_ids` lets statgpu put each entity's rows in chronological order; each `(entity_id, time_id)` pair must then be unique. Ordered categorical time labels use their declared category order.

## CPU and GPU Example

```python
from statgpu.panel import FirstDifferenceOLS

cpu = FirstDifferenceOLS(device="cpu").fit(X, y, entity_ids=entity_ids, time_ids=time_ids)
cuda = FirstDifferenceOLS(device="cuda").fit(X, y, entity_ids=entity_ids, time_ids=time_ids)
torch = FirstDifferenceOLS(device="torch").fit(X, y, entity_ids=entity_ids, time_ids=time_ids)
```

If an explicitly requested GPU backend is unavailable, `.fit()` raises an error rather than switching to CPU.

## Formula Example

Assume `df` contains `y`, `x1`, `x2`, `entity`, and `time` columns.

```python
from statgpu.panel import FirstDifferenceOLS

model = FirstDifferenceOLS().fit(
    formula="y ~ x1 + x2 - 1",
    data=df,
    entity_ids=df["entity"],
    time_ids=df["time"],
)
```

The formula removes the intercept because `FirstDifferenceOLS` does not estimate one after differencing.

## Outputs

Public results include `coef_`, `bse_`, `tvalues_`, `pvalues_`, `conf_int_`, `rsquared`, `fit_statistics_`, `nobs`, and `df_resid`. `nobs` refers to the differenced observations used in the final regression, so it is smaller than the original panel row count.

## Numerical and Strict Behavior

When `time_ids` is supplied, duplicate observations for the same entity and time raise an error because the chronological difference would be ambiguous. Calendar gaps are left as gaps: statgpu differences adjacent observed rows and does not insert missing periods or rescale by elapsed time.

If the differenced predictors are exactly collinear, fitted values can still be computed but the coefficient vector is not unique. statgpu therefore disables coefficient-level standard errors, tests, p-values, and confidence intervals for that fit. Invalid covariance choices or unavailable explicitly requested GPU backends also raise clear errors.

## FAQ

**Does a two-period calendar gap create a two-step difference?**  No. The method subtracts consecutive observed rows after chronological ordering, regardless of the calendar gap length.

**Is an intercept estimated after differencing?**  No.

## External Validation

We construct the identical differenced sample in `statsmodels==0.14.6` and compare coefficients plus HC0/HC2/HC3 covariance and standard errors. Coefficients use `rtol=5e-10, atol=5e-12`; covariance/BSE use `rtol=5e-9, atol=5e-11`. Shared covariance checks are listed in the [validation matrix](covariance.md#validation-matrix).

GPU consistency is tested separately by comparing CuPy and Torch results with NumPy at default `rtol=5e-6, atol=5e-7`.

## References

- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). The MIT Press.
