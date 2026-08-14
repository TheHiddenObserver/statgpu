# BetweenOLS

> Language: English  
> Last updated: 2026-08-14  
> Switch: [Chinese](../../cn/panel/between-ols.md)

## Overview

`BetweenOLS` collapses the panel to entity means and runs an intercept-including OLS regression across entities. Its effective sample size is the number of retained entities.

## Path

Implementation: `statgpu/panel/_between.py`.

## Objective and Estimator

For entity $i$,

$$
\bar y_i=T_i^{-1}\sum_t y_{it},
\qquad
\bar x_i=T_i^{-1}\sum_t x_{it},
$$

and with $\bar Z_i=(1,\bar x_i^\top)^\top$,

$$
\widehat\beta_B
=\arg\min_\beta\sum_i(\bar y_i-\bar Z_i^\top\beta)^2
=(\bar Z^\top\bar Z)^+\bar Z^\top\bar y.
$$

## Covariance and Inference

Covariance is computed on the entity-mean fit space. Nonrobust and HC0/1/2/3 are defined in [Panel covariance](covariance.md).

## Parameters

| Parameter | Default | Allowed / Constraint | Meaning |
|---|---:|---|---|
| `cov_type` | `"nonrobust"` | `nonrobust`, `robust`/`hc1`, `hc0`, `hc2`, `hc3` | Covariance estimator on the entity-mean fit space. |
| `alpha` | `0.05` | Significance level; `0.05` gives 95% confidence intervals. | Confidence-interval level control. |
| `device` | `"auto"` | `auto`, `cpu`, `cuda`, `torch` | Numerical backend/device. |
| `n_jobs` | `None` | integer or `None` | Shared parallelism hint. |

```python
model.fit(X, y, entity_ids=entity_ids)
```

`entity_ids` is required. The model always includes an intercept; explicit no-intercept formulas are rejected.

## CPU and GPU Example

```python
from statgpu.panel import BetweenOLS

cpu = BetweenOLS(device="cpu").fit(X, y, entity_ids=entity_ids)
cuda = BetweenOLS(device="cuda").fit(X, y, entity_ids=entity_ids)
torch = BetweenOLS(device="torch").fit(X, y, entity_ids=entity_ids)
```

## Outputs

Public results include `coef_`, `bse_`, `tvalues_`, `pvalues_`, `conf_int_`, `rsquared`, `fit_statistics_`, `nobs`, and `df_resid`.

## Numerical and Strict Behavior

The regression is solved in the entity-mean fit space with rank-aware linear algebra. There is no silent approximate-inference or backend fallback; unsupported covariance choices and unavailable explicit GPU backends raise.

## FAQ

**Why is `nobs` smaller than the original row count?**  `nobs` is the number of entity-mean observations used in the between regression.

**Are larger entities automatically weighted more heavily?**  No. The maintained estimator runs OLS on one mean observation per entity.

## External Validation

`statsmodels==0.14.6` is fit on the same entity-mean design for HC0/HC2/HC3. Coefficients are asserted with `rtol=5e-10, atol=5e-12`; covariance and BSE use `rtol=5e-9, atol=5e-11`. Shared covariance-definition checks are listed in the [validation matrix](covariance.md#validation-matrix).

The Stage-C physical runner separately compares BetweenOLS CuPy/Torch cases with NumPy at default `rtol=5e-6, atol=5e-7`.

## References

- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). The MIT Press.
