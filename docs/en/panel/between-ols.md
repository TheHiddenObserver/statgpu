# BetweenOLS

> Language: English  
> Last updated: 2026-08-15  
> Switch: [Chinese](../../cn/panel/between-ols.md)

## Overview

`BetweenOLS` first averages each variable within entity, leaving one observation per entity, and then runs an OLS regression with an intercept. It is useful when the relationship of interest is driven by differences **between** entities rather than changes within the same entity over time.

Because every retained entity contributes one averaged observation, the effective sample size is the number of retained entities, not the number of original panel rows.

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

In other words, the original panel is reduced to an ordinary cross-sectional regression of entity means.

## Covariance and Inference

Standard errors are computed from that entity-mean regression. `cov_type="nonrobust"` uses the usual homoskedastic OLS covariance; `robust`/`hc1` and HC0/HC2/HC3 provide heteroskedasticity-consistent alternatives. The shared formulas are given in [Panel covariance](covariance.md).

## Parameters

| Parameter | Default | Allowed / Constraint | Meaning |
|---|---:|---|---|
| `cov_type` | `"nonrobust"` | `nonrobust`, `robust`/`hc1`, `hc0`, `hc2`, `hc3` | How coefficient standard errors are computed after entity averaging. |
| `alpha` | `0.05` | finite and strictly between 0 and 1 | Confidence-interval significance level; `0.05` gives 95% intervals. |
| `device` | `"auto"` | `auto`, `cpu`, `cuda`, `torch` | Where numerical computation runs. |
| `n_jobs` | `None` | integer or `None` | Shared parallelism hint. |

```python
model.fit(X, y, entity_ids=entity_ids)
```

`entity_ids` is required so the observations can be averaged within entity.

## CPU and GPU Example

```python
from statgpu.panel import BetweenOLS

cpu = BetweenOLS(device="cpu").fit(X, y, entity_ids=entity_ids)
cuda = BetweenOLS(device="cuda").fit(X, y, entity_ids=entity_ids)
torch = BetweenOLS(device="torch").fit(X, y, entity_ids=entity_ids)
```

If `device="cuda"` or `device="torch"` is requested but that GPU backend is unavailable, `.fit()` raises an error rather than switching to CPU.

## Formula Example

Assume `df` contains `y`, `x1`, `x2`, and `entity` columns.

```python
from statgpu.panel import BetweenOLS

model = BetweenOLS().fit(
    formula="y ~ x1 + x2",
    data=df,
    entity_ids=df["entity"],
)
```

`BetweenOLS` always includes an intercept; explicit no-intercept formulas are rejected.

## Outputs

Public results include `coef_`, `bse_`, `tvalues_`, `pvalues_`, `conf_int_`, `rsquared`, `fit_statistics_`, `nobs`, and `df_resid`. Here `nobs` is the number of entity means used in the final regression.

## Numerical and Strict Behavior

If the entity-level regressors are exactly collinear, statgpu can still compute fitted values using a least-squares solution, but the coefficient vector is not unique. For that fit, coefficient-level standard errors, tests, p-values, and confidence intervals are disabled rather than being computed from an arbitrary coefficient representation.

Invalid covariance choices raise an error. Likewise, an explicitly requested GPU backend must be available; statgpu does not silently run the model on CPU instead.

## FAQ

**Why is `nobs` smaller than the original row count?**  `nobs` is the number of entity-mean observations used in the between regression.

**Are entities with more time observations automatically weighted more heavily?**  No. Each retained entity contributes one mean observation to the final OLS regression.

## External Validation

We compare `BetweenOLS` with `statsmodels==0.14.6` after constructing the same entity-mean regression in both packages. The checks cover coefficients and HC0/HC2/HC3 standard errors/covariances: coefficients use `rtol=5e-10, atol=5e-12`, and covariance/BSE use `rtol=5e-9, atol=5e-11`. Shared covariance checks are summarized in the [validation matrix](covariance.md#validation-matrix).

GPU consistency is tested separately by comparing CuPy and Torch results with NumPy using the Stage-C physical validation tolerance `rtol=5e-6, atol=5e-7`.

## References

- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). The MIT Press.
