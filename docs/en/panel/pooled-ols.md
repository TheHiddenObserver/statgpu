# PooledOLS

> Language: English  
> Last updated: 2026-08-16  
> Switch: [Chinese](../../cn/panel/pooled-ols.md)

## Overview

`PooledOLS` stacks all panel observations and fits one ordinary linear regression with a common intercept and common slopes. It does **not** remove entity or time effects, so it is appropriate when a single pooled conditional-mean relationship is the intended model.

`entity_ids` is optional: supplying it does not change the coefficient estimates, but it enables panel-specific fit statistics and the Breusch-Pagan LM diagnostic.

## Path

Implementation: `statgpu/panel/_pooled.py`.

## Statistical Model and Identification

The pooled linear model can be written as

$$
y_{it}=\alpha+x_{it}^{\top}\beta+u_{it}.
$$

A common sufficient condition for interpreting $\beta$ as the pooled conditional-mean slope is

$$
E(u_{it}\mid X_i)=0,
$$

where $X_i=(x_{i1},\ldots,x_{iT_i})$. The model therefore treats any unmodeled entity or time heterogeneity as part of the composite error $u_{it}$.

For example, if the underlying data satisfy

$$
y_{it}=\alpha+x_{it}^{\top}\beta+a_i+\varepsilon_{it},
$$

then pooled OLS does not remove $a_i$. To recover the same structural $\beta$, the combined error $a_i+\varepsilon_{it}$ must still be orthogonal to the regressors. If entity heterogeneity is correlated with the regressor history, pooled OLS generally does not identify the same slope as a fixed-effects estimator.

Covariance choices such as HC, clustering, HAC, or Driscoll-Kraay change how uncertainty is estimated; they do not repair a failure of this mean-model exogeneity condition.

## Estimator

With $Z=[\mathbf 1,X]$,

$$
\widehat\beta_{\mathrm{pooled}}
=\arg\min_\beta\|y-Z\beta\|_2^2
=(Z^\top Z)^+Z^\top y.
$$

Thus the coefficient estimate is the same OLS fit you would obtain from treating the stacked panel rows as one regression sample.

## Covariance and Inference

`cov_type` changes the standard-error calculation, not the OLS coefficient estimate. In addition to nonrobust and HC covariance, `PooledOLS` supports clustered covariance and two time-dependent choices:

- `cov_type="hac"` treats the observations as one ordered sequence and applies Bartlett/Newey-West HAC. If `time_index` is supplied, the rows are first sorted by its chronology; otherwise the input row order is used. Numeric and datetime labels use their natural order. An ordered pandas categorical uses the category order declared by the user. Plain string labels use lexical order; other non-categorical object labels follow their sorted comparable-value order and fail if they are not mutually comparable. Labels such as `t1, t2, t10` should therefore be supplied as an ordered categorical (or a numeric/datetime key) when lexical string order is not the intended chronology.
- `cov_type="driscoll-kraay"` groups observations by `time_index` and combines their contributions within each period before applying lag weights.

These two choices are therefore not interchangeable. Full formulas are in [Panel covariance](covariance.md).

## Parameters

| Parameter | Default | Allowed / Constraint | Meaning |
|---|---:|---|---|
| `cov_type` | `"nonrobust"` | `nonrobust`, `robust`/`hc1`, `hc0`, `hc2`, `hc3`, `clustered`, `hac`, `driscoll-kraay`/`dk`/`kernel` | How coefficient standard errors are computed. |
| `alpha` | `0.05` | finite and strictly between 0 and 1 | Confidence-interval significance level; `0.05` gives 95% intervals. |
| `bandwidth` | `None` | `None` or a non-negative integer | HAC/DK lag or smoothing bandwidth. Legacy HAC uses at most $n-1$ lags; DK follows the observed-period rule in [Panel covariance](covariance.md). |
| `kernel` | `"bartlett"` | `hac` requires Bartlett; DK also accepts Parzen and QS aliases | Kernel used by HAC/DK covariance. |
| `device` | `"auto"` | `auto`, `cpu`, `cuda`, `torch` | Where numerical computation runs. |
| `n_jobs` | `None` | integer or `None` | Shared parallelism hint. |
| `group_debias` | `False` | boolean; clustered covariance only | Apply the small-number-of-clusters correction. |

```python
model.fit(X, y, cluster=None, time_index=None, entity_ids=None)
```

Pass `cluster` when using clustered covariance and `time_index` when using Driscoll-Kraay. For legacy HAC, `time_index` is optional and, when supplied, defines the ordering used by the HAC calculation. Pass `entity_ids` when you also want standardized within/between $R^2$ or the Breusch-Pagan LM test.

## CPU and GPU Example

```python
from statgpu.panel import PooledOLS

cpu = PooledOLS(device="cpu").fit(X, y)
cuda = PooledOLS(device="cuda").fit(X, y)
torch = PooledOLS(device="torch").fit(X, y)
```

If an explicitly requested GPU backend is unavailable, `.fit()` raises an error instead of switching to CPU.

## Formula Example

Assume `df` contains `y`, `x1`, and `x2` columns.

```python
from statgpu.panel import PooledOLS

model = PooledOLS().fit(
    formula="y ~ x1 + x2",
    data=df,
)
```

`PooledOLS` always includes an intercept, so explicit no-intercept formulas are rejected.

## Outputs

Public results include `coef_`, `bse_`, `tvalues_`, `pvalues_`, `conf_int_`, `rsquared`, `fit_statistics_`, `nobs`, and `df_resid`. With `entity_ids`, `breusch_pagan_lm_test()` is also available; see [Panel diagnostics](diagnostics.md).

## Numerical and Strict Behavior

Required metadata are checked before covariance is computed. For example, clustered covariance without `cluster`, Driscoll-Kraay without `time_index`, or a cluster array with the wrong shape raises an error rather than producing a different covariance estimator. Legacy HAC also rejects missing/non-finite time metadata instead of guessing an ordering.

A new `fit()` attempt invalidates the previous fitted/inference state before work begins. If the refit fails at any later stage, partially written outputs are cleared as well; `predict()` and `summary()` then report the estimator as unfitted.

If the design matrix is exactly rank deficient, fitted values can still be computed but the coefficient vector is not unique. statgpu therefore disables coefficient-level standard errors, tests, p-values, and confidence intervals for that fit instead of reporting inference from an arbitrary coefficient representation; see [Panel covariance](covariance.md).

An explicit `device="cuda"` or `device="torch"` request also raises if that backend is unavailable rather than silently using CPU.

## FAQ

**Does supplying `entity_ids` change coefficients?**  No. It only enables panel-aware fit statistics and diagnostics.

**Is `hac` the same as Driscoll-Kraay?**  No. HAC treats observations as one ordered sequence; Driscoll-Kraay first aggregates observation contributions within each supplied time period.

## External Validation

We compare the public `PooledOLS` estimator with `linearmodels==7.0` for Driscoll-Kraay coefficients, covariance, and BSE, and for group-debiased clustered covariance. Coefficients use `rtol=2e-10, atol=2e-11`; covariance/BSE use `rtol=5e-9, atol=5e-11`. Definition-level HC, clustering, Driscoll-Kraay, default-bandwidth, and R `sandwich` checks are summarized in the [validation matrix](covariance.md#validation-matrix).

GPU consistency is tested separately by comparing CuPy and Torch outputs with NumPy at default `rtol=5e-6, atol=5e-7`; observed maximum differences are stored in the PR #126 physical validation artifacts. The dedicated `dev/benchmarks/validate_panel_hac_chronology_gpu.py` gate additionally checks ordered-categorical legacy-HAC chronology, a lexical-order negative control, formula missing-row alignment, and requested/executed CuPy/Torch backend identity on the final exact source.

## References

- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). The MIT Press.

Covariance-specific sources are collected in [Panel covariance](covariance.md).
