# FirstDifferenceOLS

> Language: English  
> Last updated: 2026-08-19  
> Switch: [Chinese](../../cn/panel/first-difference-ols.md)

## Overview

`FirstDifferenceOLS` removes a time-invariant entity effect by subtracting each entity's previous **observed** value from its current observed value. It then regresses the differenced outcome on the differenced predictors without an intercept.

This estimator can be applied to the same one-way fixed-parameter entity-effect model as a fixed-effects estimator; the difference is the transformation used to eliminate the entity effect.

## Path

Implementation: `statgpu/panel/_first_diff.py`.

## Statistical Model and Identification

Start from the one-way fixed-parameter panel model

$$
y_{it}=x_{it}^{\top}\beta+a_i+\varepsilon_{it},
$$

where $a_i$ is a fixed but unknown time-invariant entity effect. As with the fixed-effects model, $a_i$ need not be orthogonal to the regressor history. A common sufficient exogeneity condition for a static model is

$$
E\!\left(\varepsilon_{it}\mid X_i,a_i\right)=0,
\qquad
X_i=(x_{i1},\ldots,x_{iT_i}).
$$

Taking first differences removes $a_i$ exactly:

$$
\Delta y_{it}=\Delta x_{it}^{\top}\beta+\Delta\varepsilon_{it}.
$$

The slope is therefore identified from within-entity changes. A regressor that is constant over time within an entity differences to zero and cannot identify a slope in this specification. A common intercept in the level model is also removed by differencing, which is why the final regression has no intercept.

## Estimator

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

Changing the covariance estimator changes uncertainty for the differenced regression; it does not replace the exogeneity condition needed for the usual first-difference interpretation of $\beta$.

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

`entity_ids` is required. Supplying `time_ids` lets statgpu put each entity's rows in chronological order; each `(entity_id, time_id)` pair must then be unique. Numeric and datetime time labels use natural order and ordered categoricals use their declared category order. Plain string labels use lexical order; other non-categorical object labels follow their sorted comparable-value order and fail if they are not mutually comparable. Use an ordered categorical or numeric/datetime key when lexical string order is not the intended chronology.

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

The differenced regression uses the shared certified panel least-squares policy. Cancellation-sensitive response projections use the magnitude-tiered reduction path; if a nonzero coefficient is below the numerically certifiable float64 projection resolution and the candidate materially violates least-squares stationarity, `.fit()` raises `FloatingPointError` instead of returning a finite but unreliable coefficient. This is distinct from exact collinearity. If the differenced predictors are exactly collinear, fitted values can still be computed but the coefficient vector is not unique, so coefficient-level standard errors, tests, p-values, and confidence intervals are disabled.

Legacy `rsquared` is also range-safe. When the physical subtraction $\Delta y-\overline{\Delta y}$ would overflow near the float64 boundary, the response and residual are placed on a common dimensionless centering scale before the scale-invariant $R^2$ ratio is formed. Ordinary-scale centering is unchanged. Invalid covariance choices or unavailable explicitly requested GPU backends raise clear errors.

## FAQ

**Does a two-period calendar gap create a two-step difference?**  No. The method subtracts consecutive observed rows after chronological ordering, regardless of the calendar gap length.

**Is an intercept estimated after differencing?**  No.

## External Validation

We construct the identical differenced sample in `statsmodels==0.14.6` and compare coefficients plus HC0/HC2/HC3 covariance and standard errors. Coefficients use `rtol=5e-10, atol=5e-12`; covariance/BSE use `rtol=5e-9, atol=5e-11`. Shared covariance checks are listed in the [validation matrix](covariance.md#validation-matrix).

GPU consistency is tested separately by comparing CuPy and Torch results with NumPy at default `rtol=5e-6, atol=5e-7`. The current exact-head physical gate additionally exercises shared coefficient-resolution fail-closed behavior and an extreme differenced-response case whose physical centering would exceed float64 range.

## References

- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). The MIT Press.
