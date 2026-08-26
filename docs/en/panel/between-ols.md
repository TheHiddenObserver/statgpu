# BetweenOLS

> Language: English  
> Last updated: 2026-08-19  
> Switch: [Chinese](../../cn/panel/between-ols.md)

## Overview

`BetweenOLS` first averages each variable within entity, leaving one observation per entity, and then runs an OLS regression with an intercept. It is useful when the relationship of interest is driven by differences **between** entities rather than changes within the same entity over time.

Because every retained entity contributes one averaged observation, the effective sample size is the number of retained entities, not the number of original panel rows.

## Path

Implementation: `statgpu/panel/_between.py`.

## Statistical Model and Identification

The algebra of the between transformation can be seen by starting from the one-way entity-effect equation

$$
y_{it}=\alpha+x_{it}^{\top}\beta+a_i+\varepsilon_{it}.
$$

Averaging over time within entity gives

$$
\bar y_i=\alpha+\bar x_i^{\top}\beta+a_i+\bar\varepsilon_i.
$$

If $a_i$ is treated as an unrestricted fixed parameter, as in the classical fixed-effects formulation, averaging does **not** eliminate it. Therefore the fixed-parameter model alone does not justify interpreting the between-regression slope as the same structural $\beta$.

To give the between slope a population interpretation across entities, one typically adds a superpopulation or cross-sectional moment assumption. For example, if $(a_i,\bar x_i,\bar\varepsilon_i)$ are viewed as varying randomly across entities, a common sufficient condition is

$$
E\!\left(a_i+\bar\varepsilon_i\mid \bar x_i\right)=0.
$$

A weaker route, sufficient for identifying the corresponding linear-projection slope, is to impose the relevant orthogonality moment directly, for example $E[\bar x_i(a_i+\bar\varepsilon_i)]=0$ after handling the intercept. Under such an additional restriction, the between regression can recover the same $\beta$ in the underlying panel equation.

If persistent entity heterogeneity is related to the entity's average regressors, `BetweenOLS` still estimates the cross-sectional linear projection of $\bar y_i$ on $\bar x_i$, but that projection generally differs from the fixed-effects or first-difference slope.

## Estimator

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

These covariance choices change inference for the between regression; they do not replace the orthogonality condition needed to interpret its slope as the structural $\beta$ from the underlying panel model.

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

The automatically added intercept uses the same cancellation-sensitive SVD response-projection guard as `PooledOLS`. Ordinary entity-mean responses keep the historical SVD/BLAS solve; magnitude/cancellation-sensitive responses retain the same SVD, rank cutoff, design scaling, and minimum-norm solution while replacing only the response projection reduction with the shared magnitude-tiered reducer. Legacy between $R^2$ centering also uses a range-safe working scale when physical `y-mean(y)` subtraction would overflow.

If the entity-level regressors are exactly collinear, statgpu can still compute fitted values using a least-squares solution, but the coefficient vector is not unique. For that fit, coefficient-level standard errors, tests, p-values, and confidence intervals are disabled rather than being computed from an arbitrary coefficient representation.

Invalid covariance choices raise an error. Likewise, an explicitly requested GPU backend must be available; statgpu does not silently run the model on CPU instead.

## FAQ

**Why is `nobs` smaller than the original row count?**  `nobs` is the number of entity-mean observations used in the between regression.

**Are entities with more time observations automatically weighted more heavily?**  No. Each retained entity contributes one mean observation to the final OLS regression.

## External Validation

We compare `BetweenOLS` with `statsmodels==0.14.6` after constructing the same entity-mean regression in both packages. The checks cover coefficients and HC0/HC2/HC3 standard errors/covariances: coefficients use `rtol=5e-10, atol=5e-12`, and covariance/BSE use `rtol=5e-9, atol=5e-11`. Shared covariance checks are summarized in the [validation matrix](covariance.md#validation-matrix).

GPU consistency is tested separately by comparing CuPy and Torch results with NumPy using the Stage-C physical validation tolerance `rtol=5e-6, atol=5e-7`. The dedicated `dev/benchmarks/validate_panel_intercept_cancellation_gpu.py` gate additionally verifies the cancellation-sensitive entity-mean intercept path on both physical GPU backends.

## References

- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). The MIT Press.
