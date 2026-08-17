# RandomEffects

> Language: English  
> Last updated: 2026-08-15  
> Switch: [Chinese](../../cn/panel/random-effects.md)

## Overview

`RandomEffects` fits a one-way random-intercept panel model using the Swamy-Arora variance-component estimator followed by feasible GLS. Unlike fixed effects, the entity-specific effect is modeled as a random component rather than as one unrestricted fixed parameter per entity.

The chosen `cov_type` changes the reported standard errors and tests after the GLS fit; it does not change the Swamy-Arora variance components or the coefficient estimate.

## Path

Implementation: `statgpu/panel/_random_effects.py`.

## Statistical Model and Identification

A standard one-way random-effects model is

$$
y_{it}=x_{it}^{\top}\beta+a_i+\varepsilon_{it},
$$

where $x_{it}$ may include a constant. The entity effect $a_i$ is now a random variable rather than a fixed nuisance parameter. The classical error-components interpretation uses

$$
E(a_i)=0,
\qquad
\operatorname{Var}(a_i)=\sigma_a^2,
\qquad
\operatorname{Var}(\varepsilon_{it})=\sigma_e^2.
$$

A key distinction from fixed effects is the orthogonality restriction on the random effect. A common sufficient condition is

$$
E(a_i\mid X_i)=0,
\qquad
E(\varepsilon_{it}\mid X_i,a_i)=0,
$$

where $X_i=(x_{i1},\ldots,x_{iT_i})$. The classical one-way error-components covariance structure additionally treats the idiosyncratic errors as serially uncorrelated within entity, for example

$$
\operatorname{Cov}(\varepsilon_{it},\varepsilon_{is}\mid X_i)=0,
\qquad t\ne s,
$$

with the random effect orthogonal to the idiosyncratic error. These assumptions produce the familiar random-intercept covariance structure that motivates the Swamy-Arora variance-component transformation.

Under this interpretation, both within-entity and between-entity variation can be used to estimate the common slope $\beta$. If $a_i$ is systematically related to the regressor history, the random-effects GLS calculation can still be carried out numerically, but its coefficient generally need not identify the same structural $\beta$ as a fixed-effects estimator. This is the substantive distinction behind the classical FE-versus-RE Hausman comparison.

## Estimator

Swamy-Arora first estimates

$$
\widehat\sigma_e^2=\frac{RSS_W}{df_W},
\qquad
\bar T_H=\frac{N}{\sum_{i=1}^N T_i^{-1}},
$$

and

$$
\widehat\sigma_a^2=
\max\left\{0,\frac{RSS_B/df_B-\widehat\sigma_e^2}{\bar T_H}\right\}.
$$

Here $df_W=n-r_W-r_E$, with $r_E=N$ when the level design has an explicit constant and $r_E=N-1$ otherwise; $df_B=N-r_B$. If an auxiliary regression contains redundant columns, the numerical rank is used so those redundant directions are not counted as additional degrees of freedom.

For entity $i$,

$$
\theta_i=1-\sqrt{\frac{\widehat\sigma_e^2}{\widehat\sigma_e^2+T_i\widehat\sigma_a^2}},
$$

$$
y_{it}^*=y_{it}-\theta_i\bar y_i,
\qquad
x_{it}^*=x_{it}-\theta_i\bar x_i.
$$

This quasi-demeaning subtracts part of each entity mean. The amount depends on the estimated within-entity and between-entity variance components and on the entity's number of observations.

The feasible-GLS estimator is then

$$
\widehat\beta_{\mathrm{RE}}
=\arg\min_\beta\|y^*-X^*\beta\|_2^2
=(X^{*\top}X^*)^+X^{*\top}y^*.
$$

## Covariance and Inference

Standard errors are based on the quasi-demeaned data $(y^*,X^*)$ used in the GLS regression. In particular,

$$
\widehat V_{\mathrm{nonrobust}}
=\widehat\sigma_*^2(X^{*\top}X^*)^+,
\qquad
\widehat\sigma_*^2=\frac{e^{*\top}e^*}{n-\operatorname{rank}(X^*)},
$$

where $e^*=y^*-X^*\widehat\beta_{\mathrm{RE}}$. HC0/1/2/3, clustered, and Driscoll-Kraay alternatives use the same transformed regression; see [Panel covariance](covariance.md).

Robust covariance choices can relax assumptions used to estimate the **reported covariance after the GLS transformation**, but they do not change the Swamy-Arora transformation itself and do not remove the mean-model orthogonality condition $E(a_i\mid X_i)=0$ needed for the usual random-effects interpretation of $\beta$.

## Parameters

| Parameter | Default | Allowed / Constraint | Meaning |
|---|---:|---|---|
| `cov_type` | `"nonrobust"` | `nonrobust`, `robust`/`hc1`, `hc0`, `hc2`, `hc3`, `clustered`, `driscoll-kraay`/`dk`/`kernel` | How standard errors are computed after the random-effects GLS transformation. |
| `alpha` | `0.05` | finite and strictly between 0 and 1 | Confidence-interval significance level; `0.05` gives 95% intervals. |
| `device` | `"auto"` | `auto`, `cpu`, `cuda`, `torch` | Where numerical computation runs. |
| `n_jobs` | `None` | integer or `None` | Shared parallelism hint. |
| `bandwidth` | `None` | `None` or a non-negative integer; DK only | Driscoll-Kraay smoothing bandwidth. |
| `kernel` | `"bartlett"` | Bartlett/Newey-West, Parzen/Gallant, or QS/Quadratic-Spectral/Andrews aliases | Driscoll-Kraay kernel. |
| `group_debias` | `False` | boolean; clustered covariance only | Apply the small-number-of-clusters correction. |

```python
model.fit(X, y, entity_ids=entity_ids, time_ids=None, cluster=None)
```

`entity_ids` is required because the variance components and quasi-demeaning are entity-specific. Driscoll-Kraay additionally requires `time_ids`; clustered covariance requires `cluster`.

## CPU and GPU Example

```python
from statgpu.panel import RandomEffects

cpu = RandomEffects(device="cpu").fit(X, y, entity_ids=entity_ids)
cuda = RandomEffects(device="cuda").fit(X, y, entity_ids=entity_ids)
torch = RandomEffects(device="torch").fit(X, y, entity_ids=entity_ids)
```

If an explicitly requested GPU backend is unavailable, `.fit()` raises an error rather than switching to CPU.

## Formula Example

Assume `df` contains `y`, `x1`, `x2`, and `entity` columns.

```python
from statgpu.panel import RandomEffects

with_intercept = RandomEffects().fit(
    formula="y ~ x1 + x2 | entity",
    data=df,
)

without_intercept = RandomEffects().fit(
    formula="y ~ 0 + x1 + x2 | entity",
    data=df,
)
```

The first pipe variable is the entity grouping column. A second pipe variable is accepted only with `cov_type="driscoll-kraay"`, where it supplies `time_ids`; otherwise the fit fails clearly instead of silently ignoring that variable. If explicit `entity_ids`/`time_ids` are also supplied, they must match any pipe-named columns. Fixed-effect magic tokens (`EntityEffects`, `TimeEffects`, `FixedEffects`) are rejected for `RandomEffects`; use the pipe syntax to provide grouping metadata instead.

## Outputs

Public results include `coef_`, `bse_`, `tvalues_`, `pvalues_`, `conf_int_`, `theta_`, `variance_components_`, `fit_statistics_`, `nobs`, and `df_resid`. `variance_components_` stores $\widehat\sigma_e^2$ and $\widehat\sigma_a^2$. `theta_` is the entity-count-weighted average of the per-entity quasi-demeaning factors used in the fit.

## Numerical and Strict Behavior

Changing `cov_type` does not refit the random-effects model: the variance components and coefficients stay the same, while the reported uncertainty changes.

The Swamy-Arora variance-component step requires positive residual degrees of freedom in both its within and between auxiliary regressions. In particular, if the number of entities is no larger than the identified rank of the between regression, `fit()` raises instead of inventing a denominator and returning an unreliable random-effect variance.

If the transformed design is exactly rank deficient, fitted values may still be available but the coefficient vector is not uniquely identified. statgpu therefore disables coefficient-level standard errors, tests, p-values, and confidence intervals for that fit instead of reporting inference from an arbitrary coefficient representation.

The classical Hausman comparison is available only under the conditions documented in [Panel diagnostics](diagnostics.md). Invalid covariance inputs or unavailable explicitly requested GPU backends raise errors.

## FAQ

**Does `cov_type` change the Swamy-Arora coefficient estimate?**  No. It changes only the standard errors and related inference reported after the GLS fit.

**Why can $\widehat\sigma_a^2$ be zero?**  The raw Swamy-Arora estimate can be negative in finite samples; statgpu truncates that variance estimate at zero because a variance cannot be negative.

## External Validation

Random-effects coefficient estimates are **not** claimed to match another package exactly because statgpu uses its own Swamy-Arora variance-component construction. Instead, we take statgpu's quasi-demeaned $(X^*,y^*)$ regression and compare the resulting robust and Driscoll-Kraay covariance with `linearmodels==7.0`, and HC2/HC3 covariance with `statsmodels==0.14.6`. Covariance comparisons use `rtol=5e-9, atol=5e-11`; see the shared [validation matrix](covariance.md#validation-matrix).

GPU consistency is tested separately by comparing CuPy and Torch outputs with NumPy at default `rtol=5e-6, atol=5e-7`; observed differences are stored in `results/pr126_p100_fresh/panel_stage_c_correctness_p100.json`.

## References

- Swamy, P. A. V. B., & Arora, S. S. (1972). The exact finite sample properties of the estimators of coefficients in the error components regression models. *Econometrica*, 40(2), 261-275. [https://doi.org/10.2307/1909405](https://doi.org/10.2307/1909405)
- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). The MIT Press.
