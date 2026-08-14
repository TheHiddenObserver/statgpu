# RandomEffects

> Language: English  
> Last updated: 2026-08-14  
> Switch: [Chinese](../../cn/panel/random-effects.md)

## Overview

`RandomEffects` implements the one-way Swamy-Arora error-components estimator followed by feasible GLS. Covariance choices affect inference on the quasi-demeaned fit space, not the variance-component construction.

## Path

Implementation: `statgpu/panel/_random_effects.py`.

## Model and Estimator

$$
y_{it}=x_{it}^{\top}\beta+a_i+\varepsilon_{it},
\qquad
\operatorname{Var}(a_i)=\sigma_a^2,
\qquad
\operatorname{Var}(\varepsilon_{it})=\sigma_e^2.
$$

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

Here $df_W=n-r_W-r_E$, with $r_E=N$ when the level design has an explicit constant and $r_E=N-1$ otherwise; $df_B=N-r_B$. For entity $i$,

$$
\theta_i=1-\sqrt{\frac{\widehat\sigma_e^2}{\widehat\sigma_e^2+T_i\widehat\sigma_a^2}},
$$

$$
y_{it}^*=y_{it}-\theta_i\bar y_i,
\qquad
x_{it}^*=x_{it}-\theta_i\bar x_i.
$$

The feasible-GLS objective and estimator are

$$
\widehat\beta_{\mathrm{RE}}
=\arg\min_\beta\|y^*-X^*\beta\|_2^2
=(X^{*\top}X^*)^+X^{*\top}y^*.
$$

The rank-deficient extension replaces raw auxiliary-regression column counts by identified numerical ranks.

## Covariance and Inference

All residual-based covariance uses $Z=X^*$ and $e^*=y^*-X^*\widehat\beta_{\mathrm{RE}}$. In particular,

$$
\widehat V_{\mathrm{nonrobust}}
=\widehat\sigma_*^2(X^{*\top}X^*)^+,
\qquad
\widehat\sigma_*^2=\frac{e^{*\top}e^*}{n-\operatorname{rank}(X^*)}.
$$

HC0/1/2/3, clustered, and Driscoll-Kraay use [Panel covariance](covariance.md) with $X^*$ as the fit-space design.

## Parameters

| Parameter | Meaning |
|---|---|
| `cov_type` | `nonrobust`, HC0/1/2/3, `clustered`, or Driscoll-Kraay aliases. |
| `bandwidth`, `kernel` | Driscoll-Kraay controls. |
| `group_debias` | Small-group cluster correction. |
| `alpha` | Confidence-interval significance level. |
| `device` | `auto`, `cpu`, `cuda`, or `torch`. |
| `n_jobs` | Shared parallelism hint. |

```python
model.fit(X, y, entity_ids=entity_ids, time_ids=None, cluster=None)
```

`entity_ids` is required. Driscoll-Kraay requires `time_ids`; clustered covariance requires `cluster`. Formula input retains the normal R/Patsy intercept; `0 +` or `-1` requests a no-intercept random-effects model.

## CPU and GPU Example

```python
from statgpu.panel import RandomEffects

cpu = RandomEffects(device="cpu").fit(X, y, entity_ids=entity_ids)
cuda = RandomEffects(device="cuda").fit(X, y, entity_ids=entity_ids)
torch = RandomEffects(device="torch").fit(X, y, entity_ids=entity_ids)
```

Explicit GPU requests require the requested backend and do not silently fall back to CPU.

## Outputs

Public results include `coef_`, `bse_`, `tvalues_`, `pvalues_`, `conf_int_`, `theta_`, `variance_components_`, `fit_statistics_`, `nobs`, and `df_resid`. `variance_components_` stores $\widehat\sigma_e^2$ and $\widehat\sigma_a^2$.

## Numerical and Strict Behavior

Variance components and coefficients are covariance-invariant. There is no silent approximate-inference fallback. Exact rank deficiency uses the documented identified-rank extension; coordinate-wise inference still requires an identifiable coefficient representation. Classical FE-versus-RE Hausman is restricted to its documented applicability conditions.

## FAQ

**Does `cov_type` change the Swamy-Arora coefficient estimate?**  No. It changes inference after the quasi-demeaned GLS fit.

**Why can $\widehat\sigma_a^2$ be zero?**  The maintained estimator truncates the raw variance-component estimate at zero.

## External Validation

`dev/tests/test_panel_stage_c_linearmodels_estimators.py` checks HC/cluster/DK covariance on statgpu's own Swamy-Arora fit space against pinned `linearmodels==7.0` and `statsmodels==0.14.6` definitions. R panel covariance checks are maintained separately, and physical CuPy/Torch acceptance is recorded in `results/pr126_p100_fresh/validation_summary.txt`.

## References

Swamy and Arora (1972), error-components feasible GLS; Wooldridge (2010), *Econometric Analysis of Cross Section and Panel Data*.
