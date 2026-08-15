# RandomEffects

> Language: English  
> Last updated: 2026-08-15  
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

| Parameter | Default | Allowed / Constraint | Meaning |
|---|---:|---|---|
| `cov_type` | `"nonrobust"` | `nonrobust`, `robust`/`hc1`, `hc0`, `hc2`, `hc3`, `clustered`, `driscoll-kraay`/`dk`/`kernel` | Covariance estimator on the quasi-demeaned fit space. |
| `alpha` | `0.05` | finite and strictly between 0 and 1 | Confidence-interval significance level; `0.05` gives 95% intervals. |
| `device` | `"auto"` | `auto`, `cpu`, `cuda`, `torch` | Numerical backend/device. |
| `n_jobs` | `None` | integer or `None` | Shared parallelism hint. |
| `bandwidth` | `None` | `None` or a non-negative integer; DK only | Driscoll-Kraay smoothing bandwidth. |
| `kernel` | `"bartlett"` | Bartlett/Newey-West, Parzen/Gallant, or QS/Quadratic-Spectral/Andrews aliases | Driscoll-Kraay kernel. |
| `group_debias` | `False` | boolean; clustered covariance only | Apply the small-group cluster correction. |

```python
model.fit(X, y, entity_ids=entity_ids, time_ids=None, cluster=None)
```

`entity_ids` is required. Driscoll-Kraay requires `time_ids`; clustered covariance requires `cluster`.

## CPU and GPU Example

```python
from statgpu.panel import RandomEffects

cpu = RandomEffects(device="cpu").fit(X, y, entity_ids=entity_ids)
cuda = RandomEffects(device="cuda").fit(X, y, entity_ids=entity_ids)
torch = RandomEffects(device="torch").fit(X, y, entity_ids=entity_ids)
```

Explicit GPU requests require the requested backend and do not silently fall back to CPU.

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

## Outputs

Public results include `coef_`, `bse_`, `tvalues_`, `pvalues_`, `conf_int_`, `theta_`, `variance_components_`, `fit_statistics_`, `nobs`, and `df_resid`. `variance_components_` stores $\widehat\sigma_e^2$ and $\widehat\sigma_a^2$.

## Numerical and Strict Behavior

Variance components and coefficients are covariance-invariant. There is no silent approximate-inference fallback. Exact rank deficiency uses the documented identified-rank extension; coordinate-wise inference still requires an identifiable coefficient representation. Classical FE-versus-RE Hausman is restricted to its documented applicability conditions.

## FAQ

**Does `cov_type` change the Swamy-Arora coefficient estimate?**  No. It changes inference after the quasi-demeaned GLS fit.

**Why can $\widehat\sigma_a^2$ be zero?**  The maintained estimator truncates the raw variance-component estimate at zero.

## External Validation

The maintained external gate does **not** assert direct RandomEffects coefficient parity with another package, because statgpu keeps its own Swamy-Arora variance-component construction. Instead, `linearmodels==7.0` checks robust and Driscoll-Kraay covariance on statgpu's reconstructed $X^*,y^*$ fit space, while `statsmodels==0.14.6` checks HC2/HC3 on that same fit space; covariance assertions use `rtol=5e-9, atol=5e-11`. See the shared [validation matrix](covariance.md#validation-matrix).

The Stage-C physical runner separately checks RandomEffects CuPy/Torch outputs against NumPy at default `rtol=5e-6, atol=5e-7`; observed differences are stored in `results/pr126_p100_fresh/panel_stage_c_correctness_p100.json`.

## References

- Swamy, P. A. V. B., & Arora, S. S. (1972). The exact finite sample properties of the estimators of coefficients in the error components regression models. *Econometrica*, 40(2), 261-275. [https://doi.org/10.2307/1909405](https://doi.org/10.2307/1909405)
- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). The MIT Press.
