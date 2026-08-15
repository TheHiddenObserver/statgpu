# FamaMacBeth

> Language: English  
> Last updated: 2026-08-15  
> Switch: [Chinese](../../cn/panel/fama-macbeth.md)

## Overview

`FamaMacBeth` fits a separate cross-sectional regression in each time period and then averages the period-specific coefficient estimates. This is different from the other panel estimators on these pages: its target is an average cross-sectional slope, and its standard errors are based on how the estimated slopes vary **across time**.

## Path

Implementation: `statgpu/panel/_fama_macbeth.py`.

## Statistical Model and Target

A natural period-specific model is

$$
y_{it}=\alpha_t+x_{it}^{\top}\beta_t+\varepsilon_{it},
$$

where both the intercept and slope may vary by period. For the usual cross-sectional regression interpretation, a common sufficient condition within each period is

$$
E(\varepsilon_{it}\mid x_{it},t)=0,
$$

or, more generally, the corresponding cross-sectional orthogonality condition that defines $\beta_t$ as the period-$t$ linear projection coefficient.

For the $T$ periods retained by the estimator, the direct target is the equally weighted retained-period average

$$
\beta_{\mathrm{FM}}=\frac{1}{T}\sum_{t=1}^{T}\beta_t.
$$

No probability model for the sequence $\{\beta_t\}$ is needed to define this target. If the retained periods are additionally viewed as draws from a time superpopulation, the same average can be given a population interpretation such as $E_t(\beta_t)$ under the corresponding sampling assumptions. The constant-slope model $\beta_t\equiv\beta$ is a special case.

This coefficient interpretation presumes that each period-specific coefficient vector is identified by its cross-sectional design. The implementation filters periods by observation count and uses a numerical solve or Moore-Penrose solution as needed; if a retained period's design is rank deficient, a numerical coefficient vector still exists but its individual coordinates are not uniquely identified.

## Estimator

For each retained period, let $X_t$ denote the intercept-augmented period design. Then

$$
\widehat\beta_t
=\arg\min_\beta\|y_t-X_t\beta\|_2^2
=(X_t^\top X_t)^+X_t^\top y_t,
\qquad
\widehat\beta_{\mathrm{FM}}=T^{-1}\sum_{t=1}^T\widehat\beta_t.
$$

A period is retained when it satisfies `min_obs_per_period` and the implementation's minimum count rule $n_t\ge k+1$, where $k$ is the intercept-augmented design width. The final coefficient is the simple average over the retained period estimates.

## Covariance and Inference

Define the deviation of each period estimate from the final average as

$$
\nu_t=\widehat\beta_t-\widehat\beta_{\mathrm{FM}}.
$$

With `cov_type="nonrobust"`,

$$
\widehat V_{\mathrm{nonrobust}}
=\frac{1}{T(T-1)}\sum_{t=1}^T \nu_t\nu_t^\top.
$$

This treats the retained period estimates as an independent coefficient series. With `cov_type="newey-west"`, serial dependence in that coefficient series is allowed. Define

$$
\widehat\Gamma_\ell
=\frac{1}{T}\sum_{t=\ell+1}^{T}\nu_t\nu_{t-\ell}^\top,
\qquad \ell=0,\ldots,L,
$$

with Bartlett weights

$$
w_\ell=1-\frac{\ell}{L+1}.
$$

The long-run covariance of the coefficient series and the covariance of the Fama-MacBeth average are

$$
\widehat\Omega_{\mathrm{NW}}
=\widehat\Gamma_0+
\sum_{\ell=1}^{L}w_\ell
\left(\widehat\Gamma_\ell+\widehat\Gamma_\ell^\top\right),
\qquad
\widehat V_{\mathrm{NW}}(\widehat\beta_{\mathrm{FM}})
=\frac{1}{T}\widehat\Omega_{\mathrm{NW}}.
$$

If `bandwidth=None`, statgpu starts from

$$
L=\left\lfloor4(T/100)^{2/9}\right\rfloor
$$

and clips it to $0\le L\le T-1$. This Newey-West calculation is applied to the sequence of period coefficients, so it is different from the residual-based HAC/Driscoll-Kraay estimators described in [Panel covariance](covariance.md).

## Parameters

| Parameter | Default | Allowed / Constraint | Meaning |
|---|---:|---|---|
| `cov_type` | `"newey-west"` | `nonrobust` or `newey-west` | Whether period-to-period coefficient dependence is ignored or adjusted with Newey-West. |
| `bandwidth` | `None` | `None` or a non-negative integer; clipped to at most $T-1$ | Bartlett Newey-West bandwidth $L$. |
| `alpha` | `0.05` | finite and strictly between 0 and 1 | Confidence-interval significance level; `0.05` gives 95% intervals. |
| `min_obs_per_period` | `1` | positive integer | Preliminary minimum period size. A retained period must also satisfy $n_t\ge k+1$, where $k$ is the intercept-augmented design width. |
| `device` | `"auto"` | `auto`, `cpu`, `cuda`, `torch` | Where numerical computation runs. |
| `n_jobs` | `None` | integer or `None` | Shared parallelism hint. |

```python
model.fit(X, y, time_ids=time_ids, entity_ids=None)
```

`time_ids` is required to define the cross-sectional regressions. `entity_ids` is optional and is used only for standardized within/between $R^2$ calculations.

## CPU and GPU Example

```python
from statgpu.panel import FamaMacBeth

cpu = FamaMacBeth(device="cpu").fit(X, y, time_ids=time_ids)
cuda = FamaMacBeth(device="cuda").fit(X, y, time_ids=time_ids)
torch = FamaMacBeth(device="torch").fit(X, y, time_ids=time_ids)
```

If an explicitly requested GPU backend is unavailable, `.fit()` raises an error rather than switching to CPU.

## Formula Example

Assume `df` contains `y`, `x1`, `x2`, and `time` columns.

```python
from statgpu.panel import FamaMacBeth

model = FamaMacBeth().fit(
    formula="y ~ x1 + x2",
    data=df,
    time_ids=df["time"],
)
```

## Outputs

Public results include `coef_`, `bse_`, `tvalues_`, `pvalues_`, `conf_int_`, `betas_`, `cov_params_`, `fit_statistics_`, `nobs`, `n_periods`, and `df_resid`. `betas_` contains the retained period-by-period coefficient estimates, while `coef_` is their average.

## Numerical and Strict Behavior

At least two valid periods must remain after filtering; otherwise `.fit()` raises an error because the variability of the coefficient series cannot be estimated from fewer than two periods.

With `cov_type="newey-west"`, coefficient inference uses the asymptotic normal distribution. With `cov_type="nonrobust"`, it uses a Student-t reference with $T-1$ degrees of freedom. There is no hidden alternative inference method if these requirements are not met.

An explicit `device="cuda"` or `device="torch"` request also requires that backend to be available; statgpu does not silently switch to CPU.

## FAQ

**Why is this covariance not listed with HC/cluster/Driscoll-Kraay?**  Those methods are built from observation-level regression residuals. Fama-MacBeth inference is instead built from the time series of period-specific coefficient estimates $\widehat\beta_t$.

**What happens to undersized periods?**  They are excluded before the coefficient average is formed. If fewer than two valid periods remain, the fit raises an error.

## External Validation

There is currently no maintained cross-package comparison for this estimator's coefficient-series covariance, so the documentation does not claim that `linearmodels` or another package produces identical Fama-MacBeth standard errors. Estimator, covariance, filtering, and formula behavior are covered by the panel regression tests in `dev/tests/`.

GPU consistency for the `fama_macbeth_newey_west` case is tested separately by `dev/benchmarks/validate_panel_stage_a_gpu.py`, which compares CuPy and Torch with NumPy using default `rtol=5e-6, atol=5e-7`. Fama-MacBeth is not part of the Stage-C residual-covariance matrix because its covariance is defined from the coefficient series instead.

## References

- Fama, E. F., & MacBeth, J. D. (1973). Risk, return, and equilibrium: Empirical tests. *Journal of Political Economy*, 81(3), 607-636. [https://doi.org/10.1086/260061](https://doi.org/10.1086/260061)
- Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703-708. [https://doi.org/10.2307/1913610](https://doi.org/10.2307/1913610)
