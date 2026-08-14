# FamaMacBeth

> Language: English  
> Last updated: 2026-08-14  
> Switch: [Chinese](../../cn/panel/fama-macbeth.md)

## Overview

`FamaMacBeth` runs a cross-sectional regression in each retained period and averages the period-specific coefficient vectors. Its inference is based on the coefficient time series, not residual-based panel covariance.

## Path

Implementation: `statgpu/panel/_fama_macbeth.py`.

## Objective and Estimator

For each retained period, let $X_t$ denote the intercept-augmented period design. Then

$$
\widehat\beta_t
=\arg\min_\beta\|y_t-X_t\beta\|_2^2
=(X_t^\top X_t)^+X_t^\top y_t,
\qquad
\widehat\beta_{\mathrm{FM}}=T^{-1}\sum_{t=1}^T\widehat\beta_t.
$$

Periods below `min_obs_per_period` or without enough observations for the period design are omitted.

## Covariance and Inference

Let

$$
u_t=\widehat\beta_t-\widehat\beta_{\mathrm{FM}}.
$$

With `cov_type="nonrobust"`,

$$
\widehat V_{\mathrm{nonrobust}}
=\frac{1}{T(T-1)}\sum_{t=1}^T u_tu_t^\top.
$$

For `cov_type="newey-west"`, define the lag-$\ell$ coefficient covariance

$$
\widehat\Gamma_\ell
=\frac{1}{T}\sum_{t=\ell+1}^{T}u_tu_{t-\ell}^\top,
\qquad \ell=0,\ldots,L,
$$

with Bartlett weights

$$
w_\ell=1-\frac{\ell}{L+1}.
$$

The coefficient-series long-run covariance and the covariance of the Fama-MacBeth mean are

$$
\widehat\Omega_{\mathrm{NW}}
=\widehat\Gamma_0+
\sum_{\ell=1}^{L}w_\ell
\left(\widehat\Gamma_\ell+\widehat\Gamma_\ell^\top\right),
\qquad
\widehat V_{\mathrm{NW}}(\widehat\beta_{\mathrm{FM}})
=\frac{1}{T}\widehat\Omega_{\mathrm{NW}}.
$$

If `bandwidth=None`, the implementation starts from

$$
L=\left\lfloor4(T/100)^{2/9}\right\rfloor
$$

and clips it to $0\le L\le T-1$. This coefficient-series HAC path is distinct from the residual-based definitions in [Panel covariance](covariance.md).

## Parameters

| Parameter | Default | Allowed / Constraint | Meaning |
|---|---:|---|---|
| `cov_type` | `"newey-west"` | `nonrobust` or `newey-west` | Coefficient-series covariance estimator. |
| `bandwidth` | `None` | `None` or a non-negative integer; clipped to at most $T-1$ | Bartlett HAC bandwidth $L$. |
| `alpha` | `0.05` | finite and strictly between 0 and 1 | Confidence-interval significance level; `0.05` gives 95% intervals. |
| `min_obs_per_period` | `1` | positive integer | Preliminary minimum period size. A retained period must also satisfy $n_t\ge k+1$, where $k$ is the intercept-augmented design width. |
| `device` | `"auto"` | `auto`, `cpu`, `cuda`, `torch` | Numerical backend/device. |
| `n_jobs` | `None` | integer or `None` | Shared parallelism hint. |

```python
model.fit(X, y, time_ids=time_ids, entity_ids=None)
```

`time_ids` is required. `entity_ids` only enables standardized within/between $R^2$.

## CPU and GPU Example

```python
from statgpu.panel import FamaMacBeth

cpu = FamaMacBeth(device="cpu").fit(X, y, time_ids=time_ids)
cuda = FamaMacBeth(device="cuda").fit(X, y, time_ids=time_ids)
torch = FamaMacBeth(device="torch").fit(X, y, time_ids=time_ids)
```

## Outputs

Public results include `coef_`, `bse_`, `tvalues_`, `pvalues_`, `conf_int_`, `betas_`, `cov_params_`, `fit_statistics_`, `nobs`, `n_periods`, and `df_resid`.

## Numerical and Strict Behavior

At least two periods must remain after filtering. The Newey-West path uses asymptotic-normal coefficient inference; the nonrobust path uses a Student-t reference with $T-1$ degrees of freedom. There is no separate approximate estimator or inference path, and explicit GPU requests do not silently fall back to CPU.

## FAQ

**Why is this covariance not listed with HC/cluster/DK?**  Fama-MacBeth inference is constructed from the time series of $\widehat\beta_t$, not observation residuals.

**What happens to undersized periods?**  They are excluded before the coefficient average; the fit raises if no valid periods or fewer than two retained periods remain.

## External Validation

There is currently no maintained external-framework parity gate for this estimator's beta-series covariance, so the documentation does not claim `linearmodels` or another package matches it. Maintained estimator/covariance/filtering regressions are in `dev/tests/test_panel_p2.py`, with formula coverage in `dev/tests/test_panel_formula.py`.

Three-backend physical parity for the `fama_macbeth_newey_west` case is exercised by `dev/benchmarks/validate_panel_stage_a_gpu.py` against NumPy with default `rtol=5e-6, atol=5e-7`. The Stage-C residual-covariance physical matrix is separate and does not include FamaMacBeth.

## References

- Fama, E. F., & MacBeth, J. D. (1973). Risk, return, and equilibrium: Empirical tests. *Journal of Political Economy*, 81(3), 607-636. [https://doi.org/10.1086/260061](https://doi.org/10.1086/260061)
- Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703-708. [https://doi.org/10.2307/1913610](https://doi.org/10.2307/1913610)
