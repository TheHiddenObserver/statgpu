# FamaMacBeth

> Language: English  
> Last updated: 2026-08-14  
> Switch: [Chinese](../../cn/panel/fama-macbeth.md)

## Overview

`FamaMacBeth` runs a cross-sectional regression in each retained period and averages the period-specific coefficient vectors. Its inference is based on the coefficient time series, not residual-based panel covariance.

## Path

Implementation: `statgpu/panel/_fama_macbeth.py`.

## Objective and Estimator

For each retained period,

$$
\widehat\beta_t
=\arg\min_\beta\|y_t-X_t\beta\|_2^2
=(X_t^\top X_t)^+X_t^\top y_t,
\qquad
\widehat\beta_{\mathrm{FM}}=T^{-1}\sum_{t=1}^T\widehat\beta_t.
$$

Periods below `min_obs_per_period` or without enough observations for the period design are omitted.

## Covariance and Inference

Let $u_t=\widehat\beta_t-\widehat\beta_{\mathrm{FM}}$. With `cov_type="nonrobust"`,

$$
\widehat V_{\mathrm{FM}}
=\frac{1}{T}\left[\frac{1}{T-1}\sum_{t=1}^T u_tu_t^\top\right].
$$

`cov_type="newey-west"` replaces the bracketed term with a Bartlett HAC long-run covariance of the coefficient series. This path is intentionally distinct from [Panel covariance](covariance.md).

## Parameters

| Parameter | Meaning |
|---|---|
| `cov_type` | `nonrobust` or `newey-west`. |
| `bandwidth` | Bartlett HAC bandwidth for the coefficient series. |
| `min_obs_per_period` | Minimum period sample size before design-size filtering. |
| `alpha` | Confidence-interval significance level. |
| `device` | `auto`, `cpu`, `cuda`, or `torch`. |
| `n_jobs` | Shared parallelism hint. |

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

At least two periods must remain after filtering. The Newey-West path uses asymptotic-normal coefficient inference; the nonrobust path uses a Student-t reference with $T-1$ degrees of freedom. Explicit GPU requests do not silently fall back to CPU.

## FAQ

**Why is this covariance not listed with HC/cluster/DK?**  Fama-MacBeth inference is constructed from the time series of $\widehat\beta_t$, not observation residuals.

**What happens to undersized periods?**  They are excluded before the coefficient average; the fit raises if no valid periods or fewer than two retained periods remain.

## External Validation

Maintained estimator, covariance, filtering, and prediction regressions are in `dev/tests/test_panel_p2.py` and formula coverage is in `dev/tests/test_panel_formula.py`. The Stage-C residual-covariance external suite does **not** claim `linearmodels` parity for this estimator because the maintained beta-series covariance is estimator-specific.

## References

Fama and MacBeth (1973), *Risk, Return, and Equilibrium: Empirical Tests*; Newey and West (1987).
