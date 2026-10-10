# FamaMacBeth

> Language: English  
> Last updated: 2026-10-09<br>
> Switch: [Chinese](../../cn/panel/fama-macbeth.md)

## Overview

`FamaMacBeth` fits a separate cross-sectional regression in each time period and then averages the period-specific coefficient estimates. This is different from the other panel estimators on these pages: its target is an average cross-sectional slope, and its standard errors are based on how the estimated slopes vary **across time**.

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

This coefficient interpretation requires every retained period-specific coefficient vector to be identified by its cross-sectional design. After the observation-count filter, statgpu checks that the intercept-augmented $X_t$ has full column rank. If a retained period is rank deficient, `.fit()` raises a `ValueError` before coefficient averaging or inference rather than reporting coordinate-level results from a non-unique coefficient representation.

## Estimator

For each retained period, let $X_t$ denote the intercept-augmented period design. Under the required full-column-rank contract,

$$
\widehat\beta_t
=\arg\min_\beta\|y_t-X_t\beta\|_2^2
=(X_t^\top X_t)^{-1}X_t^\top y_t,
\qquad
\widehat\beta_{\mathrm{FM}}=T^{-1}\sum_{t=1}^T\widehat\beta_t.
$$

A period is retained when it satisfies `min_obs_per_period` and $n_t\ge k$, where $k$ is the intercept-augmented design width. Every retained period must also have full column rank.

A full-rank design can still exceed reliable float64 coefficient resolution, for example when extreme response values cancel. In that case `.fit()` raises `FloatingPointError` rather than returning an unreliable coefficient. Genuine zero coefficients are allowed; numerical cancellation that loses a nonzero contribution is a different case.

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

The order of the retained coefficient series follows `time_ids`. Numeric and datetime labels use their natural order. Ordered pandas categoricals preserve the category order explicitly declared by the user. Plain string labels use lexical order; other non-categorical object labels follow their sorted comparable-value order and fail if they are not mutually comparable. Semantic strings such as `t1, t2, t10` should therefore be encoded as an ordered categorical (or numeric/datetime key) when lexical string order is not the intended chronology before Newey-West lag covariance is formed.

Successful fits also publish the shared `ParameterInferenceResult` surface used by inference-capable statgpu estimators. The public `coef_`, `bse_`, `tvalues_`, `pvalues_`, and `conf_int_` arrays remain backend-native. P-values and confidence intervals are computed on the selected fit backend and, for Torch, the actual tensor device. `_inference_result` contains a NumPy snapshot for common reporting and downstream tooling after numerical inference is complete. `newey-west` uses `z`/normal inference, while `nonrobust` uses Student-t inference with $T-1$ degrees of freedom.

## Parameters

| Parameter | Default | Allowed / Constraint | Meaning |
|---|---:|---|---|
| `cov_type` | `"newey-west"` | `nonrobust` or `newey-west` | Whether period-to-period coefficient dependence is ignored or adjusted with Newey-West. |
| `bandwidth` | `None` | `None` or a non-negative integer; clipped to at most $T-1$ | Bartlett Newey-West bandwidth $L$. |
| `alpha` | `0.05` | finite and strictly between 0 and 1 | Confidence-interval significance level; `0.05` gives 95% intervals. |
| `min_obs_per_period` | `1` | positive integer | Preliminary minimum period size. A retained period must also satisfy $n_t\ge k$, where $k$ is the intercept-augmented design width, and must have full column rank. |
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

With `device="auto"`, an already NumPy/CuPy/Torch-native input may keep its native backend. An explicit `device="cpu"`, `device="cuda"`, or `device="torch"` request is authoritative even when the input container belongs to another backend: statgpu converts fit and prediction inputs to the requested/fitted backend, and an unavailable explicitly requested GPU backend raises instead of silently switching execution.

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

`FamaMacBeth` always includes a period-specific intercept, matching its array API. Explicit no-intercept formulas such as `y ~ 0 + x1 + x2` and `y ~ x1 + x2 - 1` are therefore rejected with a clear `ValueError` instead of being silently reinterpreted as intercept models.

## Outputs

Public results include `coef_`, `bse_`, `tvalues_`, `pvalues_`, `conf_int_`, `betas_`, `cov_params_`, `fit_statistics_`, `nobs`, `n_periods`, and `df_resid`. `betas_` contains the retained period-by-period coefficient estimates, while `coef_` is their average. `_inference_result` provides the standardized inference container used by common statgpu reporting and downstream tooling.

## Numerical and Strict Behavior

At least two valid periods must remain after filtering; otherwise `.fit()` raises an error because the variability of the coefficient series cannot be estimated from fewer than two periods. Every retained period must have full column rank. A rank-deficient period raises `ValueError` before inference.

Full rank does not guarantee reliable results at every response scale. If finite-precision arithmetic cannot reliably resolve a coefficient, `.fit()` raises `FloatingPointError` identifying the affected period. This can happen even for a well-conditioned design and is distinct from collinearity. Genuine zero coefficients remain valid.

The calculations use float64 arithmetic, not arbitrary-precision or exact summation. Numerical scaling protects representable coefficient averages and covariance contributions, but cannot recover every small remainder after severe cancellation. A non-finite final covariance or a negative diagonal variance causes inference to raise an error. At exactly zero estimated variance, a zero coefficient has statistic 0 while a nonzero coefficient has signed-infinite statistic; no artificial variance floor is added.

With `cov_type="newey-west"`, coefficient inference uses the asymptotic normal distribution. With `cov_type="nonrobust"`, it uses a Student-t reference with $T-1$ degrees of freedom. There is no hidden alternative inference method if these requirements are not met.

A new fit attempt invalidates the previous fitted and inference state before validation begins. If a refit fails, the estimator therefore remains unfitted instead of exposing stale coefficients or standard errors from the previous dataset.

An explicit `device="cuda"` or `device="torch"` request also requires that backend to be available; statgpu does not silently switch to CPU.

## FAQ

**Why is this covariance not listed with HC/cluster/Driscoll-Kraay?**  Those methods are built from observation-level regression residuals. Fama-MacBeth inference is instead built from the time series of period-specific coefficient estimates $\widehat\beta_t$.

**What happens to undersized periods?**  They are excluded before the coefficient average is formed. If fewer than two valid periods remain, the fit raises an error.

**What happens if a retained period is rank deficient?**  The fit raises a `ValueError`. statgpu does not average a non-unique period coefficient vector and then report coordinate-level standard errors for it.

**What happens if a period is full rank but a coefficient is below reliable float64 resolution?**  The fit raises `FloatingPointError` with a coefficient-resolution message. This reports a finite-precision limitation, not rank deficiency; a genuine zero coefficient is allowed.

<a id="external-validation"></a>

## Comparing with Other Packages

When comparing with `linearmodels` 7.0, use the same explicit period intercept, retained observations, full-rank period designs, chronological ordering, and coefficient set. Compare `betas_` with `all_params` before comparing the averaged coefficients.

For `cov_type="nonrobust"`, the corresponding linearmodels covariance is `cov_type="unadjusted", debiased=True`. Coefficients, covariance, standard errors, and coefficient t-statistics have aligned definitions. P-values and confidence intervals can still differ: statgpu uses retained-period degrees of freedom $T-1$, while linearmodels uses stacked-panel residual degrees of freedom when `debiased=True`.

For `cov_type="newey-west"`, use linearmodels `cov_type="kernel", kernel="bartlett", bandwidth=L, debiased=False` with the same fixed $L$. This aligns the coefficient-series kernel covariance and the normal-reference inference. The method remains distinct from the observation-level residual covariance estimators in [Panel covariance](covariance.md).

## References

- Fama, E. F., & MacBeth, J. D. (1973). Risk, return, and equilibrium: Empirical tests. *Journal of Political Economy*, 81(3), 607-636. [https://doi.org/10.1086/260061](https://doi.org/10.1086/260061)
- Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703-708. [https://doi.org/10.2307/1913610](https://doi.org/10.2307/1913610)