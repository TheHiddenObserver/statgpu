# FamaMacBeth

> Language: English  
> Last updated: 2026-08-16  
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

This coefficient interpretation requires every retained period-specific coefficient vector to be identified by its cross-sectional design. After the observation-count filter is applied, statgpu checks the intercept-augmented $X_t$ with the shared panel SVD rank policy. If a retained period is rank deficient, `.fit()` raises a `ValueError` before coefficient averaging or inference rather than reporting coordinate-level results from a non-unique coefficient representation.

## Estimator

For each retained period, let $X_t$ denote the intercept-augmented period design. Under the required full-column-rank contract,

$$
\widehat\beta_t
=\arg\min_\beta\|y_t-X_t\beta\|_2^2
=(X_t^\top X_t)^{-1}X_t^\top y_t,
\qquad
\widehat\beta_{\mathrm{FM}}=T^{-1}\sum_{t=1}^T\widehat\beta_t.
$$

A period is retained when it satisfies `min_obs_per_period` and the implementation's minimum count rule $n_t\ge k+1$, where $k$ is the intercept-augmented design width. The full-rank check is then applied to each retained period. The implementation uses one shared rank-revealing SVD per retained period and reuses those factors to obtain both the numerical rank and the least-squares coefficient vector. The singular-value cutoff remains on the active backend, only the final integer rank is extracted for fail-closed control flow, and the least-squares path does not materialize the covariance bread that residual-OLS inference needs.

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

The order of the retained coefficient series follows `time_ids`. Numeric and datetime labels use their natural order. Ordered pandas categoricals preserve the category order explicitly declared by the user; they are not converted to lexical string order before the Newey-West lag covariance is formed.

Successful fits also publish the shared `ParameterInferenceResult` surface used by inference-capable statgpu estimators. The public `coef_`, `bse_`, `tvalues_`, `pvalues_`, and `conf_int_` arrays remain backend-native; `_inference_result` and the `_params`, `_bse`, `_tvalues`/`_zvalues`, `_pvalues`, and `_conf_int` aliases contain NumPy snapshots for the common inference/reporting contract. `newey-west` is labeled as `z`/normal inference, while `nonrobust` is labeled as Student-t inference with $T-1$ degrees of freedom.

## Parameters

| Parameter | Default | Allowed / Constraint | Meaning |
|---|---:|---|---|
| `cov_type` | `"newey-west"` | `nonrobust` or `newey-west` | Whether period-to-period coefficient dependence is ignored or adjusted with Newey-West. |
| `bandwidth` | `None` | `None` or a non-negative integer; clipped to at most $T-1$ | Bartlett Newey-West bandwidth $L$. |
| `alpha` | `0.05` | finite and strictly between 0 and 1 | Confidence-interval significance level; `0.05` gives 95% intervals. |
| `min_obs_per_period` | `1` | positive integer | Preliminary minimum period size. A retained period must also satisfy $n_t\ge k+1$, where $k$ is the intercept-augmented design width, and must have full column rank. |
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

`FamaMacBeth` always includes a period-specific intercept, matching its array API. Explicit no-intercept formulas such as `y ~ 0 + x1 + x2` and `y ~ x1 + x2 - 1` are therefore rejected with a clear `ValueError` instead of being silently reinterpreted as intercept models.

## Outputs

Public results include `coef_`, `bse_`, `tvalues_`, `pvalues_`, `conf_int_`, `betas_`, `cov_params_`, `fit_statistics_`, `nobs`, `n_periods`, and `df_resid`. `betas_` contains the retained period-by-period coefficient estimates, while `coef_` is their average. `_inference_result` provides the standardized inference container used by common statgpu reporting and downstream tooling.

## Numerical and Strict Behavior

At least two valid periods must remain after filtering; otherwise `.fit()` raises an error because the variability of the coefficient series cannot be estimated from fewer than two periods. Every retained period must also have full column rank under the shared panel SVD cutoff; a rank-deficient retained period fails closed before inference. The rank check and coefficient solve share the same SVD, which avoids a redundant second matrix decomposition while preserving the exact rank policy.

The retained periods are currently processed serially in Python. Each period therefore performs a relatively small SVD and must expose the final rank to Python before the fit can continue. For small numbers of regressors or small cross sections, GPU kernel-launch and synchronization overhead can outweigh the linear-algebra work, so `device="cuda"` or `device="torch"` is **not** a universal speedup guarantee. Benchmark the actual panel shape when performance matters. The focused physical gate records synchronized NumPy and GPU timing on the same workload and reports the GPU/NumPy median-time ratio; a ratio above 1 means that GPU backend is slower on that fixture rather than being treated as a failed correctness result.

P-value evaluation uses the CPU distribution implementation, but the full statistic vector is transferred once and evaluated vectorially rather than synchronizing one GPU scalar per parameter. The resulting p-value vector is then copied back to the active backend, preserving the public result device contract.

With `cov_type="newey-west"`, coefficient inference uses the asymptotic normal distribution. With `cov_type="nonrobust"`, it uses a Student-t reference with $T-1$ degrees of freedom. There is no hidden alternative inference method if these requirements are not met.

A new fit attempt invalidates the previous fitted and inference state before validation begins. If a refit fails, the estimator therefore remains unfitted instead of exposing stale coefficients or standard errors from the previous dataset.

An explicit `device="cuda"` or `device="torch"` request also requires that backend to be available; statgpu does not silently switch to CPU.

## FAQ

**Why is this covariance not listed with HC/cluster/Driscoll-Kraay?**  Those methods are built from observation-level regression residuals. Fama-MacBeth inference is instead built from the time series of period-specific coefficient estimates $\widehat\beta_t$.

**What happens to undersized periods?**  They are excluded before the coefficient average is formed. If fewer than two valid periods remain, the fit raises an error.

**What happens if a retained period is rank deficient?**  The fit raises a `ValueError`. statgpu does not average a non-unique period coefficient vector and then report coordinate-level standard errors for it.

## External Validation

`dev/tests/test_fama_macbeth_linearmodels_external.py` is a maintained definition-alignment gate against pinned `linearmodels==7.0`. The fixture uses the same explicit period intercept, full-rank balanced panel, period ordering, and coefficient set in both packages. Period-by-period coefficients (`betas_` versus `all_params`) and the averaged coefficient vector are compared in both covariance modes.

For `cov_type="nonrobust"`, statgpu's covariance is aligned with linearmodels `cov_type="unadjusted", debiased=True`: covariance, standard errors, and coefficient t-statistics are compared. P-values and confidence intervals are intentionally **not** forced to match in this branch because the covariance definitions align while the two APIs use different reference degrees of freedom for post-estimation inference: statgpu uses the retained-period definition $T-1$, whereas linearmodels uses its stacked-panel residual degrees of freedom when `debiased=True`.

For `cov_type="newey-west"`, the external gate uses linearmodels `cov_type="kernel", kernel="bartlett", bandwidth=L, debiased=False` with the same fixed $L$. This aligns both the coefficient-series kernel covariance and normal-reference inference, so covariance, standard errors, test statistics, p-values, and confidence intervals are compared. The dedicated `Panel Stage C external covariance` workflow installs the pinned reference and runs this test on every relevant PR/source change.

Maintained internal regressions additionally cover formula intercept behavior, ordered-categorical versus numeric chronology on both array and formula paths, missing-row formula alignment, retained-period rank rejection, failed-refit invalidation, the single-factorization solve contract, backend-native rank-cutoff behavior, both covariance modes on Torch CPU, and vectorized p-value evaluation.

GPU consistency for the standard full-rank numeric-time `fama_macbeth_newey_west` case is tested by `dev/benchmarks/validate_panel_stage_a_gpu.py`. The focused exact-head gate `dev/benchmarks/validate_fama_macbeth_review_fix_gpu.py` additionally checks ordered-categorical chronology, formula/missing-row alignment, rank rejection, both no-intercept spellings, both covariance modes, the standard inference container/private aliases, requested/executed backend identity, and full parity for `betas_`, coefficients, covariance, standard errors, test statistics, p-values, confidence intervals, and predictions. Its performance section records same-workload synchronized NumPy and CuPy/Torch samples, the median-time ratio, and explicit optimization notes; it makes no universal speedup claim. Fama-MacBeth remains outside the Stage-C residual-covariance case matrix because its covariance is defined from the coefficient series rather than observation-level residual scores.

## References

- Fama, E. F., & MacBeth, J. D. (1973). Risk, return, and equilibrium: Empirical tests. *Journal of Political Economy*, 81(3), 607-636. [https://doi.org/10.1086/260061](https://doi.org/10.1086/260061)
- Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703-708. [https://doi.org/10.2307/1913610](https://doi.org/10.2307/1913610)
