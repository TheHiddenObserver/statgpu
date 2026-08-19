# FamaMacBeth

> Language: English  
> Last updated: 2026-08-20<br>
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

A period is retained when it satisfies `min_obs_per_period` and the implementation's minimum count rule $n_t\ge k$, where $k$ is the intercept-augmented design width. The full-rank contract is then enforced for every retained period. NumPy, CuPy, and Torch use the same conservative exact-size Gram-certificate dispatch. Retained periods are grouped by their actual row count, without zero padding. Because every period has an exact intercept, a full-rank period may first remove a safe common response anchor along that constant direction before forming the Gram right-hand side; the intercept is restored after the solve. In addition, the exact intercept coordinate of $X_t^\top y_t$ uses a magnitude-tiered response sum rather than an ordinary BLAS reduction, so a tail such as `[2**55, 1, -2**55]` produces the representable intercept contribution `1` without forcing an SVD fallback. These two protections handle different risks: anchoring removes a huge common response level, whereas the stable intercept RHS preserves cancellation around zero.

A period may consume the Gram solve only when the backend-native spectrum satisfies $\lambda_{\min}(G_t)/\lambda_{\max}(G_t)>10^{-4}$ and the Gram matrix, centered right-hand side, and candidate solution are finite. For the single-response Fama-MacBeth path, one augmented solve returns both the candidate coefficient and the inverse-Gram information used to bound coordinatewise right-hand-side roundoff, so the precision certificate does not require a second Gram factorization. If a non-intercept coefficient lies below that certified float64 resolution, the period leaves the Gram path and is checked by the maintained SVD fallback. A non-intercept right-hand-side coordinate that has rounded to exact zero while its absolute summand mass is nonzero is treated more conservatively: float64 arithmetic cannot certify whether that zero is genuine or whether a representable cancellation tail was erased, and a rounded SVD basis can also invent a finite low-order coefficient. That case therefore fails closed with `FloatingPointError` instead of publishing an uncertified period coefficient. Torch uses its documented stacked-SVD support for ordinary unsafe subsets, while NumPy/CuPy retain supported two-dimensional fallbacks.

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

Successful fits also publish the shared `ParameterInferenceResult` surface used by inference-capable statgpu estimators. The public `coef_`, `bse_`, `tvalues_`, `pvalues_`, and `conf_int_` arrays remain backend-native. Distribution inference follows the selected fit backend: NumPy uses the NumPy inference backend, CuPy uses the CuPy inference backend, and Torch uses the Torch inference backend on the actual tensor device. `_inference_result` and the `_params`, `_bse`, `_tvalues`/`_zvalues`, `_pvalues`, and `_conf_int` aliases contain NumPy snapshots only for the common inference/reporting contract; those snapshots are not used to calculate p-values or confidence intervals. For GPU fits, the reporting fields are packed on the active backend and copied in one small snapshot after numerical inference has completed. `newey-west` is labeled as `z`/normal inference, while `nonrobust` is labeled as Student-t inference with $T-1$ degrees of freedom.

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

At least two valid periods must remain after filtering; otherwise `.fit()` raises an error because the variability of the coefficient series cannot be estimated from fewer than two periods. Every retained period must also have full column rank under the shared panel SVD cutoff; a rank-deficient retained period fails closed before inference. A distinct float64 coefficient-resolution failure also fails closed: if a non-intercept coordinate is too small relative to the absolute projection error and the SVD fallback materially violates least-squares stationarity, `.fit()` raises `FloatingPointError` and identifies the retained period rather than misreporting the case as rank deficiency. The same error is used when a non-intercept Gram/SVD right-hand side is an ambiguous rounded zero with nonzero absolute summand mass. This can occur even when the period design is perfectly conditioned; the failure reflects coefficient resolution, not rank. On every maintained backend, the Gram certificate remains only a fast-path selector; true rank boundaries remain SVD-owned.

Numerical safety is fail-closed across this path. A full-rank exact period intercept can remove a range-safe midrange response anchor before the Gram/SVD response projection and restore that anchor only to the intercept afterward. Its Gram RHS is also formed from the shared magnitude-tiered response sum, so an intercept cancellation tail can remain on the `gram-certified` fast path with zero SVD fallbacks. Cancellation-sensitive SVD projections use the shared magnitude-tiered reducer for ordinary fallback cases. A non-finite Gram matrix, centered right-hand side, or candidate solution is treated as uncertified and routed through the rank-revealing SVD fallback. If an entire full-rank design is below $\sqrt{\mathrm{DBL\_MIN}}$, it is uniformly raised to that safe working scale before SVD and the final coefficient is transformed back; this positive scalar change leaves the relative rank cutoff unchanged. Period-coefficient averaging and parameter-R² scalar/group means use the shared magnitude-tiered float64 reduction policy. Ordinary-scale panel group reductions remain on the single-scatter fast path; extra magnitude tiers are activated only when dynamic range can hide a representable low-order contribution or an unscaled same-sign reduction can overflow. Each tier is reduced on the scale of the requested target: sum reductions restore only that tier's overflow factor, while mean reductions divide a safe tier after summation and predivide only a tier whose raw same-sign sum would overflow. This avoids erasing collectively representable subnormal mean contributions merely because another tier is huge. The implementation remains float64 arithmetic rather than arbitrary-precision or exact summation. Coefficient-series covariance uses per-coordinate centered scales and restores each symmetric entry with the larger scale first, preserving representable small-coordinate variance and cross-covariance. If the final covariance itself is non-finite or has a negative diagonal variance, inference fails closed. At exactly zero estimated variance, a zero coefficient has statistic 0 while a nonzero coefficient has signed-infinite statistic, so no dimensionful fake denominator or `0/0` `NaN` is introduced.

The maintained physical scaling fixture deliberately retains three resident-array workloads: micro (64×128×4; 8,192 rows), medium (128×1,024×8; 131,072 rows), and large (128×4,096×16; 524,288 rows). Historical Tesla P100 evidence on numerical source `8c60db00f5ea986aed96b1f1dce3f5c3b4f0bcd4` reports CuPy/Torch GPU-over-NumPy median-time ratios of **0.549/0.343** on micro, **0.204/0.168** on medium, and **0.114/0.109** on large, corresponding to about 1.82×/2.92×, 4.91×/5.97×, and 8.75×/9.16× speedups respectively. Every GPU backend × scale case used one `gram-certified` exact-size batch, one control synchronization, and zero SVD fallbacks. These measurements demonstrate the maintained benchmark crossover on that historical P100 resident-array protocol; they are not a universal hardware guarantee and are not physical acceptance for the current numerical head. The current head changes valid Fama-MacBeth, shared panel inference, residual-covariance arithmetic, and physical-validator paths, so fresh CuPy/Torch CUDA validation is required before those historical artifacts can be superseded.

`dev/benchmarks/benchmark_fama_macbeth_scaling_gpu.py` records synchronized NumPy/CuPy/Torch median time, rows per second, GPU/NumPy ratio, speedup, solver provenance, numerical parity, backend versions, and thread-environment provenance. GPU input arrays are transferred before warmup/timing, so the benchmark measures resident-array `fit()` performance and explicitly excludes host-to-device input-transfer time. For that historical source, numerical differences in the accepted P100 scaling artifact remain tight: coefficient/beta/prediction differences are near machine precision and the largest reported statistic difference is below $4\times10^{-11}$.

P-values and critical values are evaluated through the selected inference backend instead of transferring the statistic vector to NumPy/SciPy. Normal inference uses the backend's two-sided normal routines; general Student-t inference uses the backend Student-t routines. The exact small-degree boundaries are kept backend-native as well: df=1 uses the Cauchy identity, and df=2 uses its elementary two-sided tail and quantile formulas so Torch versions without native `betainc` do not lose the maintained high-precision contract. Only the standardized `ParameterInferenceResult` reporting snapshot crosses to NumPy after inference has been completed.

With `cov_type="newey-west"`, coefficient inference uses the asymptotic normal distribution. With `cov_type="nonrobust"`, it uses a Student-t reference with $T-1$ degrees of freedom. There is no hidden alternative inference method if these requirements are not met.

A new fit attempt invalidates the previous fitted and inference state before validation begins. If a refit fails, the estimator therefore remains unfitted instead of exposing stale coefficients or standard errors from the previous dataset.

An explicit `device="cuda"` or `device="torch"` request also requires that backend to be available; statgpu does not silently switch to CPU.

## FAQ

**Why is this covariance not listed with HC/cluster/Driscoll-Kraay?**  Those methods are built from observation-level regression residuals. Fama-MacBeth inference is instead built from the time series of period-specific coefficient estimates $\widehat\beta_t$.

**What happens to undersized periods?**  They are excluded before the coefficient average is formed. If fewer than two valid periods remain, the fit raises an error.

**What happens if a retained period is rank deficient?**  The fit raises a `ValueError`. statgpu does not average a non-unique period coefficient vector and then report coordinate-level standard errors for it.

**What happens if a period is full rank but a coefficient is below reliable float64 resolution?**  The fit raises `FloatingPointError` with a coefficient-resolution message. This is distinct from rank deficiency: the design may be well conditioned while rounded projection arithmetic cannot certify a much smaller coefficient beside a dominant direction. The same fail-closed behavior applies when a non-intercept right-hand-side coordinate has rounded to zero despite nonzero positive/negative summands, because ordinary float64 arithmetic cannot certify whether that zero is exact.

## External Validation

`dev/tests/test_fama_macbeth_linearmodels_external.py` is a maintained definition-alignment gate against pinned `linearmodels==7.0`. The fixture uses the same explicit period intercept, full-rank balanced panel, period ordering, and coefficient set in both packages. Period-by-period coefficients (`betas_` versus `all_params`) and the averaged coefficient vector are compared in both covariance modes.

For `cov_type="nonrobust"`, statgpu's covariance is aligned with linearmodels `cov_type="unadjusted", debiased=True`: covariance, standard errors, and coefficient t-statistics are compared. P-values and confidence intervals are intentionally **not** forced to match in this branch because the covariance definitions align while the two APIs use different reference degrees of freedom for post-estimation inference: statgpu uses the retained-period definition $T-1$, whereas linearmodels uses its stacked-panel residual degrees of freedom when `debiased=True`.

For `cov_type="newey-west"`, the external gate uses linearmodels `cov_type="kernel", kernel="bartlett", bandwidth=L, debiased=False` with the same fixed $L$. This aligns both the coefficient-series kernel covariance and normal-reference inference, so covariance, standard errors, test statistics, p-values, and confidence intervals are compared. The dedicated `Panel Stage C external covariance` workflow installs the pinned reference and runs this test on every relevant PR/source change.

Maintained internal regressions additionally cover formula intercept behavior, ordered-categorical versus numeric chronology on both array and formula paths, missing-row formula alignment, retained-period rank rejection, failed-refit invalidation, the SVD rank-boundary policy, conservative Gram-certificate acceptance/rejection, backend-native distribution routing, exact df=1/df=2 small-degree inference boundaries, balanced and shuffled-unbalanced exact-size GPU grouping, chronological rank-error reporting, SVD fallback ownership, direct-fit finite-validation ownership, the packed reporting snapshot, large-common-intercept Gram centering, stable exact-intercept Gram-RHS cancellation, ambiguous nonconstant zero-RHS rejection, and explicit coefficient-resolution failure reporting.

GPU consistency for the standard full-rank numeric-time `fama_macbeth_newey_west` case is tested by `dev/benchmarks/validate_panel_stage_a_gpu.py`. The historical focused gate `dev/benchmarks/validate_fama_macbeth_review_fix_gpu.py` remains the detailed chronology/formula/rank/inference correctness oracle. Final optimized-source physical acceptance uses `dev/benchmarks/validate_fama_macbeth_optimized_gpu.py`, while crossover evidence is recorded by `dev/benchmarks/benchmark_fama_macbeth_scaling_gpu.py`. The historical PR126 P100 evidence remains anchored to numerical source `8c60db00...` and is complemented by the exact-source Stage-C matrix and HAC-chronology runners; it is not current-head acceptance after the later numerical-path fixes. `dev/benchmarks/validate_panel_intercept_cancellation_gpu.py` exercises the broader shared coefficient-resolution and large-intercept panel contracts. The focused `dev/benchmarks/validate_fama_macbeth_rhs_cancellation_gpu.py` gate additionally verifies on both CuPy CUDA and Torch CUDA that an intercept cancellation tail remains `gram-certified` with zero SVD fallbacks and that an ambiguous nonconstant zero RHS fails closed. Fama-MacBeth remains outside the Stage-C residual-covariance case matrix because its covariance is defined from the coefficient series rather than observation-level residual scores.

## References

- Fama, E. F., & MacBeth, J. D. (1973). Risk, return, and equilibrium: Empirical tests. *Journal of Political Economy*, 81(3), 607-636. [https://doi.org/10.1086/260061](https://doi.org/10.1086/260061)
- Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703-708. [https://doi.org/10.2307/1913610](https://doi.org/10.2307/1913610)