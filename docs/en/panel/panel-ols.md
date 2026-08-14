# PanelOLS

> Language: English  
> Last updated: 2026-08-14  
> Switch: [Chinese](../../cn/panel/panel-ols.md)

## Overview

`PanelOLS` estimates level, one-way fixed-effect, or two-way fixed-effect linear panel regressions. Fixed effects are removed in the fitted numerical space rather than represented as a dense dummy matrix.

## Path

Implementation: `statgpu/panel/_fixed_effects.py`.

## Model and Objective

For entity and time effects,

$$
y_{it}=x_{it}^{\top}\beta+\alpha_i+\gamma_t+\varepsilon_{it}.
$$

Let $F$ denote the included fixed-effect design and $M_F=I-F(F^\top F)^+F^\top$. The slope estimate solves

$$
\widehat\beta_{\mathrm{FE}}
=\arg\min_\beta\|M_F(y-X\beta)\|_2^2
=(X^\top M_FX)^+X^\top M_Fy.
$$

One-way entity effects reduce to within demeaning. Two-way effects use backend-native alternating projection and fail closed if `demean_tol` is not reached within `demean_max_iter`.

## Covariance and Inference

The fit space is $Z=M_FX$ with residual $e=M_F(y-X\widehat\beta_{\mathrm{FE}})$. Shared nonrobust, HC, clustered, and Driscoll-Kraay formulas are in [Panel covariance](covariance.md).

For Driscoll-Kraay, the fixed-effect nuisance rank is

$$
r_F=\begin{cases}N,&\text{entity only},\\T,&\text{time only},\\N+T-C,&\text{two way},\end{cases}
$$

where $C$ is the number of connected components of the observed entity-time incidence graph.

## Parameters

| Parameter | Meaning |
|---|---|
| `entity_effects`, `time_effects` | Select fixed-effect dimensions. |
| `cov_type` | `nonrobust`, HC0/1/2/3, `clustered`, or Driscoll-Kraay aliases. |
| `bandwidth`, `kernel` | Driscoll-Kraay controls. |
| `group_debias` | Small-group cluster correction. |
| `demean_max_iter`, `demean_tol` | Two-way alternating-projection controls. |
| `alpha` | Confidence-interval significance level. |
| `device` | `auto`, `cpu`, `cuda`, or `torch`. |
| `n_jobs` | Shared parallelism hint. |

```python
model.fit(X, y, entity_ids=None, time_ids=None, cluster=None)
```

Formula input supports additive `EntityEffects` / `TimeEffects` tokens or pipe fixed-effect syntax, but not both. With no fixed effects, the normal formula intercept is retained and `0 +` / `-1` requests a no-intercept level regression; with fixed effects, the common intercept is absorbed by the effect space.

## CPU and GPU Example

```python
from statgpu.panel import PanelOLS

cpu = PanelOLS(entity_effects=True, device="cpu").fit(X, y, entity_ids=entity_ids)
cuda = PanelOLS(entity_effects=True, device="cuda").fit(X, y, entity_ids=entity_ids)
torch = PanelOLS(entity_effects=True, device="torch").fit(X, y, entity_ids=entity_ids)
```

`cuda` requires CuPy/CUDA and `torch` requires Torch CUDA; explicit GPU requests do not silently fall back to CPU.

## Outputs

Common public results include `coef_`, `bse_`, `tvalues_`, `pvalues_`, `conf_int_`, `rsquared_within`, `fit_statistics_`, `nobs`, and `df_resid`. `summary()` returns the panel summary object. Pooling F and Hausman are described in [Panel diagnostics](diagnostics.md).

## Numerical and Strict Behavior

There is no silent approximate-inference fallback. Exact rank deficiency keeps fitted-space quantities but makes coordinate-wise coefficient inference unavailable; see [Panel covariance](covariance.md). Two-way stored-effect prediction requires identified entity/time labels in the same fitted incidence component: one-sided, known-plus-unknown, and cross-component combinations fail closed; if both labels are unseen, prediction uses the linear-only fallback.

## FAQ

**Why can two-way demeaning raise instead of returning the last iterate?**  The estimator treats the convergence tolerance as part of the numerical contract and fails closed when it is not met.

**Does robust covariance change `fit_statistics_.f_statistic` into a robust Wald test?**  No. The standardized model F remains the classical fit-space joint-slope statistic; see [fit statistics](fit-statistics.md).

## External Validation

Where definitions overlap, maintained tests compare against `linearmodels==7.0`, `statsmodels==0.14.6`, and R `plm==2.6-7` / `sandwich==3.1-3`; see `dev/tests/test_panel_stage_c_linearmodels_estimators.py` and `dev/tests/test_panel_stage_c_r_external.py`. Physical CuPy/Torch acceptance is recorded in `results/pr126_p100_fresh/validation_summary.txt`.

## References

Wooldridge (2010), *Econometric Analysis of Cross Section and Panel Data*; see also the covariance and diagnostic references in [Panel covariance](covariance.md) and [Panel diagnostics](diagnostics.md).
