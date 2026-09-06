# Lasso inference

> Language: English  
> Last updated: 2026-09-06  
> Model guide: [Lasso](lasso.md)  
> Switch: [简体中文](../../cn/models/lasso-inference.md)

This page is the statistical-inference companion to the learner-first [Lasso guide](lasso.md). It explains what statgpu's post-fit inference modes compute, what the reported intervals mean, and where the current implementation deliberately stops making a stronger claim.

## Why Lasso inference needs its own page

The fitted Lasso coefficient vector solves a penalized prediction/selection problem. That is not the same object as an ordinary least-squares estimator from a model fixed before looking at the data. Once the same data are used to choose a sparse active set and estimate its coefficients, attaching ordinary OLS standard errors to the selected model generally ignores selection uncertainty.

statgpu therefore exposes several inference modes with different purposes. They should not be treated as interchangeable ways to print the same p-values.

## Choose the inference path by the claim you need

| `inference_method` | What statgpu computes | Appropriate interpretation | Main limitation |
|---|---|---|---|
| `debiased` | de-biased/de-sparsified coefficient estimator, standard errors, z statistics, p-values and marginal confidence intervals | coefficient-wise high-dimensional inference under de-biasing assumptions | validity depends on sparsity/design/noise conditions and on the regularization/inference construction |
| `cpu_ols_inference` | OLS-style refit on the selected active set on CPU | engineering/post-selection diagnostic | not a general selective-inference confidence procedure |
| `gpu_ols_inference` | GPU-oriented OLS-style diagnostic path | same diagnostic role while reducing avoidable host/device movement | same post-selection validity limitation |
| `bootstrap` | residual-bootstrap refits of the penalized model | resampling-based uncertainty diagnostic | expensive and not, by itself, a general correction for data-driven model selection |

`naive_ols` and `gpu_naive_ols` are maintained compatibility aliases for the OLS-style paths.

For formal coefficient inference after Lasso, `debiased` is the main statgpu path. For prediction or feature selection only, set `compute_inference=False` and avoid paying for an inference procedure you do not need.

## Minimal de-biased inference example

```python
import numpy as np
from statgpu.linear_model import Lasso

rng = np.random.default_rng(7)
X = rng.normal(size=(700, 10))
beta = np.zeros(10)
beta[[1, 4, 8]] = [1.5, -1.1, 0.7]
y = 0.4 + X @ beta + rng.normal(scale=1.0, size=X.shape[0])

model = Lasso(
    alpha=0.08,
    inference_method="debiased",
    compute_inference=True,
    device="cpu",
).fit(X, y)

print(model._params)
print(model._bse)
print(model._pvalues)
print(model._conf_int)
```

The penalized coefficients remain available in `coef_`. Inference reporting uses the de-biased parameter vector stored in the inference result, so a penalized coefficient and its de-biased inferential estimate need not be numerically identical.

## What de-biasing changes

Let the fitted Lasso coefficient vector be \(\hat\beta\), with residual vector

$$
r = y - b - X\hat\beta.
$$

The L1 penalty creates shrinkage bias. statgpu corrects the coefficient vector with

$$
\hat\theta^{\mathrm{db}}
= \hat\beta + \frac{1}{n} M X^\top r,
$$

where \(M\) is a data-dependent decorrelation matrix intended to approximate an inverse of the feature Gram/covariance matrix.

The correction does **not** mean that the original sparse Lasso estimate has become unpenalized. `coef_` is still the fitted penalized model used for prediction. The de-biased vector is an inference object used for coefficient-wise uncertainty reporting.

## How statgpu constructs the decorrelation matrix

For each feature \(j\), statgpu performs a node-wise Lasso regression of \(x_j\) on the remaining columns \(X_{-j}\). In the CPU implementation, the node-wise penalty scale is

$$
\lambda_{\mathrm{nw}}
= \hat\sigma\sqrt{\frac{2\log(\max(p,2))}{n}}.
$$

Writing the node-wise coefficient vector as \(\hat\gamma_j\), define

$$
z_j = x_j - X_{-j}\hat\gamma_j,
\qquad
C_j = \frac{z_j^\top x_j}{n}.
$$

The corresponding row of \(M\) is assembled from \(1/C_j\) on the diagonal and \(-\hat\gamma_j/C_j\) on the other coordinates. This is the implementation bridge between the sparse precision-matrix idea in de-sparsified Lasso theory and the actual statgpu computation.

If a node-wise system is numerically singular, statgpu uses rank-failure-aware inverse/pseudoinverse fallbacks where the affected inference path defines one. Numerical convergence of the node-wise optimization is necessary, but it does not replace the statistical assumptions behind de-biasing.

## Standard errors, z statistics, and marginal intervals

With

$$
\widehat\Sigma = \frac{X^\top X}{n},
\qquad
V = M\widehat\Sigma M^\top,
$$

statgpu uses

$$
\widehat{\mathrm{se}}_j
= \sqrt{\frac{\hat\sigma^2 V_{jj}}{n}},
\qquad
Z_j = \frac{\hat\theta_j^{\mathrm{db}}}{\widehat{\mathrm{se}}_j}.
$$

Two-sided p-values use a standard-normal reference distribution. The current marginal confidence interval stored in `_conf_int` is a 95% interval based on the normal critical value.

Important distinctions:

- `_conf_int[j]` is a **marginal** interval for one parameter at a time.
- A collection of 95% marginal intervals does not automatically have 95% simultaneous coverage over the whole coefficient vector.
- `simultaneous_alpha` does not change these marginal intervals; it controls the separate simultaneous procedure described below.

When an intercept is fitted, statgpu reports intercept uncertainty as well, but the feature de-biasing construction and the intercept calculation are not the same algebraic operation. Do not interpret the intercept row as another node-wise-Lasso coordinate.

## Simultaneous max-|Z| inference

Lasso can optionally calibrate a common critical value with a multiplier bootstrap. Enable it with:

```python
model = Lasso(
    alpha=0.08,
    inference_method="debiased",
    compute_inference=True,
    enable_simultaneous_inference=True,
    simultaneous_method="maxz_bootstrap",
    simultaneous_alpha=0.05,
    simultaneous_n_bootstrap=2000,
    simultaneous_random_state=123,
).fit(X, y)

marginal = model._conf_int
simultaneous = model._conf_int_simultaneous
critical = model._simultaneous_critical_value
```

The current method draws independent standard-normal multipliers \(\xi_i\), forms bootstrap score perturbations from the fitted residuals, and records the maximum absolute standardized perturbation over the target feature set. Schematically,

$$
T^*
= \max_{j\in\mathcal J}
\left|
\frac{n^{-1}\sum_i \xi_i r_i (MX_i)_j}
{\widehat{\mathrm{se}}_j}
\right|.
$$

The empirical \((1-\alpha)\)-quantile of \(T^*\) becomes the common critical value \(c_{1-\alpha}\), giving

$$
\hat\theta_j^{\mathrm{db}}
\pm c_{1-\alpha}\widehat{\mathrm{se}}_j.
$$

Because the same critical value protects the target family, these intervals are normally wider than the corresponding marginal intervals.

### Simultaneous-inference controls

| Parameter | Default | Meaning |
|---|---:|---|
| `enable_simultaneous_inference` | `False` | Run simultaneous calibration after successful de-biased inference. |
| `simultaneous_method` | `"maxz_bootstrap"` | Current supported simultaneous calibration method. |
| `simultaneous_alpha` | `0.05` | Family-wise error level used to select the max-|Z| critical value. |
| `simultaneous_n_bootstrap` | `1000` | Number of multiplier-bootstrap draws. |
| `simultaneous_random_state` | `None` | Seed for reproducible multiplier draws. |
| `simultaneous_include_intercept` | `False` | Requests that the reported simultaneous target include the intercept; see the implementation boundary below. |

`enable_simultaneous_inference=True` requires `compute_inference=True`, `inference_method="debiased"`, and `simultaneous_method="maxz_bootstrap"`. Unsupported combinations fail instead of silently returning ordinary marginal intervals.

### Current intercept boundary

The recommended/default simultaneous family is the feature vector (`simultaneous_include_intercept=False`). In the current implementation, setting `simultaneous_include_intercept=True` applies the feature-calibrated max-|Z| critical value to the intercept row, but the bootstrap maximum itself is still constructed from feature-score coordinates. Therefore, do **not** interpret that option as a separately bootstrapped intercept-inclusive max-|Z| family. If a scientifically essential family must include the intercept, treat that case as requiring additional validation rather than relying on the default feature-family guarantee.

## Simultaneous intervals are not the same as p-value adjustment

These operations answer related but different questions:

- `_conf_int` gives coefficient-wise marginal intervals.
- `_conf_int_simultaneous` uses a common max-|Z| critical value for an interval family.
- `model.adjust_pvalues(method="bh")` or `adjust_pvalues(...)` applies a multiple-testing correction to an existing vector of p-values.
- `model.combine_pvalues(...)` asks whether a set of p-values contains global evidence; it does not construct simultaneous coefficient intervals.

The generic estimator-bound and module-level multiple-testing APIs are documented in the [Inference API reference](../guides/inference-api.md).

## Backend behavior and reporting boundary

De-biased coefficient inference has dedicated numerical paths for NumPy CPU, CuPy CUDA, and Torch CUDA. The expensive node-wise/decorrelation work is performed on the selected numerical backend where the corresponding path is implemented.

The reporting layer is NumPy-oriented: final inferential arrays such as `_bse`, `_pvalues`, `_conf_int`, and structured result metadata are exposed as host-side reporting objects.

Simultaneous calibration has an additional boundary: after backend-native de-biased inference, the state required by the max-|Z| multiplier bootstrap is represented on the CPU/NumPy side and the bootstrap calibration runs there. Thus `device="cuda"` or `device="torch"` does **not** mean that every simultaneous-bootstrap draw stays on the GPU.

Explicit unavailable GPU devices should fail rather than silently becoming CPU estimation paths. The CPU reporting/calibration boundary described above is an explicit part of inference reporting, not a hidden estimator fallback.

## Sample weights and inferential scope

`Lasso.fit(..., sample_weight=...)` is a supported fitting surface. Weighted high-dimensional inference, however, has stronger modeling and normalization questions than the unweighted formulas shown above. The equations on this page describe the core de-biased construction and should not be read as an automatic theorem for arbitrary analytic-weight designs. If weighted coefficient inference is scientifically central, validate the exact weighting convention and target estimand for that application rather than assuming that unweighted asymptotics transfer unchanged.

## Residual bootstrap path

`inference_method="bootstrap"` performs repeated residual resampling and penalized refits. The constructor controls are:

- `n_bootstrap` (default `200`);
- `bootstrap_random_state` (default `None`).

The current implementation derives standard errors from bootstrap variability, sign-based two-sided p-values, and percentile confidence intervals. This is computationally much more expensive than one de-biased fit. It is also not advertised as a complete selective-inference procedure: resampling the fitted residual model does not automatically account for every consequence of choosing the active set and tuning parameters from the same data.

## OLS-style post-selection paths

The OLS-style paths refit or diagnose the selected active set using ordinary linear-model machinery. They are useful for engineering comparison and for seeing what an unpenalized active-set refit would report, but their intervals must not be described as valid selective-inference intervals merely because the selected model is sparse.

Use them when that diagnostic is exactly what you want; do not choose them as a shortcut around the assumptions of de-biased or selection-aware inference.

## Fitted inference outputs

When the selected inference path succeeds, the reporting surface can include:

| Attribute | Meaning |
|---|---|
| `_params` | parameter vector used by inference reporting; for `debiased`, feature entries are de-biased estimates |
| `_bse` | standard errors |
| `_tvalues` | historical/statistic storage used by some paths; de-biased reporting has z semantics |
| `_zvalues` | z-style statistic field when populated through the structured result layer |
| `_pvalues` | two-sided coefficient p-values |
| `_conf_int` | marginal confidence intervals |
| `_conf_int_simultaneous` | simultaneous intervals after successful max-|Z| calibration |
| `_simultaneous_critical_value` | calibrated common critical value |
| `_inference_result` | structured inference result and method/backend metadata |

`summary()` uses the inference result available for the fitted model. Underscore-prefixed arrays are established reporting surfaces in the current release, but their meaning remains method-specific.

## Reproducibility and cost

For reproducible resampling, set `bootstrap_random_state` or `simultaneous_random_state` as appropriate. Increasing `simultaneous_n_bootstrap` reduces Monte Carlo noise in the critical-value estimate at the cost of more CPU work and memory traffic.

De-biased inference can be substantially more expensive than fitting the original Lasso because it solves many node-wise sparse regressions. This is especially noticeable as the feature dimension grows. Backend acceleration changes the numerical cost profile but not the statistical assumptions.

## What the current implementation does not claim

- Ordinary post-selection OLS intervals are not general selective-inference intervals.
- Residual bootstrap is not advertised as a universal correction for model-selection uncertainty.
- Marginal de-biased intervals do not provide joint coverage without simultaneous calibration.
- Simultaneous max-|Z| intervals do not replace FDR procedures such as Benjamini-Hochberg when FDR is the scientific target.
- Numerical convergence, GPU execution, and a small KKT residual do not prove that the high-dimensional assumptions required for de-biased inference hold for a dataset.
- The default simultaneous target is the feature family; see the intercept boundary above before requesting intercept inclusion.

## References

- Zhang, C.-H., & Zhang, S. S. (2014). Confidence intervals for low-dimensional parameters in high-dimensional linear models. *Journal of the Royal Statistical Society: Series B*, 76(1), 217–242.
- van de Geer, S., Bühlmann, P., Ritov, Y., & Dezeure, R. (2014). On asymptotically optimal confidence regions and tests for high-dimensional models. *The Annals of Statistics*, 42(3), 1166–1202.
- Javanmard, A., & Montanari, A. (2014). Confidence intervals and hypothesis testing for high-dimensional regression. *Journal of Machine Learning Research*, 15, 2869–2909.
- Bühlmann, P., & van de Geer, S. (2011). *Statistics for High-Dimensional Data*. Springer.
