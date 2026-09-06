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
| `cpu_ols` | OLS-style refit on the selected active set through the current CPU-oriented helper | engineering/post-selection diagnostic | not a general selective-inference confidence procedure |
| `gpu_ols` | compatibility selector accepted by the unified wrapper; it currently routes to the same CPU-oriented post-selection OLS helper | diagnostic only | not a backend-native GPU OLS-inference path; GPU-resident inputs can be unsupported |
| `bootstrap` | residual-bootstrap refits of the penalized model | resampling-based uncertainty diagnostic | expensive and not, by itself, a general correction for data-driven model selection |

The older `cpu_ols_inference` / `gpu_ols_inference` names belong to the legacy Lasso surface and historical documentation; they are not the active `inference_method` values of the current unified `Lasso` wrapper.

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
| `enable_simultaneous_inference` | `False` | Run simultane