# Lasso inference

> Language: English  
> Last updated: 2026-09-09  
> Model guide: [Lasso](lasso.md)  
> Switch: [简体中文](../../cn/models/lasso-inference.md)

This page is the statistical-inference companion to the learner-first [Lasso guide](lasso.md). It explains what statgpu's post-fit inference modes compute, how to interpret the reported intervals, and where the current implementation deliberately stops making a stronger claim.

## Why Lasso inference needs its own page

The fitted Lasso coefficient vector solves a penalized prediction/selection problem. That is not the same object as an ordinary least-squares estimator from a model fixed before looking at the data. Once the same data are used to choose a sparse active set and estimate its coefficients, attaching ordinary OLS standard errors to the selected model generally ignores selection uncertainty.

statgpu therefore exposes several inference modes with different purposes. They should not be treated as interchangeable ways to print the same p-values.

## Choose the inference path by the claim you need

| `inference_method` | What statgpu computes | Appropriate interpretation | Main limitation |
|---|---|---|---|
| `debiased` | De-biased/de-sparsified coefficient estimator, standard errors, z statistics, p-values, and marginal confidence intervals | Coefficient-wise high-dimensional inference under de-biasing assumptions | Validity depends on sparsity/design/noise conditions and on the regularization/inference construction |
| `post_selection_ols` | Unpenalized OLS/WLS refit on the active set selected by the penalized fit, using the fit-resolved backend | Engineering/post-selection diagnostic | Not a general selective-inference confidence procedure |
| `bootstrap` | Residual-bootstrap refits of the penalized model | Resampling-based uncertainty diagnostic | Expensive and not, by itself, a general correction for data-driven model selection |

`post_selection_ols` is the canonical hardware-neutral spelling. The legacy unified aliases `cpu_ols` and `gpu_ols` are deprecated together and remain accepted for one compatibility cycle with `FutureWarning`; both normalize to `post_selection_ols`. `LassoCV` additionally recognizes the older `cpu_ols_inference` / `gpu_ols_inference` spellings at its compatibility boundary and normalizes them to the same statistical method.

The method name does **not** choose the execution device. `device="cpu"`, `device="cuda"`, and `device="torch"` are authoritative; only genuine `device="auto"` may preserve a backend-native CuPy or Torch-CUDA input as part of automatic routing.

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

The penalized coefficients remain available in `coef_`. Inference reporting uses the de-biased parameter vector, so a penalized coefficient and its de-biased inferential estimate need not be numerically identical.

## What de-biasing changes

Let the fitted Lasso coefficient vector be $\hat\beta$, with residual vector

$$
r = y - b - X\hat\beta.
$$

The L1 penalty creates shrinkage bias. statgpu corrects the coefficient vector with

$$
\hat\theta^{\mathrm{db}}
= \hat\beta + \frac{1}{n} M X^\top r,
$$

where $M$ is a data-dependent decorrelation matrix intended to approximate an inverse of the feature Gram/covariance matrix.

The correction does **not** mean that the original sparse Lasso estimate has become unpenalized. `coef_` is still the fitted penalized model used for prediction. The de-biased vector is an inference object used for coefficient-wise uncertainty reporting.

## How statgpu constructs the decorrelation matrix

For each feature $j$, statgpu performs a node-wise Lasso regression of $x_j$ on the remaining columns $X_{-j}$. In the CPU implementation, the node-wise penalty scale is

$$
\lambda_{\mathrm{nw}}
= \hat\sigma\sqrt{\frac{2\log(\max(p,2))}{n}}.
$$

Writing the node-wise coefficient vector as $\hat\gamma_j$, define

$$
z_j = x_j - X_{-j}\hat\gamma_j,
\qquad
C_j = \frac{z_j^\top x_j}{n}.
$$

The corresponding row of $M$ is assembled from $1/C_j$ on the diagonal and $-\hat\gamma_j/C_j$ on the other coordinates. This is the implementation bridge between the sparse precision-matrix idea in de-sparsified Lasso theory and the actual statgpu computation.

Numerical convergence of the node-wise optimization is necessary, but it does not replace the statistical assumptions behind de-biasing.

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

When an intercept is fitted, prediction and inference deliberately have separate parameter ownership. Public `coef_` and `intercept_` remain the penalized prediction fit. The inference slopes are de-biased, and the reported original-coordinate intercept is the coherent centered value

$$
\hat\theta^{\mathrm{db}}_0
= \bar y_w - \bar x_w^\top\hat\theta^{\mathrm{db}}.
$$

Its standard error and influence representation are therefore tied to the same centered de-biased parameterization rather than treating the intercept as another node-wise-Lasso feature coordinate.

## Simultaneous max-|Z| inference

Lasso can optionally calibrate a common critical value with a multiplier bootstrap:

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

The method draws independent standard-normal multipliers $\xi_i$, forms bootstrap score perturbations from the fitted residuals, and records the maximum absolute standardized perturbation over the requested target family. Schematically,

$$
T^*
= \max_{j\in\mathcal J}
\left|
\frac{n^{-1}\sum_i \xi_i r_i (MX_i)_j}
{\widehat{\mathrm{se}}_j}
\right|.
$$

The empirical $(1-\alpha)$-quantile of $T^*$ becomes the common critical value $c_{1-\alpha}$, giving

$$
\hat\theta_j^{\mathrm{db}}
\pm c_{1-\alpha}\widehat{\mathrm{se}}_j.
$$

Because the same critical value protects the target family, simultaneous intervals are normally wider than the corresponding marginal intervals.

### Simultaneous-inference controls

| Parameter | Default | Meaning |
|---|---:|---|
| `enable_simultaneous_inference` | `False` | Run simultaneous calibration after successful de-biased inference. |
| `simultaneous_method` | `"maxz_bootstrap"` | Current supported simultaneous calibration method. |
| `simultaneous_alpha` | `0.05` | Family-wise error level used to select the common critical value. |
| `simultaneous_n_bootstrap` | `1000` | Number of multiplier-bootstrap draws. |
| `simultaneous_random_state` | `None` | Seed for reproducible multiplier draws. |
| `simultaneous_include_intercept` | `False` | Include the centered de-biased intercept in both the bootstrap max-|Z| target and the reported joint interval family. |

`enable_simultaneous_inference=True` requires `compute_inference=True`, `inference_method="debiased"`, and `simultaneous_method="maxz_bootstrap"`. `simultaneous_alpha` must lie strictly in `(0, 1)` and `simultaneous_n_bootstrap` must be positive; unsupported combinations fail instead of silently returning ordinary marginal intervals.

### Intercept-inclusive simultaneous family

When `simultaneous_include_intercept=True`, the same centered-nodewise original-coordinate intercept influence used by the marginal standard error participates in the bootstrap maximum itself. The intercept is therefore not merely an extra row receiving a feature-only critical value: it is part of the calibrated joint target family.

The default remains `False`, so callers who only need simultaneous coverage over the feature vector do not pay for an enlarged target family.

## Simultaneous intervals are not p-value adjustment

These operations answer related but different questions:

- `_conf_int` gives coefficient-wise marginal intervals.
- `_conf_int_simultaneous` uses a common max-|Z| critical value for an interval family.
- `model.adjust_pvalues(method="bh")` or module-level `adjust_pvalues(...)` applies a multiple-testing correction to an existing vector of p-values.
- `model.combine_pvalues(...)` asks whether a set of p-values contains global evidence; it does not construct simultaneous coefficient intervals.

The generic estimator-bound and module-level multiple-testing APIs are documented in the [Inference API reference](../guides/inference-api.md).

## Backend behavior and reporting boundary

Post-fit method identity and backend identity are separate.

For `post_selection_ols`, inference always reuses the successful penalized fit's recorded backend and concrete device. The active-set OLS/WLS refit, covariance calculation, reference-distribution inference, and confidence-interval numerics run on NumPy, CuPy, or Torch according to that fit-resolved backend. Explicit GPU requests fail closed when the requested backend is unavailable; they do not silently become CPU OLS inference.

For `debiased`, the maintained marginal CuPy/Torch paths likewise keep their numerical coefficient inference on the executed GPU backend. The established reporting layer remains NumPy-oriented: final arrays such as `_bse`, `_pvalues`, `_conf_int`, and structured result metadata are snapshotted to host-side reporting objects only after numerical inference is complete.

For centered `fit_intercept=True` simultaneous inference on CuPy/Torch, the expensive B×n multiplier draws, feature/intercept scores, max-|Z| reduction, quantile calibration, and joint confidence-interval numerics remain on the same concrete GPU device before the final reporting snapshot. The historical `fit_intercept=False` simultaneous path still uses the pre-existing generic reporting-stage helper and is not claimed as GPU-native by the PR #138 contract.

Residual `bootstrap` is different: its current residual-refit implementation is CPU-native. A GPU `device` on the estimator therefore does not imply that residual-bootstrap resampling/refits are GPU-native.

## Sample weights and inferential scope

`Lasso.fit(..., sample_weight=...)` is a supported fitting surface. The maintained NumPy/CuPy/Torch `debiased` paths use the same weighted-centered average-loss working problem, so multiplying all analytic weights by the same positive constant leaves the penalized fit and maintained de-biased inference unchanged.

That implementation contract is not, by itself, a theorem for every scientific interpretation of analytic weights. If weighted coefficient inference is central to the application, validate the weighting convention and target estimand against the assumptions of the intended inferential analysis.

## Residual bootstrap path

`inference_method="bootstrap"` performs repeated residual resampling and penalized refits. Its constructor controls are `n_bootstrap` (default `200`) and `bootstrap_random_state` (default `None`).

The current implementation derives standard errors from bootstrap variability, sign-based two-sided p-values, and percentile confidence intervals. It is computationally much more expensive than one de-biased fit, and it is not advertised as a complete selective-inference procedure.

## OLS-style post-selection path

`inference_method="post_selection_ols"` refits an **unpenalized OLS or WLS model on exactly the active set** chosen by the penalized fit. `coef_` and `intercept_` remain the penalized prediction fit; `_params` and `_inference_result` own the active-set refit used for inferential reporting.

The refit uses the fit-resolved NumPy/CuPy/Torch backend. Rank-deficient active designs use an effective-rank Moore-Penrose/SVD calculation instead of ordinary normal equations. Classical `cov_type="nonrobust"` reporting uses the maintained Student-t convention; robust/HAC covariance choices use the shared Gaussian robust-covariance layer and its normal-reference convention where exposed by the estimator.

Inactive full-space coordinates retain compatibility placeholders (`SE=0`, statistic `0`, `p=1`, and `[0, 0]` intervals). These are not claims that an omitted coefficient is known exactly; use the selected-feature metadata to identify coordinates that received the active-set refit.

Most importantly, ordinary OLS/WLS intervals formed after choosing variables from the same data remain a **post-selection diagnostic**, not a general selective-inference confidence procedure.

## Fitted inference outputs

When the selected inference path succeeds, the reporting surface can include:

| Attribute | Meaning |
|---|---|
| `_params` | Parameter vector used by inference reporting; for `debiased`, feature entries are de-biased estimates; for `post_selection_ols`, active entries belong to the unpenalized refit |
| `_bse` | Standard errors |
| `_tvalues` | Historical/statistic storage used by some paths; de-biased reporting has z semantics |
| `_zvalues` | z-style statistic field when populated through the structured result layer |
| `_pvalues` | Two-sided coefficient p-values |
| `_conf_int` | Marginal confidence intervals |
| `_conf_int_simultaneous` | Simultaneous intervals after successful max-|Z| calibration |
| `_simultaneous_critical_value` | Calibrated common critical value |
| `_inference_result` | Structured inference result and method/backend metadata |

`summary()` uses the inference result available for the fitted model. Underscore-prefixed arrays are established reporting surfaces in the current release, but their meaning remains method-specific.

## Reproducibility and cost

For reproducible resampling, set `bootstrap_random_state` or `simultaneous_random_state` as appropriate. Increasing `simultaneous_n_bootstrap` reduces Monte Carlo noise in the critical-value estimate at the cost of more work and memory traffic on the path's actual numerical backend.

De-biased inference can be substantially more expensive than fitting the original Lasso because it solves many node-wise sparse regressions. Backend acceleration changes the numerical cost profile but not the statistical assumptions.

## What the current implementation does not claim

- Ordinary post-selection OLS/WLS intervals are not general selective-inference intervals.
- Residual bootstrap is not a universal correction for model-selection uncertainty.
- Marginal de-biased intervals do not provide joint coverage without simultaneous calibration.
- Simultaneous max-|Z| intervals do not replace FDR procedures such as Benjamini-Hochberg when FDR is the scientific target.
- Numerical convergence, GPU execution, and a small KKT residual do not prove that the high-dimensional assumptions required for de-biased inference hold for a dataset.
- The historical `fit_intercept=False` simultaneous path is not claimed as GPU-native merely because a GPU was used for the penalized/de-biased fit.

## References

- Zhang, C.-H., & Zhang, S. S. (2014). Confidence intervals for low-dimensional parameters in high-dimensional linear models. *Journal of the Royal Statistical Society: Series B*, 76(1), 217–242.
- van de Geer, S., Bühlmann, P., Ritov, Y., & Dezeure, R. (2014). On asymptotically optimal confidence regions and tests for high-dimensional models. *The Annals of Statistics*, 42(3), 1166–1202.
- Javanmard, A., & Montanari, A. (2014). Confidence intervals and hypothesis testing for high-dimensional regression. *Journal of Machine Learning Research*, 15, 2869–2909.
- Bühlmann, P., & van de Geer, S. (2011). *Statistics for High-Dimensional Data*. Springer.
