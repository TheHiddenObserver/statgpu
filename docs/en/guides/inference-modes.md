# Inference Modes

> Language: English  
> Last updated: 2026-09-17  
> This page: choosing and interpreting coefficient-inference methods  
> Switch: [Chinese](../../cn/guides/inference-modes.md)

## What this guide is for

statgpu exposes several inference procedures because a classical low-dimensional regression, a fixed-penalty GLM, and a sparse model selected by L1 regularization are different statistical problems.

Use this page to answer two questions:

1. **Which inference method applies to my fitted model?**
2. **What parameter or refit does the reported interval describe?**

Detailed implementation, backend kernels, internal result storage, and validation evidence are intentionally outside this guide.

## Method overview

| Fitted model / situation | Inference method | Interpretation |
|---|---|---|
| Gaussian linear/Ridge model | classical or robust covariance | inference for the fitted linear-model coefficients |
| smooth non-Gaussian L2/no-penalty penalized GLM | `m_estimation` | fixed-penalty estimating-equation inference |
| Gaussian Lasso/ElasticNet | `debiased` | de-biased/de-sparsified coefficient inference |
| Gaussian Lasso/ElasticNet | `post_selection_ols` | OLS/WLS diagnostic refit on the selected active set |
| supported Gaussian penalized models | `bootstrap` | residual-bootstrap distribution with tuning held fixed |
| supported SCAD/MCP models | `oracle` when explicitly requested | inference conditional on the selected active set |
| unsupported loss/penalty/method combination | — | raises an error instead of substituting another inferential target |

The exact support matrix is documented in [Penalized GLM inference](penalized-glm-inference.md) and the relevant model page.

## Gaussian linear-model inference

For ordinary Gaussian linear models and the shared squared-error L2/Ridge path, `cov_type` determines the covariance estimator and reference distribution.

Common choices are:

- `nonrobust` — classical covariance with Student-t reference inference;
- `hc0`, `hc1`, `hc2`, `hc3` — heteroskedasticity-consistent covariance with normal reference inference;
- `hac` — Bartlett-kernel HAC covariance with normal reference inference.

The numerical inference follows the fitted model's supported backend. Small reporting arrays may be returned as NumPy after numerical inference is complete; this reporting conversion does not mean that an explicit CUDA/Torch fit was silently re-run on CPU.

For model-specific covariance choices, see the corresponding model page.

## Fixed-penalty M-estimation for penalized GLMs

For supported smooth non-Gaussian models with L2 or no penalty,

```python
inference_method="auto"
```

resolves to `m_estimation`.

A positive L2 fit targets the **penalized estimating equation** at the chosen penalty strength. An unpenalized fit targets the ordinary unpenalized model parameter. The current non-Gaussian fixed-penalty covariance choices are `nonrobust`, `hc0`, and `hc1`; requesting HC2/HC3/HAC on a row that does not support them raises an error.

### Analytic weights

Where the selected smooth solver supports analytic `sample_weight`, the fit and the corresponding M-estimation calculation use the same normalized weighted objective,

$$
L_w(\beta)
=
\frac{\sum_i w_i\,\ell_i(\beta)}{\sum_i w_i}.
$$

Multiplying all positive analytic weights by the same constant therefore leaves the statistical target unchanged. A loss that does not define weighted fitting, such as the unsupported weighted Cox routes, rejects the request rather than discarding the weights.

For the covariance formula and exact loss/penalty support, see [Penalized GLM inference](penalized-glm-inference.md).

## Sparse Gaussian inference

For Gaussian `Lasso`, `ElasticNet`, and equivalent squared-error L1/ElasticNet penalized models, three public modes serve different purposes.

### `debiased`

De-biased/de-sparsified inference corrects the first-order regularization bias by estimating an approximate inverse design precision matrix and applying a one-step correction.

This is the appropriate coefficient-inference mode when you need high-dimensional marginal inference and its assumptions are reasonable for the application. Node-wise precision tuning is controlled separately from the main model penalty; see [Node-wise Lasso inference tuning migration](nodewise-alpha-migration.md).

When analytic weights are supported, de-biased inference uses the same weighted-centered statistical problem as the sparse Gaussian fit. Global positive rescaling of the weights does not change that target.

If simultaneous inference is enabled, statgpu uses max-|Z| calibration rather than treating ordinary marginal intervals as simultaneous intervals. Check the Lasso/ElasticNet model documentation for the relevant simultaneous-inference controls.

### `post_selection_ols`

`post_selection_ols` first takes the active set chosen by the penalized model and then fits an **unpenalized OLS or WLS regression on exactly that active set**.

This distinction is important:

- prediction still uses the original penalized `coef_` / `intercept_`;
- the coefficient table produced by this inference mode belongs to the active-set OLS/WLS refit;
- standard OLS/WLS intervals after selecting variables on the same data are **not** general selective-inference confidence intervals.

Use this mode as a post-selection diagnostic or conditional refit, not as an automatic correction for variable-selection uncertainty.

```python
from statgpu.linear_model import Lasso

model = Lasso(
    alpha=0.1,
    solver="fista",
    compute_inference=True,
    inference_method="post_selection_ols",
)
model.fit(X, y)

# Prediction parameters remain the penalized fit.
prediction_coef = model.coef_
```

The historical `cpu_ols` and `gpu_ols` spellings are deprecated compatibility aliases for `post_selection_ols`; they do not choose the execution device.

### `bootstrap`

The Gaussian residual-bootstrap mode keeps the fitted design and tuning configuration fixed. For each draw it:

1. computes fitted values and residuals;
2. resamples residuals with replacement;
3. constructs a bootstrap response;
4. refits the same penalized model with the same tuning configuration; and
5. summarizes the resulting coefficient distribution.

This is not a universal bootstrap for every GLM family. The supported residual-bootstrap path requires `sample_weight=None` and `cov_type="nonrobust"`; weighted residual bootstrap, robust/HAC block bootstrap, non-Gaussian bootstrap, and Cox bootstrap are not supplied by this mode.

The resulting uncertainty describes the fixed-design, fixed-tuning bootstrap procedure. It does not automatically account for tuning or variable-selection uncertainty.

## SCAD/MCP active-set inference

Where `inference_method="oracle"` is supported, statgpu performs inference conditional on the active set selected by the non-convex penalized fit. `auto` does not silently choose this interpretation because conditioning on a selected support is a substantive inferential assumption.

Check [Penalized GLM inference](penalized-glm-inference.md) for the current model/backend restrictions before requesting `oracle`.

## Inference after cross-validation

CV selection and coefficient inference are separate stages:

```text
fit candidates on folds
    -> select tuning parameter
    -> refit selected model on all observations
    -> run inference once on the final refit
```

The reported uncertainty is therefore conditional on the CV-selected tuning configuration unless the method explicitly says otherwise. It does not automatically adjust for tuning-selection uncertainty.

See [Cross-Validation](cross-validation.md) for the selection/refit contract.

## Method names do not select hardware

`inference_method` chooses a statistical procedure; `device` chooses execution hardware where that procedure is supported.

- `device="cpu"` requests NumPy CPU execution;
- `device="cuda"` requires the CuPy CUDA route;
- `device="torch"` requires the Torch CUDA route;
- `device="auto"` may choose among available backends.

An unsupported explicit backend/method combination raises rather than pretending that a CPU result came from the requested accelerator. Some methods have narrower backend support than the parent estimator, so check the method-specific page when device placement is important.

## Interpreting fitted and inference parameters

For ordinary unpenalized models, fitted coefficients and inference coefficients usually refer to the same estimator. Penalized post-fit methods can differ:

- `post_selection_ols` reports an active-set unpenalized refit while prediction remains penalized;
- `debiased` reports bias-corrected inferential parameters while prediction remains penalized;
- `bootstrap` describes repeated penalized refits under the fixed tuning configuration.

Do not infer the target solely from the shape of a coefficient table. Use `inference_method_`, `inference_target_`, and the relevant model documentation when the distinction matters.

## Choosing a method

A useful decision rule is:

1. **Ordinary low-dimensional Gaussian model:** use the model's classical/robust covariance options.
2. **Smooth non-Gaussian L2/no-penalty GLM:** start with `inference_method="auto"`, which resolves to fixed-penalty M-estimation where supported.
3. **Sparse Gaussian L1/ElasticNet model:** use `debiased` for high-dimensional coefficient inference when its assumptions are appropriate; use `post_selection_ols` only when an active-set diagnostic/refit is the intended target.
4. **Need a Gaussian fixed-tuning resampling view:** use `bootstrap` only within its documented scope.
5. **SCAD/MCP active-set interpretation:** request `oracle` explicitly only when that conditional target is intended and supported.

## Related documentation

- [Penalized GLM inference](penalized-glm-inference.md) — formulas, support matrix, and detailed statistical targets
- [Cross-Validation](cross-validation.md) — tuning selection and final-refit behavior
- [Node-wise Lasso inference tuning migration](nodewise-alpha-migration.md) — `nodewise_alpha`
- [Device and GPU Memory](device-and-memory.md) — backend/device semantics
- [Lasso](../models/lasso.md) and [ElasticNet](../models/elastic-net.md) — model-specific sparse inference
