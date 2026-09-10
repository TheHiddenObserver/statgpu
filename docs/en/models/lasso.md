# Lasso

> Language: English  
> Last updated: 2026-09-10  
> Switch: [简体中文](../../cn/models/lasso.md)

## What problem does it solve?

`Lasso` is linear regression with an L1 penalty. It shrinks coefficients and can drive weak ones exactly to zero, so one fitted model can support prediction and sparse feature selection.

Use it when you expect a relatively small active set among many candidate predictors. If most predictors are expected to have small but real effects, [Ridge](ridge.md) is often more stable. If important predictors occur in strongly correlated groups, [Elastic Net](elastic-net.md) is often a better starting point.

## Intuition

Lasso solves a compromise: fit the data while paying for coefficient magnitude. The L1 penalty has a sharp corner at zero, so weak updates are soft-thresholded all the way to zero rather than merely being shrunk.

```text
OLS   : fit without regularization
Ridge : shrink everything smoothly
Lasso : shrink and remove weak coefficients
```

Sparsity is useful, but selection is data- and tuning-dependent. A zero coefficient is a property of the fitted model at the chosen `alpha`, not proof that the population effect is exactly zero.

## Model and objective

With an unpenalized intercept $b$, statgpu minimizes

$$
\frac{1}{2n}\sum_{i=1}^{n}(y_i-b-x_i^\top\beta)^2
+\alpha\lVert\beta\rVert_1.
$$

Larger `alpha` means more shrinkage and usually more zeros. Because L1 acts directly on coefficient magnitude, continuous predictors should usually be put on comparable scales before regularization.

## Minimal runnable example

```python
import numpy as np
from statgpu.linear_model import Lasso

rng = np.random.default_rng(1)
X = rng.normal(size=(500, 12))
true_coef = np.zeros(12)
true_coef[[1, 5, 9]] = [2.0, -1.5, 0.8]
y = 0.7 + X @ true_coef + rng.normal(scale=0.7, size=500)

model = Lasso(
    alpha=0.08,
    device="cpu",
    compute_inference=False,
).fit(X, y)

print(model.coef_)
print(np.flatnonzero(np.abs(model.coef_) > 1e-8))
print(model.score(X, y))
```

`coef_` and `intercept_` always belong to the penalized prediction fit. Nonzero coefficients remain shrunken; they are not ordinary OLS estimates.

## Key parameters

| Parameter | Default | How to think about it |
|---|---:|---|
| `alpha` | `1.0` | Main prediction/selection tuning parameter. Prefer validation or `LassoCV` to training fit for predictive choice. |
| `fit_intercept` | `True` | Usually keep unless the model is known to have no intercept or the design already encodes one. |
| `device` | `"auto"` | Select CPU, CuPy CUDA, Torch CUDA, or automatic routing. Explicit GPU requests fail rather than silently falling back to CPU. |
| `solver` | `"fista"` | Authoritative direct-fit numerical solver on every backend. |
| `stopping` | `"coef_delta"` | Use `kkt` when an optimality-based stopping diagnostic is preferred. |
| `compute_inference` | `True` | Turn off for prediction/selection-only work. |
| `inference_method` | `"debiased"` | Choose the statistical post-fit procedure; it is independent of `device`. |
| `nodewise_alpha` | `None` | Separate tuning parameter for the node-wise precision problems used only by `debiased` inference. |

## CPU, GPU, Formula, and weights

```python
model = Lasso(
    alpha=0.08,
    device="cuda",
    solver="fista",
    stopping="kkt",
    compute_inference=False,
).fit(X, y)
```

Explicit `device="cuda"` uses CuPy CUDA and explicit `device="torch"` uses Torch CUDA. Unavailable explicit devices fail visibly. `fit()` also supports `sample_weight=` and the shared `formula=` / `data=` interface.

Analytic weights use the maintained average-loss convention. Global positive rescaling of all weights does not change the intended weighted sparse-Gaussian problem.

## Compare with nearby methods

| Method | Exact zeros? | Correlated predictors | Typical reason to choose it |
|---|:---:|---|---|
| OLS / `LinearRegression` | no | can be unstable | unpenalized estimation |
| [Ridge](ridge.md) | no | strong stabilization | prediction without feature deletion |
| **Lasso** | yes | may choose one member of a group | sparse prediction / selection |
| [Elastic Net](elastic-net.md) | yes when L1 share > 0 | more group-friendly | sparse model with correlated features |

## Advanced: solver support

For direct `Lasso.fit`, `solver` chooses the algorithm and `device` chooses the execution backend. The historical `cpu_solver` argument is retained for compatibility but does not replace `solver` for a direct fit.

| `solver` | CPU | CuPy / Torch | Notes |
|---|:---:|:---:|---|
| `fista` | yes | yes | default proximal path |
| `auto` | yes | yes | current L1 Gaussian automatic route |
| `fista_bb` | yes | yes | spectral-step variant |
| `admm` | yes | yes | alternative split solver; weight restrictions apply |
| `coordinate_descent` | yes | no | CPU-only direct-fit path |

## Advanced: inference after Lasso

Inference after a data-selected sparse fit is not ordinary fixed-model OLS inference. statgpu exposes several distinct procedures:

| `inference_method` | What it does | Main limitation |
|---|---|---|
| `debiased` | one-step de-biased/de-sparsified coefficient inference | validity depends on high-dimensional sparsity/design/noise and tuning assumptions |
| `post_selection_ols` | OLS/WLS refit on the active set on the fit-resolved backend | diagnostic after selection; not a general selective-inference guarantee |
| `bootstrap` | residual-bootstrap refits | computationally heavier and not a universal selection correction |

`post_selection_ols` is the canonical hardware-neutral spelling. Legacy `cpu_ols` and `gpu_ols` aliases are deprecated and normalize to the same statistical method; they do not choose the device.

For `LassoCV(compute_inference=True)`, the main `alpha` is selected first and inference runs only on the final full-data refit. Current inference is conditional on the selected tuning value rather than correcting separately for CV tuning uncertainty.

### `alpha` versus `nodewise_alpha`

The main `alpha` controls the penalized prediction/selection fit. `nodewise_alpha` controls only the node-wise Lasso regressions used to approximate the design precision matrix inside `inference_method="debiased"`. Changing only `nodewise_alpha` must not change the penalized `coef_`, the `LassoCV` alpha grid, fold scores, or selected `alpha_`.

An explicit finite positive `nodewise_alpha` is authoritative. With `nodewise_alpha=None` and $p\ge2$, statgpu standardizes the canonical centered/weighted working design and uses

$$
\lambda_{\mathrm{nw}}
=\sqrt{\frac{2\log(\max(p,2))}{n_{\mathrm{nw}}}},
$$

where $n_{\mathrm{nw}}=n$ without analytic weights and a Kish-style effective sample size is used for non-uniform analytic weights. The order $\sqrt{\log(p)/n}$ is theory-motivated, while the exact constant and weighted effective-sample-size convention are statgpu defaults.

This is intentionally **response-scale independent**. The historical internal rule multiplied the node-wise penalty by a residual estimate of the response scale; that behavior is superseded and is not exposed as a legacy public mode.

The node-wise solve is performed on the standardized design, checked by an independent full KKT publication gate, and transformed back to the original working-feature scale. For `p=1`, no nuisance node-wise regression exists: statgpu uses analytic univariate precision and leaves `nodewise_alpha_` as `None`.

For `LassoCV`, `nodewise_alpha` is final-refit inference configuration only. See [Lasso inference](lasso-inference.md) for the construction and [the node-wise tuning migration guide](../guides/nodewise-alpha-migration.md) for the migration contract.

### Backend and reporting boundary

NumPy, CuPy, and Torch implement the same maintained node-wise statistical definition. Explicit CUDA/Torch debiased numerical inference does not silently substitute the CPU implementation. Small reporting arrays are converted to NumPy only after the backend-native numerical inference boundary; metadata records the numerical backend/device.

With `fit_intercept=True`, debiased reporting uses a coherent centered parameterization. `coef_` / `intercept_` remain the penalized prediction fit, while `_params` contains the debiased reporting parameters. Optional intercept-inclusive simultaneous inference uses max-|Z| multiplier-bootstrap calibration and reports `_conf_int_simultaneous` separately from marginal `_conf_int`.

## Common pitfalls

- Do not interpret a selected feature as causal or certainly nonzero in the population.
- Do not ignore feature scaling before L1 regularization.
- Do not tune `alpha` by maximizing training $R^2$.
- Do not treat `post_selection_ols` as general selective inference.
- Do not assume `LassoCV` automatically corrects inference for tuning uncertainty.
- Do not confuse `alpha` with `nodewise_alpha`: the latter is inference-only.
- Do not treat a small numerical KKT residual as proof that high-dimensional inferential assumptions hold.

## Complete API reference

The runtime public constructor is the wrapper constructor plus the `nodewise_alpha=None` parameter installed by the maintained node-wise inference compatibility contract:

```python
Lasso(
    alpha=1.0,
    fit_intercept=True,
    max_iter=1000,
    tol=1e-4,
    stopping="coef_delta",
    inference_method="debiased",
    n_bootstrap=200,
    bootstrap_random_state=None,
    enable_simultaneous_inference=False,
    simultaneous_method="maxz_bootstrap",
    simultaneous_alpha=0.05,
    simultaneous_n_bootstrap=1000,
    simultaneous_random_state=None,
    simultaneous_include_intercept=False,
    device="auto",
    n_jobs=None,
    compute_inference=True,
    solver="fista",
    cpu_solver="coordinate_descent",
    lipschitz_L=None,
    admm_rho=1.0,
    gpu_memory_cleanup=False,
    nodewise_alpha=None,
)
```

The marked table below remains the source-synchronized static wrapper inventory used by the draft documentation checker. The runtime-installed public extension is listed immediately after it.

<!-- API-CONSTRUCTOR-START:Lasso -->
| Parameter | Default | Reference meaning |
|---|---:|---|
| `alpha` | `1.0` | L1 penalty strength. |
| `fit_intercept` | `True` | Fit an unpenalized intercept. |
| `max_iter` | `1000` | Maximum solver iterations. |
| `tol` | `1e-4` | Numerical convergence tolerance. |
| `stopping` | `"coef_delta"` | `coef_delta` or `kkt` convergence criterion where supported. |
| `inference_method` | `"debiased"` | `debiased`, canonical `post_selection_ols`, or `bootstrap`; legacy `cpu_ols` / `gpu_ols` aliases are deprecated. |
| `n_bootstrap` | `200` | Residual-bootstrap draws. |
| `bootstrap_random_state` | `None` | Residual-bootstrap RNG seed. |
| `enable_simultaneous_inference` | `False` | Enable simultaneous max-|Z| intervals after debiased inference. |
| `simultaneous_method` | `"maxz_bootstrap"` | Simultaneous calibration method. |
| `simultaneous_alpha` | `0.05` | Family-wise error level. |
| `simultaneous_n_bootstrap` | `1000` | Multiplier-bootstrap draws. |
| `simultaneous_random_state` | `None` | Simultaneous bootstrap RNG seed. |
| `simultaneous_include_intercept` | `False` | Include the coherent debiased intercept in the simultaneous target family. |
| `device` | `"auto"` | `auto`, `cpu`, `cuda`, or `torch`. |
| `n_jobs` | `None` | Parallelism hint where supported. |
| `compute_inference` | `True` | Compute the selected post-fit inference path. |
| `solver` | `"fista"` | Backend-neutral direct-fit solver. |
| `cpu_solver` | `"coordinate_descent"` | Legacy/shared compatibility control; not authoritative for direct fit. |
| `lipschitz_L` | `None` | Optional precomputed Lipschitz constant. |
| `admm_rho` | `1.0` | ADMM penalty parameter. |
| `gpu_memory_cleanup` | `False` | Best-effort GPU cache cleanup after fit. |
<!-- API-CONSTRUCTOR-END:Lasso -->

**Runtime-installed public extension:** `nodewise_alpha=None` — `None` uses the standardized design-side automatic rule; a finite positive real scalar explicitly sets the node-wise penalty. Successful multi-feature debiased inference publishes the resolved value in `nodewise_alpha_`.

### `fit` and core methods

`fit(X=None, y=None, sample_weight=None, formula=None, data=None)` returns `self`. `predict(X, return_cpu=True)` produces continuous predictions; `score(X, y, sample_weight=None)` returns $R^2$; `summary()` reports fitted inference when available. `get_params` / `set_params` and sklearn cloning preserve the requested `nodewise_alpha`; changing it invalidates stale fitted inference state.

Inherited inference utilities such as `adjust_pvalues`, `combine_pvalues`, `bootstrap_statistic`, and `permutation_test` are documented in the [Inference API](../guides/inference-api.md).

### Important fitted fields

| Attribute | Meaning |
|---|---|
| `coef_`, `intercept_` | penalized prediction fit |
| `n_iter_` | numerical iteration count |
| `nodewise_alpha_` | resolved node-wise tuning after successful multi-feature debiased inference; otherwise `None` |
| `_params`, `_bse`, `_zvalues`, `_pvalues`, `_conf_int` | inference/reporting arrays when inference succeeds |
| `_conf_int_simultaneous` | simultaneous intervals when explicitly enabled |
| `_inference_result` | structured inference result including node-wise and backend provenance |

Penalized-fit `rsquared_adj`, `fvalue`, `f_pvalue`, `aic`, and `bic`, when available, are compatibility/plugin diagnostics using ordinary parameter-count/residual-DoF conventions. They are not selection-, tuning-, or effective-DoF-aware criteria.

## Validation

Maintained coverage includes direct/API/clone/set-params contracts, response-scale invariance, feature-scale equivariance, analytic `p=1`, KKT failure behavior, weighted invariants, CV final-refit isolation, formula parity, independent two-feature reference checks, cache provenance, NumPy/CuPy/Torch parity, and physical CUDA validation of the accepted node-wise implementation.

## References

- Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *JRSS B*, 58(1), 267–288.
- Bühlmann, P., & van de Geer, S. (2011). *Statistics for High-Dimensional Data*. Springer.
- van de Geer, S., Bühlmann, P., Ritov, Y., & Dezeure, R. (2014). On asymptotically optimal confidence regions and tests for high-dimensional models. *Annals of Statistics*, 42(3), 1166–1202.
- Zhang, C.-H., & Zhang, S. S. (2014). Confidence intervals for low-dimensional parameters in high-dimensional linear models. *JRSS B*, 76(1), 217–242.
- Javanmard, A., & Montanari, A. (2014). Confidence intervals and hypothesis testing for high-dimensional regression. *JMLR*, 15, 2869–2909.
