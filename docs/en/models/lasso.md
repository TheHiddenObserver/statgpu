# Lasso

> Language: English  
> Last updated: 2026-09-08  
> This page: Model documentation  
> Switch: [Chinese](../../cn/models/lasso.md)

Language switch: [Chinese](../../cn/models/lasso.md)

## Overview

`Lasso` provides L1-regularized linear regression with CPU/GPU execution and multiple inference modes. Direct fitting uses one backend-neutral `solver` interface; device selection, solver selection, and statistical inference method are separate choices.

## Path

`statgpu.linear_model.Lasso`

## Objective Function

Estimate
$$
\min_{\beta}\frac{1}{2n}\|y - X\beta\|_2^2 + \alpha\|\beta\|_1
$$
with iterative optimization (`fista`, `admm`, or coordinate descent where supported).

## Estimating Equation

The model is solved by iterative optimization rather than a closed-form normal equation. Stopping can be based on coefficient change (`coef_delta`) or KKT consistency (`kkt`), depending on `stopping`.

`solver` is the authoritative direct-fit algorithm selector on every backend:

- CPU coordinate descent: `solver="coordinate_descent"`
- CPU or GPU proximal path: `solver="fista"` (or another supported solver)
- backend location: selected separately with `device="cpu"`, `"cuda"`, or `"torch"`

The historical `cpu_solver` constructor argument is deprecated. It remains accepted for a compatibility cycle but does not select the direct-fit algorithm in the unified solver engine. Migrate legacy code to `solver=...`; see the [penalized solver API migration guide](../guides/penalized-solver-api-migration.md).

## Covariance/Inference

`Lasso` inference is controlled by `inference_method`:

- `post_selection_ols`: hardware-neutral active-set OLS/WLS refit diagnostic.
- `debiased`: de-biased (de-sparsified) Lasso inference with z-statistic semantics.
- `bootstrap`: residual bootstrap; typically slower and still not a universal selection-aware correction.

The unified spellings `cpu_ols` and `gpu_ols` are deprecated together. During the compatibility window they emit `FutureWarning` and normalize to `post_selection_ols`; they are **not** separate CPU and GPU statistical procedures. `LassoCV` additionally accepts the older `cpu_ols_inference` / `gpu_ols_inference` spellings at its compatibility boundary and normalizes them to the same method.

### What `post_selection_ols` computes

The penalized fit first chooses an active feature set. statgpu then refits an **unpenalized OLS model on exactly those selected columns**, or WLS when `sample_weight` is supplied, using the backend/device recorded by the successful penalized fit. Gaussian covariance and reference-distribution inference are computed on that same numerical backend before the established NumPy reporting snapshot is taken.

The original penalized `coef_` remains the coefficient vector used for prediction. The active-set OLS/WLS refit is exposed for inference/reporting through `_params`, `_inference_result`, `_bse`, `_tvalues`/`_zvalues`, `_pvalues`, and `_conf_int`.

Validity notes:

- `post_selection_ols` is a heuristic post-selection diagnostic. Its intervals should not be interpreted as general selective-inference confidence intervals after choosing variables from the same data.
- The ordinary `debiased` `_conf_int` is marginal per coefficient. Simultaneous/joint family-wise coverage requires the dedicated simultaneous inference path.
- Rank-deficient active refits use the effective design rank for residual degrees of freedom and a design-level Moore-Penrose/SVD refit rather than squaring the condition number through normal equations.

### Device/backend rule

`inference_method` describes **what statistical procedure is computed**; it does not choose hardware.

- explicit `device="cpu"` -> NumPy CPU;
- explicit `device="cuda"` -> CuPy CUDA only, failing closed if unavailable;
- explicit `device="torch"` -> Torch CUDA only, failing closed if unavailable;
- only genuine estimator/global `device="auto"` may preserve an already backend-native CuPy or Torch-CUDA input during automatic routing.

Backend reuse is method-specific. `post_selection_ols` reuses the successful fit's
`_selected_backend_name` / `_selected_backend_device`. Maintained CuPy/Torch
**marginal** `debiased` inference stays on the executed GPU backend, including
scalar normal-reference critical values.

For centered `fit_intercept=True` debiased inference, the expensive simultaneous
multiplier-bootstrap stage also stays on the same concrete CuPy/Torch device. The
coherent marginal result has already taken its established O(p) NumPy reporting
snapshot; only those small marginal parameter/SE arrays are mapped back to the
execution device. The B×n multiplier draws, feature/intercept scores, max-|Z|
reduction, quantile calibration, and joint confidence-interval numerics then
remain backend-native before the joint result is snapshotted for reporting. The
result records `simultaneous_numerical_backend`,
`simultaneous_numerical_device`, `simultaneous_reporting_backend="numpy"`, and
`simultaneous_reporting_boundary="post_numerical_inference"`. The historical
`fit_intercept=False` simultaneous path still uses its pre-existing generic
reporting-stage helper and is not claimed as GPU-native by this PR.

Residual `bootstrap` currently uses CPU-native residual refits, so an explicit
GPU `device` controls the penalized fit but does not make bootstrap GPU-native.

With analytic weights, direct Lasso and debiased inference use the same
weighted-centered average-loss convention on NumPy/CuPy/Torch, so multiplying all
weights by one positive constant does not change the statistical problem.
`LassoCV` uses the same convention for the default alpha grid, every weighted
training fold, validation MSE, and final refit. Constant positive weights take the
exact unweighted CV path. Once AUTO resolves a concrete backend for CV, the final
selected-alpha `Lasso` refit remains on that backend.

### Debiased intercept ownership

With `inference_method="debiased"`, prediction and inference intentionally expose
different intercept estimates. Public `coef_` and `intercept_` remain the
**penalized prediction fit**. Inference reporting uses
`theta_db = _params[1:]` and the matching original-coordinate intercept
`_params[0] = ybar_w - xbar_w @ theta_db`.

Consequently, `_bse[0]`, the first z-statistic/p-value, and `_conf_int[0]` describe
the debiased reporting intercept, not `intercept_`. Shifting every feature by a
constant vector `c` leaves the debiased slopes unchanged and shifts `_params[0]`
by `-c @ theta_db`, preserving one coherent parameterization. The structured
result records `intercept_estimator="centered_debiased"` and
`intercept_influence="centered_nodewise"` in metadata.

For `LassoCV(compute_inference=True, inference_method="debiased")`, the outer CV
estimator exposes the same final-refit `_inference_result` and matching
`_params`/SE/statistic/p-value/CI reporting surface as `estimator_`. Its public
`coef_`/`intercept_` still belong to the penalized selected-alpha prediction
refit.

### Simultaneous debiased inference

With `enable_simultaneous_inference=True`, Lasso calibrates a multiplier-bootstrap
max-|Z| critical value. The ordinary `_conf_int` remains marginal; the joint
intervals are stored separately in `_conf_int_simultaneous`.

`simultaneous_alpha` must lie strictly in `(0, 1)` and
`simultaneous_n_bootstrap` must be a positive integer. These controls are validated before
NumPy/CuPy/Torch backend dispatch.

`simultaneous_include_intercept=False` calibrates the family over feature
coefficients only. With `simultaneous_include_intercept=True`, the same centered-
nodewise original-coordinate intercept influence used by the marginal debiased
SE is part of the bootstrap maximum itself. It is therefore not merely an extra
output row receiving a feature-only critical value. On CuPy/Torch with
`fit_intercept=True`, this centered simultaneous calculation is backend-native as
described above. Every successful refit clears any previous simultaneous critical
value, target mask, intervals, and precision/influence state before publishing
the new result.

## Parameters

This table is the complete public constructor inventory for `statgpu.linear_model.Lasso`.

| Parameter | Default | Description |
|---|---:|---|
| `alpha` | `1.0` | L1 regularization strength. |
| `fit_intercept` | `True` | Whether to fit an intercept. |
| `max_iter` | `1000` | Maximum optimization iterations. |
| `tol` | `1e-4` | Convergence tolerance. |
| `stopping` | `"coef_delta"` | Stopping rule: `coef_delta` / `kkt`. |
| `inference_method` | `"debiased"` | `post_selection_ols` / `debiased` / `bootstrap`; deprecated `cpu_ols` and `gpu_ols` aliases remain temporarily accepted. |
| `n_bootstrap` | `200` | Bootstrap draws for the residual-bootstrap inference path. |
| `bootstrap_random_state` | `None` | RNG seed for residual-bootstrap inference. |
| `enable_simultaneous_inference` | `False` | Enable simultaneous inference (debiased only). |
| `simultaneous_method` | `"maxz_bootstrap"` | Simultaneous-inference method; currently `maxz_bootstrap`. |
| `simultaneous_alpha` | `0.05` | Simultaneous family-wise error level; must be strictly in `(0, 1)` when simultaneous inference is enabled. |
| `simultaneous_n_bootstrap` | `1000` | Positive integer multiplier-bootstrap draw count for max-|Z| calibration when simultaneous inference is enabled. |
| `simultaneous_random_state` | `None` | RNG seed for simultaneous bootstrap. |
| `simultaneous_include_intercept` | `False` | Whether the debiased intercept is included in both the simultaneous target set and max-|Z| calibration family. |
| `device` | `"auto"` | Execution device: `auto`, `cpu`, `cuda` (CuPy), or `torch` (Torch CUDA). |
| `n_jobs` | `None` | CPU parallelism where supported. |
| `compute_inference` | `True` | Whether to compute post-fit inference. |
| `solver` | `"fista"` | Backend-neutral direct-fit solver; use `coordinate_descent` for the CPU CD path or another supported solver as appropriate. |
| `cpu_solver` | `"coordinate_descent"` | **Deprecated compatibility parameter.** It does not select the current direct-fit algorithm; use `solver` instead. |
| `lipschitz_L` | `None` | Optional user-supplied Lipschitz constant for compatible iterative solvers. |
| `admm_rho` | `1.0` | ADMM penalty parameter when the ADMM path is selected. |
| `gpu_memory_cleanup` | `False` | Best-effort GPU memory cleanup after fit where supported. |

## CPU+GPU Examples

```python
from statgpu.linear_model import Lasso

# CPU coordinate descent: solver selects the algorithm, device selects CPU.
m_cpu = Lasso(
    alpha=0.1,
    device="cpu",
    solver="coordinate_descent",
    stopping="kkt",
)
m_cpu.fit(X, y)

# GPU FISTA + the same hardware-neutral post-selection inference method.
m_gpu = Lasso(
    alpha=0.1,
    device="cuda",
    solver="fista",
    stopping="kkt",
    inference_method="post_selection_ols",
    gpu_memory_cleanup=True,
)
m_gpu.fit(X, y)

# Prediction uses the penalized model; inference reports the active-set refit.
penalized_coef = m_gpu.coef_
post_selection_params = m_gpu._params
```

Simultaneous inference example:

```python
m_sim = Lasso(
    alpha=0.1,
    device="cpu",
    inference_method="debiased",
    enable_simultaneous_inference=True,
    simultaneous_method="maxz_bootstrap",
    simultaneous_alpha=0.05,
    simultaneous_n_bootstrap=1000,
    simultaneous_random_state=7,
    simultaneous_include_intercept=True,
)
m_sim.fit(X, y)
ci_marginal = m_sim._conf_int
ci_simul = m_sim._conf_int_simultaneous
```

## strict/approx difference

`debiased` is the main high-dimensional coefficient-inference path. `post_selection_ols` is a lighter active-set OLS/WLS diagnostic, while `bootstrap` is a more expensive resampling path. Their statistical claims are different and should not be treated as interchangeable.

## Outputs

- Penalized prediction fit: `intercept_`, `coef_`, `n_iter_`
- Inference (if enabled): `_params`, `_bse`, `_tvalues` / `_zvalues`, `_pvalues`, `_conf_int`, `_inference_result`
- Under `inference_method="post_selection_ols"`, `coef_` remains penalized while `_params` contains the active-set OLS/WLS refit embedded in the full parameter layout.
- Under `inference_method="debiased"`, `_params[1:]` contains debiased slopes and `_params[0]` contains their matching original-coordinate debiased intercept; `_conf_int` is marginal per reported parameter.
- With simultaneous inference enabled, `_conf_int_simultaneous` stores joint intervals over the configured target family (`maxz_bootstrap`); when the debiased intercept is included it also participates in the max-|Z| calibration.
- Methods: `fit`, `predict`, `score`, `summary`
- Common diagnostics include `aic` and `bic` when available.

## FAQ

- Why can CPU and GPU iteration counts differ under the same `tol`? Different numerical backends and solver implementations can converge differently; compare under fixed `solver` and `stopping`.
- Should CPU users set `cpu_solver`? No. Use `solver`; `cpu_solver` is a deprecated compatibility argument from the previous CPU/GPU-split API.
- Should I choose `cpu_ols` or `gpu_ols` based on hardware? No. Both are deprecated aliases for `post_selection_ols`. Choose the statistical method with `inference_method` and the execution location with `device`.
- Does `post_selection_ols` change `coef_`? No. Prediction keeps the penalized coefficients; the active-set refit lives in inference/reporting fields such as `_params` and `_inference_result`.
- Why can `intercept_` differ from `_params[0]` under `debiased`? `intercept_` belongs to the penalized prediction fit, while `_params[0]` is the intercept paired with the debiased slope vector used by statistical reporting.
- When should I use `debiased`? Prefer it when you need coefficient-level inference in high-dimensional sparse settings, subject to the method's assumptions.
- Is `post_selection_ols` a valid selective-inference confidence procedure? No. Treat it as a post-selection diagnostic.
- Are ordinary `debiased` intervals simultaneous/joint confidence regions? No. Ordinary `_conf_int` values are marginal. Enable the dedicated simultaneous path when family-wise intervals are required.
- How do I include the intercept in simultaneous coverage? Set `simultaneous_include_intercept=True`; the debiased intercept then participates in the bootstrap max-|Z| calibration as well as the reported joint interval set.

## External Validation

- `dev/benchmarks/validate_post_selection_ols_gpu.py`
- `dev/benchmarks/benchmark_lasso_inference_gpu_vs_cpu.py` — canonical `post_selection_ols` CPU/CuPy end-to-end parity and complete fit+inference timing benchmark.
- `dev/benchmarks/benchmark_lasso_cpu_gpu_tol.py`
- `dev/comparisons/compare_lasso_kkt_stopping.py`
- `dev/tests/test_lasso_debiased_inference.py`
- `dev/tests/test_post_selection_ols_inference_api.py`
- `dev/tests/test_penalized_solver_api_cleanup.py`

The physical post-selection OLS validator requires both CuPy CUDA and Torch CUDA. Its presence is not itself physical-GPU evidence; exact-head GPU acceptance must be recorded separately when executed on physical CUDA hardware.

## References

- Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *Journal of the Royal Statistical Society: Series B*, 58(1), 267-288. [https://doi.org/10.1111/j.2517-6161.1996.tb02080.x](https://doi.org/10.1111/j.2517-6161.1996.tb02080.x)
- Buhlmann, P., & van de Geer, S. (2011). *Statistics for High-Dimensional Data*. Springer.
- Zhang, C.-H., & Zhang, S. S. (2014). Confidence intervals for low-dimensional parameters in high-dimensional linear models. *Journal of the Royal Statistical Society: Series B*, 76(1), 217-242. [https://doi.org/10.1111/rssb.12026](https://doi.org/10.1111/rssb.12026)
- Javanmard, A., & Montanari, A. (2014). Confidence intervals and hypothesis testing for high-dimensional regression. *Journal of Machine Learning Research*, 15, 2869-2909. [https://jmlr.org/papers/v15/javanmard14a.html](https://jmlr.org/papers/v15/javanmard14a.html)
