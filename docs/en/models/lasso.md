# Lasso

> Language: English  
> Last updated: 2026-09-06  
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

### Device/backend rule

`inference_method` describes **what statistical procedure is computed**; it does not choose hardware.

- explicit `device="cpu"` -> NumPy CPU;
- explicit `device="cuda"` -> CuPy CUDA only, failing closed if unavailable;
- explicit `device="torch"` -> Torch CUDA only, failing closed if unavailable;
- only genuine estimator/global `device="auto"` may preserve an already backend-native CuPy or Torch-CUDA input during automatic routing.

Backend reuse is method-specific. `post_selection_ols` reuses the successful fit's
`_selected_backend_name` / `_selected_backend_device`, and maintained CuPy/Torch
`debiased` routes keep their numerical inference on the executed GPU backend.
Residual `bootstrap` currently uses CPU-native residual refits, so an explicit GPU
`device` controls the penalized fit but does not make bootstrap GPU-native.

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
| `simultaneous_alpha` | `0.05` | Simultaneous family-wise error level. |
| `simultaneous_n_bootstrap` | `1000` | Multiplier-bootstrap draws for max-|Z| calibration. |
| `simultaneous_random_state` | `None` | RNG seed for simultaneous bootstrap. |
| `simultaneous_include_intercept` | `False` | Whether the simultaneous target set includes the intercept. |
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
)
m_sim.fit(X, y)
ci_marginal = m_sim._conf_int
ci_simul = m_sim._conf_int_simultaneous
```

## strict/approx difference

`debiased` is the main high-dimensional coefficient-inference path. `post_selection_ols` is a lighter active-set OLS/WLS diagnostic, while `bootstrap` is a more expensive resampling path. Their statistical claims are different and should not be treated as interchangeable.

## Outputs

- Penalized fit: `intercept_`, `coef_`, `n_iter_`
- Inference (if enabled): `_params`, `_bse`, `_tvalues` / `_zvalues`, `_pvalues`, `_conf_int`, `_inference_result`
- Under `inference_method="post_selection_ols"`, `coef_` remains penalized while `_params` contains the active-set OLS/WLS refit embedded in the full parameter layout.
- Under `inference_method="debiased"`, summary/statistical reporting uses z-style semantics (`z`, `P>|z|`), and `_conf_int` is marginal per coefficient.
- With simultaneous inference enabled, `_conf_int_simultaneous` stores joint intervals over the configured target set (`maxz_bootstrap`).
- Methods: `fit`, `predict`, `score`, `summary`
- Common diagnostics include `aic` and `bic` when available.

## FAQ

- Why can CPU and GPU iteration counts differ under the same `tol`? Different numerical backends and solver implementations can converge differently; compare under fixed `solver` and `stopping`.
- Should CPU users set `cpu_solver`? No. Use `solver`; `cpu_solver` is a deprecated compatibility argument from the previous CPU/GPU-split API.
- Should I choose `cpu_ols` or `gpu_ols` based on hardware? No. Both are deprecated aliases for `post_selection_ols`. Choose the statistical method with `inference_method` and the execution location with `device`.
- Does `post_selection_ols` change `coef_`? No. Prediction keeps the penalized coefficients; the active-set refit lives in inference/reporting fields such as `_params` and `_inference_result`.
- When should I use `debiased`? Prefer it when you need coefficient-level inference in high-dimensional sparse settings, subject to the method's assumptions.
- Is `post_selection_ols` a valid selective-inference confidence procedure? No. Treat it as a post-selection diagnostic.
- Are ordinary `debiased` intervals simultaneous/joint confidence regions? No. Ordinary `_conf_int` values are marginal. Enable the dedicated simultaneous path when family-wise intervals are required.
- How do I enable simultaneous intervals? Set `enable_simultaneous_inference=True` with `inference_method="debiased"` and `simultaneous_method="maxz_bootstrap"`.

## External Validation

- `dev/benchmarks/validate_post_selection_ols_gpu.py`
- `dev/benchmarks/benchmark_lasso_inference_gpu_vs_cpu.py` (historical hardware-bearing API benchmark)
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
