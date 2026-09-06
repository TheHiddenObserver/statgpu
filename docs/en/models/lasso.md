# Lasso

> Language: English  
> Last updated: 2026-09-06  
> This page: Model documentation  
> Switch: [Chinese](../../cn/models/lasso.md)

Language switch: [Chinese](../../cn/models/lasso.md)

## Overview

`Lasso` provides L1-regularized linear regression with CPU/GPU execution and multiple inference backends. Direct fitting uses one backend-neutral `solver` interface; device selection and solver selection are separate choices.

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

- `inference_method="cpu_ols_inference"`: CPU-side OLS-style post-selection inference surface.
- `inference_method="gpu_ols_inference"`: GPU-side inference path to reduce host/device transfer overhead.
- `inference_method="debiased"`: de-biased (de-sparsified) Lasso inference with z-statistic semantics.
- `inference_method="bootstrap"`: residual bootstrap; typically more robust and slower.
- `compute_inference=True` enables `_bse`, `_tvalues`, `_pvalues`, `_conf_int`.
- Legacy aliases are accepted: `naive_ols -> cpu_ols_inference`, `gpu_naive_ols -> gpu_ols_inference`.

Validity notes:
- `cpu_ols_inference` / `gpu_ols_inference` intervals are heuristic post-selection intervals and should not be interpreted as valid selective-inference confidence intervals.
- The current `debiased` implementation returns per-coefficient marginal confidence intervals only; simultaneous/joint coverage is not guaranteed unless simultaneous inference is explicitly enabled.

## Parameters

This table is the complete public constructor inventory for `statgpu.linear_model.Lasso`.

| Parameter | Default | Description |
|---|---:|---|
| `alpha` | `1.0` | L1 regularization strength. |
| `fit_intercept` | `True` | Whether to fit an intercept. |
| `max_iter` | `1000` | Maximum optimization iterations. |
| `tol` | `1e-4` | Convergence tolerance. |
| `stopping` | `"coef_delta"` | Stopping rule: `coef_delta` / `kkt`. |
| `inference_method` | `"debiased"` | `cpu_ols_inference` / `gpu_ols_inference` / `debiased` / `bootstrap`. |
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

# GPU FISTA: the same solver interface is used on GPU.
m_gpu = Lasso(
    alpha=0.1,
    device="cuda",
    solver="fista",
    stopping="kkt",
    inference_method="gpu_ols_inference",
    gpu_memory_cleanup=True,
)
m_gpu.fit(X, y)
```

Simultaneous inference example (supports `device="cpu"` and `device="cuda"`, with device-consistent computation):

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

`debiased` is the strict mainline inference path for high-dimensional statistical inference. `cpu_ols_inference` and `gpu_ols_inference` are lighter approximate paths for engineering throughput, while `bootstrap` is usually more robust but materially slower.

## Outputs

- Coefficients: `intercept_`, `coef_`, `n_iter_`
- Inference (if enabled): `_bse`, `_tvalues` / `_zvalues`, `_pvalues`, `_conf_int`
- Under `inference_method="debiased"`, summary/statistical reporting uses z-style semantics (`z`, `P>|z|`), and `_conf_int` is marginal per coefficient.
- With simultaneous inference enabled, `_conf_int_simultaneous` stores joint intervals over the configured target set (`maxz_bootstrap`).
- Methods: `fit`, `predict`, `score`, `summary`
- Common diagnostics include `aic` and `bic` when available.

## FAQ

- Why can CPU and GPU iteration counts differ under the same `tol`? Different numerical backends and solver implementations can converge differently; compare under fixed `solver` and `stopping`.
- Should CPU users set `cpu_solver`? No. Use `solver`; `cpu_solver` is a deprecated compatibility argument from the previous CPU/GPU-split API.
- When should I use `gpu_ols_inference`? Prefer it for larger GPU-trained workloads to reduce transfer overhead.
- When should I use `debiased`? Prefer it when you need inferential quantities (SE/p-values/intervals) in high-dimensional sparse settings.
- Are `cpu_ols_inference` / `gpu_ols_inference` intervals statistically valid confidence intervals? Not in a strict selective-inference sense; treat them as engineering diagnostics.
- Are `debiased` intervals simultaneous/joint confidence regions? The ordinary `_conf_int` values are marginal. Enable the dedicated simultaneous path when family-wise intervals are required.
- How do I enable simultaneous intervals? Set `enable_simultaneous_inference=True` with `inference_method="debiased"` and `simultaneous_method="maxz_bootstrap"`.

## External Validation

- `dev/benchmarks/benchmark_lasso_inference_gpu_vs_cpu.py`
- `dev/benchmarks/benchmark_lasso_cpu_gpu_tol.py`
- `dev/comparisons/compare_lasso_kkt_stopping.py`
- `dev/tests/test_lasso_debiased_inference.py`
- `dev/tests/test_penalized_solver_api_cleanup.py`

## References

- Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *Journal of the Royal Statistical Society: Series B*, 58(1), 267-288. [https://doi.org/10.1111/j.2517-6161.1996.tb02080.x](https://doi.org/10.1111/j.2517-6161.1996.tb02080.x)
- Buhlmann, P., & van de Geer, S. (2011). *Statistics for High-Dimensional Data*. Springer.
- Zhang, C.-H., & Zhang, S. S. (2014). Confidence intervals for low-dimensional parameters in high-dimensional linear models. *Journal of the Royal Statistical Society: Series B*, 76(1), 217-242. [https://doi.org/10.1111/rssb.12026](https://doi.org/10.1111/rssb.12026)
- Javanmard, A., & Montanari, A. (2014). Confidence intervals and hypothesis testing for high-dimensional regression. *Journal of Machine Learning Research*, 15, 2869-2909. [https://jmlr.org/papers/v15/javanmard14a.html](https://jmlr.org/papers/v15/javanmard14a.html)
