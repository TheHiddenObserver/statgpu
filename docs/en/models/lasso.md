# Lasso

> Language: English  
> Last updated: 2026-04-17  
> This page: Model documentation  
> Switch: [Chinese](../../cn/models/lasso.md)

Language switch: [Chinese](../../cn/models/lasso.md)

## Overview

`Lasso` provides L1-regularized linear regression with selectable CPU/GPU solvers and inference backends. It targets sparse feature selection while preserving a familiar estimator interface.

## Path

`statgpu.linear_model.Lasso`

## Objective Function

Estimate
$$
\min_{\beta}\frac{1}{2n}\|y - X\beta\|_2^2 + \alpha\|\beta\|_1
$$
with iterative optimization (`fista`, `admm`, or coordinate descent depending on backend configuration).

## Estimating Equation

The model is solved by iterative optimization rather than a closed-form normal equation. Stopping can be based on coefficient change (`coef_delta`) or KKT consistency (`kkt`), depending on `stopping`.

## Covariance/Inference

- `inference_method="cpu_ols_inference"`: CPU-side OLS-style post-selection inference surface.
- `inference_method="gpu_ols_inference"`: GPU-side inference path to reduce host/device transfer overhead.
- `inference_method="debiased"`: de-biased (de-sparsified) Lasso inference with z-statistic semantics.
- `inference_method="bootstrap"`: residual bootstrap; typically more robust and slower.
- `compute_inference=True` enables `_bse`, `_tvalues`, `_pvalues`, `_conf_int`.
- Legacy aliases are accepted: `naive_ols -> cpu_ols_inference`, `gpu_naive_ols -> gpu_ols_inference`.

Validity notes:
- `cpu_ols_inference` / `gpu_ols_inference` intervals are heuristic post-selection intervals and should not be interpreted as valid selective-inference confidence intervals.
- The current `debiased` implementation returns per-coefficient marginal confidence intervals only; simultaneous/joint coverage is not guaranteed.

## Parameters

| Parameter | Default | Description |
|---|---:|---|
| `alpha` | `1.0` | L1 regularization strength |
| `solver` | `"fista"` | GPU solver: `fista` / `admm` |
| `cpu_solver` | `"coordinate_descent"` | CPU solver: `coordinate_descent` / `fista` |
| `stopping` | `"coef_delta"` | Stopping rule: `coef_delta` / `kkt` |
| `inference_method` | `"cpu_ols_inference"` | `cpu_ols_inference` / `gpu_ols_inference` / `debiased` / `bootstrap` |
| `compute_inference` | `True` | Whether to compute inference stats |
| `enable_simultaneous_inference` | `False` | Enable simultaneous inference (debiased only) |
| `simultaneous_method` | `"maxz_bootstrap"` | Currently only `maxz_bootstrap` is supported |
| `simultaneous_alpha` | `0.05` | Simultaneous coverage level parameter |
| `simultaneous_n_bootstrap` | `1000` | Number of multiplier-bootstrap draws for max-|Z| calibration |
| `simultaneous_random_state` | `None` | RNG seed for simultaneous bootstrap |
| `simultaneous_include_intercept` | `False` | Whether the simultaneous target set includes intercept |
| `gpu_memory_cleanup` | `False` | Best-effort CuPy pool cleanup after each fit |

## CPU+GPU Examples

```python
from statgpu.linear_model import Lasso

# CPU
m_cpu = Lasso(alpha=0.1, device="cpu", cpu_solver="coordinate_descent", stopping="kkt")
m_cpu.fit(X, y)

# GPU
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
- Inference (if enabled): `_bse`, `_tvalues`, `_pvalues`, `_conf_int`
- Under `inference_method="debiased"`, summary/statistical reporting uses z-style semantics (`z`, `P>|z|`), and `_conf_int` is marginal per coefficient.
- With simultaneous inference enabled, `_conf_int_simultaneous` stores joint intervals over the configured target set (`maxz_bootstrap`).
- Methods: `fit`, `predict`, `score`, `summary`
- Common diagnostics include `aic` and `bic` when available.

## FAQ

- Why can CPU and GPU iteration counts differ under the same `tol`? Different solvers and numeric paths converge differently; compare under fixed `solver` and `stopping`.
- When should I use `gpu_ols_inference`? Prefer it for larger GPU-trained workloads to reduce transfer overhead.
- When should I use `debiased`? Prefer it when you need inferential quantities (SE/p-values/intervals) in high-dimensional sparse settings.
- Are `cpu_ols_inference` / `gpu_ols_inference` intervals statistically valid confidence intervals? Not in a strict selective-inference sense; treat them as engineering diagnostics.
- Are `debiased` intervals simultaneous/joint confidence regions? No. They are currently marginal per-coefficient intervals.
- How do I enable simultaneous intervals? Set `enable_simultaneous_inference=True` with `inference_method="debiased"` and `simultaneous_method="maxz_bootstrap"`.

## External Validation

- `dev/benchmarks/benchmark_lasso_inference_gpu_vs_cpu.py`
- `dev/benchmarks/benchmark_lasso_cpu_gpu_tol.py`
- `dev/comparisons/compare_lasso_kkt_stopping.py`
- `dev/tests/test_lasso_debiased_inference.py`

## References

- Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *Journal of the Royal Statistical Society: Series B*, 58(1), 267-288. [https://doi.org/10.1111/j.2517-6161.1996.tb02080.x](https://doi.org/10.1111/j.2517-6161.1996.tb02080.x)
- Buhlmann, P., & van de Geer, S. (2011). *Statistics for High-Dimensional Data*. Springer.
- Zhang, C.-H., & Zhang, S. S. (2014). Confidence intervals for low-dimensional parameters in high-dimensional linear models. *Journal of the Royal Statistical Society: Series B*, 76(1), 217-242. [https://doi.org/10.1111/rssb.12026](https://doi.org/10.1111/rssb.12026)
- Javanmard, A., & Montanari, A. (2014). Confidence intervals and hypothesis testing for high-dimensional regression. *Journal of Machine Learning Research*, 15, 2869-2909. [https://jmlr.org/papers/v15/javanmard14a.html](https://jmlr.org/papers/v15/javanmard14a.html)
