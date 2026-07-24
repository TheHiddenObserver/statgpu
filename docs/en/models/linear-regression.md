# LinearRegression

> Language: English  
> Last updated: 2026-04-17  
> This page: Model documentation  
> Switch: [Chinese](../../cn/models/linear-regression.md)

Language switch: [Chinese](../../cn/models/linear-regression.md)

## Overview

`LinearRegression` implements OLS with unified CPU/GPU fitting and inference. It is the baseline linear model used across consistency tests and robust covariance comparisons. Multi-output estimation is supported, but textual `summary()` is single-output only.

## Path

`statgpu.linear_model.LinearRegression`

## Objective Function

Estimate
$$
\min_{\beta} \|y - X\beta\|_2^2
$$
with optional intercept handling, then compute diagnostics and inference from residual-based covariance estimators.

## Estimating Equation

The estimator solves the normal equations:
$$
X^\top(y - X\hat\beta)=0
$$
equivalently \(\hat\beta=(X^\top X)^{-1}X^\top y\) when the inverse exists (or numerically stable equivalent linear algebra in implementation).

## Covariance/Inference

- `cov_type="nonrobust"`: classical OLS covariance.
- `cov_type="hc0"|"hc1"|"hc2"|"hc3"`: heteroskedasticity-robust sandwich variants.
- `cov_type="hac"`: Newey-West (Bartlett) covariance; `hac_maxlags` controls lag truncation.
- `compute_inference=True` returns `_bse`, `_tvalues`, `_pvalues`, `_conf_int`.
- Inference is available on CPU and CUDA paths under aligned settings.

## Parameters

| Parameter | Default | Description |
|---|---:|---|
| `fit_intercept` | `True` | Whether to fit an intercept |
| `device` | `"auto"` | `cpu` / `cuda` / `auto` |
| `compute_inference` | `True` | Whether to compute inference stats (SE/t/p/CI) |
| `cov_type` | `"nonrobust"` | `nonrobust` / `hc0` / `hc1` / `hc2` / `hc3` / `hac` |
| `hac_maxlags` | `None` | Max lag for `cov_type="hac"`; default follows Newey-West style heuristic |
| `gpu_memory_cleanup` | `False` | Best-effort CuPy pool cleanup after each fit |

## CPU+GPU Examples

```python
from statgpu.linear_model import LinearRegression

# CPU with HAC covariance
m_cpu = LinearRegression(device="cpu", cov_type="hac", hac_maxlags=4, compute_inference=True)
m_cpu.fit(X, y)
print(m_cpu._bse)

# GPU with HC1 covariance
m_gpu = LinearRegression(device="cuda", cov_type="hc1", compute_inference=True)
m_gpu.fit(X, y)
print(m_gpu._pvalues)
```

## strict/approx difference

There is no separate public approx inference mode for this model. The default path is the release path used in external consistency tests; CPU/GPU differences are expected to be small floating-point effects.

## Outputs

- Coefficients: `intercept_`, `coef_`
- Inference: `_bse`, `_tvalues`, `_pvalues`, `_conf_int`
- Diagnostics: `r_squared`, `adj_r_squared`, `f_statistic`, `aic`, `bic`
- Methods: `fit`, `predict`, `score`, `summary`

Multi-output `y` support:
- `coef_`: `(n_targets, n_features)`, `intercept_`: `(n_targets,)`
- `_bse/_tvalues/_pvalues`: `(n_params, n_targets)`
- `_conf_int`: `(n_params, n_targets, 2)`
- `summary()` raises for multi-output fits.

## FAQ

- Why do CPU and GPU p-values differ slightly? Different numeric kernels and floating-point paths can produce tiny differences.
- When should I use `hac` instead of `hc*`? Use `hac` for serial correlation; use `hc1`/`hc3` for heteroskedasticity without explicit time dependence.

## External Validation

- `dev/tests/test_external_consistency.py`
  - `test_linear_estimation_and_inference_match_statsmodels`
  - `test_linear_robust_covariance_matches_statsmodels`
  - `test_linear_robust_covariance_gpu_matches_statsmodels`
  - `test_linear_hac_covariance_matches_statsmodels`

## References

- Greene, W. H. (2018). *Econometric Analysis* (8th ed.). Pearson.
- White, H. (1980). A heteroskedasticity-consistent covariance matrix estimator and a direct test for heteroskedasticity. *Econometrica*, 48(4), 817-838. [https://doi.org/10.2307/1912934](https://doi.org/10.2307/1912934)
- MacKinnon, J. G., & White, H. (1985). Some heteroskedasticity-consistent covariance matrix estimators with improved finite sample properties. *Journal of Econometrics*, 29(3), 305-325. [https://doi.org/10.1016/0304-4076(85)90158-7](https://doi.org/10.1016/0304-4076(85)90158-7)
- Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703-708. [https://doi.org/10.2307/1913610](https://doi.org/10.2307/1913610)
