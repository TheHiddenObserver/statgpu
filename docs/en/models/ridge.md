# Ridge

> Language: English  
> Last updated: 2026-04-17  
> This page: Model documentation  
> Switch: [Chinese](../../models/ridge.md)

Language switch: [Chinese](../../models/ridge.md)

## Overview

`Ridge` provides L2-regularized linear regression with the same inference surface as `LinearRegression` (including robust covariance options). It is used when multicollinearity or shrinkage is required while keeping interpretable coefficient inference in aligned settings.

## Path

`statgpu.linear_model.Ridge`

## Objective Function

Estimate
\[
\min_{\beta} \|y - X\beta\|_2^2 + \alpha\|\beta\|_2^2
\]
with optional intercept handling.

## Estimating Equation

The ridge first-order condition is
\[
(X^\top X + \alpha I)\hat\beta = X^\top y
\]
solved by stable linear algebra routines on CPU/GPU backends.

## Covariance/Inference

- `cov_type="nonrobust"`: classical ridge covariance.
- `cov_type="hc0"|"hc1"|"hc2"|"hc3"`: sandwich-style robust covariance variants.
- `cov_type="hac"`: Newey-West (Bartlett) covariance with optional `hac_maxlags`.
- `compute_inference=True` returns `_bse`, `_tvalues`, `_pvalues`, `_conf_int`.

## Parameters

| Parameter | Default | Description |
|---|---:|---|
| `alpha` | `1.0` | L2 regularization strength |
| `fit_intercept` | `True` | Whether to fit an intercept |
| `device` | `"auto"` | `cpu` / `cuda` / `auto` |
| `n_jobs` | `None` | Number of parallel jobs |
| `compute_inference` | `True` | Whether to compute inference stats (SE/t/p/CI) |
| `cov_type` | `"nonrobust"` | `nonrobust` / `hc0` / `hc1` / `hc2` / `hc3` / `hac` |
| `hac_maxlags` | `None` | Max lag for `cov_type="hac"`; default follows Newey-West style heuristic |
| `gpu_memory_cleanup` | `False` | Best-effort CuPy pool cleanup after each fit |

## CPU+GPU Examples

```python
from statgpu.linear_model import Ridge

# CPU
m_cpu = Ridge(alpha=1.0, device="cpu", cov_type="hc3", compute_inference=True)
m_cpu.fit(X, y)

# GPU
m_gpu = Ridge(alpha=1.0, device="cuda", cov_type="hc3", compute_inference=True, gpu_memory_cleanup=True)
m_gpu.fit(X, y)
```

## strict/approx difference

No separate public approx mode is exposed. The default inference path is the validated release path; backend differences are expected to be limited to floating-point effects.

## Outputs

- Coefficients: `intercept_`, `coef_`
- Inference: `_bse`, `_tvalues`, `_pvalues`, `_conf_int`
- Diagnostics: `r_squared`, `adj_r_squared`, `f_statistic`, `aic`, `bic`
- Methods: `fit`, `predict`, `score`, `summary`

## FAQ

- How should `alpha` be chosen? Start with log-grid cross-validation (for example, `1e-4` to `1e2`) and then fix a task-specific value.
- When should I set `hac_maxlags`? When using `cov_type="hac"` with time dependence; otherwise leave default.

## External Validation

- Inference/covariance behavior follows the same robust covariance implementation surface used by `LinearRegression`.
- Related consistency checks are maintained in `dev/tests/test_external_consistency.py`.

## References

- Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression: Biased estimation for nonorthogonal problems. *Technometrics*, 12(1), 55-67. [https://doi.org/10.1080/00401706.1970.10488634](https://doi.org/10.1080/00401706.1970.10488634)
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.
