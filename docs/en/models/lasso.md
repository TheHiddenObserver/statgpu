# Lasso

> Language: English  
> Last updated: 2026-04-02  
> This page: Model documentation  
> Switch: [中文](../../models/lasso.md)

Language switch: [中文](../../models/lasso.md)

Path: `statgpu.linear_model.Lasso`

## Parameter Table

| Parameter | Default | Description |
|---|---:|---|
| `alpha` | `1.0` | L1 regularization strength |
| `solver` | `"fista"` | GPU solver: `fista` / `admm` |
| `cpu_solver` | `"coordinate_descent"` | CPU solver: `coordinate_descent` / `fista` |
| `stopping` | `"coef_delta"` | Stopping rule: `coef_delta` / `kkt` |
| `inference_method` | `"cpu_ols_inference"` | `cpu_ols_inference` / `gpu_ols_inference` / `bootstrap` |
| `compute_inference` | `True` | Whether to compute inference stats |
| `gpu_memory_cleanup` | `False` | Best-effort CuPy pool cleanup after each fit |

## Example

```python
from statgpu.linear_model import Lasso

m = Lasso(
    alpha=0.1,
    device="cuda",
    solver="fista",
    stopping="kkt",
    inference_method="gpu_ols_inference",
)
m.fit(X, y)
```

## Outputs

- Coefficients: `intercept_`, `coef_`, `n_iter_`
- Inference: `_bse`, `_tvalues`, `_pvalues`, `_conf_int` (if enabled)

## Inference Modes

- `cpu_ols_inference`: CPU-side OLS-style inference
- `gpu_ols_inference`: GPU-side inference to reduce host/device transfers
- `bootstrap`: residual bootstrap, usually more robust but slower

## Returns and Properties

- `fit(X, y)` returns `self`
- `predict(X)` returns predictions
- `score(X, y)` returns `R^2`
