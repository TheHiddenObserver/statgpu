# LinearRegression

> Language: English  
> Last updated: 2026-04-02  
> This page: Model documentation  
> Switch: [中文](../../models/linear-regression.md)

Language switch: [中文](../../models/linear-regression.md)

Path: `statgpu.linear_model.LinearRegression`

## Parameter Table

| Parameter | Default | Description |
|---|---:|---|
| `fit_intercept` | `True` | Whether to fit an intercept |
| `device` | `"auto"` | `cpu` / `cuda` / `auto` |
| `compute_inference` | `True` | Whether to compute inference stats (SE/t/p/CI) |
| `cov_type` | `"nonrobust"` | `nonrobust` / `hc0` / `hc1` |
| `gpu_memory_cleanup` | `False` | Best-effort CuPy pool cleanup after each fit |

## Example

```python
from statgpu.linear_model import LinearRegression

m = LinearRegression(device="cuda", cov_type="hc1", compute_inference=True)
m.fit(X, y)
m.summary()
```

## Outputs

- Coefficients: `intercept_`, `coef_`
- Inference: `_bse`, `_tvalues`, `_pvalues`, `_conf_int`
- Diagnostics: `aic`, `bic`, `r_squared`, `adj_r_squared`

## Returns and Properties

- `fit(X, y)` returns `self`
- `predict(X)` returns predictions
- `score(X, y)` returns `R^2`

## External Consistency

- `dev/tests/test_external_consistency.py`
  - `test_linear_estimation_and_inference_match_statsmodels`
  - `test_linear_robust_covariance_matches_statsmodels`
  - `test_linear_robust_covariance_gpu_matches_statsmodels`
