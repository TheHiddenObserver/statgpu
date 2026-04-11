# Ridge

> Language: English  
> Last updated: 2026-04-10  
> This page: Model documentation  
> Switch: [Chinese](../../models/ridge.md)

Language switch: [Chinese](../../models/ridge.md)

Path: `statgpu.linear_model.Ridge`

## Parameter Table

| Parameter | Default | Description |
|---|---:|---|
| `alpha` | `1.0` | L2 regularization strength |
| `fit_intercept` | `True` | Whether to fit an intercept |
| `device` | `"auto"` | `cpu` / `cuda` / `auto` |
| `n_jobs` | `None` | Number of parallel jobs |
| `compute_inference` | `True` | Whether to compute inference stats (SE/t/p/CI) |
| `cov_type` | `"nonrobust"` | `nonrobust` / `hc0` / `hc1` / `hc2` / `hc3` / `hac` |
| `hac_maxlags` | `None` | Max lag used when `cov_type="hac"`; if omitted, Newey-West style default is used |
| `gpu_memory_cleanup` | `False` | Best-effort CuPy pool cleanup after each fit |

## Example

```python
from statgpu.linear_model import Ridge

m = Ridge(alpha=1.0, device="cuda", cov_type="hc3", compute_inference=True, gpu_memory_cleanup=True)
m.fit(X, y)
m.summary()
```

## Robust Covariance (HC0/HC1/HC2/HC3/HAC)

- `cov_type="nonrobust"`: classical ridge covariance
- `cov_type="hc0"`: White/sandwich robust covariance
- `cov_type="hc1"`: HC0 with DOF correction `n/(n-k)`
- `cov_type="hc2"`: leverage-adjusted robust covariance
- `cov_type="hc3"`: more conservative jackknife-style robust covariance
- `cov_type="hac"`: Newey-West (Bartlett kernel) covariance with optional `hac_maxlags`

## Outputs

- Coefficients: `intercept_`, `coef_`
- Inference: `_bse`, `_tvalues`, `_pvalues`, `_conf_int`
- Diagnostics: `r_squared`, `adj_r_squared`, `f_statistic`, `aic`, `bic`
- Prediction: `predict(X)`
- Score: `score(X, y)`
- Summary: `summary()`

## Returns and Properties

- `fit(X, y)` returns `self`
- `predict(X)` returns predictions
- `score(X, y)` returns `R^2`
