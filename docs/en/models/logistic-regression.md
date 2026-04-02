# LogisticRegression

> Language: English  
> Last updated: 2026-04-02  
> This page: Model documentation  
> Switch: [中文](../../models/logistic-regression.md)

Language switch: [中文](../../models/logistic-regression.md)

Path: `statgpu.linear_model.LogisticRegression`

## Parameter Table

| Parameter | Default | Description |
|---|---:|---|
| `fit_intercept` | `True` | Whether to fit an intercept |
| `C` | `1.0` | Inverse regularization strength |
| `max_iter` | `100` | Max IRLS iterations |
| `tol` | `1e-4` | Convergence tolerance |
| `device` | `"auto"` | `cpu` / `cuda` / `auto` |
| `compute_inference` | `True` | Whether to compute inference stats |
| `cov_type` | `"nonrobust"` | `nonrobust` / `hc0` / `hc1` |
| `gpu_memory_cleanup` | `False` | Best-effort CuPy pool cleanup after each fit |

## Example

```python
from statgpu.linear_model import LogisticRegression

m = LogisticRegression(device="cuda", cov_type="hc1", compute_inference=True)
m.fit(X, y_binary)
proba = m.predict_proba(X)
```

## Robust Covariance (HC0/HC1)

- `cov_type="nonrobust"`: classical information-matrix covariance
- `cov_type="hc0"`: White/sandwich robust covariance
- `cov_type="hc1"`: HC0 with DOF correction `n/(n-k)`

## Outputs

- Coefficients: `intercept_`, `coef_`, `n_iter_`
- Inference: `_bse`, `_zvalues`, `_pvalues`, `_conf_int`
- Metrics: `aic`, `bic`, `pseudo_rsquared`, `accuracy`, `precision`, `recall`, `f1`

## Returns and Properties

- `fit(X, y)` returns `self`
- `predict_proba(X)` returns class probabilities
- `predict(X)` returns labels

## External Consistency

- `tests/test_external_consistency.py`
  - `test_logistic_robust_covariance_matches_statsmodels`
  - `test_logistic_robust_covariance_gpu_matches_statsmodels`
