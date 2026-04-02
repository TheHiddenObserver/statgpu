# Ridge

> Language: English  
> Last updated: 2026-04-02  
> This page: Model documentation  
> Switch: [中文](../../models/ridge.md)

Language switch: [中文](../../models/ridge.md)

Path: `statgpu.linear_model.Ridge`

## Parameter Table

| Parameter | Default | Description |
|---|---:|---|
| `alpha` | `1.0` | L2 regularization strength |
| `fit_intercept` | `True` | Whether to fit an intercept |
| `device` | `"auto"` | `cpu` / `cuda` / `auto` |
| `gpu_memory_cleanup` | `False` | Best-effort CuPy pool cleanup after each fit |

## Example

```python
from statgpu.linear_model import Ridge

m = Ridge(alpha=1.0, device="cuda", gpu_memory_cleanup=True)
m.fit(X, y)
```

## Outputs

- Coefficients: `intercept_`, `coef_`
- Prediction: `predict(X)`
- Score: `score(X, y)`

## Returns and Properties

- `fit(X, y)` returns `self`
- `predict(X)` returns predictions
- `score(X, y)` returns `R^2`
