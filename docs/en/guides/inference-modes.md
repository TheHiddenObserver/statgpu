# Inference Modes (Lasso)

> Language: English  
> Last updated: 2026-04-02  
> This page: Guide  
> Switch: [中文](../../guides/inference-modes.md)

Language switch: [中文](../../guides/inference-modes.md)

`Lasso.inference_method` options:
- `cpu_ols_inference` (default)
- `gpu_ols_inference`
- `bootstrap`

Backward-compatible aliases:
- `naive_ols` -> `cpu_ols_inference`
- `gpu_naive_ols` -> `gpu_ols_inference`

Recommended usage:

```python
from statgpu.linear_model import Lasso

model = Lasso(
    alpha=0.1,
    device="cuda",
    solver="fista",
    stopping="kkt",
    compute_inference=True,
    inference_method="gpu_ols_inference",
)
model.fit(X, y)
```

Related robust covariance support:
- `LinearRegression(cov_type="nonrobust" | "hc0" | "hc1")`
- `LogisticRegression(cov_type="nonrobust" | "hc0" | "hc1")`
