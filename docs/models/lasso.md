# Lasso

路径：`statgpu.linear_model.Lasso`

## 主要参数
- `alpha`
- `solver` (`fista` / `admm`)
- `cpu_solver` (`coordinate_descent` / `fista`)
- `stopping` (`coef_delta` / `kkt`)
- `inference_method` (`cpu_ols_inference` / `gpu_ols_inference` / `bootstrap`)
- `compute_inference`
- `gpu_memory_cleanup`

## 示例

```python
from statgpu.linear_model import Lasso

m = Lasso(
    alpha=0.1,
    device="cuda",
    solver="fista",
    stopping="kkt",
    inference_method="gpu_ols_inference",
    gpu_memory_cleanup=True,
)
m.fit(X, y)
```
