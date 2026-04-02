# LinearRegression

路径：`statgpu.linear_model.LinearRegression`

## 主要参数
- `fit_intercept`
- `device`
- `compute_inference`
- `gpu_memory_cleanup`
- `cov_type` (`nonrobust` / `hc0` / `hc1`)

## 示例

```python
from statgpu.linear_model import LinearRegression

m = LinearRegression(device="cuda", compute_inference=True, gpu_memory_cleanup=False)
m.fit(X, y)
m.summary()
```

## 稳健协方差

```python
from statgpu.linear_model import LinearRegression

# White robust SE
m = LinearRegression(device="cpu", cov_type="hc1")
m.fit(X, y)
```
