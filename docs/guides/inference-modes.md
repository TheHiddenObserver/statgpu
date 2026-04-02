# 推断配置（Lasso）

`Lasso` 的 `inference_method`：

- `cpu_ols_inference`（默认）
- `gpu_ols_inference`
- `bootstrap`

兼容旧名：
- `naive_ols` -> `cpu_ols_inference`
- `gpu_naive_ols` -> `gpu_ols_inference`

## 推荐用法

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

## 选型建议

- `cpu_ols_inference`：兼容性优先
- `gpu_ols_inference`：减少大块 CPU 回传，推断速度优先
- `bootstrap`：更稳健，但计算开销更大

## 基准脚本

参见 `examples/benchmark_lasso_inference_gpu_vs_cpu.py`。
