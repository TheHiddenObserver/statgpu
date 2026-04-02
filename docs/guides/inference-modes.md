# 推断配置（Lasso）

> 语言: 中文  
> 最后更新: 2026-04-02  
> 页面定位: 指南文档  
> 切换: [English](../en/guides/inference-modes.md)

语言切换：[English](../en/guides/inference-modes.md)

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

## 相关模型的稳健协方差

除了 `Lasso` 的推断模式外，以下模型也支持协方差配置：

- `LinearRegression(cov_type="nonrobust" | "hc0" | "hc1")`
- `LogisticRegression(cov_type="nonrobust" | "hc0" | "hc1")`

可参考：
- `docs/models/linear-regression.md`
- `docs/models/logistic-regression.md`
