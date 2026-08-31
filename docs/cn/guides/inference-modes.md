# 推断配置

> 语言: 中文  
> 最后更新: 2026-08-28  
> 页面定位: 指南文档  
> 切换: [English](../../en/guides/inference-modes.md)

语言切换：[English](../../en/guides/inference-modes.md)

## Gaussian 线性模型推断

对于 squared-error L2/Ridge 使用的共享 Gaussian 推断路径，数值协方差与参考分布推断现在在实际完成模型拟合的 backend 上执行，即 NumPy、CuPy 或 Torch。这里的数值阶段包括 bread/协方差计算、标准误、检验统计量、p 值以及置信区间临界值。

既有公开 reporting 契约保持不变：所有数值推断完成后，推断结果以及 estimator 的 reporting 属性（`_params`、`_bse`、`_tvalues`、`_pvalues`、`_conf_int`）才进行一次最终 NumPy snapshot。这个转换是 reporting boundary，而不是 CPU inference fallback。共享路径会在 `_inference_result.metadata` 中记录 `numerical_backend`、`numerical_device`、`reporting_backend="numpy"` 和 `reporting_boundary="post_numerical_inference"`。

显式 `device="cuda"` 或 `device="torch"` 时，Gaussian L2 推断不会静默降级到 NumPy。若缺失或出现非法的实际执行 backend provenance，则直接 fail closed。`device="auto"` 继续沿用 estimator 原有的 backend 选择策略。

上述说明只针对已经迁移的共享 Gaussian inference path，**不表示 statgpu 中所有 inference 实现都已经迁移到同一生命周期**。

Gaussian 路径支持：

- `nonrobust`：经典协方差，使用 Student-t 参考分布；
- `hc0`、`hc1`、`hc2`、`hc3`：异方差稳健 sandwich 协方差，使用正态参考分布；
- `hac`：Bartlett kernel HAC 协方差，使用正态参考分布。

backend-native reference helper 同时保留残差自由度为 1 和 2 时的稳定 Student-t 恒等式，避免极端但仍可表示的尾概率因减法消去或不必要的 `t**2` overflow 被错误压成 0。

## Lasso 的 `inference_method`

- `cpu_ols_inference`（默认）
- `gpu_ols_inference`
- `bootstrap`

兼容旧名：

- `naive_ols` -> `cpu_ols_inference`
- `gpu_naive_ols` -> `gpu_ols_inference`

### 推荐用法

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

### 选型建议

- `cpu_ols_inference`：兼容性优先；
- `gpu_ols_inference`：减少大块 CPU 回传，推断速度优先；
- `bootstrap`：更稳健，但计算开销更大。

### 基准脚本

参见 `dev/benchmarks/benchmark_lasso_inference_gpu_vs_cpu.py`。

## 相关模型的稳健协方差

- `LinearRegression(cov_type="nonrobust" | "hc0" | "hc1" | "hc2" | "hc3" | "hac")`
- `Ridge(cov_type="nonrobust" | "hc0" | "hc1" | "hc2" | "hc3" | "hac")`
- `LogisticRegression(cov_type="nonrobust" | "hc0" | "hc1" | "hc2" | "hc3" | "hac")`
