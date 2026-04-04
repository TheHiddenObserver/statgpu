# Lasso

> 语言: 中文  
> 最后更新: 2026-04-02  
> 页面定位: 模型文档  
> 切换: [English](../en/models/lasso.md)

语言切换：[English](../en/models/lasso.md)

路径：`statgpu.linear_model.Lasso`

## 参数表

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `alpha` | `1.0` | L1 正则强度 |
| `max_iter` | `1000` | 最大迭代数 |
| `tol` | `1e-4` | 收敛阈值 |
| `solver` | `"fista"` | GPU 求解器：`fista` / `admm` |
| `cpu_solver` | `"coordinate_descent"` | CPU 求解器：`coordinate_descent` / `fista` |
| `stopping` | `"coef_delta"` | 停止准则：`coef_delta` / `kkt` |
| `inference_method` | `"cpu_ols_inference"` | `cpu_ols_inference` / `gpu_ols_inference` / `bootstrap` |
| `compute_inference` | `True` | 是否计算推断统计 |
| `gpu_memory_cleanup` | `False` | 每次 `fit` 后尝试释放 CuPy memory pool |

## 主要参数
- `alpha`: L1 正则强度
- `solver`: GPU 求解器（`fista` / `admm`）
- `cpu_solver`: CPU 求解器（`coordinate_descent` / `fista`）
- `stopping`: `coef_delta` / `kkt`
- `inference_method`: `cpu_ols_inference` / `gpu_ols_inference` / `bootstrap`
- `compute_inference`
- `gpu_memory_cleanup`

## 快速示例

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

## 推断模式说明

- `cpu_ols_inference`：CPU 侧 OLS 风格推断
- `gpu_ols_inference`：GPU 侧推断，减少大块 host/device 传输
- `bootstrap`：残差重采样，通常更稳健但更慢

兼容旧名：
- `naive_ols` -> `cpu_ols_inference`
- `gpu_naive_ols` -> `gpu_ols_inference`

## 输出

- 系数：`intercept_`, `coef_`, `n_iter_`
- 推断：`_bse`, `_tvalues`, `_pvalues`, `_conf_int`（在 `compute_inference=True` 时）
- 汇总：`summary()`

## 返回与属性

- `fit(X, y)`：返回 `self`
- `predict(X)`：返回预测值向量
- `score(X, y)`：返回 `R^2`
- 常用属性：`coef_`, `intercept_`, `n_iter_`, `aic`, `bic`

## 常见问题

- **Q: 为什么同样 `tol` CPU/GPU 时间和迭代不完全一致？**  
  A: 不同求解器和数值路径会带来差异。建议在对比实验中固定 `solver/stopping` 并使用 repeats。
- **Q: 何时用 `gpu_ols_inference`？**  
  A: 大样本、GPU 训练时优先，用于减少 host/device 传输开销。

## 相关基准脚本

- `dev/benchmarks/benchmark_lasso_inference_gpu_vs_cpu.py`
- `dev/benchmarks/benchmark_lasso_cpu_gpu_tol.py`
- `dev/comparisons/compare_lasso_kkt_stopping.py`
