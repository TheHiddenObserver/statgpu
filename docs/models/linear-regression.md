# LinearRegression

> 语言: 中文  
> 最后更新: 2026-04-02  
> 页面定位: 模型文档  
> 切换: [English](../en/models/linear-regression.md)

语言切换：[English](../en/models/linear-regression.md)

路径：`statgpu.linear_model.LinearRegression`

## 参数表

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `fit_intercept` | `True` | 是否拟合截距 |
| `device` | `"auto"` | `cpu` / `cuda` / `auto` |
| `compute_inference` | `True` | 是否计算推断统计（SE/t/p/CI） |
| `cov_type` | `"nonrobust"` | `nonrobust` / `hc0` / `hc1` |
| `gpu_memory_cleanup` | `False` | 每次 `fit` 后尝试释放 CuPy memory pool |

## 主要参数
- `fit_intercept`: 是否拟合截距
- `device`: `cpu` / `cuda` / `auto`
- `compute_inference`: 是否计算推断统计（SE/t/p/CI）
- `cov_type`: `nonrobust` / `hc0` / `hc1`
- `gpu_memory_cleanup`: 每次 `fit` 后尝试释放 CuPy memory pool

## 快速示例

```python
from statgpu.linear_model import LinearRegression

m = LinearRegression(device="cuda", compute_inference=True, gpu_memory_cleanup=False)
m.fit(X, y)
m.summary()
```

## 稳健协方差（HC0/HC1）

- `cov_type="nonrobust"`：经典 OLS 协方差（t 分布口径）
- `cov_type="hc0"`：White 异方差稳健协方差
- `cov_type="hc1"`：HC0 + 自由度修正 `n/(n-k)`

CUDA 下 `hc0/hc1` 已实现纯 GPU 推断路径。

```python
from statgpu.linear_model import LinearRegression

# White robust SE (HC1)
m = LinearRegression(device="cpu", cov_type="hc1")
m.fit(X, y)
```

## 输出统计（`compute_inference=True`）

- 系数：`intercept_`, `coef_`
- 推断：`_bse`, `_tvalues`, `_pvalues`, `_conf_int`
- 诊断：`r_squared`, `adj_r_squared`, `f_statistic`, `aic`, `bic`
- 汇总：`summary()`

多目标 `y`（shape=`(n_samples, n_targets)`）时：

- 估计支持：`coef_` 为 `(n_targets, n_features)`，`intercept_` 为 `(n_targets,)`
- 推断支持：`_bse/_tvalues/_pvalues` 形状为 `(n_params, n_targets)`，`_conf_int` 为 `(n_params, n_targets, 2)`
- `summary()` 仅支持单目标，会在多目标下抛出错误

## 返回与属性

- `fit(X, y)`：返回 `self`
- `predict(X)`：返回预测值向量
- `score(X, y)`：返回 `R^2`
- 常用属性：`coef_`, `intercept_`, `nobs`, `df_resid`, `aic`, `bic`

## 常见问题

- **Q: GPU 下和 CPU 下 p 值有轻微差异？**  
  A: 浮点路径和线性代数实现不同会带来微小差异，通常在数值误差范围内。
- **Q: 什么时候用 `hc0/hc1`？**  
  A: 样本存在异方差时优先考虑 `hc0/hc1`；`hc1`包含小样本修正。

## 外部一致性

已补充与 `statsmodels.OLS` 的一致性测试：

- `dev/tests/test_external_consistency.py`
  - `test_linear_estimation_and_inference_match_statsmodels`
  - `test_linear_robust_covariance_matches_statsmodels`
  - `test_linear_robust_covariance_gpu_matches_statsmodels`
