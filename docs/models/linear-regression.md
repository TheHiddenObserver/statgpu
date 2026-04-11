# LinearRegression

> 语言: 中文  
> 最后更新: 2026-04-10  
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
| `cov_type` | `"nonrobust"` | `nonrobust` / `hc0` / `hc1` / `hc2` / `hc3` / `hac` |
| `hac_maxlags` | `None` | `cov_type="hac"` 时的最大滞后阶；为空时使用 Newey-West 风格默认规则 |
| `gpu_memory_cleanup` | `False` | 每次 `fit` 后尝试释放 CuPy memory pool |

## 主要参数
- `fit_intercept`: 是否拟合截距
- `device`: `cpu` / `cuda` / `auto`
- `compute_inference`: 是否计算推断统计（SE/t/p/CI）
- `cov_type`: `nonrobust` / `hc0` / `hc1` / `hc2` / `hc3` / `hac`
- `hac_maxlags`: 仅 `cov_type="hac"` 生效；未指定时按样本规模自动推断
- `gpu_memory_cleanup`: 每次 `fit` 后尝试释放 CuPy memory pool

## 快速示例

```python
from statgpu.linear_model import LinearRegression

m = LinearRegression(device="cuda", compute_inference=True, gpu_memory_cleanup=False)
m.fit(X, y)
m.summary()
```

## 稳健协方差（HC0/HC1/HC2/HC3/HAC）

- `cov_type="nonrobust"`：经典 OLS 协方差（t 分布口径）
- `cov_type="hc0"`：White 异方差稳健协方差
- `cov_type="hc1"`：HC0 + 自由度修正 `n/(n-k)`
- `cov_type="hc2"`：基于 leverage 的稳健修正
- `cov_type="hc3"`：更保守的 jackknife 风格修正
- `cov_type="hac"`：Newey-West（Bartlett kernel）自相关稳健协方差，可通过 `hac_maxlags` 控制滞后阶

CUDA 下上述 `cov_type` 均支持 GPU 推断路径。

```python
from statgpu.linear_model import LinearRegression

# White robust SE (HC1)
m = LinearRegression(device="cpu", cov_type="hc1")
m.fit(X, y)

# HAC robust SE (Newey-West, maxlags=4)
m_hac = LinearRegression(device="cpu", cov_type="hac", hac_maxlags=4)
m_hac.fit(X, y)
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
- **Q: 什么时候用 `hc0/hc1/hc2/hc3/hac`？**  
  A: 异方差场景可优先 `hc1/hc3`；高 leverage 样本可考虑 `hc2/hc3`；存在时序相关时优先 `hac` 并显式设置 `hac_maxlags`。

## 外部一致性

已补充与 `statsmodels.OLS` 的一致性测试：

- `dev/tests/test_external_consistency.py`
  - `test_linear_estimation_and_inference_match_statsmodels`
  - `test_linear_robust_covariance_matches_statsmodels`
  - `test_linear_robust_covariance_gpu_matches_statsmodels`
  - `test_linear_hac_covariance_matches_statsmodels`
