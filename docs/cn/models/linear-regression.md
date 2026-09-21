# LinearRegression

> 语言: 中文  
> 最后更新: 2026-09-21  
> 页面定位: 模型文档  
> 切换: [English](../../en/models/linear-regression.md)

## 概览

`LinearRegression` 提供 OLS 估计与可选推断，支持 CPU/GPU 统一接口。当前版本覆盖单目标与多目标回归；多目标下支持参数与推断张量输出，但 `summary()` 仅支持单目标。

## 导入路径

`statgpu.linear_model.LinearRegression`

## 目标函数

最小化残差平方和：

$$
\min_{\beta, b} \|y - X\beta - b\|_2^2
$$

其中 `fit_intercept=True` 时同时估计截距项 `b`。

## 估计方程

采用 OLS 正规方程/等价线性代数求解；`compute_inference=True` 时基于残差与设计矩阵计算标准误、统计量与置信区间。

## 协方差与推断

- `cov_type="nonrobust"`：经典 OLS 协方差（t 统计口径）
- `cov_type="hc0"`：White 异方差稳健
- `cov_type="hc1"`：HC0 + 自由度修正 `n/(n-k)`
- `cov_type="hc2"`：基于杠杆值（leverage）的修正
- `cov_type="hc3"`：更保守的 jackknife（留一法）风格修正
- `cov_type="hac"`：Newey-West（Bartlett 核）自相关稳健协方差
- `hac_maxlags`：仅 `cov_type="hac"` 生效；不设时按样本规模自动推断

上述协方差类型在 CPU/GPU 路径均可用。

## 参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `fit_intercept` | `True` | 是否拟合截距 |
| `device` | `"auto"` | `cpu` / `cuda` / `auto` |
| `compute_inference` | `True` | 是否计算推断统计（SE/t/p/CI） |
| `cov_type` | `"nonrobust"` | `nonrobust` / `hc0` / `hc1` / `hc2` / `hc3` / `hac` |
| `hac_maxlags` | `None` | `cov_type="hac"` 时的最大滞后阶 |
| `gpu_memory_cleanup` | `False` | `fit` 后尝试释放 CuPy 内存池 |

## CPU/GPU 示例

```python
from statgpu.linear_model import LinearRegression

# CPU：HC1 稳健标准误
m_cpu = LinearRegression(device="cpu", cov_type="hc1", compute_inference=True)
m_cpu.fit(X, y)

# GPU：HAC 稳健标准误
m_gpu = LinearRegression(
    device="cuda",
    cov_type="hac",
    hac_maxlags=4,
    compute_inference=True,
)
m_gpu.fit(X, y)
```

## 严格计算与近似计算

`LinearRegression` 默认使用同一套推断定义，并未提供单独的近似计算开关。CPU 与 GPU 之间若出现微小数值差异，通常来自浮点运算和底层线性代数实现。

## 输出

- 拟合返回：`fit(X, y) -> self`
- 预测与评分：`predict(X)`、`score(X, y)`（`R^2`）
- 主要属性：`coef_`, `intercept_`, `nobs`, `df_resid`, `aic`, `bic`
- 推断属性（`compute_inference=True`）：`_bse`, `_tvalues`, `_pvalues`, `_conf_int`
- 诊断统计：`r_squared`, `adj_r_squared`, `f_statistic`
- 汇总：`summary()`（仅单目标）

多目标 `y` 为 `(n_samples, n_targets)` 时：
- `coef_` 为 `(n_targets, n_features)`，`intercept_` 为 `(n_targets,)`
- `_bse/_tvalues/_pvalues` 为 `(n_params, n_targets)`，`_conf_int` 为 `(n_params, n_targets, 2)`

## 常见问题（FAQ）

- **GPU 与 CPU 的 p 值轻微不同是否正常？**  
  正常。通常由浮点累计误差和底层线性代数实现差异引起。
- **`cov_type` 如何选择？**  
  异方差场景优先 `hc1/hc3`；高杠杆值场景可用 `hc2/hc3`；存在时序相关时使用 `hac` 并显式设置 `hac_maxlags`。

## 与外部实现的对照

估计结果、稳健协方差和 HAC 协方差均与 `statsmodels.OLS` 做数值对照；CPU 与 GPU 路径也有对应的一致性测试。

## 参考文献

- Greene, W. H. (2018). *Econometric Analysis* (8th ed.). Pearson.
- White, H. (1980). A heteroskedasticity-consistent covariance matrix estimator and a direct test for heteroskedasticity. *Econometrica*, 48(4), 817-838. [https://doi.org/10.2307/1912934](https://doi.org/10.2307/1912934)
- MacKinnon, J. G., & White, H. (1985). Some heteroskedasticity-consistent covariance matrix estimators with improved finite sample properties. *Journal of Econometrics*, 29(3), 305-325. [https://doi.org/10.1016/0304-4076(85)90158-7](https://doi.org/10.1016/0304-4076(85)90158-7)
- Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703-708. [https://doi.org/10.2307/1913610](https://doi.org/10.2307/1913610)
