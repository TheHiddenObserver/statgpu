# LinearRegression

> 语言: 中文  
> 最后更新: 2026-04-17  
> 页面定位: 模型文档  
> 切换: [English](../en/models/linear-regression.md)

语言切换：[English](../en/models/linear-regression.md)

## 概览（Overview）

`LinearRegression` 提供 OLS 估计与可选推断，支持 CPU/GPU 统一接口。当前版本覆盖单目标与多目标回归；多目标下支持参数与推断张量输出，但 `summary()` 仅支持单目标。

## 路径（Path）

`statgpu.linear_model.LinearRegression`

## 目标函数（Objective Function）

最小化残差平方和：

\[
\min_{\beta, b} \|y - X\beta - b\|_2^2
\]

其中 `fit_intercept=True` 时同时估计截距项 `b`。

## 估计方程（Estimating Equation）

采用 OLS 正规方程/等价线性代数求解；`compute_inference=True` 时基于残差与设计矩阵计算标准误、统计量与置信区间。

## 协方差与推断（Covariance/Inference）

- `cov_type="nonrobust"`：经典 OLS 协方差（t 统计口径）
- `cov_type="hc0"`：White 异方差稳健
- `cov_type="hc1"`：HC0 + 自由度修正 `n/(n-k)`
- `cov_type="hc2"`：基于 leverage 的修正
- `cov_type="hc3"`：更保守的 jackknife 风格修正
- `cov_type="hac"`：Newey-West（Bartlett kernel）自相关稳健协方差
- `hac_maxlags`：仅 `cov_type="hac"` 生效；不设时按样本规模自动推断

上述协方差类型在 CPU/GPU 路径均可用。

## 参数（Parameters）

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `fit_intercept` | `True` | 是否拟合截距 |
| `device` | `"auto"` | `cpu` / `cuda` / `auto` |
| `compute_inference` | `True` | 是否计算推断统计（SE/t/p/CI） |
| `cov_type` | `"nonrobust"` | `nonrobust` / `hc0` / `hc1` / `hc2` / `hc3` / `hac` |
| `hac_maxlags` | `None` | `cov_type="hac"` 时的最大滞后阶 |
| `gpu_memory_cleanup` | `False` | `fit` 后尝试释放 CuPy memory pool |

## CPU+GPU 示例（CPU+GPU Examples）

```python
from statgpu.linear_model import LinearRegression

# CPU: HC1 robust SE
m_cpu = LinearRegression(device="cpu", cov_type="hc1", compute_inference=True)
m_cpu.fit(X, y)

# GPU: HAC robust SE
m_gpu = LinearRegression(
    device="cuda",
    cov_type="hac",
    hac_maxlags=4,
    compute_inference=True,
)
m_gpu.fit(X, y)
```

## strict/approx 差异（strict/approx difference）

当前文档对应默认 strict 推断路径。`LinearRegression` 未提供独立 `approx` 开关；若出现 CPU/GPU 微小差异，主要来源于浮点与线性代数实现差异。

## 输出（Outputs）

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
  异方差场景优先 `hc1/hc3`；高 leverage 场景可用 `hc2/hc3`；存在时序相关时使用 `hac` 并显式设置 `hac_maxlags`。

## 外部验证（External Validation）

与 `statsmodels.OLS` 的对齐测试位于：

- `dev/tests/test_external_consistency.py`
  - `test_linear_estimation_and_inference_match_statsmodels`
  - `test_linear_robust_covariance_matches_statsmodels`
  - `test_linear_robust_covariance_gpu_matches_statsmodels`
  - `test_linear_hac_covariance_matches_statsmodels`

## 参考（References）

- Greene, W. H. (2018). *Econometric Analysis* (8th ed.). Pearson.
- White, H. (1980). A heteroskedasticity-consistent covariance matrix estimator and a direct test for heteroskedasticity. *Econometrica*, 48(4), 817-838. [https://doi.org/10.2307/1912934](https://doi.org/10.2307/1912934)
- MacKinnon, J. G., & White, H. (1985). Some heteroskedasticity-consistent covariance matrix estimators with improved finite sample properties. *Journal of Econometrics*, 29(3), 305-325. [https://doi.org/10.1016/0304-4076(85)90158-7](https://doi.org/10.1016/0304-4076(85)90158-7)
- Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703-708. [https://doi.org/10.2307/1913610](https://doi.org/10.2307/1913610)
