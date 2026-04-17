# Ridge

> 语言: 中文  
> 最后更新: 2026-04-17  
> 页面定位: 模型文档  
> 切换: [English](../en/models/ridge.md)

语言切换：[English](../en/models/ridge.md)

## 概览（Overview）

`Ridge` 在 OLS 基础上引入 L2 正则，支持 CPU/GPU 统一训练与推断。当前实现已对齐 `LinearRegression` 的稳健协方差选项（含 `hc2/hc3/hac`）。

## 路径（Path）

`statgpu.linear_model.Ridge`

## 目标函数（Objective Function）

最小化带 L2 惩罚项的平方损失：

\[
\min_{\beta, b}\ \|y - X\beta - b\|_2^2 + \alpha \|\beta\|_2^2
\]

`alpha` 越大，系数收缩越强。

## 估计方程（Estimating Equation）

通过带惩罚的正规方程/等价数值线性代数求解参数；`compute_inference=True` 时对估计结果计算标准误、统计量与区间估计。

## 协方差与推断（Covariance/Inference）

- `cov_type="nonrobust"`：经典 ridge 协方差
- `cov_type="hc0"`：White/sandwich 稳健
- `cov_type="hc1"`：HC0 + 自由度修正 `n/(n-k)`
- `cov_type="hc2"`：基于 leverage 修正
- `cov_type="hc3"`：更保守的 jackknife 风格修正
- `cov_type="hac"`：Newey-West（Bartlett kernel）自相关稳健协方差
- `hac_maxlags`：仅 `cov_type="hac"` 生效；不设时自动推断

## 参数（Parameters）

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `alpha` | `1.0` | L2 正则强度 |
| `fit_intercept` | `True` | 是否拟合截距 |
| `device` | `"auto"` | `cpu` / `cuda` / `auto` |
| `n_jobs` | `None` | 并行任务数 |
| `compute_inference` | `True` | 是否计算推断统计（SE/t/p/CI） |
| `cov_type` | `"nonrobust"` | `nonrobust` / `hc0` / `hc1` / `hc2` / `hc3` / `hac` |
| `hac_maxlags` | `None` | `cov_type="hac"` 时最大滞后阶 |
| `gpu_memory_cleanup` | `False` | `fit` 后尝试释放 CuPy memory pool |

## CPU+GPU 示例（CPU+GPU Examples）

```python
from statgpu.linear_model import Ridge

# CPU
m_cpu = Ridge(alpha=1.0, device="cpu", cov_type="hc1", compute_inference=True)
m_cpu.fit(X, y)

# GPU
m_gpu = Ridge(
    alpha=1.0,
    device="cuda",
    cov_type="hc3",
    compute_inference=True,
    gpu_memory_cleanup=True,
)
m_gpu.fit(X, y)
```

## strict/approx 差异（strict/approx difference）

当前使用默认 strict 推断口径；未提供单独 `approx` 开关。CPU/GPU 结果的细小差异通常来自数值路径差异。

## 输出（Outputs）

- `fit(X, y) -> self`
- `predict(X)`、`score(X, y)`（`R^2`）
- 参数与诊断：`intercept_`, `coef_`, `r_squared`, `adj_r_squared`, `f_statistic`, `aic`, `bic`
- 推断属性（`compute_inference=True`）：`_bse`, `_tvalues`, `_pvalues`, `_conf_int`
- 汇总：`summary()`

## 常见问题（FAQ）

- **`alpha` 如何选择？**  
  建议先用对数网格（如 `1e-4` 到 `1e2`）做外部交叉验证，再固化到生产参数。
- **什么时候设置 `hac_maxlags`？**  
  当 `cov_type="hac"` 且存在时序相关时建议显式设置；否则使用默认规则。

## 外部验证（External Validation）

当前文档未单列 Ridge 的独立外部一致性脚本，建议复用线性模型一致性框架与基准脚本进行回归验证。

## 参考（References）

- Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression: Biased estimation for nonorthogonal problems. *Technometrics*, 12(1), 55-67. [https://doi.org/10.1080/00401706.1970.10488634](https://doi.org/10.1080/00401706.1970.10488634)
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.
