# LinearRegression

> 语言：简体中文
> 最后更新：2026-09-03
> 切换：[English](../../en/models/linear-regression.md)

## 它解决什么问题？

`LinearRegression` 用一个或多个特征解释或预测连续型结果。它实现普通最小二乘法（OLS），提供统一的 CPU/GPU 接口，并可计算经典、异方差稳健或自相关稳健推断。

常见问题包括：

- 某个特征增加一个单位时，结果的期望值会变化多少？
- 多个特征的线性组合能多好地预测连续目标？
- 考虑估计不确定性后，哪些系数仍明显不为零？

## 什么时候使用？

当响应变量连续，且线性条件均值是合理的第一近似时，可以使用线性回归。它尤其适合作为可解释的基线模型。

以下情况应考虑其他方法：

- 响应是二分类、计数、正值且强偏态，或生存时间：从[广义线性模型](generalized-linear-model.md)或 Cox 模型开始；
- 关系明显非线性且没有加入合适的变换特征；
- 需要自动筛选特征：考虑 Ridge、Lasso、Elastic Net 或 SCAD；
- 想解释因果效应，但研究设计本身并不支持因果识别。

## 直观理解

OLS 寻找一个超平面，使观测响应与拟合值之间的纵向距离平方和最小。每个系数表示一个“控制其他已纳入特征不变后的关联”：该特征增加一个单位时，`y` 的期望变化量。

## 模型、目标函数与假设

带截距的模型为

$$
y_i = \beta_0 + x_i^\top\beta + \varepsilon_i,
$$

OLS 估计量为

$$
(\hat\beta_0,\hat\beta)
=
\arg\min_{\beta_0,\beta}
\sum_{i=1}^{n}
\left(y_i-\beta_0-x_i^\top\beta\right)^2.
$$

其中，`n` 是观测数，`x_i` 是第 `i` 个特征向量，`\varepsilon_i` 是已纳入特征未解释的部分。

若要解释系数并进行常规推断，应检查：

- 条件均值关系是否足够接近线性；
- 变量尺度与零点是否具有可解释意义；
- 设计矩阵是否满秩；
- 误差的条件均值是否为零；
- 协方差估计是否匹配误差结构。

只有经典 `nonrobust` 标准误要求同方差；HC 或 HAC 推断不要求同方差。

## 最小可运行示例

下面的示例会自行生成数据，可以直接复制运行。

```python
import numpy as np
from statgpu.linear_model import LinearRegression

rng = np.random.default_rng(0)
X = rng.normal(size=(500, 3))
true_coef = np.array([2.0, -1.0, 0.5])
y = 1.5 + X @ true_coef + rng.normal(scale=0.7, size=500)

model = LinearRegression(
    device="cpu",
    cov_type="hc1",
    compute_inference=True,
).fit(X, y)

print("截距：", model.intercept_)
print("系数：", model.coef_)
print("R²：", model.rsquared)
print("标准误：", model._bse)
print("p 值：", model._pvalues)
print("前三个预测：", model.predict(X[:3]))
```

拟合系数应接近 `[2.0, -1.0, 0.5]`，截距应接近 `1.5`。响应中包含随机噪声，因此每次具体估计不会与真值完全相同。

## 如何读取结果？

- `coef_[j]`：控制其他特征不变时，第 `j` 个特征增加一个单位对应的响应期望变化量。
- `intercept_`：所有特征取零时的响应期望；只有当“全为零”有实际含义时才值得直接解释。
- `rsquared`：样本内变异被线性模型解释的比例；高 $R^2$ 不代表因果关系，也不保证样本外预测准确。
- `_bse`、`_tvalues`、`_pvalues`、`_conf_int`：`compute_inference=True` 时的系数不确定性；拟合截距时，推断数组第一项为截距。
- `predict(X_new)` 返回预测值，`score(X, y)` 返回 $R^2$。

## 关键参数怎么选？

| 参数 | 默认值 | 选择建议 |
|---|---:|---|
| `fit_intercept` | `True` | 通常保留；只有理论模型确实过原点或设计矩阵已有截距列时才关闭 |
| `device` | `"auto"` | 小数据与兼容性优先用 `"cpu"`；数据规模足以抵消传输开销时再用 `"cuda"` 或 `"torch"` |
| `compute_inference` | `True` | 只需系数或预测、希望减少推断开销时关闭 |
| `cov_type` | `"nonrobust"` | 异方差使用 `"hc1"` 或 `"hc3"`；有顺序且存在序列相关时使用 `"hac"` |
| `hac_maxlags` | `None` | 按领域知识设置最大相关滞后；默认值使用样本量启发式规则 |
| `gpu_memory_cleanup` | `False` | 只有每次拟合后归还 CuPy 缓存显存比重复拟合速度更重要时才开启 |

## 与其他方法比较

| 方法 | 更适合的情况 |
|---|---|
| Ridge | 特征高度相关，希望用收缩改善预测 |
| Lasso / Elastic Net | 需要稀疏系数或自动特征选择 |
| GLM | 响应不适合用高斯连续模型表示 |
| 稳健回归 | 少量极端响应值严重影响 OLS |
| Panel OLS | 同一实体或时期存在重复观测 |

## CPU、GPU 与推断支持

同一个估计器可在 NumPy/CPU、CuPy/CUDA 或受支持的 Torch 路径上拟合：

```python
gpu_model = LinearRegression(
    device="cuda",
    cov_type="hc1",
    compute_inference=True,
).fit(X, y)
```

支持 `nonrobust`、`hc0`、`hc1`、`hc2`、`hc3` 和 `hac` 协方差。不同后端的浮点累加顺序和线性代数实现不同，结果可能有微小差异。该模型没有独立的公开近似推断模式。

## 常见误区

- 比较系数大小前，应确认特征量纲一致，必要时先标准化。
- 小 p 值不等于实际影响很大。
- HC 标准误不能修复非线性、遗漏变量或依赖结构；它只改变不确定性估计。
- 只有观测顺序具有实际含义时才使用 HAC，并在拟合前正确排序。
- 应检查残差，并用留出数据评估预测。

## API 与验证

导入路径：`statgpu.linear_model.LinearRegression`

主要输出：`coef_`、`intercept_`、`rsquared`、`rsquared_adj`、`fvalue`、`f_pvalue`、`aic`、`bic`、`_bse`、`_tvalues`、`_pvalues` 和 `_conf_int`。

模型支持多目标响应，但 `summary()` 只支持单目标。与 `statsmodels.OLS` 的外部一致性测试位于 `dev/tests/test_external_consistency.py`。

## 参考文献

- Greene, W. H. (2018). *Econometric Analysis* (8th ed.). Pearson.
- White, H. (1980). A heteroskedasticity-consistent covariance matrix estimator and a direct test for heteroskedasticity. *Econometrica*, 48(4), 817-838.
- Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703-708.
