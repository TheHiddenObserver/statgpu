# 线性回归（LinearRegression）

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

其中，$n$ 是观测数，$x_i$ 是第 $i$ 个观测的特征向量，$\varepsilon_i$ 是已纳入特征未解释的部分。

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
model.summary()
```

拟合系数应接近 `[2.0, -1.0, 0.5]`，截距应接近 `1.5`。响应中包含随机噪声，因此每次具体估计不会与真值完全相同。

`summary()` 会把拟合统计量和系数表集中打印出来。上面这个固定随机种子的示例开头如下：

```text
Linear Regression Results
Covariance Type:                        hc1
No. Observations:                       500
R-squared:                           0.9064
Adj. R-squared:                      0.9058
                        coef      std err          t      P>|t|
(Intercept)           1.4562       0.0327     44.512     0.0000
x1                    2.0288       0.0314     64.660     0.0000
x2                   -0.9906       0.0349    -28.382     0.0000
x3                    0.5440       0.0312     17.422     0.0000
```

`summary()` 只适用于已经拟合、`compute_inference=True` 的单目标模型。完整输出还包括 F 统计量、对数似然、AIC、BIC 与 95% 置信区间。

## Formula 与 DataFrame 示例

先安装可选的公式解析依赖，再直接使用列名拟合：

```bash
pip install "statgpu[formula]"
```

```python
import pandas as pd
from statgpu.linear_model import LinearRegression

frame = pd.DataFrame(X, columns=["x1", "x2", "x3"])
frame["y"] = y

formula_model = LinearRegression(
    device="cpu",
    cov_type="hc3",
).fit(
    formula="y ~ x1 + x2 + x3",
    data=frame,
)

formula_model.summary()
pred = formula_model.predict(frame.iloc[:3])
```

模型会保存 Patsy 设计信息，使预测时的分类变量编码与交互项列保持一致。其他估计器未必接受 `formula=`，请先查看 [Formula 接口与支持矩阵](../guides/formula-interface.md)。

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

六种协方差的单独公式、适用假设、全部参数、输出形状与后端行为见[线性回归推断与完整 API](linear-regression-inference.md)。源码级加速设计单独放在 [CPU/GPU 加速实现指南](../guides/acceleration-internals.md)。

## 常见误区

- 比较系数大小前，应确认特征量纲一致，必要时先标准化。
- 小 p 值不等于实际影响很大。
- HC 标准误不能修复非线性、遗漏变量或依赖结构；它只改变不确定性估计。
- 只有观测顺序具有实际含义时才使用 HAC，并在拟合前正确排序。
- 应检查残差，并用留出数据评估预测。

## API 与验证

导入路径：`statgpu.linear_model.LinearRegression`

本入门页只保留常用工作流需要的参数。[完整 API 页面](linear-regression-inference.md#完整-api-参考)会列出全部构造参数、拟合参数、拟合后统计量、推断数组、方法、失败条件和多目标形状。

模型支持多目标响应，但 `summary()` 只支持单目标。与 `statsmodels.OLS` 的外部一致性测试位于 `dev/tests/test_external_consistency.py`。

## 参考文献

- Greene, W. H. (2018). *Econometric Analysis* (8th ed.). Pearson.
- White, H. (1980). A heteroskedasticity-consistent covariance matrix estimator and a direct test for heteroskedasticity. *Econometrica*, 48(4), 817-838.
- Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703-708.
