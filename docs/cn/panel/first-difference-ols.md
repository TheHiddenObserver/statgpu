# FirstDifferenceOLS

> 语言：中文  
> 最后更新：2026-10-13  
> 切换：[English](../../en/panel/first-difference-ols.md)

<a id="概述"></a>

## 何时用个体内变化回答问题

`FirstDifferenceOLS` 对同一个体的相邻**已观测时期**做一阶差分，用当前观测减去上一条已观测记录，从而消除不随时间变化的个体效应；随后对差分后的响应和解释变量进行无截距 OLS 回归。

它与固定效应模型可以从同一个单向个体效应方程出发，区别在于消除个体效应所采用的数据变换不同。

适合用同一个体的变化来回答问题，并希望消除不随时间变化的个体效应。它不能识别时间不变解释变量的效应。与[固定效应](panel-ols.md)一样，一个常见的充分假设是严格外生性：给定完整解释变量历史和个体效应后，误差的条件均值为零。做差分本身不能保证该假设成立。

从 $y_{it}=\alpha+x_{it}^{\top}\beta+a_i+\varepsilon_{it}$ 出发。其中 $i$ 表示个体，$t$ 表示时期，$y_{it}$ 是响应，$x_{it}$ 是解释变量向量，$\alpha$ 是公共截距，$\beta$ 是公共斜率向量，$a_i$ 是不随时间变化的个体效应，$\varepsilon_{it}$ 是剩余误差。个体内差分后得到 $\Delta y_{it}=\Delta x_{it}^{\top}\beta+\Delta\varepsilon_{it}$，$\alpha$ 和 $a_i$ 都被消除。

这里 $\Delta$ 表示同一个体当前值减去上一条已观测记录的值，不除以日历时间间隔。拟合最小化该差分方程中的残差平方和，不估计截距；传给 `.fit()` 的仍是原始水平值。

## 准备水平值并确定时间顺序

采用长表格式，每行是一条个体—时期观测。`X` 的形状为 `(n, p)`，`y` 为 `(n,)`，`entity_ids` 的长度为 `n`，三者按同一行序对齐。一维 `X` 也可作为单个解释变量。ID 用于标识观测属于哪个个体，不是解释变量列。数组中的数值必须有限。

传入不含常数列的原始水平值，由 `.fit()` 完成差分。建议提供长度为 `n` 的 `time_ids` 明确时间顺序；省略时，模型会使用每个个体在输入中的原始行序。一个个体至少有两条记录才会贡献差分。

提供 `time_ids` 时，每个 `(entity_id, time_id)` 组合必须唯一。数值和日期时间标签按自然顺序排列；有序 categorical 使用用户声明的类别顺序。普通字符串按字典序排序；其他非 categorical 对象标签按可比较值的顺序排序，标签无法相互比较时会报错。若字符串的字典序不能表示真实时间顺序，应改用数值、日期时间或有序 categorical 标签。

<a id="使用示例"></a>
<a id="数组接口"></a>

## 先在 CPU 上拟合一个模型

每行是一个个体在一个时期的观测：12 个个体各有 5 条按时间标记的记录。差分会消除公共截距和不随时间变化的个体效应。

<a id="cpu-example"></a>

<!-- example: cpu-basic -->
```python
import numpy as np
from statgpu.panel import FirstDifferenceOLS

rng = np.random.default_rng(7)
entity_ids = np.repeat(np.arange(12), 5)
time_ids = np.tile(np.arange(5), 12)
x = np.repeat(np.linspace(-2, 2, 12), 5) + rng.normal(size=60)
X = x[:, None]
entity_effect = np.repeat(np.linspace(-1, 1, 12), 5)
y = 1.0 + 2.0 * x + entity_effect + 0.2 * rng.normal(size=60)

model = FirstDifferenceOLS(device="cpu").fit(
    X, y, entity_ids=entity_ids, time_ids=time_ids
)
print("slope:", np.round(model.coef_, 3))
print("nobs, df_resid:", model.nobs, model.df_resid)
```
<!-- example-end: cpu-basic -->

<a id="输出与解释"></a>

## 理解差分回归并预测变化

`coef_` 只包含斜率，接近生成数据时设定的 2。每个个体贡献四条相邻观测差分，因此 `nobs=48`、`df_resid=47`。即使水平误差彼此独立，差分后也会产生序列相关；可用的 HC 选项处理异方差，并不处理一般的序列相关。

常用结果包括 `coef_`、`bse_`、`tvalues_`、`pvalues_`、`conf_int_`、`rsquared`、`fit_statistics_`、`nobs` 与 `df_resid`。其中 `nobs` 是实际进入最终回归的差分观测数，因此通常小于原始面板数据行数。

`predict(X_new)` 只计算 `X_new @ coef_`，返回 NumPy 数组；它不会对新输入做差分，也不会还原个体效应。若要预测响应变化，应自行构造所需的解释变量差分再传入。直接传入水平值只得到斜率部分，不是完整的水平预测。若公式包含变换或交互项，应对生成后的设计列做差分并传入数值矩阵；先对原始 dataframe 列差分再重建公式，可能对应不同的模型。

`bse_` 是系数标准误，`conf_int_` 每行为相应系数的区间下限和上限。`nonrobust` 使用自由度为 `df_resid` 的 Student-t 推断；HC 选项使用正态参考分布，此时 `tvalues_` 实际存放 z 统计量。用 `print(model.summary())` 可同时查看估计与不确定性。公开系数和推断数组均为 NumPy 数组。标准化的组内/组间/总体指标见[拟合统计量](fit-statistics.md)，不应假定它们等于变换后回归的 `rsquared`。

<a id="协方差与统计推断"></a>

## 选择差分误差的推断方式

标准误基于实际进入回归的 $(\Delta y,\Delta X)$ 计算。当前支持 `nonrobust`、`robust`/`hc1`、`hc0`、`hc2` 和 `hc3`。统一公式见 [面板协方差](covariance.md)。

这些协方差选项只改变差分回归的不确定性估计，不能替代基础面板模型对误差项所需的外生性条件。

本估计器不提供 cluster 或 HAC 选项。超过两个时期时，相邻差分会共享水平误差；即使水平误差独立，差分误差也会序列相关。需要按这种依赖结构选择模型和协方差方案，不能把 HC 当作序列相关修正。

## 检查样本和数值限制

最终差分回归需要正的残差自由度，即差分观测数大于设计矩阵的秩。提供 `time_ids` 时，同一个体同一时间标签出现重复观测会报错，因为差分顺序不唯一。日历缺口不会补齐，也不会按照相隔时间对差分缩放。

若有限精度运算无法可靠区分差分回归中的某个非零系数，`.fit()` 会抛出 `FloatingPointError`。这不同于精确共线性：差分设计精确秩亏时仍可计算拟合值，但系数向量不唯一，因此不提供系数级标准误、检验、p 值和置信区间。

原有 `rsquared` 使用避免溢出的中心化计算，即使差分响应尺度极端，也保持同一个尺度不变的 $R^2$ 定义。

无效协方差选项或不可用的显式 GPU 后端会报错，不会静默换成其他选项。

<a id="公式接口"></a>

## 从数据框列拟合

假设 `df` 已包含下面公式及分组参数中使用的列。

```python
from statgpu.panel import FirstDifferenceOLS

model = FirstDifferenceOLS().fit(
    formula="y ~ x1 + x2 - 1",
    data=df,
    entity_ids=df["entity"],
    time_ids=df["time"],
)
```

这里显式去掉截距，因为差分后的回归本身不估计截距。

## 在 GPU 上运行同一个模型

在 GPU 上运行时，把 `device="cpu"` 换成 `device="cuda"`（CuPy/CUDA）或 `device="torch"`（Torch CUDA）。显式请求的后端不可用时会报错，不会切换到 CPU。

<a id="统计模型与识别"></a>

## 为什么差分能消除个体效应

从

$$
y_{it}=x_{it}^{\top}\beta+a_i+\varepsilon_{it}
$$

出发，其中 $a_i$ 是固定但未知、且不随时间变化的个体效应；它不必与解释变量历史正交。对静态线性面板模型，一个常见的充分外生性条件是

$$
E\!\left(\varepsilon_{it}\mid X_i,a_i\right)=0,
\qquad
X_i=(x_{i1},\ldots,x_{iT_i}).
$$

做一阶差分后，$a_i$ 被消除：

$$
\Delta y_{it}=\Delta x_{it}^{\top}\beta+\Delta\varepsilon_{it}.
$$

因此，斜率由同一个体内部随时间发生的变化识别。若某个解释变量在个体内部始终不变，其差分恒为 0，无法在该模型中识别对应斜率。原水平模型中的公共截距也会被差分消掉，所以最终回归不估计截距。

<a id="估计量"></a>

## 相邻观测差分的最小二乘问题

对个体 $i$ 的相邻已观测记录，定义

$$
\Delta y_{it}=y_{it}-y_{i,t^-},
\qquad
\Delta x_{it}=x_{it}-x_{i,t^-},
$$

其中 $t^-$ 表示该个体上一条已观测时间。于是

$$
\widehat\beta_{\mathrm{FD}}
=\arg\min_\beta\|\Delta y-\Delta X\beta\|_2^2
=(\Delta X^\top\Delta X)^+\Delta X^\top\Delta y.
$$

缺失的日历时期不会被补齐，差分也不会按照时间间隔长度再次缩放。

<a id="参数"></a>

## 完整 API 参考

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `cov_type` | `"nonrobust"` | `nonrobust`、`robust`/`hc1`、`hc0`、`hc2`、`hc3` |
| `alpha` | `0.05` | 有限且严格位于 `(0,1)`；置信区间显著性水平；`0.05` 对应 95% 区间 |
| `device` | `"auto"` | `auto`、`cpu`、`cuda`、`torch` |
| `n_jobs` | `None` | 整数或 `None`；共享并行参数 |

### 拟合输入

`fit(X=None, y=None, entity_ids=None, time_ids=None, formula=None, data=None)` 返回拟合后的估计器。

| 参数 | 含义 |
|---|---|
| `X`、`y` | 数组接口：解释变量为 `(n, p)`，响应为 `(n,)`；一维 `X` 视为单个解释变量。数值必须有限。 |
| `entity_ids` | 必需，与 `X`、`y` 按行对齐的个体标签。 |
| `time_ids` | 可选的时间标签；省略时保留每个个体在输入中的原始行序。 |
| `formula`、`data` | 可替代数组输入的公式字符串和 dataframe；需要可选的 `pandas`/`patsy` 依赖。提供的 ID 数组会与公式保留的观测行对齐。 |

使用数组输入或 `formula` 加 `data` 中的一种。模型的截距约定见[数据框示例](#公式接口)。

### 其他公开方法

- `predict(X)`：每条输入行返回一个预测值，具体解释见前文；特征数量和列顺序必须与训练时一致。
- `summary()`：返回可打印的系数与拟合摘要，使用 `print(model.summary())` 查看。
- `get_params(deep=True)` 与 `set_params(**params)`：读取或更新构造参数；改变设置后应重新拟合。
- 继承的 `adjust_pvalues`、`combine_pvalues`、`bootstrap_statistic` 和 `permutation_test` 的完整签名见[共享估计器 API](../reference/estimator-api.md)。这些工具不会自动选择适用于面板依赖结构的重采样方案。

<a id="常见问题"></a>

## 关于时间缺口和截距的常见疑问

**跨两个日历时期的间隔会形成“两步差分”吗？**  
不会。模型只使用当前已观测记录减去上一条已观测记录，不论两者在日历时间上相隔多久。

**差分后是否估计截距？**  
不估计。公共截距与时间不变的个体效应都会被一阶差分消除。

**为什么时间不变解释变量无法估计？**  
因为它在同一个体内的一阶差分恒为 0，不再提供识别斜率所需的变化。

<a id="external-validation"></a>
<a id="comparing-with-other-packages"></a>

## 与其他软件包比较

与 statsmodels 等 OLS 实现比较时，应按相同的个体内时间顺序构造相同的差分样本，拟合时不加截距，并对齐 HC0/HC2/HC3 等协方差选项。两边都应保留日历缺口，不能把相邻已观测记录的差分改成除以时间间隔的变化率。详见[协方差定义的比较](covariance.md#comparing-covariance-definitions)。

<a id="相关文档"></a>

## 继续阅读

- [面板模型总览](README.md)
- [面板协方差](covariance.md)
- [面板拟合统计量](fit-statistics.md)

<a id="参考文献"></a>

## 统计参考文献

- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). The MIT Press.
