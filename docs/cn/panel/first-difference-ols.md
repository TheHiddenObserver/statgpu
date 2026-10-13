# FirstDifferenceOLS

> 语言：中文  
> 最后更新：2026-10-13  
> 切换：[English](../../en/panel/first-difference-ols.md)

## 概述

`FirstDifferenceOLS` 对同一个体的相邻**已观测时期**做一阶差分，用当前观测减去上一条已观测记录，从而消除不随时间变化的个体效应；随后对差分后的响应和解释变量进行无截距 OLS 回归。

它与固定效应模型可以从同一个单向个体效应方程出发，区别在于消除个体效应所采用的数据变换不同。

## 统计模型与识别

从

$$
y_{it}=x_{it}^{\top}\beta+a_i+\varepsilon_{it}
$$

出发，其中 $a_i$ 是固定但未知、且不随时间变化的个体效应。对静态线性面板模型，一个常见的充分外生性条件是

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

## 估计量

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

## 协方差与统计推断

标准误基于实际进入回归的 $(\Delta y,\Delta X)$ 计算。当前支持 `nonrobust`、`robust`/`hc1`、`hc0`、`hc2` 和 `hc3`。统一公式见 [面板协方差](covariance.md)。

这些协方差选项只改变差分回归的不确定性估计，不能替代基础面板模型对误差项所需的外生性条件。

如果差分后的设计矩阵精确秩亏，拟合值仍可能唯一，但系数向量不唯一。此时 statgpu 保留拟合结果，同时不发布系数级标准误、检验、p 值和置信区间。

## 参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `cov_type` | `"nonrobust"` | `nonrobust`、`robust`/`hc1`、`hc0`、`hc2`、`hc3` |
| `alpha` | `0.05` | 置信区间显著性水平；`0.05` 对应 95% 区间 |
| `device` | `"auto"` | `auto`、`cpu`、`cuda`、`torch` |
| `n_jobs` | `None` | 适用路径的并行任务数 |

`entity_ids` 是必需输入。提供 `time_ids` 后，statgpu 会先在每个个体内部按时间排序：

```python
model.fit(
    X,
    y,
    entity_ids=entity_ids,
    time_ids=time_ids,
)
```

此时每个 `(entity_id, time_id)` 组合必须唯一。数值和日期时间标签按自然顺序排列；有序 categorical 使用用户声明的类别顺序。普通字符串按字典序排序；其他非 categorical 对象标签按可比较值的顺序排序，标签无法相互比较时会报错。若字符串的字典序不能表示真实时间顺序，应改用数值、日期时间或有序 categorical 标签。

## 使用示例

### 数组接口

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

`coef_` 只包含斜率，接近生成数据时设定的 2。每个个体贡献四条相邻观测差分，因此 `nobs=48`、`df_resid=47`。即使水平误差彼此独立，差分后也会产生序列相关；可用的 HC 选项处理异方差，并不处理一般的序列相关。

在 GPU 上运行时，把 `device="cpu"` 换成 `device="cuda"`（CuPy/CUDA）或 `device="torch"`（Torch CUDA）。显式请求的后端不可用时会报错，不会切换到 CPU。

### 公式接口

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

## 输出与解释

常用结果包括 `coef_`、`bse_`、`tvalues_`、`pvalues_`、`conf_int_`、`rsquared`、`fit_statistics_`、`nobs` 与 `df_resid`。其中 `nobs` 是实际进入最终回归的差分观测数，因此通常小于原始面板数据行数。

若同一个体在同一时间标签下出现重复观测，模型会报错，因为无法唯一确定差分顺序。极端数值尺度下，statgpu 会使用更稳定的投影与中心化计算；如果 float64 精度不足以可靠区分某个非零系数，会显式报错，而不是发布有限但不可靠的结果。

## 常见问题

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

## 相关文档

- [面板模型总览](README.md)
- [面板协方差](covariance.md)
- [面板拟合统计量](fit-statistics.md)

## 参考文献

- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). The MIT Press.
