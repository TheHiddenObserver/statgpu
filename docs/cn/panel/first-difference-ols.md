# FirstDifferenceOLS

> 语言：中文  
> 最后更新：2026-09-17  
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

此时每个 `(entity_id, time_id)` 组合必须唯一。数值和日期时间标签按自然顺序排列；有序 categorical 使用用户声明的类别顺序。若字符串的字典序不能表示真实时间顺序，应改用数值、日期时间或有序 categorical 标签。

## 使用示例

### 数组接口

```python
from statgpu.panel import FirstDifferenceOLS

model = FirstDifferenceOLS(device="cpu").fit(
    X,
    y,
    entity_ids=entity_ids,
    time_ids=time_ids,
)
```

GPU 路径可以通过 `device="cuda"` 或 `device="torch"` 显式请求；对应后端不可用时直接报错，不会切换到 CPU。

### 公式接口

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

## 相关文档

- [面板模型总览](../models/panel.md)
- [面板协方差](covariance.md)
- [面板拟合统计量](fit-statistics.md)

## 参考文献

- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). The MIT Press.
