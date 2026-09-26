# BetweenOLS

> 语言：中文  
> 最后更新：2026-09-17  
> 切换：[English](../../en/panel/between-ols.md)

## 概述

`BetweenOLS` 先在每个个体内部对响应和解释变量取时间均值，把面板数据压缩为“每个个体一行”，再对这些个体均值进行带截距的 OLS 回归。它关注的是**不同个体长期平均水平之间的关系**，而不是同一个体随时间的变化。

因此，最终回归的有效样本量是进入均值回归的个体数量，而不是原始面板数据的总行数。

## 统计模型与识别

可以从单向个体效应模型出发：

$$
y_{it}=\alpha+x_{it}^{\top}\beta+a_i+\varepsilon_{it}.
$$

在个体内部对时间取平均后得到

$$
\bar y_i=\alpha+\bar x_i^{\top}\beta+a_i+\bar\varepsilon_i.
$$

如果把 $a_i$ 视为不受限制的固定参数，取时间均值并不会消除它。因此，仅由固定效应模型本身，不能推出 BetweenOLS 的斜率一定等于原始结构参数 $\beta$。

若希望把个体间回归的斜率解释为基础模型中的同一个 $\beta$，还需要额外的跨个体正交条件。例如，一个常见的充分条件是

$$
E\!\left(a_i+\bar\varepsilon_i\mid \bar x_i\right)=0.
$$

如果只关心线性投影参数，也可以直接对个体均值施加相应的矩条件。若长期个体异质性与平均解释变量相关，`BetweenOLS` 仍然定义了 $\bar y_i$ 对 $\bar x_i$ 的横截面线性投影，但该投影通常不会与固定效应或一阶差分估计的斜率相同。

## 估计量

对个体 $i$，定义

$$
\bar y_i=T_i^{-1}\sum_t y_{it},
\qquad
\bar x_i=T_i^{-1}\sum_t x_{it}.
$$

令 $\bar Z_i=(1,\bar x_i^\top)^\top$，则

$$
\widehat\beta_B
=\arg\min_\beta\sum_i(\bar y_i-\bar Z_i^\top\beta)^2
=(\bar Z^\top\bar Z)^+\bar Z^\top\bar y.
$$

也就是说，原始面板数据最终被转换为一个以个体均值为观测单位的普通横截面 OLS 问题。

## 协方差与统计推断

标准误直接基于个体均值回归计算。当前支持：

- `cov_type="nonrobust"`：通常的同方差 OLS 协方差；
- `robust` / `hc1`：HC1 异方差稳健协方差；
- `hc0`、`hc2`、`hc3`：相应的 HC 稳健协方差。

这些选项只改变个体均值回归的不确定性估计，不能替代把斜率解释为基础面板模型结构参数时所需的识别条件。统一公式见 [面板协方差](covariance.md)。

如果个体均值设计矩阵精确秩亏，拟合值仍可能唯一，但系数向量不唯一。此时 statgpu 保留可解释的拟合结果，同时不发布系数级标准误、检验、p 值和置信区间。

## 参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `cov_type` | `"nonrobust"` | `nonrobust`、`robust`/`hc1`、`hc0`、`hc2`、`hc3` |
| `alpha` | `0.05` | 置信区间显著性水平；`0.05` 对应 95% 区间 |
| `device` | `"auto"` | `auto`、`cpu`、`cuda`、`torch` |
| `n_jobs` | `None` | 适用路径的并行任务数 |

`entity_ids` 是必需输入，因为模型需要先按个体计算时间均值：

```python
model.fit(X, y, entity_ids=entity_ids)
```

## 使用示例

### 数组接口

```python
from statgpu.panel import BetweenOLS

cpu = BetweenOLS(device="cpu").fit(
    X, y, entity_ids=entity_ids
)

cuda = BetweenOLS(device="cuda").fit(
    X, y, entity_ids=entity_ids
)
```

显式 `device="cuda"` 或 `device="torch"` 请求要求对应 GPU 后端实际可用；不可用时直接报错，不会自动改用 CPU。

### 公式接口

```python
from statgpu.panel import BetweenOLS

model = BetweenOLS().fit(
    formula="y ~ x1 + x2",
    data=df,
    entity_ids=df["entity"],
)
```

`BetweenOLS` 始终包含截距，因此显式无截距公式会被拒绝。

## 输出与解释

常用结果包括：

- `coef_`：个体均值回归的系数；
- `bse_`、`tvalues_`、`pvalues_`、`conf_int_`：可用时的系数推断；
- `rsquared`、`fit_statistics_`：拟合优度统计量；
- `nobs`：最终进入回归的个体均值观测数；
- `df_resid`：残差自由度。

数值实现会对极端尺度和抵消情形采用稳定的线性代数路径，但这些数值保护不会改变上面的统计目标或协方差定义。

## 常见问题

**为什么 `nobs` 小于原始数据行数？**  
因为最终回归以“个体均值”为观测单位，每个保留个体只贡献一行。

**时间观测更多的个体会自动获得更高权重吗？**  
不会。当前 BetweenOLS 的最终 OLS 中，每个保留个体贡献一个均值观测。

**BetweenOLS 与固定效应回归可以互换吗？**  
不能。两者利用的数据变异来源和识别条件不同：BetweenOLS 使用个体间长期平均差异，固定效应回归使用去除固定效应后的个体内变化。

## 相关文档

- [面板模型总览](../models/panel.md)
- [面板协方差](covariance.md)
- [面板拟合统计量](fit-statistics.md)

## 参考文献

- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). The MIT Press.
