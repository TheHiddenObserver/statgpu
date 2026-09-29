# RandomEffects

> 语言：中文  
> 最后更新：2026-09-17  
> 页面定位：面板模型文档  
> 切换：[English](../../en/panel/random-effects.md)

## 概览

`RandomEffects` 使用 Swamy–Arora 方法估计单向随机截距面板模型的方差分量，再通过可行 GLS 得到系数。与固定效应模型不同，它把个体专属效应建模为随机成分，而不是为每个个体设置一个不受约束的固定参数。

`cov_type` 只改变 GLS 拟合之后报告的标准误和检验，不会改变 Swamy–Arora 方差分量，也不会改变系数估计。

## 统计模型与识别

标准的单向随机效应模型可以写成

$$
y_{it}=x_{it}^{\top}\beta+a_i+\varepsilon_{it},
$$

其中 $x_{it}$ 可以包含常数项。个体效应 $a_i$ 是随机变量，而不是固定的干扰参数。经典误差分量解释通常假设

$$
E(a_i)=0,
\qquad
\operatorname{Var}(a_i)=\sigma_a^2,
\qquad
\operatorname{Var}(\varepsilon_{it})=\sigma_e^2.
$$

随机效应与固定效应最关键的区别在于：随机效应模型要求 $a_i$ 与解释变量正交。一个常见的充分条件是

$$
E(a_i\mid X_i)=0,
\qquad
E(\varepsilon_{it}\mid X_i,a_i)=0,
$$

其中 $X_i=(x_{i1},\ldots,x_{iT_i})$。

经典单向误差分量结构还通常假设同一个体内不同时间的特质误差不相关，例如

$$
\operatorname{Cov}(\varepsilon_{it},\varepsilon_{is}\mid X_i)=0,
\qquad t\ne s,
$$

并要求随机效应与特质误差正交。

在这些条件下，个体内和个体间变异都可以用来估计同一个公共斜率 $\beta$。如果 $a_i$ 与解释变量历史存在系统性关系，随机效应 GLS 在数值上仍然可以计算，但其系数一般不再保证识别与固定效应估计量相同的结构参数。这也是经典 Hausman 比较背后的核心区别。

## Swamy–Arora 估计

Swamy–Arora 首先估计

$$
\widehat\sigma_e^2
=
\frac{RSS_W}{df_W},
\qquad
\bar T_H
=
\frac{N}{\sum_{i=1}^N T_i^{-1}},
$$

再计算

$$
\widehat\sigma_a^2
=
\max\left\{
0,
\frac{RSS_B/df_B-\widehat\sigma_e^2}{\bar T_H}
\right\}.
$$

其中 $df_W=n-r_W-r_E$。当水平设计矩阵中有显式常数列时 $r_E=N$，否则 $r_E=N-1$；同时 $df_B=N-r_B$。如果辅助回归中存在重复或线性依赖的列，自由度按实际数值秩计算，避免把同一个有效方向重复计数。

对个体 $i$，定义

$$
\theta_i
=
1-
\sqrt{
\frac{\widehat\sigma_e^2}
{\widehat\sigma_e^2+T_i\widehat\sigma_a^2}
},
$$

并进行准去均值变换：

$$
y_{it}^*=y_{it}-\theta_i\bar y_i,
\qquad
x_{it}^*=x_{it}-\theta_i\bar x_i.
$$

随后在 $(y^*,X^*)$ 上进行可行 GLS：

$$
\widehat\beta_{\mathrm{RE}}
=
\arg\min_\beta\|y^*-X^*\beta\|_2^2
=
(X^{*\top}X^*)^+X^{*\top}y^*.
$$

准去均值会从每个观测中减去一部分个体均值，减去多少由估计得到的个体内/个体间方差分量以及该个体的观测数共同决定。

## 协方差与推断

标准误基于真正用于 GLS 的准去均值数据 $(y^*,X^*)$ 计算。经典协方差为

$$
\widehat V_{\mathrm{nonrobust}}
=
\widehat\sigma_*^2
(X^{*\top}X^*)^+,
\qquad
\widehat\sigma_*^2
=
\frac{e^{*\top}e^*}
{n-\operatorname{rank}(X^*)},
$$

其中

$$
e^*=y^*-X^*\widehat\beta_{\mathrm{RE}}.
$$

HC0–HC3、聚类稳健以及 Driscoll–Kraay 协方差也都基于同一个变换后回归计算；完整公式见 [面板模型协方差](covariance.md)。

稳健协方差可以放宽**GLS 变换完成后不确定性估计**的部分假设，但不会改变 Swamy–Arora 变换本身，也不能消除随机效应解释对 $E(a_i\mid X_i)=0$ 的要求。如果这一条件均值正交性失效，仅仅换用稳健标准误并不会自动修复系数识别问题。

## 参数

| 参数 | 默认值 | 可选值 / 约束 | 含义 |
|---|---:|---|---|
| `cov_type` | `"nonrobust"` | `nonrobust`、`robust`/`hc1`、`hc0`、`hc2`、`hc3`、`clustered`、`driscoll-kraay`/`dk`/`kernel` | GLS 变换完成后如何计算标准误 |
| `alpha` | `0.05` | 有限且严格位于 `(0,1)` | 置信区间显著性水平；`0.05` 对应 95% 区间 |
| `device` | `"auto"` | `auto` / `cpu` / `cuda` / `torch` | 数值计算的执行后端与设备 |
| `n_jobs` | `None` | 整数或 `None` | 共享并行参数 |
| `bandwidth` | `None` | `None` 或非负整数；仅 DK 使用 | Driscoll–Kraay 平滑带宽 |
| `kernel` | `"bartlett"` | Bartlett/Newey–West、Parzen/Gallant、QS/Quadratic-Spectral/Andrews 等别名 | Driscoll–Kraay 核函数 |
| `group_debias` | `False` | 布尔值；仅聚类稳健协方差使用 | 是否应用小聚类数修正 |

拟合接口：

```python
model.fit(X, y, entity_ids=entity_ids, time_ids=None, cluster=None)
```

- `entity_ids` 必需，因为方差分量估计和准去均值都按个体进行；
- Driscoll–Kraay 还需要 `time_ids`；
- 聚类稳健协方差需要 `cluster`。

## CPU 与 GPU 示例

```python
from statgpu.panel import RandomEffects

cpu = RandomEffects(device="cpu").fit(X, y, entity_ids=entity_ids)
cuda = RandomEffects(device="cuda").fit(X, y, entity_ids=entity_ids)
torch = RandomEffects(device="torch").fit(X, y, entity_ids=entity_ids)
```

如果显式请求的 GPU 后端不可用，`.fit()` 会直接报错，而不会切换到 CPU。

## 公式接口

假设 `df` 包含 `y`、`x1`、`x2` 与 `entity`：

```python
from statgpu.panel import RandomEffects

with_intercept = RandomEffects().fit(
    formula="y ~ x1 + x2 | entity",
    data=df,
)

without_intercept = RandomEffects().fit(
    formula="y ~ 0 + x1 + x2 | entity",
    data=df,
)
```

管道符（`|`）后的第一个变量表示个体分组列。只有当 `cov_type="driscoll-kraay"` 时才接受第二个管道变量，并把它作为 `time_ids`；其他协方差类型下会明确报错，而不是静默忽略该变量。

如果同时显式传入 `entity_ids` / `time_ids`，它们必须与公式中管道部分指定的对应列一致。`RandomEffects` 会拒绝固定效应专用标记 `EntityEffects`、`TimeEffects` 和 `FixedEffects`；分组元数据应通过管道语法提供。

## 输出

常用结果包括：

- `coef_`、`bse_`、`tvalues_`、`pvalues_`、`conf_int_`；
- `theta_`；
- `variance_components_`；
- `fit_statistics_`；
- `nobs`、`df_resid`。

`variance_components_` 保存 $\widehat\sigma_e^2$ 与 $\widehat\sigma_a^2$；`theta_` 是拟合中各个体准去均值因子按个体数量加权后的平均值。

## 数值行为与失败语义

改变 `cov_type` 不会重新拟合随机效应模型：方差分量与系数保持不变，只改变报告的不确定性。

用户可以依赖以下公开行为：

- 个体内和个体间辅助回归都必须具有正的残差自由度；如果个体数量不足以支持个体间回归，`.fit()` 会报错，而不会人为构造不可靠的方差分量；
- 极端尺度下，如果 float64 无法可靠表示个体内/个体间残差方差或准去均值结果，会抛出 `FloatingPointError`，而不会返回有限但错误的 GLS 系数；
- 如果变换后的设计矩阵精确秩亏，拟合值仍可能有意义，但系数向量不唯一。此时不会继续发布依赖唯一系数表示的标准误、检验、p 值与置信区间；
- 不合法的协方差输入或不可用的显式 GPU 后端会直接报错；
- 每次新的拟合尝试都会先失效旧的已拟合/推断状态，失败后对象不会保留看起来像成功结果的部分状态。

这些边界描述的是用户可观察的数值行为，不要求应用代码依赖内部使用哪一种 SVD、缩放、归约或验证脚本。

## Hausman 比较

经典 Hausman 比较只在 [面板诊断](diagnostics.md) 说明的条件下可用。它检验的核心正是随机效应模型中“个体效应与解释变量不相关”这一额外假设是否与固定效应估计结果相容。

## 常见问题

**`cov_type` 会改变 Swamy–Arora 系数估计吗？**  
不会；它只改变 GLS 拟合后的标准误与相关推断。

**为什么 $\widehat\sigma_a^2$ 可能等于 0？**  
有限样本下未经截断的 Swamy–Arora 估计可能为负；由于方差不能为负，statgpu 会把该估计截断为 0。

**稳健标准误能解决随机效应模型的内生性吗？**  
不能。稳健协方差只改变不确定性估计，无法替代 $E(a_i\mid X_i)=0$ 这样的识别条件。

## 相关文档

- [面板模型总览](../models/panel.md) — Panel 模型选择与解释
- [PanelOLS](panel-ols.md) — 固定效应估计
- [面板模型协方差](covariance.md) — HC、cluster 与 Driscoll–Kraay
- [面板诊断](diagnostics.md) — Hausman 等检验
- [设备与 GPU 内存](../guides/device-and-memory.md) — 后端与设备语义

## 参考文献

- Swamy, P. A. V. B., & Arora, S. S. (1972). The exact finite sample properties of the estimators of coefficients in the error components regression models. *Econometrica*, 40(2), 261-275. [https://doi.org/10.2307/1909405](https://doi.org/10.2307/1909405)
- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). The MIT Press.
