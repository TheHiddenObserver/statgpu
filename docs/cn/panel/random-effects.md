# RandomEffects

> 语言：中文  
> 最后更新：2026-10-13  
> 页面定位：面板模型文档  
> 切换：[English](../../en/panel/random-effects.md)

<a id="概览"></a>

## 什么时候适合随机截距

当你希望同时利用个体内部的变化与个体之间的差异估计公共斜率，并用随机截距描述个体的基准水平时，可以使用 `RandomEffects`。例如，它可以结合一家企业内部的变化与不同企业之间的比较。

先把解释变量未解释的部分分成两类：同一家企业各期共享的长期偏差 $a_i$，以及每条观测各自剩下的误差 $\varepsilon_{it}$（个体特异误差）。“解释变量历史”指这家企业所有时期的解释变量，不是额外传入的一组数据。

这里的核心假设比固定效应更强：给定这些解释变量后，随机效应 $a_i$ 的条件均值必须为零。例如，若投入长期较高的企业也恰好有持续较高、未记录的管理能力，这个条件就可能不成立。此外，给定这些解释变量和 $a_i$ 后，各期 $\varepsilon_{it}$ 的条件均值也必须为零。经典误差分量结构还假设各分量方差恒定、特异误差在个体内不随时间相关，且两个误差分量正交。下文用公式给出精确条件。

如果个体的稳定差异与解释变量存在系统性联系，可考虑带个体效应的 [PanelOLS](panel-ols.md)。稳健标准误不能消除这一识别问题。`RandomEffects` 拟合的是每个个体的一个随机截距，不包含时间随机效应或随机斜率。

模型根据估计的两类误差方差，决定从每行减去多少个体均值。拟合先用 Swamy–Arora 估计方差分量，再减去一部分个体均值（准去均值），最后用可行 GLS 估计系数。

模型为 $y_{it}=x_{it}^{\top}\beta+a_i+\varepsilon_{it}$，其中 $i$ 表示个体，$t$ 表示时期。$y_{it}$ 是响应，$x_{it}$ 是该行的解释变量向量（若传入常数列，也包含该列），$\beta$ 包含公共系数。随机截距 $a_i$ 由个体 $i$ 的所有观测共享，$\varepsilon_{it}$ 则是每条观测各自的误差。

准去均值构造 $y_{it}^{*}=y_{it}-\theta_i\bar y_i$ 和 $x_{it}^{*}=x_{it}-\theta_i\bar x_i$，再对这些变换后的观测做最小二乘。横线表示该个体已观测时期的均值；$\theta_i$ 根据估计的方差分量和该个体的观测次数，决定减去多少均值。

## 准备个体分组与设计矩阵

使用长表，每行是一个个体在一个时期的观测：

- `X` 的形状为 `(n_observations, n_features)`，`y` 为 `(n_observations,)`；所有数组按行对齐。
- `entity_ids` 必须提供，每行一个标签，用于确定准去均值及方差分量估计中的个体分组。
- 数组接口不自动添加常数。需要公共截距时，请在 `X` 中加入一列 1。
- Driscoll–Kraay 协方差需要 `time_ids`，聚类协方差需要 `cluster`。提供这些标签不会额外加入随机效应。

示例使用 12 个个体、5 个时期；个体效应独立于 `x` 生成，符合随机效应的正交性假设。`X.shape == (60, 2)` 包含显式常数列和一个解释变量。

<a id="cpu-与-gpu-示例"></a>

## 拟合随机截距模型

<a id="cpu-example"></a>

<!-- example: cpu-basic -->
```python
import numpy as np
from statgpu.panel import RandomEffects

rng = np.random.default_rng(7)
entity_ids = np.repeat(np.arange(12), 5)
time_ids = np.tile(np.arange(5), 12)
x = rng.normal(size=60)
X = np.column_stack([np.ones(60), x])
entity_effect = rng.normal(scale=0.6, size=12)
y = 1.0 + 2.0 * x + entity_effect[entity_ids] + 0.2 * rng.normal(size=60)

model = RandomEffects(device="cpu").fit(X, y, entity_ids=entity_ids)
print("coef:", np.round(model.coef_, 3))
print("theta:", round(float(model.theta_), 3))
print("nobs:", model.nobs)
```
<!-- example-end: cpu-basic -->

`coef_` 与 `X` 的列顺序一致：先截距、后斜率，斜率接近生成数据时设定的 2。由于各个体观测数相同，这里的 `theta_` 就是共同的准去均值因子；介于 0 与 1 的值表示减去一部分个体均值。`nobs=60` 仍是原始观测行数。这个小型模拟用于说明输出，不是对随机效应假设的检验。

在 GPU 上运行时，把 `device="cpu"` 换成 `device="cuda"`（CuPy/CUDA）或 `device="torch"`（Torch CUDA）。显式请求的后端不可用时会报错，不会切换到 CPU。

<a id="输出"></a>

## 读懂系数与方差分量

`coef_` 按原始水平设计矩阵的列顺序排列。示例中第一个系数是公共截距，第二个是结合个体内与个体间变化估计的公共斜率。

`bse_` 按 `coef_` 的顺序衡量各系数的不确定性，并不是单次预测的波动大小。`conf_int_` 每行给出一个系数的区间下限与上限；`alpha=0.05` 对应 95% 区间。`tvalues_` 是系数与标准误之比：默认 `nonrobust` 拟合使用自由度为 `df_resid` 的 Student-t 参考分布，其他支持的协方差类型使用正态参考分布计算 p 值与区间。比较推断结果前，请核对[面板协方差](covariance.md)中的约定。

`variance_components_` 是字典：`sigma2_e` 为个体特异误差方差，`sigma2_a` 为随机截距方差。`theta_` 概括减去了多少个体均值。非平衡面板中，拟合会使用各个体自己的因子；`theta_` 是这些因子在个体之间的平均值，不是对所有行统一使用的因子。等价地，相同观测数对应的因子按该类个体的数量加权。

`nobs` 计数原始行数，`df_resid` 是最终变换后回归的残差自由度，`fit_statistics_` 包含标准化的拟合统计量，见[拟合统计量](fit-statistics.md)。`summary()` 打印并返回结构化面板摘要。

### 预测结果的公共部分

`model.predict(X_new)` 返回原始水平设计矩阵与 `coef_` 的乘积，类型为 NumPy 数组。它不是准去均值后的结果，也不会估计或加入个体专属的随机截距。`predict()` 不接受个体标签。

使用与拟合时相同的列及顺序，包括示例中显式加入的常数列。公式拟合后，也可传入包含所需解释变量的 DataFrame。只有在身份明确时，缩短一列的数组才允许省略拟合时的显式常数；其他列数不匹配会报错。数组预测时保留完整的训练设计最为清楚。

## 选择标准误并重新拟合

更换 `cov_type` 改变拟合后的不确定性，不改变方差分量与系数。

先确认上述随机效应假设，再选择不确定性模型。默认是 `cov_type="nonrobust"`，也可使用 HC0–HC3、聚类稳健与 Driscoll–Kraay。

- 若按个体聚类，设置 `cov_type="clustered"`，并在 `fit()` 中传入 `cluster=entity_ids`。仅提供 `entity_ids` 不会自动启用聚类标准误；`group_debias=True` 可选地加入小簇数修正。
- Driscoll–Kraay 需要 `time_ids`；通过 `bandwidth` 与 `kernel` 选择时间平滑。本模型没有独立的 `hac` 选项。
- 比较协方差选择时，须对相同数据及其余设置重新拟合。`set_params()` 会重置已拟合状态，直接给属性赋值也不会重算已有结果。

这些选项针对同一个准去均值 GLS 回归计算协方差，不会改变 Swamy–Arora 变换，也不能修复随机效应与解释变量之间的相关性。

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

<a id="数值行为与失败语义"></a>

## 检查数据是否支持这次拟合

个体内与个体间辅助回归的残差自由度都必须为正。尤其是，个体数必须大于个体间回归的识别秩；仅在少数个体上收集很多行数据并不足够。

最终变换后的设计若精确秩亏，拟合值仍可能可用，但系数向量不唯一；此时系数级标准误、检验、p 值与置信区间不可用。

辅助回归与最终 GLS 的数值保护会在极端尺度或抵消情形下保留原有 Swamy–Arora 与最小二乘定义。但如果 float64 无法可靠表示个体内/个体间残差方差贡献或准去均值数据，`fit()` 会抛出 `FloatingPointError`。输入有限、设计满秩并不保证计算结果一定可表示。

每次新的拟合尝试都会先清除旧状态；失败后不会保留可当作成功结果使用的部分输出。协方差输入不合法或显式请求的 GPU 后端不可用时也会报错。

## Hausman 比较

经典 Hausman 比较只在 [面板诊断](diagnostics.md) 说明的条件下可用。它检验的核心正是随机效应模型中“个体效应与解释变量不相关”这一额外假设是否与固定效应估计结果相容。

## 常见问题

**`cov_type` 会改变 Swamy–Arora 系数估计吗？**  
不会。在同一数据和其余设置下换用另一种协方差并重新拟合，只改变 GLS 拟合后的标准误与相关推断。

**为什么 $\widehat\sigma_a^2$ 可能等于 0？**  
有限样本下未经截断的 Swamy–Arora 估计可能为负；由于方差不能为负，statgpu 会把该估计截断为 0。

**稳健标准误能解决随机效应模型的内生性吗？**  
不能。稳健协方差只改变不确定性估计，无法替代 $E(a_i\mid X_i)=0$ 这样的识别条件。

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

### 拟合与其他方法

```python
model.fit(X=None, y=None, entity_ids=None, time_ids=None,
          formula=None, data=None, cluster=None)
```

可传入数组，或配合 `data` 使用 `formula`；`fit()` 返回已拟合的估计器。个体标签必需，且每行一个；DK 还需要 `time_ids`，聚类协方差需要[面板协方差](covariance.md)说明的一维或二维 `cluster` 分组。公式旁路元数据需要与保留观测对齐，见[本页公式接口](#公式接口)。

- `predict(X)`：返回公共水平预测，不加入个体专属随机截距。
- `summary()`：打印并返回结构化面板摘要。
- `hausman_test(fixed_effects_model)`：与兼容的固定效应拟合比较；见[面板诊断](diagnostics.md)。
- `get_params(deep=True)` / `set_params(**params)`：读取或修改构造参数；`set_params()` 清除拟合状态后须重新拟合。

拟合属性：`coef_`、`bse_`、`tvalues_`、`pvalues_`、`conf_int_`、`theta_`、`variance_components_`、`fit_statistics_`、`nobs`、`df_resid`。变换后设计精确秩亏时，系数推断数组不可用。

继承的 `adjust_pvalues`、`combine_pvalues`、`bootstrap_statistic` 和 `permutation_test` 的完整签名见[共享估计器 API](../reference/estimator-api.md)。这些工具不会自动选择适用于面板依赖结构的重采样方案。

<a id="external-validation"></a>

<a id="comparing-with-other-packages"></a>

## 与其他软件包比较

随机效应系数**不保证与其他软件包一致**，因为 statgpu 使用自己的 Swamy–Arora 方差分量构造。比较协方差时，应使用 statgpu 实际得到的准去均值回归 $(X^*,y^*)$：在对齐协方差设置和自由度后，可与 linearmodels 比较稳健及 Driscoll–Kraay 协方差，与 statsmodels 比较 HC2/HC3。

这样的比较只针对该变换后回归的不确定性，不能据此认定不同随机效应实现具有等价的系数估计。详见[协方差定义的比较](covariance.md#comparing-covariance-definitions)。

## 相关文档

- [面板模型总览](README.md) — Panel 模型选择与解释
- [PanelOLS](panel-ols.md) — 固定效应估计
- [面板模型协方差](covariance.md) — HC、cluster 与 Driscoll–Kraay
- [面板诊断](diagnostics.md) — Hausman 等检验
- [设备与 GPU 内存](../guides/device-and-memory.md) — 后端与设备语义

## 参考文献

- Swamy, P. A. V. B., & Arora, S. S. (1972). The exact finite sample properties of the estimators of coefficients in the error components regression models. *Econometrica*, 40(2), 261-275. [https://doi.org/10.2307/1909405](https://doi.org/10.2307/1909405)
- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). The MIT Press.
