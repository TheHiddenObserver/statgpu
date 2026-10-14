# PanelOLS

> 语言：中文  
> 最后更新：2026-10-13  
> 切换：[English](../../en/panel/panel-ols.md)

<a id="概述"></a>

## 什么时候使用固定效应

当你想研究解释变量的变化与结果变化之间的关系，同时控制个体之间长期稳定的差异、某一时期的共同冲击，或同时控制两者时，可以使用 `PanelOLS`。例如，个体固定效应让你在剔除企业稳定的基准差异后，比较同一家企业的投入与产出变化。

个体效应去除个体特有的水平差异，因此个体内部从不变化的变量无法单独识别斜率。时间效应去除所有个体共享的时期水平差异；双向效应同时控制这两类差异。

固定效应可以与解释变量历史相关，但不能消除随时间变化的混杂因素。静态模型的通常解释仍要求：给定解释变量历史和纳入的效应后，个体特异误差的条件均值为零。斜率只能由去除所有指定效应后剩余的变化识别。

模型写作 $y_{it}=x_{it}^{\top}\beta+a_i+\gamma_t+\varepsilon_{it}$，未选择的效应项省略。其中 $i$ 表示个体，$t$ 表示时期，$y_{it}$ 是响应，$x_{it}$ 是该行的解释变量向量，$\beta$ 是公共斜率向量。个体水平 $a_i$ 和时期水平 $\gamma_t$ 是固定但未知的参数；$\varepsilon_{it}$ 是剩余的观测层面误差。

下面的示例只使用个体效应，对每个个体减去其已观测时期的均值后，得到 $y_{it}-\bar y_i=(x_{it}-\bar x_i)^{\top}\beta+(\varepsilon_{it}-\bar\varepsilon_i)$。对这些去均值后的观测做最小二乘即可估计斜率；个体水平以及其中包含的公共截距被消除。横线表示该个体已观测时期的平均值。

## 先准备面板数据

使用长表形式：每行是一个个体在一个时期的观测，而不是把各时期展开成多列。

- `X` 的形状为 `(n_observations, n_features)`，`y` 为 `(n_observations,)`。所有数组的行必须一一对应。
- `entity_ids` 标识每行属于哪个个体，`time_ids` 标识所属时期；每个标签数组与观测数等长。它们用于分组，不是额外的解释变量。
- 设置 `entity_effects=True` 时传入 `entity_ids`，设置 `time_effects=True` 时传入 `time_ids`；双向效应需同时开启两项并提供两类标签。两项默认均为 `False`。数组接口中，仅提供标签并不会自动开启相应效应。
- 数组接口不会自动加入截距。有固定效应时，不要额外加入常数列，因为它已被效应吸收；无固定效应而又需要截距时，应自行加入常数列。

下面的 `X.shape == (60, 1)`：12 个个体，各观测 5 个时期。个体效应与 `x` 的个体间水平差异相关，但个体内部的变化仍可用于识别斜率。

<a id="使用示例"></a>

## 拟合一个个体固定效应模型

<a id="数组接口"></a>

<a id="cpu-example"></a>

<!-- example: cpu-basic -->
```python
import numpy as np
from statgpu.panel import PanelOLS

rng = np.random.default_rng(7)
entity_ids = np.repeat(np.arange(12), 5)
time_ids = np.tile(np.arange(5), 12)
x = np.repeat(np.linspace(-2, 2, 12), 5) + rng.normal(size=60)
X = x[:, None]
entity_effect = np.repeat(np.linspace(-1, 1, 12), 5)
y = 1.0 + 2.0 * x + entity_effect + 0.2 * rng.normal(size=60)

model = PanelOLS(entity_effects=True, device="cpu").fit(
    X, y, entity_ids=entity_ids
)
print("slope:", np.round(model.coef_, 3))
print("nobs, df_resid:", model.nobs, model.df_resid)
```
<!-- example-end: cpu-basic -->

`coef_` 只包含一个斜率，接近生成数据时设定的 2。公共截距和各个体的水平已被固定效应吸收，因此该系数数组不含单独的截距。`nobs=60` 保留原始观测行数；`df_resid=47` 则扣除了 12 个个体效应方向和一个斜率。

在 GPU 上运行时，把 `device="cpu"` 换成 `device="cuda"`（CuPy/CUDA）或 `device="torch"`（Torch CUDA）。显式请求的后端不可用时会报错，不会切换到 CPU。

<a id="输出与解释"></a>

## 读懂结果并进行预测

`coef_` 与 `X` 的列顺序一致；示例中的系数是个体内斜率。在模型假设成立时，同一个体的 `x` 增加一个单位，对应 `y` 的估计变化约为 2。各固定效应的水平不是 `coef_` 中的独立系数项。

`bse_` 按 `coef_` 的顺序衡量各系数的不确定性，并不是单次预测的波动大小。`conf_int_` 每行给出一个系数的区间下限与上限；`alpha=0.05` 对应 95% 区间。`tvalues_` 是系数与标准误之比：默认 `nonrobust` 拟合使用自由度为 `df_resid` 的 Student-t 参考分布，其他支持的协方差类型使用正态参考分布计算 p 值与区间。比较推断结果前，请核对[面板协方差](covariance.md)中的约定。

`nobs` 计数原始观测行数。`df_resid` 还会扣除固定效应占用的维度，因此不能只用行数减去可见的斜率个数。`rsquared_within` 衡量变换后数据上的拟合优度；`fit_statistics_` 提供标准化的组内、组间、整体 $R^2$ 及可用的联合检验统计量，定义见[拟合统计量](fit-statistics.md)。`summary()` 打印并返回面板回归摘要。

### 预测时是否加入固定效应

`model.predict(X_new)` 只返回线性部分 $X_{new}\widehat\beta$。要在示例中加入拟合的个体水平，应调用 `model.predict(X_new, entity_ids=new_entity_ids)`。预测特征的列及顺序应与拟合一致；返回值是 NumPy 数组。

单向模型中，提供已知标签会加入对应效应；未提供或从未见过的标签不加入效应。双向模型若要加入效应，须同时提供 `entity_ids` 和 `time_ids`，且每行的两个标签都已知、属于拟合时个体—时间关联结构的同一连通分量。已知标签与未知或未提供的标签配对，或两个标签属于不同连通分量时，都会报错。如果两个标签均未知（包括两个数组都不传），预测只使用线性部分；这不等于估计出了缺失个体或时期的效应。

可以把个体和时期分别看作两类节点，每条训练观测把所属个体与时期连起来；沿这些连线能够互相到达的标签属于同一连通分量。例如，观测 `(A, 1)`、`(A, 2)`、`(B, 1)`、`(B, 2)` 把企业 A/B 与时期 1/2 连在一起；另一组 `(C, 3)`、`(C, 4)`、`(D, 3)`、`(D, 4)` 把 C/D 与 3/4 连在一起。若没有观测连接这两组数据，即使 A 和 3 都出现过，预测 `(A, 3)` 并加入效应仍会报错，因为训练数据无法确定两组固定效应的相对水平。某个配对本身不必在训练中出现过；在满足上述规则的前提下，两个已知标签属于同一分量即可。

公式拟合后，`predict(new_df)` 会按公式重建设计矩阵，但用于加入固定效应的预测标签仍需显式传入。

## 选择固定效应与标准误

使用 `entity_effects=True` 比较个体内部的变化，使用 `time_effects=True` 控制共同的时期冲击；同时开启两项即可纳入双向效应。两项默认均为 `False`，都不开启时是普通水平回归；如果需要自动加入截距的合并模型，可使用 [PooledOLS](pooled-ols.md)。

先根据希望进行的比较选择固定效应，再根据剩余误差的结构选择标准误：

- 默认 `cov_type="nonrobust"` 使用经典协方差。也支持 HC0–HC3，但在固定短 $T>2$ 的个体效应模型中，即使原始误差没有序列相关，直接用于组内回归的普通 HC0/HC1 仍可能不一致。HC2/HC3 也不是通用补救，详见 [HC 适用条件](covariance.md#nonrobust-and-hc-covariance)。
- 若误差可能在个体内相关，可设置 `cov_type="clustered"`，并在 `fit()` 中传入 `cluster=entity_ids`。固定效应标签不会自动决定聚类层级。`group_debias=True` 是可选的小簇数修正，仅适用于聚类协方差。
- 使用 Driscoll–Kraay 时设置 `cov_type="driscoll-kraay"` 并传入 `time_ids`，即使模型没有时间固定效应也需要时间标签。`bandwidth` 与 `kernel` 控制时间平滑。此模型不支持 `PooledOLS` 的独立 `hac` 选项。

协方差设置改变标准误及相关推断，不改变去除哪些固定效应。假设、多维聚类与带宽规则见[面板协方差](covariance.md)。[Pooling F 与 Hausman 检验](diagnostics.md)可辅助比较模型，但不能替代外生性假设的实质判断。

## 公式接口

假设 `df` 已包含下面公式及分组参数中使用的列。

```python
from statgpu.panel import PanelOLS

# 双向固定效应：pipe 语法
two_way = PanelOLS().fit(
    formula="y ~ x1 + x2 | entity + time",
    data=df,
)

# 等价的固定效应标记写法
two_way_tokens = PanelOLS().fit(
    formula="y ~ x1 + x2 + EntityEffects + TimeEffects",
    data=df,
)

# 无固定效应、无截距的水平回归
level_no_intercept = PanelOLS().fit(
    formula="y ~ 0 + x1 + x2",
    data=df,
)
```

固定效应可以使用 pipe 语法，也可以使用 `EntityEffects` / `TimeEffects` 等固定效应标记，但同一个公式中不能混用。如果同时从公式和显式参数提供个体/时间标签，两者必须在公式筛选后的样本上完全一致。

<a id="预测与数值边界"></a>

## 解释结果前检查这些边界

双向去均值必须达到 `demean_tol`；若在 `demean_max_iter` 内未收敛，`fit()` 会报错。先检查个体/时间标签及数据结构，再决定是否调整迭代上限或容差，不应把未收敛的变换当作已拟合模型。

如果变换后的解释变量精确共线，拟合值仍可能唯一，但系数向量不唯一。此时系数级标准误、检验、p 值与置信区间不可用，应先处理被吸收或重复的变量；见[面板协方差](covariance.md)。

公式接口不接受混用两种固定效应语法、在 pipe 中指定超过两个固定效应变量，或固定效应模型中没有任何非截距解释变量。

## 常见问题

**为什么双向去均值不能在未收敛时直接返回最后一次迭代？**  
因为未充分去均值会改变实际拟合的回归问题。只有达到 `demean_tol` 后结果才会发布。

**使用稳健协方差后，`fit_statistics_.f_statistic` 会自动变成稳健 Wald 检验吗？**  
不会。该字段仍表示经典的斜率联合检验；见 [面板拟合统计量](fit-statistics.md)。

**固定效应会改变斜率的识别来源吗？**  
会。系数只由去除相应固定效应后仍存在的解释变量变化识别。

## 统计模型与识别

含个体和时间固定效应的经典模型可以写为

$$
y_{it}=x_{it}^{\top}\beta+a_i+\gamma_t+\varepsilon_{it}.
$$

其中 $a_i$ 与 $\gamma_t$ 是固定但未知的干扰参数，不要求来自某个概率分布。对静态线性面板模型，一个常见的充分外生性条件是

$$
E\!\left(\varepsilon_{it}\mid X_i,a_i,\gamma_1,\ldots,\gamma_T\right)=0.
$$

这里 $X_i=(x_{i1},\ldots,x_{iT_i})$ 表示个体的解释变量历史；期望以纳入的固定效应为条件，模型不要求这些效应与 $X_i$ 正交。因此，核心限制落在个体特异误差 $\varepsilon_{it}$ 上，而不是要求固定效应与解释变量独立。

$\beta$ 只能由去除固定效应后仍然存在的解释变量变化识别。例如，仅包含个体固定效应时，一个在个体内部完全不随时间变化的解释变量会被固定效应吸收，无法单独识别其斜率。

当 `entity_effects=False` 且 `time_effects=False` 时，`PanelOLS` 退化为普通水平回归，此时不再使用上述固定效应解释。

## 估计量

statgpu 直接进行去均值等变换，不构造庞大的虚拟变量矩阵；固定效应模型的统计含义不变。

令 $F$ 表示纳入模型的固定效应设计矩阵，并定义

$$
M_F=I-F(F^\top F)^+F^\top.
$$

则

$$
\widehat\beta_{\mathrm{FE}}
=\arg\min_\beta\|M_F(y-X\beta)\|_2^2
=(X^\top M_FX)^+X^\top M_Fy.
$$

仅有个体固定效应时，这就是通常的个体内去均值。双向固定效应下，statgpu 交替去除个体均值和时间均值，直到满足 `demean_tol`。如果在 `demean_max_iter` 次迭代内仍未收敛，`.fit()` 会直接报错，而不会返回尚未充分去均值的近似结果。

## 协方差与统计推断

标准误使用与系数估计完全相同的变换后解释变量和残差。记

$$
Z=M_FX,
\qquad
e=M_F(y-X\widehat\beta_{\mathrm{FE}}),
$$

则 `nonrobust`、HC、聚类稳健和 Driscoll–Kraay 协方差的统一定义见 [面板协方差](covariance.md)。

Driscoll–Kraay 和残差自由度都需要计入固定效应占用的维度。固定效应秩为

$$
r_F=\begin{cases}
N,&\text{仅个体固定效应},\\
T,&\text{仅时间固定效应},\\
N+T-C,&\text{双向固定效应},
\end{cases}
$$

其中 $C$ 是已观测个体—时间关联图的连通分量数。残差自由度使用

$$
df_{\mathrm{resid}}
=n-\operatorname{rank}(Z)-r_F.
$$

这一自由度同时用于公开的 `df_resid`、同方差残差方差尺度、HC1 修正以及 nonrobust 的 Student-t 推断；它与显式加入固定效应虚拟变量回归的残差自由度一致，不是在变换后简单扣除 `N-1` 或 `T-1`。

如果去除固定效应后的设计矩阵精确秩亏，拟合值仍可能唯一，但系数向量不唯一。此时 statgpu 保留拟合结果，同时不发布系数级标准误、检验、p 值和置信区间。

## 参数

| 参数 | 默认值 | 可选值 / 约束 | 含义 |
|---|---:|---|---|
| `entity_effects` | `False` | 布尔值 | 是否去除个体固定效应 |
| `time_effects` | `False` | 布尔值 | 是否去除时间固定效应 |
| `cov_type` | `"nonrobust"` | `nonrobust`、`robust`/`hc1`、`hc0`、`hc2`、`hc3`、`clustered`、`driscoll-kraay`/`dk`/`kernel` | 标准误的计算方式 |
| `alpha` | `0.05` | 有限且严格位于 `(0,1)` | 置信区间显著性水平；`0.05` 对应 95% 区间 |
| `device` | `"auto"` | `auto`、`cpu`、`cuda`、`torch` | 执行后端与设备 |
| `n_jobs` | `None` | 整数或 `None` | 共享并行参数 |
| `bandwidth` | `None` | `None` 或非负整数；仅 DK 使用 | Driscoll–Kraay 带宽；`None` 使用自动规则 |
| `kernel` | `"bartlett"` | Bartlett/Newey–West、Parzen/Gallant、QS/Quadratic-Spectral/Andrews 等别名 | Driscoll–Kraay 核函数 |
| `group_debias` | `False` | 布尔值；仅聚类协方差使用 | 是否应用小簇数修正 |
| `demean_max_iter` | `1_000_000` | 正整数 | 双向去均值的最大迭代次数 |
| `demean_tol` | `1e-10` | 有限正数 | 双向去均值的收敛容差 |

### 拟合与其他方法

```python
model.fit(X=None, y=None, entity_ids=None, time_ids=None,
          cluster=None, formula=None, data=None)
```

可传入数组，或配合 `data` 使用 `formula`；`fit()` 返回已拟合的估计器。标签数组与观测一一对应；`cluster` 提供[面板协方差](covariance.md)说明的一维或二维聚类分组。公式旁路元数据需要与保留观测对齐，见[本页公式接口](#公式接口)。

- `predict(X, entity_ids=None, time_ids=None)`：线性预测，并按上文规则加入已提供且可识别的固定效应。
- `summary()`：打印并返回结构化面板摘要。
- `pooling_f_test()`：比较纳入的固定效应与合并模型。
- `hausman_test(random_effects_model)`：与兼容的随机效应拟合比较；见[适用条件](diagnostics.md)。
- `get_params(deep=True)` / `set_params(**params)`：读取或修改构造参数。`set_params()` 会清除拟合结果，之后须重新拟合才能预测或读取摘要。

拟合属性：`coef_`、`bse_`、`tvalues_`、`pvalues_`、`conf_int_`、`rsquared_within`、`fit_statistics_`、`nobs`、`df_resid`。变换后设计精确秩亏时，系数推断数组不可用。

继承的 `adjust_pvalues`、`combine_pvalues`、`bootstrap_statistic` 和 `permutation_test` 的完整签名见[共享估计器 API](../reference/estimator-api.md)。这些工具不会自动选择适用于面板依赖结构的重采样方案。

<a id="external-validation"></a>

<a id="comparing-with-other-packages"></a>

## 与其他软件包比较

与 linearmodels 比较单向或双向固定效应时，应对齐样本、纳入的固定效应、变换后设计矩阵，以及被吸收效应占用的自由度。与 R plm 比较单向固定效应时也需要这些设置一致。Driscoll–Kraay 还需对齐时间顺序、核和带宽。无固定效应时，可与 statsmodels 等普通 OLS 实现比较，但截距设定必须相同。详见[协方差定义的比较](covariance.md#comparing-covariance-definitions)。

## 相关文档

- [面板模型总览](README.md)
- [面板协方差](covariance.md)
- [面板诊断](diagnostics.md)
- [面板拟合统计量](fit-statistics.md)

## 参考文献

- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). The MIT Press.
