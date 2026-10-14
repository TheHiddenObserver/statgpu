# 面板模型协方差估计

> 语言：中文  
> 最后更新：2026-10-13  
> 页面定位：面板模型协方差与系数推断参考  
> 切换：[English](../../en/panel/covariance.md)

## 概览

先选均值模型，再选协方差。协方差改变不确定性的估计，不会改变回归消除哪些遗漏效应，也不能保证解释变量外生。首次拟合从[模型指南](README.md)开始；本页用于选择协方差选项和查阅精确定义。

## 按误差相关结构选择

| 需要允许的误差结构 | 选项及适用模型 | 需要准备的信息 |
|---|---|---|
| 符合相应模型的经典同方差协方差假设 | 下表五类残差 OLS 模型的 `nonrobust` | 模型本身所需的个体或时间标签 |
| 异方差，不额外允许任意行间相关 | 这五类模型的 `robust`/`hc1`、`hc0`、`hc2`、`hc3` | 无额外协方差标签；HC2/HC3 还要求杠杆值有效 |
| 给定组内的误差相关 | `PanelOLS`、`PooledOLS`、`RandomEffects` 的 `clustered` | 与观测逐行对齐的一列或两列聚类标签；每个维度至少两组 |
| 时间维度足够长时的跨个体相关与弱时间依赖 | 上述三类模型的 `driscoll-kraay` | FE/RE 使用 `time_ids`，合并 OLS 使用 `time_index`；另选核函数与带宽 |
| 单条有序观测序列中的序列相关 | 仅 `PooledOLS` 的 `hac` | 可选 `time_index`；不提供时使用输入行序 |
| 横截面平均系数的不确定性 | `FamaMacBeth` 的 `nonrobust` 或 `newey-west` | 各时期的系数序列，而非堆叠后的残差协方差 |

选项能够运行，不代表其推断假设已经满足；应按实际回归或系数序列判断。HC 不处理任意的个体内序列相关，直接用于组内回归时还存在固定短面板的额外限制，详见下文。聚类稳健协方差需要足够多的簇；时间序列协方差需要足够长且符合相应依赖条件的序列；两组只是输入下限。`group_debias=True` 是有限样本调整，不能保证少量聚类下的推断可靠。`BetweenOLS` 和 `FirstDifferenceOLS` 只公开非稳健与 HC 选项，不接受聚类或 DK 选项。

构造模型时选择 `cov_type`，在 `fit()` 中提供所需标签，修改协方差设置后重新拟合。确认系数推断可用后，再读取 `bse_`、`tvalues_`、`pvalues_` 和 `conf_int_`。[公开 API 部分](#公开-api-与别名)列出别名与独立函数；下文保留各估计量的定义和限制。

## 协方差对应的回归空间

不同面板估计器真正用于最终回归的数据并不相同：固定效应模型会先进行组内变换，随机效应模型会做准去均值处理，一阶差分模型会先进行差分。因此，标准误和协方差必须基于**实际参与系数估计的变换后数据**计算，而不能统一套用原始 $X$ 和 $y$。

为了用统一符号描述共享公式，记某个模型最终回归使用的设计矩阵为 $Z$，相应残差为 $e$，并定义

$$
B=(Z^\top Z)^+,
\qquad
\psi_i=Bz_i e_i.
$$

对不同模型，$Z$ 的含义如下：

| 模型 | 协方差所对应的回归 |
|---|---|
| `PooledOLS` | 原始水平数据上的合并 OLS |
| `PanelOLS` | 去除所选固定效应后的回归 |
| `RandomEffects` | 准去均值后的设计矩阵 $X^*$ |
| `BetweenOLS` | 每个个体一个均值观测的回归 |
| `FirstDifferenceOLS` | 一阶差分后的回归 |

`FamaMacBeth` 不使用上述基于单个残差回归的协方差形式。它根据各时期估计系数构成的时间序列计算不确定性，详见 [FamaMacBeth](fama-macbeth.md)。每个保留时期的设计矩阵必须满列秩，否则 `.fit()` 会抛出 `ValueError`。

对于上表五类基于残差 OLS 的估计器，如果实际回归空间中的设计矩阵精确秩亏，拟合值仍可能是唯一的，但系数向量本身并不唯一。此时 statgpu 会保留仍有明确含义的拟合结果，但不会报告依赖唯一系数表示的标准误、检验统计量、p 值和置信区间。

<a id="nonrobust-and-hc-covariance"></a>

## 经典协方差与 HC0–HC3

`cov_type="nonrobust"` 使用经典模型协方差，需要满足相应模型的同方差等方差结构假设。对于固定效应，给定解释变量和固定效应后，原始特异误差同方差且互不相关，可以支持使用扣除适当残差自由度的经典协方差；去均值本身会使变换后的误差相关，经典固定效应方差公式并不要求它们彼此独立。随机效应还依赖误差分量设定及相应的 GLS 变换。该选项通常不能抵御异方差或错误的相关结构设定。

HC0–HC3 使用上表相应回归中每行残差的平方。在适当的独立观测渐近条件下，它们允许异方差；HC1 调整自由度，HC2/HC3 调整杠杆值。模型支持这些选项，不代表所有面板变换后都适用。

**固定短面板的个体效应尤其需要注意。** 当 $T>2$ 固定、个体数不断增加时，即使原始特异误差不存在序列相关，直接对组内回归使用普通 HC0/HC1 仍可能不一致（Stock 和 Watson，2008，第 1 节）。这里实现的 HC2/HC3 杠杆值调整也不是这一固定 $T$ 问题的通用补救。如果个体之间独立且个体数足够多，按个体聚类通常是允许个体内相关的合适替代方案；它仍不允许任意的跨个体相关。

对于 `BetweenOLS`，独立抽样单位是个体均值，而非原面板中的每一行。HC 允许不同个体均值的误差方差不同，包括个体观测数不等带来的方差差异，但不修正个体之间的相关性。对于 `FirstDifferenceOLS`，相邻差分共享一个水平误差项；即使水平误差独立，一个体超过两期时，差分误差通常仍有序列相关。非稳健与 HC 选项都不会自动处理这种差分诱发的相关。这两个估计器均不提供聚类、HAC 或 DK 选项；若其适用假设不成立，应另选模型与协方差方案。

经典协方差为：

$$
\widehat V_{\mathrm{nonrobust}}
=
\widehat\sigma^2 B,
\qquad
\widehat\sigma^2
=
\frac{e^\top e}{df_{\mathrm{resid}}}.
$$

HC0–HC3 是异方差稳健协方差。HC0 与 HC1 为

$$
\widehat V_{\mathrm{HC0}}
=
\sum_i\psi_i\psi_i^\top,
\qquad
\widehat V_{\mathrm{HC1}}
=
\frac{n}{df_{\mathrm{resid}}}\widehat V_{\mathrm{HC0}}.
$$

令杠杆值

$$
h_i=z_i^\top Bz_i,
$$

则

$$
\widehat V_{\mathrm{HC2}}
=
\sum_i\frac{\psi_i\psi_i^\top}{1-h_i},
\qquad
\widehat V_{\mathrm{HC3}}
=
\sum_i\frac{\psi_i\psi_i^\top}{(1-h_i)^2}.
$$

HC2/HC3 要求 $1-h_i$ 在数值上为正。如果某个观测的杠杆值在数值上等于 1，statgpu 会报错，而不是返回无穷大或不稳定的协方差。如果模型本身已经在系数空间中秩亏，则系数层推断整体不可用，也不会继续强行构造 HC2/HC3。

`nonrobust` 系数推断使用 Student-t 参考分布；HC、聚类稳健和 Driscoll–Kraay 协方差使用渐近正态参考分布。

对于精确为 0 的对角方差，statgpu 不会人为加入绝对方差下限来制造一个很小但非零的标准误：零系数对应统计量 0，非零系数对应带符号无穷，并由这一显式结果继续得到 p 值与置信区间。

## 聚类稳健协方差

聚类标签应由希望允许的相关结构决定，不能仅根据均值模型包含哪些固定效应来选择：

- **按个体聚类**允许同一个体内的异方差与任意跨期相关，但要求不同个体簇相互独立，或满足相应的渐近弱依赖条件。它不覆盖使不同个体相关的共同冲击。
- **按时间聚类**允许同一期内不同个体任意相关，但要求各时期簇相互独立。它不覆盖跨期序列相关。
- **按个体和时间双向聚类**允许共享同一个体或同一时期的观测相关，不允许对两种标签都不同的观测任意设定相关，例如不受限制的跨个体、跨时期溢出效应。

常规聚类稳健推断要求**每个**聚类维度都有足够多的簇，并满足适当的矩条件、没有单个簇主导整体等要求。双向聚类中，个体数很多不能弥补时期簇太少。没有适用于所有问题的簇数门槛，簇大小不均与杠杆值也很重要。至少两组的输入下限和 `group_debias=True` 都不保证小簇数推断可靠；报告的参考分布仍是渐近正态，而非专门的小簇数修正。

对聚类组 $g$，令

$$
s_g=\sum_{i\in g}\psi_i,
$$

则基本形式为

$$
\widehat V_G=\sum_g s_gs_g^\top.
$$

每一个用户指定的聚类维度都必须至少包含两个不同的组。只有一个聚类组时，聚类稳健方差无法从组间变异中识别，因此 statgpu 会直接报错，即使 `group_debias=False` 也是如此。

当 `group_debias=True` 时，会应用小样本聚类数修正

$$
\frac{G}{G-1}\frac{n-1}{n}.
$$

### 双向聚类

双向聚类使用包含—排除形式：

$$
\widehat V_{1,2}
=
\widehat V_1+\widehat V_2-\widehat V_{12},
$$

其中 $\widehat V_{12}$ 对两个聚类标签组成的联合分组计算。

当一个聚类划分嵌套于另一个时，较细划分与交集项相消，结果等于较粗划分的协方差。这一性质取决于各组的成员关系，而不是标签的整数编码；两个边际划分本身不必相同。

极端数值尺度下，如果双向聚类计算中的非零得分分量或其乘积超出 float64 的工作范围，statgpu 会抛出 `FloatingPointError`，而不是把该分量当成 0。

## Driscoll–Kraay 协方差

当误差既有跨个体相关，也有跨期相关，而且时间维度足够长时，可以考虑 Driscoll–Kraay。它的理论依据是时间聚合得分序列的大 $T$ 渐近，需要有限矩、足够弱的时间依赖等正则条件（Driscoll 和 Kraay，1998）。它不保证能处理任意相关或非平稳共同冲击，也不能仅凭个体很多就保证短面板的推断可靠。

带宽需要在遗漏序列相关与远期滞后估计过于嘈杂之间折中。在通常的 HAC 一致性条件下，带宽随时间维度增加，但增长速度应受适当约束，不能一直占据序列长度的较大固定比例。应结合时期数，检查合理带宽范围内结果是否稳定；自动带宽只是经验规则，并不能证明这些条件成立。

statgpu 按时间索引聚合同一期内各观测的得分贡献。定义

$$
g_t=\sum_{i:t_i=t}\psi_i,
$$

并用核权重 $w_\ell$ 对不同时间滞后加权，则

$$
\widehat V_{\mathrm{DK}}
=
\frac{n}{n-\mathrm{extra\_df}-r_Z}
\left[
\sum_tg_tg_t^\top+
\sum_{\ell=1}^{T-1}w_\ell
\sum_{t=\ell+1}^{T}
\left(g_tg_{t-\ell}^\top+g_{t-\ell}g_t^\top\right)
\right].
$$

这里 $r_Z$ 表示回归中实际可识别的方向数：满列秩时等于 $Z$ 的列数，秩亏时等于 $\operatorname{rank}(Z)$。`PanelOLS` 还需要通过 `extra_df` 计入被吸收的固定效应；`PooledOLS` 与 `RandomEffects` 的这一项为 0。

当 `bandwidth=None` 时，默认带宽为

$$
\left\lfloor4(T/100)^{2/9}\right\rfloor.
$$

支持的核包括 Bartlett、Parzen 和 Quadratic Spectral（QS）。Bartlett 与 Parzen 在带宽之外权重为 0；QS 把带宽作为平滑尺度，并可对更远的已观测滞后赋予非零权重。

### 时间顺序

时间标签的顺序会直接影响 Driscoll–Kraay：

- 数值和 datetime 标签使用自然顺序；
- 有序 pandas categorical 使用用户声明的类别顺序；
- 普通字符串按字典序排序；
- 其他可以比较的对象标签按其自然比较顺序排序；
- 标签之间无法比较时会报错。

如果字符串字典序并不代表真实时间顺序，例如 `t1, t2, t10`，应改用数值/datetime 键，或显式设置为有序 categorical。

## `PooledOLS` 的 HAC 与 Driscoll–Kraay 不同

`PooledOLS(cov_type="hac")` 是独立的按序列计算的 Bartlett/Newey–West HAC，不应与 Driscoll–Kraay 混为一谈。如果提供 `time_index`，`PooledOLS` 会先按照该索引排序再计算 HAC。

Driscoll–Kraay 则先按照面板时间索引聚合同一期观测的得分贡献，再对期与期之间的滞后相关进行核加权。

<a id="ordered-sequence-hac"></a>

## 合并 OLS 的 HAC 针对单条有序序列

`PooledOLS(cov_type="hac")` 对单条观测级得分序列计算 Bartlett/Newey–West 协方差。传入 `time_index` 时先按它排序，否则按输入行序计算滞后。适用的例子是单条时间序列回归：观测间距有明确含义，序列依赖足够弱，观测数足够多，带宽选择合适。

它不会在个体边界重置滞后，也不会把同一期的观测先聚合。仅将堆叠面板按时间排序，并不能得到一般的面板 HAC：相邻行可能来自不同个体，相同时间标签的多行仍占据不同序列位置。若相关仅发生在相互独立的个体内部，可考虑个体聚类；若符合两维相关结构，可考虑双向聚类；若有足够长的时间维度且满足相应条件，可考虑 DK。`PanelOLS` 和 `RandomEffects` 不接受单独的 `hac` 选项。

<a id="coefficient-series-conditions"></a>

## FamaMacBeth 针对时期系数序列

`FamaMacBeth` 的 `nonrobust` 根据各期系数的样本波动估计平均值的方差，把时期估计视为独立。`newey-west` 则估计该系数序列的长期协方差，需要有意义的时间顺序、足够弱的序列依赖、适当的矩与稳定性条件、合适的带宽，以及足够多的保留时期。两个有效时期只是输入下限，并不保证推断可靠。

两种选项都不能修复各期回归的内生性、未识别的横截面系数或系数序列中任意的结构变化。单期个体很多，不能替代时间序列推断所需的足够时期数。平均斜率目标、时期筛选和参考分布见 [FamaMacBeth 指南](fama-macbeth.md)。

## 传入两个聚类维度

下面的完整示例让 `cluster` 的每行分别存放一个个体标签和一个时间标签。均值模型包含个体效应；增加时间聚类不会增加时间固定效应。模拟数据含有各期共同冲击，因此同时按个体和时间聚类。这里的样本规模用于说明调用方式，不是保证有效推断的簇数门槛。

<!-- example: covariance-two-way -->
```python
import numpy as np
from statgpu.panel import PanelOLS

rng = np.random.default_rng(42)
n_entities, n_periods = 80, 60
entity_ids = np.repeat(np.arange(n_entities), n_periods)
time_ids = np.tile(np.arange(n_periods), n_entities)
X = rng.normal(size=(len(entity_ids), 1))
entity_shocks = rng.normal(size=n_entities)
time_shocks = rng.normal(size=n_periods)
y = (2.0 * X[:, 0] + entity_shocks[entity_ids]
     + time_shocks[time_ids] + rng.normal(size=len(entity_ids)))
cluster = np.column_stack([entity_ids, time_ids])
model = PanelOLS(
    entity_effects=True, cov_type="clustered", device="cpu"
).fit(X, y, entity_ids=entity_ids, time_ids=time_ids, cluster=cluster)
print("coef:", np.round(model.coef_, 3))
print("standard errors:", np.round(model.bse_, 3))
```
<!-- example-end: covariance-two-way -->

两列分别定义个体与时间分组。如果把个体—时间配对标签压成一列，得到的会是按交集分组的单向聚类，与双向聚类不同。

## 公开 API 与别名

`statgpu.panel` 公开导出的协方差辅助函数包括：

- `clustered_covariance`
- `two_way_clustered_covariance`
- `hac_covariance`
- `driscoll_kraay_covariance`

若需要模型的系数推断，请选择该估计器支持的 `cov_type`，拟合后读取 `bse_`、`tvalues_`、`pvalues_` 和 `conf_int_`。这些面板估计器中，`FamaMacBeth` 还通过 `cov_params_` 公开完整的系数协方差矩阵；上表五类基于残差 OLS 的估计器不提供该属性。各模型支持的协方差选项见对应的模型说明。

估计器的 `cov_type` 支持以下常用别名：

- `hc1` 与 `robust` 表示同一 HC1 路径；
- `dk` 与 `kernel` 是 `driscoll-kraay` 的别名。

Driscoll–Kraay 核名称支持 Bartlett/Newey–West、Parzen/Gallant 与 QS/Quadratic-Spectral/Andrews 等常用写法。

## 数值行为与失败语义

聚类、HAC 与 Driscoll–Kraay 协方差计算会减少可避免的中间溢出和下溢，但仍受 float64 精度限制。最终结果可表示，并不保证能准确恢复剧烈抵消后的任意小余量；这些数值处理不改变前面的协方差定义。

系数检验使用实际的正标准误，不加入绝对方差下限。精确零方差的统计量规则见上文，不能把零标准误简单解释为没有估计不确定性。

显式设备请求遵循 [设备与 GPU 内存](../guides/device-and-memory.md) 中的后端语义。

<a id="validation-matrix"></a>
<a id="comparing-covariance-definitions"></a>

## 协方差定义的比较

只有使用相同观测、变换后回归、截距、残差自由度和协方差设置，包间比较才有明确含义：

- HC0–HC3：使用相同设计矩阵和残差，并对齐杠杆值及有限样本修正。
- 单向或双向聚类：使用相同的分组划分和 `group_debias` 设置。
- Driscoll–Kraay：对齐时间顺序、核、带宽及固定效应占用的自由度。
- `BetweenOLS` 与 `FirstDifferenceOLS`：先构造相同的个体均值或差分样本，再与 statsmodels 等 OLS 实现比较。
- `RandomEffects`：基于 statgpu 的准去均值回归 $(X^*,y^*)$ 比较协方差；使用不同方差分量构造的软件包不一定得到相同的 GLS 系数。
- `FamaMacBeth`：分别对齐系数序列协方差和推断参考分布，见 [FamaMacBeth 的包间比较](fama-macbeth.md#comparing-with-other-packages)。

若参考分布或其自由度不同，系数或协方差一致，并不意味着 p 值和置信区间也相同。

## 相关文档

- [面板模型总览](README.md) — 模型选择与统计解释
- [面板模型架构](architecture.md) — 各估计器如何共享输入、拟合与推断基础设施
- [设备与 GPU 内存](../guides/device-and-memory.md) — 后端与设备语义
- [PooledOLS](pooled-ols.md)、[PanelOLS](panel-ols.md)、[RandomEffects](random-effects.md)、[BetweenOLS](between-ols.md)、[FirstDifferenceOLS](first-difference-ols.md)、[FamaMacBeth](fama-macbeth.md) — 模型专属行为

## 参考文献

- White, H. (1980). A heteroskedasticity-consistent covariance matrix estimator and a direct test for heteroskedasticity. *Econometrica*, 48(4), 817-838.
- MacKinnon, J. G., & White, H. (1985). Some heteroskedasticity-consistent covariance matrix estimators with improved finite sample properties. *Journal of Econometrics*, 29(3), 305-325.
- Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703-708.
- Andrews, D. W. K. (1991). Heteroskedasticity and autocorrelation consistent covariance matrix estimation. *Econometrica*, 59(3), 817-858.
- Driscoll, J. C., & Kraay, A. C. (1998). Consistent covariance matrix estimation with spatially dependent panel data. *The Review of Economics and Statistics*, 80(4), 549-560. [doi:10.1162/003465398557825](https://doi.org/10.1162/003465398557825)
- Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011). Robust inference with multiway clustering. *Journal of Business & Economic Statistics*, 29(2), 238-249.

- Stock, J. H., & Watson, M. W. (2008). Heteroskedasticity-robust standard errors for fixed effects panel data regression. *Econometrica*, 76(1), 155–174. [doi:10.1111/j.0012-9682.2008.00821.x](https://doi.org/10.1111/j.0012-9682.2008.00821.x)。[作者稿](https://www.princeton.edu/~mwatson/papers/Stock_Watson_Ecta_2008.pdf)。
