# 面板模型协方差估计

> 语言：中文  
> 最后更新：2026-09-17  
> 页面定位：面板模型协方差与系数推断参考  
> 切换：[English](../../en/panel/covariance.md)

## 概览

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

`FamaMacBeth` 不使用上述基于单个残差回归的协方差形式。它根据各时期估计系数构成的时间序列计算不确定性，详见 [FamaMacBeth](fama-macbeth.md)。

如果实际回归空间中的设计矩阵精确秩亏，拟合值仍可能是唯一的，但系数向量本身并不唯一。此时 statgpu 会保留仍有明确含义的拟合结果，但不会继续发布依赖唯一系数表示的标准误、检验统计量、p 值和置信区间。

## 经典协方差与 HC0–HC3

`cov_type="nonrobust"` 对应通常的同方差 OLS 协方差：

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

聚类稳健协方差允许同一聚类组内的观测具有相关误差。对聚类组 $g$，令

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

如果一个聚类维度嵌套在另一个维度中，statgpu 比较的是二者诱导出的分组划分是否等价，而不是要求用户提供的整数编码逐元素相同。

极端数值尺度下，如果某个数学上非零的协方差分量无法在 float64 中可靠表示，statgpu 会显式报错，而不是静默把该分量当成 0。普通用户不需要依赖内部如何缩放或归约这些分量；可依赖的是最终统计定义和明确的失败行为。

## Driscoll–Kraay 协方差

Driscoll–Kraay 协方差按时间索引聚合同一期内各观测的得分贡献。定义

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

## 公开 API 与别名

`statgpu.panel` 公开导出的协方差辅助函数包括：

- `clustered_covariance`
- `two_way_clustered_covariance`
- `hac_covariance`
- `driscoll_kraay_covariance`

`ols_covariance` 是面板估计器内部复用的分发函数，不属于公开的 `statgpu.panel` 导出接口。

估计器的 `cov_type` 支持以下常用别名：

- `hc1` 与 `robust` 表示同一 HC1 路径；
- `dk` 与 `kernel` 是 `driscoll-kraay` 的别名。

Driscoll–Kraay 核名称支持 Bartlett/Newey–West、Parzen/Gallant 与 QS/Quadratic-Spectral/Andrews 等常用写法。

## 数值行为与失败语义

面板协方差计算需要在有限精度下处理可能非常大的残差、杠杆值、聚类得分和滞后协方差项。statgpu 会在不改变统计公式的前提下使用数值稳定的归约与缩放方式。

用户可以依赖以下公开行为：

- 有限且可表示的协方差结果不会仅因为中间量尺度很大就被任意截断；
- 无法可靠表示的结果会显式报错，而不是静默返回 0、无穷大或伪造的微小方差；
- 整体缩放响应变量时，系数与标准误应按相应尺度变化，而有限的 t/z 统计量不因人为方差下限而改变；
- 显式设备请求仍遵循 [设备与 GPU 内存](../guides/device-and-memory.md) 中的后端语义。

具体的内部归约顺序、工作尺度、外部比较版本、测试容差、硬件验证与历史验证产物属于工程验证层，不是本用户参考页的稳定接口。

## 相关文档

- [面板模型总览](../models/panel.md) — 模型选择与统计解释
- [面板模型架构](architecture.md) — 各估计器如何共享输入、拟合与推断基础设施
- [设备与 GPU 内存](../guides/device-and-memory.md) — 后端与设备语义
- [PooledOLS](pooled-ols.md)、[PanelOLS](panel-ols.md)、[RandomEffects](random-effects.md)、[BetweenOLS](between-ols.md)、[FirstDifferenceOLS](first-difference-ols.md)、[FamaMacBeth](fama-macbeth.md) — 模型专属行为

## 参考文献

- White, H. (1980). A heteroskedasticity-consistent covariance matrix estimator and a direct test for heteroskedasticity. *Econometrica*, 48(4), 817-838.
- MacKinnon, J. G., & White, H. (1985). Some heteroskedasticity-consistent covariance matrix estimators with improved finite sample properties. *Journal of Econometrics*, 29(3), 305-325.
- Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703-708.
- Andrews, D. W. K. (1991). Heteroskedasticity and autocorrelation consistent covariance matrix estimation. *Econometrica*, 59(3), 817-858.
- Driscoll, J. C., & Kraay, A. C. (1998). Consistent covariance matrix estimation with spatially dependent panel data. *The Review of Economics and Statistics*, 80(4), 549-560.
- Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2011). Robust inference with multiway clustering. *Journal of Business & Economic Statistics*, 29(2), 238-249.
