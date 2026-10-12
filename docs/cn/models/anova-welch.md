# Welch 单因素方差分析

> 语言：中文\
> 最后更新：2026-10-12\
> 切换：[English](../../en/models/anova-welch.md)

<a id="the-question"></a>

## 不合并组内方差，比较各组均值

假设不同处理组之间相互独立，但数据的变异程度不同，样本量也可能不同。
`f_welch` 检验的总体原假设为

$$H_0:\mu_1=\mu_2=\cdots=\mu_k,$$

同时利用各组各自的方差估计均值的不确定性。模型允许

$$y_{ij}=\mu_i+\varepsilon_{ij},\qquad
\varepsilon_{ij}\sim\mathcal N(0,\sigma_i^2),$$

其中误差相互独立，各组可以有不同的总体方差 $\sigma_i^2$。
Welch 检验据此调整 F 统计量及其参考分布的自由度。
共同概念和方法选择见 [ANOVA 总览](anova.md)。

<a id="assumptions"></a>

## 什么时候适合使用 Welch 检验

- 观测相互独立、响应变量为数值型，且不宜假设各组总体方差相等时，可以使用 Welch 检验。
  各组样本量可以不同。
- F 参考分布是基于组内正态误差模型的近似。不等方差调整不能消除严重偏态、
  强影响离群值或观测相关性带来的问题。
- 重复测量、配对和聚类观测需要能够处理相关性的模型。
  `f_welch` 不提供重复测量方差分析、协方差分析（ANCOVA）或多元方差分析（MANOVA）。
- 每组至少需要两个观测，才能估计组内方差。
  下面的加权公式要求各组方差均为正；[常数组的处理](#parameters-and-pitfalls)另有规定。

<a id="welch-intuition"></a>

## Welch 检验：按均值的精度加权

均值估计越精确的组，在 Welch 检验中占的权重越大。
精度取决于样本量和组内方差：样本越多、方差越小，均值通常估计得越精确。
设共有 $k$ 个组，第 $i$ 组的样本量为 $n_i$，总样本量为 $N=\sum_i n_i$。
记该组均值为 $\bar y_i$，样本方差为
$s_i^2=\sum_j(y_{ij}-\bar y_i)^2/(n_i-1)$。
均值的估计方差是 $s_i^2/n_i$，取其倒数得到权重 $w_i=n_i/s_i^2$。
各组方差为正时，记

$$W=\sum_i w_i,\qquad \bar y_w=\frac{\sum_i w_i\bar y_i}{W},\qquad
U=\sum_i\frac{(1-w_i/W)^2}{n_i-1}.$$

统计量及其近似参考分布的自由度为

$$F_W=\frac{\sum_i w_i(\bar y_i-\bar y_w)^2/(k-1)}
{1+\frac{2(k-2)}{k^2-1}U},\qquad
\nu_1=k-1,\qquad \nu_2=\frac{k^2-1}{3U}.$$

分母中的修正用于考虑各组方差估计本身的不确定性。
$\nu_2$ 是有效自由度，通常不是整数，也不等于经典检验中的 $N-k$。
该方法来自 [Welch（1951）](https://doi.org/10.2307/2332579)。

<a id="anova-welch"></a>

## 完整的不等方差示例

在新的 Python 会话中依次运行以下三个代码块即可。
下面生成三个组，它们的样本量和总体标准差都不同。
本例使用 NumPy 在 CPU 上运行，不需要 GPU，也不依赖其他页面中的变量。

<!-- example: anova-welch -->
```python
import numpy as np
from statgpu.anova import f_welch
```

分别生成 20、28、36 个观测。注意 `rng.normal` 的第二个参数是标准差，不是方差。

```python
rng = np.random.default_rng(2026)
unequal_groups = [
    rng.normal(10.0, 0.7, 20),
    rng.normal(10.8, 1.5, 28),
    rng.normal(13.0, 3.0, 36),
]
```

用 `*unequal_groups` 将列表中的数组逐一作为组参数传入。这次还要留意分母自由度。

```python
welch = f_welch(*unequal_groups, backend="numpy")
print(f"F={welch.statistic:.3f}, p={welch.pvalue:.3g}")
print(f"df=({welch.df_between}, {welch.df_within:.3f})")
print(f"eta_squared={welch.eta_squared}")
```
<!-- example-end: anova-welch -->

输出为 `F=24.825`、`p=3.82e-08`，自由度为 `(2, 48.370)`，
且 `eta_squared=nan`。结果支持“三个总体均值并非全部相等”，但仍需两两比较来判断差异来自哪里。
若要在不等方差条件下比较各对均值，见
[Bonferroni 两两 Welch 检验](anova-posthoc.md#anova-bonferroni)。
后续比较应沿用合适的方差假设，不宜仅因 Welch 检验显著就改用合并方差的 Tukey 检验。

<a id="effect-sizes"></a>

## 为什么效应量字段是 NaN

`f_welch` 始终返回 `eta_squared=NaN`，因为此接口不提供 Welch 专用效应量。
在本例中，这不表示 F 检验失败，也不表示效应为零。
对同一批观测调用 `cohens_f`，仍会执行经典检验的计算，
不能把结果当作经不等方差调整的 Welch 效应量。
报告时应给出各组均值、以响应变量原始单位表示的差异，以及适合该比较的不确定性估计。
统计显著性与效应大小的区别，见[总览中的效应量说明](anova.md#effect-sizes)。

<a id="parameters-and-pitfalls"></a>

## 重要参数和常见问题

调用 `f_welch(*groups, backend="auto", dtype=None)` 时，每组分别传入一个数组。
默认计算精度为 float64。接口没有截尾选项，也不能关闭 Welch 修正。
完整签名、结果字段、输入展平规则和通用输入要求，见
[API 参考](../reference/anova-api.md#f-welch)。

- 至少需要两个组，每组至少包含两个有限的实数观测值。
  NaN 和无穷值会触发 `ValueError`，缺失值不会被静默删除。
- 仅部分组的样本方差为零时，方差倒数加权无法计算，函数会触发 `ValueError`。
- 所有组内部都是常数时，若各组均值被判定为相等，F 和 P 值均为 `NaN`；
  否则 F 为无穷大、P 值为 0。这一特殊情形返回 `df_within=N-k`，
  eta 平方始终为 `NaN`。均值相等的数值容差，见
  [退化输入规则](../reference/anova-api.md#finite-and-degenerate-inputs)。
- 对通常的正方差样本，应保留 `df_within` 的小数部分。
  将其取整或换成 $N-k$，会改变参考分布。
- 总体检验显著不能指出具体哪两组不同；未拒绝原假设也不等于证明等效。

<a id="backend-execution"></a>

## 后端选择

`f_welch` 支持 NumPy、CuPy 和 Torch 数组。本例显式使用 NumPy。
若要在 GPU 上计算组内统计量，应将各组数据放在同一设备上，使用 CuPy 数组
或 Torch CUDA 张量，并选择对应后端。仅设置 `backend="torch"` 不会把数据移到 CUDA。
结果字段是 Python 标量，分布计算不保证完全在 GPU 上执行。
自动选择、计算精度和尾部概率精度的限制，见
[API 执行说明](../reference/anova-api.md#backend-dtype-device)。

<a id="comparing-implementations"></a>

## 如何与其他实现比较

直接比较时，应使用相同的有限一维观测值、相同的分组、float64 精度，
并确保各组样本方差均为正。

- [statsmodels `anova_oneway`](https://www.statsmodels.org/stable/generated/statsmodels.stats.oneway.anova_oneway.html)：
  设置 `use_var="unequal"`、`welch_correction=True` 和 `trim_frac=0`，
  核对 F、P 值以及分子、分母两项自由度。
- [SciPy `f_oneway`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f_oneway.html)：
  支持 `equal_var=False` 的版本可执行 Welch 检验；显式设置该选项后，核对 F 和 P 值。
  默认的等方差计算属于另一种检验，且 SciPy 的返回结果不包含自由度。

截尾均值检验改变了比较目标，关闭 Welch 修正则改变了统计量，
都不适合作为本页计算的直接对照。

<a id="references"></a>

## 参考资料与后续阅读

- [Welch（1951）：On the Comparison of Several Mean Values: An Alternative Approach](https://doi.org/10.2307/2332579)。
- [statsmodels：单因素方差分析方法](https://www.statsmodels.org/stable/generated/statsmodels.stats.oneway.anova_oneway.html)。
- [SciPy：`f_oneway`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f_oneway.html)。
- 方法教程：[经典单因素](anova-oneway.md)、[Welch 单因素](anova-welch.md)
  和[平衡双因素](anova-twoway.md)。
- [ANOVA 总览](anova.md)、[事后比较](anova-posthoc.md)
  和[完整 API 参考](../reference/anova-api.md)。
