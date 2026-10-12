# 单因素方差分析：经典检验与 Welch 检验

> 语言：中文\
> 最后更新：2026-10-12\
> 切换：[English](../../en/models/anova-oneway.md)

<a id="the-question"></a>

## 先明确研究问题

假设不同实验单位分别接受若干种处理，每个单位提供一个数值型响应。
你想知道各组的总体均值是否相同。单因素方差分析检验一个整体原假设：

$$H_0:\mu_1=\mu_2=\cdots=\mu_k.$$

较小的 p 值支持“至少有一组均值不同”，却不能直接告诉你哪两组不同、
是否每一对都不同，也不能说明差异有多大实际价值。报告检验结果时，应同时
展示各组均值、不确定性和效应量。要回答具体两组的比较问题，请继续阅读
[事后比较](anova-posthoc.md)；其他设计见 [ANOVA 总览](anova.md)。

<a id="choose-the-test"></a>

## 根据方差假设选择检验

- 各组观测独立、总体方差相等的假设合理时，使用 `f_oneway`。各组样本量可以不同。
- 不宜假设总体方差相等时，尤其当样本量也不同时，使用 `f_welch`。
  它分别估计各组均值的不确定性。
- 两种检验都要求观测独立，并以组内正态误差模型为参考；Welch 使用近似 F 分布。
  它们都不能自动解决相关观测、严重离群值或不合适的响应模型。
  应结合数据分布和实验设计选择方法，而不是选择 p 值更小的方法。
- 同一个人的重复测量、配对数据和聚类观测需要显式处理相关性。
  这些函数不提供重复测量方差分析、协方差分析（ANCOVA）或多元方差分析（MANOVA）。

正态性假设针对各组内部的误差，而不是所有观测混在一起后的直方图。
独立性来自观测的收集方式；增加样本量不能补救把相关观测当作独立重复的问题。
[SciPy 的 ANOVA 说明](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f_oneway.html)
总结了经典检验的假设及其与 Welch 检验的区别。

<a id="classical-anova"></a>

## 为什么用方差之比检验均值

第 $i$ 组有 $n_i$ 个观测，均值为 $\bar y_i$，总样本量为 $N=\sum_i n_i$。
总体样本均值按各组样本量加权：

$$\bar y=\frac{\sum_i n_i\bar y_i}{N}.$$

组间平方和衡量各组均值的分散程度，组内平方和衡量每组剩余的波动：

$$SS_B=\sum_i n_i(\bar y_i-\bar y)^2,\qquad
SS_W=\sum_i\sum_j(y_{ij}-\bar y_i)^2.$$

分别除以自由度后得到两个方差估计。在总体均值相等、方差相同的原假设下，
二者估计同一个噪声水平。如果组间变异相对于组内噪声明显偏大，就有理由怀疑均值相等：

$$F=\frac{SS_B/(k-1)}{SS_W/(N-k)},\qquad
(df_1,df_2)=(k-1,N-k).$$

`f_oneway` 返回 F 统计量及其右尾 p 值，也返回
$\eta^2=SS_B/(SS_B+SS_W)$，见[效应量](#effect-sizes)。
计算直接使用平方和，不需要迭代求解器或调参网格。

<a id="anova-basic"></a>

## 完整的经典检验 CPU 示例

下面比较两个独立组，数据生成时使用相同的总体标准差。
在新的 Python 会话中依次运行这三个代码块。首先导入本例用到的全部函数。

<!-- example: anova-basic -->
```python
import numpy as np
from statgpu.anova import f_oneway, cohens_f
```

固定随机种子便于重现。每个数组包含 100 个观测，参数顺序用于标识组别。

```python
rng = np.random.default_rng(7)
g1 = rng.normal(0.0, 1.0, 100)
g2 = rng.normal(0.5, 1.0, 100)
```

计算整体检验，并查看实际样本均值和效应量。
显式选择 NumPy 后端，因此运行本例不需要安装 GPU 软件。

```python
result = f_oneway(g1, g2, backend="numpy")
f_effect = cohens_f(g1, g2, backend="numpy")
print(np.round([g1.mean(), g2.mean()], 3))
print(f"F={result.statistic:.3f}, p={result.pvalue:.3g}")
print(f"df=({result.df_between}, {result.df_within})")
print(f"eta_squared={result.eta_squared:.3f}, Cohen f={f_effect:.3f}")
```
<!-- example-end: anova-basic -->

两组均值为 `[-0.173, 0.409]`，检验得到 `F=22.079`、`p=4.89e-06`，
自由度为 `(1, 198)`。在事先确定的 5% 显著性水平下，拒绝均值相等的原假设。
只有两个组时，经典 F 统计量等于等方差双样本 t 统计量的平方。
有三个或更多组时，同一个整体检验无法指出具体哪两组不同。

`eta_squared=0.100` 表示总样本变异中约有 10.0% 与分组有关。
这不是原假设为假的概率、因果结论，也不是没有估计误差的总体方差比例。
两组均值相差约 `0.582` 个响应单位，其实际意义需要结合应用场景判断。

<a id="welch-intuition"></a>

## Welch 检验：按均值的精度加权

在样本量相同时，均值估计更不稳定的组应提供较少证据。
Welch 检验使用各组样本方差 $s_i^2$，令 $w_i=n_i/s_i^2$。
各组方差为正时，定义

$$W=\sum_i w_i,\qquad \bar y_w=\frac{\sum_i w_i\bar y_i}{W},\qquad
U=\sum_i\frac{(1-w_i/W)^2}{n_i-1}.$$

统计量及其近似参考分布的自由度为

$$F_W=\frac{\sum_i w_i(\bar y_i-\bar y_w)^2/(k-1)}
{1+\frac{2(k-2)}{k^2-1}U},\qquad
\nu_1=k-1,\qquad \nu_2=\frac{k^2-1}{3U}.$$

分母中的修正考虑了分别估计各组方差带来的不确定性。
$\nu_2$ 是有效自由度，通常不是整数，也不等于经典检验中的 $N-k$。
该方法来自 [Welch（1951）](https://doi.org/10.2307/2332579)。

<a id="anova-welch"></a>

## 完整的不等方差示例

本例可以独立运行，不依赖前一个示例。三个组的样本量和总体标准差都不同。

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

用 `*unequal_groups` 将每个数组作为一组传入，并明确查看分母自由度。

```python
welch = f_welch(*unequal_groups, backend="numpy")
print(f"F={welch.statistic:.3f}, p={welch.pvalue:.3g}")
print(f"df=({welch.df_between}, {welch.df_within:.3f})")
print(f"eta_squared={welch.eta_squared}")
```
<!-- example-end: anova-welch -->

输出为 `F=24.825`、`p=3.82e-08`，自由度为 `(2, 48.370)`，
且 `eta_squared=nan`。结果支持三个总体均值并非全部相等，但没有直接定位差异。
若要在不等方差条件下比较各对均值，见
[Bonferroni 两两 Welch 检验](anova-posthoc.md#anova-bonferroni)。
不要因为 Welch 检验显著，就直接接上采用合并方差的 Tukey 检验。

<a id="effect-sizes"></a>

## 区分统计显著性与效应量

- 经典单因素检验的 `eta_squared` 为 $SS_B/SS_T$，其中 $SS_T=SS_B+SS_W$。
  它描述样本中的变异比例；大样本中，较小的效应也可能得到很小的 p 值。
- `cohens_f(*groups)` 通过经典检验计算
  $f=\sqrt{\eta^2/(1-\eta^2)}$，本例为 `0.334`。
  “小”“大”等常见划分只是约定，不能替代具体领域的判断。
- `partial_eta_squared(ss_effect, ss_error)` 计算
  $\eta_p^2=SS_{effect}/(SS_{effect}+SS_{error})$。
  经典单因素设计只有一个效应，所以偏 eta 平方等于 eta 平方。
  多因素设计中，偏 eta 平方的分母不包含其他效应，两者通常不同；
  见[双因素示例](anova-twoway.md#effect-sizes)。
- Welch 结果中的 `eta_squared` 有意返回 `NaN`：此接口没有给出 Welch 专用效应量。
  对同一批数据调用 `cohens_f` 仍然得到经典描述量，不能将它当作经不等方差调整的 Welch 效应量。

这些辅助函数不提供效应量的置信区间。报告时应明确响应尺度、比较对象及适当的不确定性。

<a id="parameters-and-pitfalls"></a>

## 重要参数和常见问题

`f_oneway`、`f_welch` 和 `cohens_f` 接受 `*groups`、`backend="auto"`
和 `dtype=None`。`None` 使用对应后端的 float64。统计推断建议保留这个默认值；
当组间差异相对于响应数值很小时，较低精度可能影响计算。
完整签名及结果字段见 [ANOVA API 参考](../reference/anova-api.md)。

- 每个组使用独立的一维数组。输入会被展平，矩阵不会被解释为多项独立检验。
  接口没有 `axis` 或 `nan_policy` 参数。
- 经典检验至少需要两个非空组，且 $N>k$。Welch 要求每组至少有两个观测。
- NaN 和无穷值会触发 `ValueError`，不会被静默删除。分析前应先明确缺失数据处理方案。
- 所有观测完全相同时，经典检验的统计量、p 值和 eta 平方均为 `NaN`。
  若各组内部都是常数而组间均值不同，则返回无限大的 F、零 p 值及 eta 平方一。
- Welch 在仅部分组方差为零时触发 `ValueError`。若全部组都是常数，
  均值相同时 F 和 p 值未定义，均值不同时 F 为无穷、p 值为零；eta 平方始终为 `NaN`。
- 未拒绝原假设不等于证明等效。函数也不会替你诊断离群值、正态性或独立性。

<a id="gpu-execution"></a>

## 可选的 GPU 运行方式

先运行[经典 CPU 示例](#anova-basic)，再根据已安装且可用的 CUDA 环境选择下面一个代码块。
复用相同数据，统计问题不会改变。对于这种小数据，GPU 的传输和启动开销可能超过计算收益。

### CuPy

<!-- example-requires: anova-basic -->
<!-- example: anova-cupy -->
```python
import cupy as cp

g1_gpu, g2_gpu = cp.asarray(g1), cp.asarray(g2)
result_gpu = f_oneway(g1_gpu, g2_gpu, backend="cupy")
```
<!-- example-end: anova-cupy -->

### Torch CUDA

<!-- example-requires: anova-basic -->
<!-- example: anova-torch -->
```python
import torch

g1_torch = torch.as_tensor(g1, device="cuda", dtype=torch.float64)
g2_torch = torch.as_tensor(g2, device="cuda", dtype=torch.float64)
result_torch = f_oneway(g1_torch, g2_torch, backend="torch")
```
<!-- example-end: anova-torch -->

这里 `backend` 选择数组库，Torch 张量则显式放到 CUDA 设备上。
显式指定后端时，不会静默改用另一个数组库。`"auto"` 根据组数组类型推断数组库，
普通列表和 NumPy 输入使用 NumPy。归约计算使用所选后端，结果字段是 Python 标量。
分布函数可能涉及 CPU 标量计算，但不会改变所用的统计检验。
这些函数没有分别选择“严格模式”和“近似模式”的开关。
完整说明见 [API 执行说明](../reference/anova-api.md)。

<a id="comparing-implementations"></a>

## 如何与其他实现比较

应对齐有限观测、分组边界和 float64 精度。经典 `f_oneway` 可与 SciPy 的
等方差 `f_oneway` 比较。Welch 可与
[statsmodels `anova_oneway`](https://www.statsmodels.org/stable/generated/statsmodels.stats.oneway.anova_oneway.html)
比较，设置 `use_var="unequal"`、`welch_correction=True` 和 `trim_frac=0`，
核对 F、p 值及两项自由度。等方差检验或截尾均值检验回答的是不同问题。
函数名相似，并不意味着参数选项或返回对象完全相同。

<a id="references"></a>

## 参考资料与后续阅读

- [NIST：单因素方差分析](https://itl.nist.gov/div898/handbook/ppc/section2/ppc231.htm)。
- [Welch（1951）：On the Comparison of Several Mean Values: An Alternative Approach](https://doi.org/10.2307/2332579)。
- Cohen, J.（1988）。*Statistical Power Analysis for the Behavioral Sciences*，第 2 版。
- 继续阅读[平衡双因素方差分析](anova-twoway.md)、[事后比较](anova-posthoc.md)
  或[完整 API](../reference/anova-api.md)。
