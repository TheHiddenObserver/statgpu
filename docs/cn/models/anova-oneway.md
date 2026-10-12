# 单因素方差分析：经典检验与 Welch 检验

> 语言：中文\
> 最后更新：2026-10-12\
> 切换：[English](../../en/models/anova-oneway.md)

<a id="the-question"></a>

## 先明确研究问题

假设若干种处理分别用于不同的实验单位，每个单位记录一个数值型结果。
你想知道这些处理对应的总体均值是否相同。单因素方差分析把这个问题写成一个原假设：

$$H_0:\mu_1=\mu_2=\cdots=\mu_k.$$

在独立性、正态性等模型假设成立时，较小的 P 值支持“至少有一组总体均值不同”。它并不说明每一对
均值都不同，也不能直接回答差异有多大实际意义。报告结果时，应同时展示各组均值、
估计的不确定性和效应量。要判断具体哪两组不同，请继续阅读
[事后比较](anova-posthoc.md)；其他设计见 [ANOVA 总览](anova.md)。

<a id="choose-the-test"></a>

## 根据方差假设选择检验

- 各组观测独立、总体方差相等的假设合理时，使用 `f_oneway`。各组样本量可以不同。
- 不宜假设总体方差相等时，尤其当样本量也不同时，使用 `f_welch`。
  它利用各组各自的方差估计均值的不确定性。
- 两种检验都要求观测独立，并以组内误差服从正态分布为前提；Welch 使用近似 F 分布。
  对于相关观测、严重离群值或不适合模型的响应变量，还需另行选择合适的处理方法。
  应结合数据分布和实验设计作判断，而不是选择 P 值更小的方法。
- 同一个人的重复测量、配对数据和聚类观测，需要使用能够处理相关性的方法。
  这些函数不提供重复测量方差分析、协方差分析（ANCOVA）或多元方差分析（MANOVA）。

正态性假设针对各组内部的误差，而不是所有观测混在一起后的直方图。
观测是否独立，取决于数据的收集方式；把相关观测当作独立重复，即使数量很多也会影响推断。
[SciPy 的 ANOVA 说明](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f_oneway.html)
总结了经典检验的假设及其与 Welch 检验的区别。

<a id="classical-anova"></a>

## 为什么用方差之比检验均值

第 $i$ 组有 $n_i$ 个观测，均值为 $\bar y_i$，总样本量为 $N=\sum_i n_i$。
所有观测的总均值按各组样本量加权：

$$\bar y=\frac{\sum_i n_i\bar y_i}{N}.$$

组间平方和衡量各组均值相对于总均值的差异，组内平方和衡量观测值相对于本组均值的波动：

$$SS_B=\sum_i n_i(\bar y_i-\bar y)^2,\qquad
SS_W=\sum_i\sum_j(y_{ij}-\bar y_i)^2.$$

分别除以自由度后，得到组间均方和组内均方。在等方差假设成立、各组总体均值相等时，
二者估计的是同一个误差方差。如果组间均方明显大于组内均方，就有证据反对均值相等的原假设：

$$F=\frac{SS_B/(k-1)}{SS_W/(N-k)},\qquad
(df_1,df_2)=(k-1,N-k).$$

`f_oneway` 返回 F 统计量及其右尾 P 值，也返回
$\eta^2=SS_B/(SS_B+SS_W)$，见[效应量](#effect-sizes)。
这些量都由平方和直接计算，无需设置迭代次数或搜索调参网格。

<a id="anova-basic"></a>

## 完整的经典检验 CPU 示例

下面比较两个独立组，数据生成时使用相同的总体标准差。
在新的 Python 会话中依次运行以下三个代码块即可。先导入所需函数。

<!-- example: anova-basic -->
```python
import numpy as np
from statgpu.anova import f_oneway, cohens_f
```

固定随机种子，便于重现结果。每个数组包含 100 个观测，传入函数的顺序就是组别顺序。

```python
rng = np.random.default_rng(7)
g1 = rng.normal(0.0, 1.0, 100)
g2 = rng.normal(0.5, 1.0, 100)
```

运行总体检验，同时查看样本均值和效应量。
本例显式选择 NumPy 后端，不需要安装 GPU 软件。

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
有三个或更多组时，总体检验本身无法指出具体哪两组不同。

`eta_squared=0.100` 表示总样本变异中约有 10.0% 与分组有关。
这里描述的是样本中的变异比例，不能当作原假设为假的概率，也不能据此作出因果结论。
若据此估计总体中的相应比例，还需考虑估计误差。两组均值相差约 `0.582` 个响应变量的原始单位，
这个差距是否重要，需要结合应用场景判断。

<a id="welch-intuition"></a>

## Welch 检验：按均值的精度加权

均值估计越精确的组，在 Welch 检验中占的权重越大。
精度取决于样本量和组内方差：样本越多、方差越小，均值通常估计得越精确。
用各组样本方差 $s_i^2$ 定义权重 $w_i=n_i/s_i^2$。各组方差为正时，记

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

本例可以独立运行。下面生成三个组，它们的样本量和总体标准差都不同。

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

## 区分统计显著性与效应量

- 经典单因素检验的 `eta_squared` 为 $SS_B/SS_T$，其中 $SS_T=SS_B+SS_W$。
  它描述样本中的变异比例；大样本中，较小的效应也可能得到很小的 P 值。
- `cohens_f(*groups)` 通过经典检验计算
  $f=\sqrt{\eta^2/(1-\eta^2)}$，本例为 `0.334`。
  “小效应”“大效应”等常见划分只是经验约定，实际意义仍需结合研究领域判断。
- `partial_eta_squared(ss_effect, ss_error)` 计算
  $\eta_p^2=SS_{effect}/(SS_{effect}+SS_{error})$。
  经典单因素设计只有一个效应，所以偏 eta 平方等于 eta 平方。
  多因素设计中，偏 eta 平方的分母不包含其他效应，两者通常不同；
  见[双因素示例](anova-twoway.md#effect-sizes)。
- Welch 结果中的 `eta_squared` 始终为 `NaN`，因为此接口不提供 Welch 专用效应量。
  对同一批数据调用 `cohens_f`，得到的仍是经典 ANOVA 的描述性效应量，
  不能将它当作经不等方差调整的 Welch 效应量。

这些辅助函数不提供效应量的置信区间。报告时应说明响应变量的单位、比较对象，
并根据研究问题补充适当的不确定性估计。

<a id="parameters-and-pitfalls"></a>

## 重要参数和常见问题

`f_oneway`、`f_welch` 和 `cohens_f` 接受 `*groups`、`backend="auto"`
和 `dtype=None`。`dtype=None` 表示使用所选后端的 float64。
统计推断建议保留这一默认精度；当观测数值很大、组间差异很小时，较低精度可能影响结果。
完整签名及结果字段见 [ANOVA API 参考](../reference/anova-api.md)。

- 每组分别用一个一维数组传入。输入会被展平，矩阵不会被解释为多项独立检验。
  接口没有 `axis` 或 `nan_policy` 参数。
- 经典检验至少需要两个非空组，且 $N>k$。Welch 要求每组至少有两个观测。
- NaN 和无穷值会触发 `ValueError`，不会被静默删除。分析前应先明确缺失数据处理方案。
- 所有观测完全相同时，经典检验的统计量、P 值和 eta 平方均为 `NaN`。
  若各组内部都是常数而组间均值不同，则 F 为无穷大、P 值为 0、eta 平方为 1。
- Welch 在仅部分组方差为零时触发 `ValueError`。若全部组都是常数，
  均值相同时 F 和 P 值未定义，均值不同时 F 为无穷大、P 值为 0；eta 平方始终为 `NaN`。
- 未拒绝原假设不等于证明等效。函数也不会替你诊断离群值、正态性或独立性。

<a id="gpu-execution"></a>

## 可选的 GPU 运行方式

先运行[经典 CPU 示例](#anova-basic)，再根据已安装且可用的 CUDA 环境选择下面一个代码块。
两种方式都复用前面的数据，检验的统计含义不变。对于这样的小数据集，
GPU 的数据传输和启动开销可能超过计算收益。

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
普通列表和 NumPy 输入使用 NumPy。各组统计量使用所选后端计算，结果字段是 Python 标量。
部分分布计算可能在 CPU 上完成，但不会改变检验的统计含义。
这些函数没有分别选择“严格模式”和“近似模式”的开关。
完整说明见 [API 执行说明](../reference/anova-api.md)。

<a id="comparing-implementations"></a>

## 如何与其他实现比较

比较时应使用相同的有限观测值、相同的分组和 float64 精度。经典 `f_oneway` 可与 SciPy 的
等方差 `f_oneway` 比较。Welch 可与
[statsmodels `anova_oneway`](https://www.statsmodels.org/stable/generated/statsmodels.stats.oneway.anova_oneway.html)
比较，设置 `use_var="unequal"`、`welch_correction=True` 和 `trim_frac=0`，
核对 F、P 值及两项自由度。等方差检验采用不同的方差假设，截尾均值检验则改变了比较目标，
都不适合作为这里的直接对照。函数名相似，也不意味着参数选项或返回对象完全相同。

<a id="references"></a>

## 参考资料与后续阅读

- [NIST：单因素方差分析](https://itl.nist.gov/div898/handbook/ppc/section2/ppc231.htm)。
- [Welch（1951）：On the Comparison of Several Mean Values: An Alternative Approach](https://doi.org/10.2307/2332579)。
- Cohen, J.（1988）。*Statistical Power Analysis for the Behavioral Sciences*，第 2 版。
- 继续阅读[平衡双因素方差分析](anova-twoway.md)、[事后比较](anova-posthoc.md)
  或[完整 API](../reference/anova-api.md)。
