# 经典单因素方差分析

<a id="单因素方差分析经典检验与-welch-检验"></a>

> 语言：中文\
> 最后更新：2026-10-12\
> 切换：[English](../../en/models/anova-oneway.md)

<a id="the-question"></a>

## 先明确研究问题

如果各组观测独立，且总体方差相等的假设合理，可以用经典单因素方差分析
检验各组总体均值是否全部相等：

$$H_0:\mu_1=\mu_2=\cdots=\mu_k.$$

例如，对不同实验单位施加不同处理后，比较各组的数值型响应。
`f_oneway` 检验这一总体原假设，不直接判断具体哪两组存在差异。
共同概念和方法选择见 [ANOVA 总览](anova.md)。

<a id="choose-the-test"></a>
<a id="根据方差假设选择检验"></a>

## 什么时候适合使用经典模型

- 观测相互独立，各组内部的误差服从正态分布，并且总体方差相等。
  各组样本量可以不同。
- 检验前应检查研究设计和组内数据分布。函数不会自动诊断观测相关性、
  离群值或模型假设是否成立。
- 不宜假设总体方差相等时，参阅独立的 [Welch 单因素检验教程](anova-welch.md)。
  重复测量、配对或聚类观测需要能够处理相关性的模型，`f_oneway` 不提供这类分析。

<a id="classical-anova"></a>

## 为什么用方差之比检验均值

第 $i$ 组第 $j$ 个观测的经典模型为

$$y_{ij}=\mu_i+\varepsilon_{ij},\qquad
\varepsilon_{ij}\overset{\mathrm{iid}}{\sim}\mathcal N(0,\sigma^2).$$

每组有自己的均值 $\mu_i$；所有误差相互独立，服从相同的正态分布，
共用一个总体方差 $\sigma^2>0$。
设共有 $k$ 个组，第 $i$ 组有 $n_i$ 个观测，均值为 $\bar y_i$，总样本量为 $N=\sum_i n_i$。
所有观测的总均值按各组样本量加权：

$$\bar y=\frac{\sum_i n_i\bar y_i}{N}.$$

组间平方和衡量各组均值相对于总均值的差异，组内平方和衡量观测值相对于本组均值的波动：

$$SS_B=\sum_i n_i(\bar y_i-\bar y)^2,\qquad
SS_W=\sum_i\sum_j(y_{ij}-\bar y_i)^2.$$

$k$ 个组均值提供 $k-1$ 个独立的组间对比；每组各估计一个均值后，
组内还剩 $\sum_i(n_i-1)=N-k$ 个自由度。
两项平方和分别除以相应自由度，得到组间均方和组内均方。在等方差假设成立、各组总体均值相等时，
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
效应量的解释和估计不确定性见[总览中的效应量说明](anova.md#effect-sizes)。
两组均值相差约 `0.582` 个响应变量的原始单位，
这个差距是否重要，需要结合应用场景判断。

<a id="effect-sizes"></a>
<a id="区分统计显著性与效应量"></a>

## 经典检验中的效应量计算

`f_oneway` 返回 $\eta^2=SS_B/(SS_B+SS_W)$。
`cohens_f(*groups)` 先运行同样的经典计算，再按
$f=\sqrt{\eta^2/(1-\eta^2)}$ 转换，本例得到 `0.334`。
在这个只有一个效应的设计中，偏 eta 平方等于 eta 平方。
这些指标的含义、在其他设计中的区别，以及辅助函数的限制，见
[总览中的效应量说明](anova.md#effect-sizes)。

<a id="parameters-and-pitfalls"></a>

## 重要参数和常见问题

调用 `f_oneway(*groups, backend="auto", dtype=None)` 时，每组分别传入一个数组。
`cohens_f` 接受相同的输入。`dtype=None` 使用 float64，统计推断建议保留这一默认精度。
完整签名、结果字段、输入展平规则和后端行为，见
[ANOVA API 参考](../reference/anova-api.md#f-oneway)。

- 至少需要两个非空组，且总样本量 $N>k$。满足总样本量条件时，允许某组只有一个观测。
- 使用有限的实数观测值。NaN 和无穷值会触发 `ValueError`；分析前应先确定缺失数据处理方案。
- 所有观测完全相同时，统计量、P 值和 eta 平方均为 `NaN`。
  若各组内部都是常数而组间均值不同，则 F 为无穷大、P 值为 0、eta 平方为 1。
- 未拒绝原假设不等于证明等效，较小的 P 值也不代表差异有实际意义。
  结果解读见 [ANOVA 总览](anova.md)。

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
显式指定后端时，不会静默改用另一个数组库。结果字段是 Python 标量。
自动选择后端、设备位置、标量分布计算及数值限制，见
[API 执行说明](../reference/anova-api.md#backend-dtype-device)。

<a id="comparing-implementations"></a>

## 如何与其他实现比较

比较时应使用相同的有限一维观测值、相同的分组和 float64 精度。
可将 F 和 P 值与
[SciPy `f_oneway`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f_oneway.html)
的等方差计算比较：提供该选项的版本使用 `equal_var=True`，旧版默认也执行等方差检验。
SciPy 的数组轴、缺失值及无效输入处理方式与 statgpu 不同；应对齐有效样本，
不要把两个接口当作完全可互换。

<a id="welch-intuition"></a>
<a id="welch-检验按均值的精度加权"></a>
<a id="anova-welch"></a>
<a id="完整的不等方差示例"></a>

## 不等方差时的替代方法

Welch 检验的[精度加权与 F 统计量修正](anova-welch.md#welch-intuition)
以及[完整的不等方差示例](anova-welch.md#anova-welch)，见独立的 Welch 教程。

<a id="references"></a>

## 参考资料与后续阅读

- [NIST：单因素方差分析](https://itl.nist.gov/div898/handbook/ppc/section2/ppc231.htm)。
- [SciPy：经典单因素方差分析](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.f_oneway.html)。
- Cohen, J.（1988）。*Statistical Power Analysis for the Behavioral Sciences*，第 2 版。
- 方法教程：[经典单因素](anova-oneway.md)、[Welch 单因素](anova-welch.md)
  和[平衡双因素](anova-twoway.md)。
- [ANOVA 总览](anova.md)、[事后比较](anova-posthoc.md)
  和[完整 API 参考](../reference/anova-api.md)。
