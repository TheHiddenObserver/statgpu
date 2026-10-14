# 平衡双因素方差分析

> 语言：中文\
> 最后更新：2026-10-12\
> 切换：[English](../../../en/models/testing/anova-twoway.md)

<a id="the-question"></a>

## 两个因素对应三个不同问题

双因素方差分析研究按两个因素分类的数值型响应。输入是各因素水平组合下的响应样本，
检验目标包括两个因素各自的主效应，以及纳入模型时的交互作用。输出为各效应的 F 统计量、
P 值、自由度与效应量摘要。本页的平衡实现要求每个水平组合具有相同样本量。

例如，研究两种肥料与三种浇水安排对植物生长的影响。让每种肥料都分别配合三种
浇水安排，就得到六个处理组合，也称为单元格。各单元格使用独立植株时，
双因素方差分析可以分别回答三个问题：

1. **因素 A：** 对三种浇水水平等权平均后，两种肥料对应的总体均值是否相同？
2. **因素 B：** 对两种肥料等权平均后，不同浇水安排对应的总体均值是否相同？
3. **交互作用：** 肥料之间的差异是否随浇水水平改变？

前两个问题检验的是主效应，第三个问题检验的是交互作用。没有交互作用时，
可以把两种因素的效应相加；有交互作用时，一个因素的效应会随另一个因素的水平变化。
例如，某种肥料相对于另一种肥料的优势在频繁浇水时很大、在少量浇水时很小，
仅报告平均的肥料差异就会掩盖这种变化。因此，解释主效应之前，应先查看交互作用和各单元格均值。

`f_twoway` 支持平衡、交叉的固定效应设计。这里“交叉”表示两因素的每种水平组合
都有观测，“平衡”表示各单元格的样本量相同；每个因素至少需要两个水平。
固定效应检验关注的是这些指定水平之间的差异。
它不提供不平衡设计的 Type I/II/III 方差分析、重复测量方差分析、随机效应方差分析、
协方差分析（ANCOVA）或多元方差分析（MANOVA）。
其他设计应如何选择分析方法，见 [ANOVA 总览](anova.md)。

<a id="assumptions"></a>

## 检验之前先核对设计

经典 F 检验假设误差独立、各单元格内误差服从正态分布，并且各单元格的总体方差相同。
分析时应查看各单元格的均值、波动情况和模型残差；函数不会自动完成这些诊断。

- 单元格样本量相同是此接口的计算要求，不代表总体方差相同。
- 独立重复需要来自不同的实验单位。同一株植物的多次读数不能当作多株独立植株。
- 包含交互项的完整模型需要每格有重复观测。每格只有一个观测时，没有可用于估计误差的自由度，
  会触发 `ValueError`。
- 省略交互项意味着假定两因素的效应可以相加，应有研究依据，不能为了隐藏交互作用或获得更小的 P 值而这样设置。

<a id="model-and-tests"></a>

## 模型、平方和与检验

对 A 的第 $i$ 个水平、B 的第 $j$ 个水平组合下的第 $r$ 个观测，完整模型为

$$y_{ijr}=\mu+\alpha_i+\beta_j+(\alpha\beta)_{ij}+\varepsilon_{ijr}.$$

设 A 有 $a$ 个水平，B 有 $b$ 个水平，每格有 $n$ 个观测，则 $N=abn$。
记 $\bar y_{ij}$ 为单元格均值，$\bar y_{i\cdot}$ 与 $\bar y_{\cdot j}$
分别为行均值与列均值，$\bar y$ 为总均值。平衡设计中的变异可分解为

$$SS_A=bn\sum_i(\bar y_{i\cdot}-\bar y)^2,\qquad
SS_B=an\sum_j(\bar y_{\cdot j}-\bar y)^2,$$

$$SS_{AB}=n\sum_{ij}(\bar y_{ij}-\bar y_{i\cdot}-\bar y_{\cdot j}+\bar y)^2,
\qquad SS_E=\sum_{ijr}(y_{ijr}-\bar y_{ij})^2.$$

总平方和满足 $SS_T=SS_A+SS_B+SS_{AB}+SS_E$。
交互平方和衡量两个主效应相加后仍无法解释的单元格均值差异。
每个效应的平方和除以相应自由度，得到该效应的均方；再除以残差均方
$MS_E=SS_E/df_E$，就得到 F 统计量：

| 项 | 分子自由度 | F 统计量 |
|---|---:|---|
| A | $a-1$ | $(SS_A/(a-1))/MS_E$ |
| B | $b-1$ | $(SS_B/(b-1))/MS_E$ |
| A × B | $(a-1)(b-1)$ | $(SS_{AB}/((a-1)(b-1)))/MS_E$ |
| 误差 | $ab(n-1)$ | 所有检验的共同分母 |

各 P 值是相应 F 统计量的右尾概率，尚未对多个效应检验进行联合校正。
如果研究结论涉及多个效应、响应变量或实验，并需要控制整个检验族的错误率，应提前定义检验族，
并阅读[多重检验](multiple-testing.md)。
[NIST 双因素方差分析说明](https://www.itl.nist.gov/div898/handbook/prc/section4/prc438.htm)
给出了平衡析因设计的平方和分解。

<a id="anova-twoway"></a>

## 完整的析因设计 CPU 示例

下面先分析包含交互项的完整模型，再解释效应量，最后与省略交互项的加性模型比较。
示例可以独立运行，按顺序执行代码即可。先导入所需函数。

<!-- example: anova-twoway -->
```python
import numpy as np
from statgpu.anova import f_twoway, partial_eta_squared
```

用嵌套列表组织数据：行表示肥料（A），列表示浇水（B），每格是一维数组，
包含 12 个独立观测。每一行需要用 Python 列表或元组组织，不能直接传入三维数组。
六个单元格的样本量完全相同。

```python
rng = np.random.default_rng(11)
target_means = [[10.0, 10.5, 11.0], [10.0, 11.5, 13.0]]
data = [[rng.normal(mean, 1.2, 12) for mean in row]
        for row in target_means]
cell_means = np.array([[cell.mean() for cell in row] for row in data])
print(np.round(cell_means, 3))
```

拟合包含交互项的模型。各检验使用同一个残差方差估计，返回的 eta 平方均以总平方和为分母。

```python
result = f_twoway(data, interaction=True, backend="numpy")
print(f"A: F={result.factor_a_statistic:.3f}, p={result.factor_a_pvalue:.3g}")
print(f"B: F={result.factor_b_statistic:.3f}, p={result.factor_b_pvalue:.3g}")
print(f"AB: F={result.interaction_statistic:.3f}, p={result.interaction_pvalue:.3g}")
print(f"residual df={result.df_within}, SS={result.ss_within:.3f}")
print(np.round([result.factor_a_eta_squared,
                result.factor_b_eta_squared,
                result.interaction_eta_squared], 3))
```

生成数据后，各单元格的样本均值约为

| 肥料 | 浇水 0 | 浇水 1 | 浇水 2 |
|---|---:|---:|---:|
| A 水平 0 | 10.217 | 10.412 | 10.487 |
| A 水平 1 | 9.787 | 11.745 | 13.170 |

完整模型的结果为

| 效应 | F | P 值 | 分子自由度 | eta 平方 |
|---|---:|---:|---:|---:|
| A | 24.963 | 4.55e-06 | 1 | 0.157 |
| B | 19.649 | 2.02e-07 | 2 | 0.248 |
| A × B | 14.192 | 7.47e-06 | 2 | 0.179 |

残差自由度为 `66`，残差平方和为 `67.960`。
交互检验支持“肥料之间的差异随浇水水平变化”。从样本均值看，第二种肥料相对第一种，
在浇水 0 下约低 `0.430` 个单位，在浇水 2 下却约高 `2.682` 个单位。
主效应检验把另一因素各水平的结果取平均，因此不能据此认为每个水平下都有相同差异。
这些总体检验也没有直接给出上述单元格均值差的置信区间或 P 值。

<a id="effect-sizes"></a>

## 普通 eta 平方与偏 eta 平方的分母不同

返回的 `factor_a_eta_squared`、`factor_b_eta_squared` 和
`interaction_eta_squared` 都是普通 eta 平方：

$$\eta_A^2=SS_A/SS_T,\qquad \eta_{AB}^2=SS_{AB}/SS_T.$$

例如，交互项约占总样本变异的 17.9%。偏 eta 平方则为
$SS_{effect}/(SS_{effect}+SS_E)$，其分母排除了其他效应。
结果中包含残差平方和，因此可以先计算残差均方，再结合 F 和分子自由度，求出相应效应的平方和：

$$SS_{effect}=F_{effect}\,df_{effect}\,(SS_E/df_E).$$

继续使用上面的 `result`，计算交互作用的偏 eta 平方。
这一步要求 F 统计量为有限值，且数据没有出现零残差等退化情形。

```python
mse = result.ss_within / result.df_within
ss_interaction = result.interaction_statistic * result.interaction_df * mse
partial_interaction = partial_eta_squared(ss_interaction, result.ss_within)
print(f"interaction SS={ss_interaction:.3f}")
print(f"partial eta_squared={partial_interaction:.3f}")
```

交互平方和为 `29.226`，偏 eta 平方为 `0.301`，不同于普通 eta 平方 `0.179`。
这两个数都描述样本中的效应大小，不能解释为因果贡献，也不反映估计的不确定性。
不同效应的偏 eta 平方不能直接相加；跨研究或跨软件比较时，也应先确认定义一致。

<a id="additive-model"></a>

## 省略交互作用会发生什么

设置 `interaction=False` 后，模型变为 $\mu+\alpha_i+\beta_j$。
原先由交互项解释的变异进入残差：

$$SS_{E,add}=SS_E+SS_{AB},\qquad
df_{E,add}=N-a-b+1=df_E+(a-1)(b-1).$$

如果仍只把单元格内部的波动作为误差，就会漏掉加性模型无法解释的交互变异。
因此，残差平方和与残差自由度都要调整，主效应 F 检验也要重新计算。
设置这个选项是在改变模型，而不只是隐藏输出中的交互项。

```python
additive = f_twoway(data, interaction=False, backend="numpy")
print(f"A: F={additive.factor_a_statistic:.3f}")
print(f"B: F={additive.factor_b_statistic:.3f}")
print(f"residual df={additive.df_within}, SS={additive.ss_within:.3f}")
print(additive.interaction_statistic, additive.interaction_pvalue)
```
<!-- example-end: anova-twoway -->

加性模型的残差平方和为 `97.186`，由原残差平方和与交互平方和相加得到
（表中显示为 `67.960 + 29.226`，计算时使用未舍入值），
残差自由度为 `68`。主效应 F 值变为 `17.985` 和 `14.156`，交互字段为 `None`。
普通主效应 eta 平方保持不变，因为平衡设计的主效应平方和及总平方和没有改变。
本例的单元格均值呈现明显交互变化，检验也提供了相应证据，因此不宜仅用加性模型概括结果。

<a id="parameters-and-pitfalls"></a>

## 重要参数与使用边界

- `data`：矩形的嵌套列表或元组，每格存放一维样本，必须包含所有 A × B 组合。
  空单元格、各行长度不同、单元格样本量不等、任一因素少于两个水平，
  以及非有限观测都会触发 `ValueError`。
- `interaction=True`：包含并检验交互作用。设为 `False` 后，四个交互结果字段均为
  `None`，残差包括交互变异。每格只有一个观测时，加性模型仍可能有正的残差自由度，
  但无法区分交互变异与单元格内部误差，也就不能单独检验交互作用；此时推断依赖加性假设。
- `backend="auto"`：本教程显式使用 `backend="numpy"`。GPU 单元格请显式选择
  `"cupy"` 或 `"torch"`；此函数的 `"auto"` 不检查嵌套单元格数组，会选择 NumPy。
  Torch 张量应已放到目标设备，完整行为见 [API 参考](../../reference/anova-api.md)。
- `dtype=None`：默认使用后端的 float64。当较大数值中的微小差异容易受舍入影响时，
  建议保留默认精度。
- 当残差均方为零时，正的效应均方对应无穷大的 F 和为 0 的 P 值；若两者均为零，
  F 和 P 值未定义。这是数据退化导致的结果，不能理解为已经毫无疑问地证实了效应。
- 此接口不提供 Type I、II 或 III 平方和选项，因此拒绝不等单元格样本量。
  不要单纯为了满足平衡要求而删除观测，应选择适合真实设计和研究假设的方法。
- 这里检验的是固定效应，不估计随机效应方差分量，也不处理重复测量。
  `f_twoway` 不提供公式接口，也不自动计算简单效应或对比。

发现显著交互作用后，应先明确要比较什么，例如在某个固定浇水水平下比较两种肥料，
再选择相应的多重检验控制方法。把每个单元格当作一个组进行全部两两比较，
得到的是另一个比较族，不能替代主效应或交互作用检验。

<a id="references"></a>

## 参考资料与后续阅读

- [NIST：双因素方差分析的模型与计算](https://www.itl.nist.gov/div898/handbook/prc/section4/prc438.htm)。
- [NIST：方差分析所检验的假设](https://www.itl.nist.gov/div898/handbook/prc/section4/prc43.htm)。
- Cohen, J.（1988）。*Statistical Power Analysis for the Behavioral Sciences*，第 2 版。
- [经典单因素检验](anova-oneway.md)、[Welch 检验](anova-welch.md)、[事后比较](anova-posthoc.md)
  和 [ANOVA 完整 API 参考](../../reference/anova-api.md)。
