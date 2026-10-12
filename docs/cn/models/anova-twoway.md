# 平衡双因素方差分析

> 语言：中文\
> 最后更新：2026-10-12\
> 切换：[English](../../en/models/anova-twoway.md)

<a id="the-question"></a>

## 两个因素对应三个不同问题

假设你研究两种肥料与三种浇水安排对植物生长的影响。每种肥料都配合每种浇水安排，
构成六个处理组合，也称为单元格。各单元格使用独立植株时，双因素方差分析可分别回答：

1. **因素 A：** 对浇水水平等权平均后，不同肥料的总体均值是否相同？
2. **因素 B：** 对肥料水平等权平均后，不同浇水安排的总体均值是否相同？
3. **交互作用：** 肥料之间的差异是否随浇水水平改变？

交互作用表示偏离可加性的模式。如果某种肥料在频繁浇水时作用很强、在少量浇水时作用很小，
一个平均肥料差异就不能概括完整规律。有交互作用时，应先查看各单元格均值，
不要仅用主效应总结结果。

`f_twoway` 支持平衡、交叉的固定效应设计：每个因素至少有两个水平，
每种水平组合均有观测，且各单元格的观测数量相同。
它不提供不平衡设计的 Type I/II/III 方差分析、重复测量方差分析、随机效应方差分析、
协方差分析（ANCOVA）或多元方差分析（MANOVA）。
可用分析方法的选择见 [ANOVA 总览](anova.md)。

<a id="assumptions"></a>

## 检验之前先核对设计

经典 F 检验假设误差独立、各单元格内误差服从正态分布，并且各单元格的总体方差相同。
应检查单元格模式和残差特征；函数不会自动完成这些诊断。

- 单元格样本量相同是此接口的计算要求，不代表总体方差相同。
- 独立重复意味着不同的实验单位。同一株植物的重复读数不能当作更多独立植株。
- 完整交互模型需要单元格内重复。每格只有一个观测时，模型没有剩余误差自由度，
  会触发 `ValueError`。
- 省略交互作用意味着采用可加模型假设，不是隐藏重要交互作用或自动获取更小 p 值的方法。

<a id="model-and-tests"></a>

## 模型、平方和与检验

对于 A 的第 $i$ 个水平、B 的第 $j$ 个水平和第 $r$ 个重复，完整模型为

$$y_{ijr}=\mu+\alpha_i+\beta_j+(\alpha\beta)_{ij}+\varepsilon_{ijr}.$$

设 A 有 $a$ 个水平，B 有 $b$ 个水平，每格有 $n$ 个观测，则 $N=abn$。
记 $\bar y_{ij}$ 为单元格均值，$\bar y_{i\cdot}$ 与 $\bar y_{\cdot j}$
为行均值与列均值，$\bar y$ 为总均值。平衡设计可将变异分解为

$$SS_A=bn\sum_i(\bar y_{i\cdot}-\bar y)^2,\qquad
SS_B=an\sum_j(\bar y_{\cdot j}-\bar y)^2,$$

$$SS_{AB}=n\sum_{ij}(\bar y_{ij}-\bar y_{i\cdot}-\bar y_{\cdot j}+\bar y)^2,
\qquad SS_E=\sum_{ijr}(y_{ijr}-\bar y_{ij})^2.$$

总平方和满足 $SS_T=SS_A+SS_B+SS_{AB}+SS_E$。
交互项对应两个主效应都无法解释的单元格均值变化。
对每个效应，将平方和除以自由度得到均方，再除以 $MS_E=SS_E/df_E$：

| 项 | 分子自由度 | F 统计量 |
|---|---:|---|
| A | $a-1$ | $(SS_A/(a-1))/MS_E$ |
| B | $b-1$ | $(SS_B/(b-1))/MS_E$ |
| A × B | $(a-1)(b-1)$ | $(SS_{AB}/((a-1)(b-1)))/MS_E$ |
| 误差 | $ab(n-1)$ | 所有检验的共同分母 |

各 p 值来自相应效应的 F 分布右尾概率，没有对这些检验联合进行多重校正。
如果决策需要控制多个效应、响应或实验构成的检验族错误率，应提前定义检验族，
并阅读[多重检验](multiple-testing.md)。
[NIST 双因素方差分析说明](https://www.itl.nist.gov/div898/handbook/prc/section4/prc438.htm)
给出了平衡析因设计的平方和分解。

<a id="anova-twoway"></a>

## 完整的析因设计 CPU 示例

本示例依次演示完整模型、效应量解释和可加模型比较。
按顺序运行代码即可，不需要其他页面中的变量。首先导入函数。

<!-- example: anova-twoway -->
```python
import numpy as np
from statgpu.anova import f_twoway, partial_eta_squared
```

建立单元格的嵌套列表：行表示肥料（A），列表示浇水（B），每格是一维数组，
包含 12 个独立观测。请用 Python 列表或元组组织行，而不要直接传入三维数组。
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

实际单元格均值约为

| 肥料 | 浇水 0 | 浇水 1 | 浇水 2 |
|---|---:|---:|---:|
| A 水平 0 | 10.217 | 10.412 | 10.487 |
| A 水平 1 | 9.787 | 11.745 | 13.170 |

完整模型的结果为

| 效应 | F | p 值 | 分子自由度 | eta 平方 |
|---|---:|---:|---:|---:|
| A | 24.963 | 4.55e-06 | 1 | 0.157 |
| B | 19.649 | 2.02e-07 | 2 | 0.248 |
| A × B | 14.192 | 7.47e-06 | 2 | 0.179 |

残差自由度为 `66`，残差平方和为 `67.960`。
交互检验支持肥料差异随浇水水平变化：第二种肥料在浇水 0 下约低 `0.430` 个单位，
而在浇水 2 下约高 `2.682` 个单位。主效应检验总结跨另一因素的平均差异，
不意味着每个水平下都有相同差异。这些整体检验也没有直接提供上述单元格差值的
置信区间或 p 值。

<a id="effect-sizes"></a>

## 普通 eta 平方与偏 eta 平方的分母不同

返回的 `factor_a_eta_squared`、`factor_b_eta_squared` 和
`interaction_eta_squared` 都是普通 eta 平方：

$$\eta_A^2=SS_A/SS_T,\qquad \eta_{AB}^2=SS_{AB}/SS_T.$$

例如，交互项约占总样本变异的 17.9%。偏 eta 平方则为
$SS_{effect}/(SS_{effect}+SS_E)$，其分母排除了其他效应。
接口提供残差平方和，因此可用 F、分子自由度及残差均方还原相应效应的平方和：

$$SS_{effect}=F_{effect}\,df_{effect}\,(SS_E/df_E).$$

继续使用上面的 `result`，计算交互作用的偏 eta 平方。
这里要求 F 统计量有限且数据不是退化情况。

```python
mse = result.ss_within / result.df_within
ss_interaction = result.interaction_statistic * result.interaction_df * mse
partial_interaction = partial_eta_squared(ss_interaction, result.ss_within)
print(f"interaction SS={ss_interaction:.3f}")
print(f"partial eta_squared={partial_interaction:.3f}")
```

交互平方和为 `29.226`，偏 eta 平方为 `0.301`，不同于普通 eta 平方 `0.179`。
两者都不是因果比例或不确定性区间，不同效应的偏 eta 平方也不应直接相加。
跨研究或跨软件比较时，应使用相同定义。

<a id="additive-model"></a>

## 省略交互作用会发生什么

设置 `interaction=False` 后，模型变为 $\mu+\alpha_i+\beta_j$。
原先由交互项解释的变异进入残差：

$$SS_{E,add}=SS_E+SS_{AB},\qquad
df_{E,add}=N-a-b+1=df_E+(a-1)(b-1).$$

若此时仍只使用单元格内部误差，就会丢掉可加模型无法解释的变异。
所以残差平方和和残差自由度都需要改变，主效应 F 检验也必须重新计算。
这个选项并不只是隐藏输出中的交互项。

```python
additive = f_twoway(data, interaction=False, backend="numpy")
print(f"A: F={additive.factor_a_statistic:.3f}")
print(f"B: F={additive.factor_b_statistic:.3f}")
print(f"residual df={additive.df_within}, SS={additive.ss_within:.3f}")
print(additive.interaction_statistic, additive.interaction_pvalue)
```
<!-- example-end: anova-twoway -->

可加模型的残差平方和为 `97.186`，对应未舍入的 `67.960 + 29.226`，
残差自由度为 `68`。主效应 F 值变为 `17.985` 和 `14.156`，交互字段为 `None`。
普通主效应 eta 平方保持不变，因为平衡设计的主效应平方和及总平方和没有改变。
本例中可见且有统计证据支持的交互模式，说明不宜仅用可加模型概括数据。

<a id="parameters-and-pitfalls"></a>

## 重要参数与使用边界

- `data`：矩形的嵌套列表或元组，每格存放一维样本，必须包含所有 A × B 组合。
  空单元格、各行长度不同、单元格样本量不等、任一因素少于两个水平，
  以及非有限观测都会触发 `ValueError`。
- `interaction=True`：包含并检验交互作用。设为 `False` 后，四个交互结果字段均为
  `None`，残差包括交互变异。每格只有一个观测时，可加模型仍可能有正残差自由度，
  但无法单独检验交互作用与误差，推断依赖可加性假设。
- `backend="auto"`：本教程显式使用 `backend="numpy"`。GPU 单元格请显式选择
  `"cupy"` 或 `"torch"`；此函数的 `"auto"` 不检查嵌套单元格数组，会选择 NumPy。
  Torch 张量应已放到目标设备，完整行为见 [API 参考](../reference/anova-api.md)。
- `dtype=None`：默认使用后端的 float64。当较大数值中的微小差异容易受舍入影响时，
  建议保留默认精度。
- 当残差均方为零时，正的效应均方对应无限大的 F 和零 p 值；若两者均为零，
  F 和 p 值未定义。这是退化数据配置，不代表无限强的推断确定性。
- 此接口不提供 Type I、II 或 III 平方和选项，因此拒绝不等单元格样本量。
  不要单纯为了满足平衡要求而删除观测，应选择适合真实设计和研究假设的方法。
- 这里检验的是固定效应，不估计随机效应方差分量，也不处理重复测量。
  `f_twoway` 不提供公式接口，也不自动计算简单效应或对比。

发现显著交互作用后，应先定义有科学意义的条件比较，再选择多重检验控制方案。
将每个单元格当作一个组进行全体两两比较，会改变比较族，不能直接复现主效应或交互检验。

<a id="references"></a>

## 参考资料与后续阅读

- [NIST：双因素方差分析的模型与计算](https://www.itl.nist.gov/div898/handbook/prc/section4/prc438.htm)。
- [NIST：方差分析所检验的假设](https://www.itl.nist.gov/div898/handbook/prc/section4/prc43.htm)。
- Cohen, J.（1988）。*Statistical Power Analysis for the Behavioral Sciences*，第 2 版。
- [单因素与 Welch 检验](anova-oneway.md)、[事后比较](anova-posthoc.md)
  和 [ANOVA 完整 API 参考](../reference/anova-api.md)。
