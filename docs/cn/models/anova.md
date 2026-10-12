# ANOVA：比较组均值

> 语言：中文\
> 最后更新：2026-10-12\
> 切换：[English](../../en/models/anova.md)

<a id="anova"></a>
<a id="概览"></a>

## ANOVA 回答什么问题？

假设你想比较几种处理下的连续测量结果。即使各组总体均值相同，抽到的样本均值
也往往不同。方差分析（ANOVA）要回答的是：与组内观测值的自然波动相比，
组间均值差异是否大到难以用随机误差解释？

对于有 $k$ 个水平的单因素设计，总体检验的原假设为

$$
H_0: \mu_1=\mu_2=\cdots=\mu_k.
$$

在模型假设成立时，较小的 P 值支持“各组总体均值并非全部相等”这一结论。
至于具体哪两组不同、差异有多大、是否存在因果关系，还需要进一步分析。
反过来，未拒绝原假设也不能证明均值相等：样本过少或测量噪声较大时，
有实际意义的差异也可能难以检出。

## 为什么比较方差能判断均值差异？

经典模型把每个观测值写成所属组的总体均值与随机误差之和。组内变异反映随机噪声，
组间变异则可能还包含总体均值不同带来的差异。将两类变异各自的平方和除以自由度，
再用组间均方除以组内均方，就得到 F 统计量。若总体均值相等，这个比值通常不会太大。
[单因素教程](anova-oneway.md)将具体说明这一步分解，以及如何结合效应量解释结果。

使用前需要核对采样设计与误差假设：观测应相互独立，通常的 F 检验要求组内误差
近似正态，经典 ANOVA 还要求各组总体方差相等。正态性针对各组内部，
并不要求把所有组混在一起后仍呈正态分布。Welch ANOVA 放宽了等方差假设，
但仍需另行处理相关观测、重复测量、严重离群值，以及结果变量不适合模型的情况。

## 根据设计选择分析方法

| 问题与设计 | 从这里开始 | 使用前注意 |
|---|---|---|
| 按一个因素分组，各组独立且总体方差相等时，均值是否相等？ | [经典单因素 ANOVA](anova-oneway.md#anova-basic)，`f_oneway` | 各组样本量可以不同；仍需独立性与总体等方差假设 |
| 独立组的总体方差不等时，均值是否相等？ | [Welch ANOVA](anova-oneway.md#anova-welch)，`f_welch` | 使用各组方差估计，分母自由度通常为小数 |
| 两个交叉因素如何影响结果？一个因素的作用是否取决于另一个因素？ | [平衡双因素 ANOVA](anova-twoway.md)，`f_twoway` | 所有因素组合需要相同样本量；事先判断模型是否需要交互项 |
| 总体方差相等时，具体哪些组均值不同？ | [Tukey HSD](anova-posthoc.md#anova-tukey)，`tukey_hsd` | 全部两两比较共用合并方差；样本量不等时采用 Tukey–Kramer 调整 |
| 不要求等方差时，具体哪些组均值不同？ | [Bonferroni 两两 Welch 检验](anova-posthoc.md#anova-bonferroni)，`bonferroni` | 全部两两检验组成一个比较族；返回的 P 值是原始值 |
| 样本中的分组差异有多大？ | [效应量](anova-oneway.md#effect-sizes)，`cohens_f`、`partial_eta_squared` | 效应大小与统计显著性回答不同问题 |

选择模型应从研究问题与采样设计出发，不要尝试多个方法后只保留最小的 P 值。
解释结果前，先查看各组样本量、分布与方差差异。同一受试者出现在多个组中时，
其观测往往相关，需要使用能够处理这种相关性的方法。

## 从总体检验走向有用结论

1. 明确要比较的总体均值或因素效应，在查看结果前选定显著性水平。
2. 检查设计与假设，再查看检验统计量、P 值与自由度。双因素模型中，应先考察交互作用，
   再解释主效应，也就是对另一个因素各水平取平均后的差异。
3. 同时报告观测均值与合适的效应量。统计上可检出的差异，实际意义未必大。
4. 需要判断具体组对是否有差异时，先明确哪些比较属于同一个比较族，再选择合适的
   多重比较方法。看过 P 值后只挑最可能显著的几对，并不能消除多重比较问题。

总体检验显著，并不意味着后续可以直接使用未校正的两两检验。
预先规划好、并已考虑多重比较校正的两两分析，也不必等总体检验显著后才进行。
[事后比较教程](anova-posthoc.md)解释比较目标与置信区间；
[多重检验原理](multiple-testing.md)进一步解释族错误率与错误发现率。

## 支持范围

模块包含**七个公开函数**：三个总体检验函数、两个两两比较函数和两个效应量辅助函数。
这七个函数并不对应七种 ANOVA 设计。[完整 API 参考](../reference/anova-api.md)
列出全部函数与五种结果类型，包括表示单项比较的 `PairwiseComparison`。

支持的设计是独立组单因素 ANOVA、Welch 单因素 ANOVA 和平衡双因素 ANOVA。
`f_twoway` 没有 Type I/II/III 平方和选择参数，单元格样本量不等时会报错。
这些函数不提供重复测量 ANOVA、ANCOVA、MANOVA、任意对比或更多因素的 ANOVA。
设置 `interaction=False` 会拟合加性模型，并将交互变异并入残差，主效应检验也随之改变。

所有基于观测数据的函数都会拒绝 NaN 与无穷输入，而不会静默删除观测值。
即使输入不含 NaN 或无穷值，常数数据仍可能使统计量未定义或变为无穷大。
遇到这类结果时，请查阅各函数的[输入与退化情形说明](../reference/anova-api.md)。

<a id="后端与执行边界"></a>
<a id="strict-与-approximate"></a>
<a id="限制与失败行为"></a>
<a id="torch-输入是否必须指定-backendtorch"></a>
<a id="为什么返回的-p-值可能是-python-标量"></a>
<a id="为什么拒绝不平衡双因素设计"></a>
<a id="faq"></a>

### 数组、设备与数值限制

各组均值、方差等统计量可以使用 NumPy、CuPy 或 Torch 计算。函数直接返回由
Python 标量和比较记录组成的结果对象，不采用估计器的 `fit`/`predict` 接口。
部分最终的分布计算在 CPU 上执行，其中 Tukey 的学生化极差分布使用 SciPy。
`backend="torch"` 选择 Torch 后端，本身并不要求 CUDA。

双因素数据以嵌套单元格传入时，请显式选择后端，不要依赖输入类型推断。
dtype、数组转换与设备限制见 [API 参考](../reference/anova-api.md)；
估计器的 `device` 参数采用另一套约定，见[设备指南](../guides/device-and-memory.md)。

ANOVA 没有供用户选择的严格模式或近似模式。把标量计算移到 CPU 不会改变
检验的方差假设，也不会把 Welch 推断变成经典 ANOVA。

## 继续阅读完整示例

初次使用可以从单因素示例开始；已有明确设计时，直接选择对应教程。
需要查参数、返回字段或异常时，使用完整 API 参考。

<a id="单因素-anova"></a>
<a id="welch-anova"></a>
<a id="anova-basic"></a>
<a id="在-cpu-上比较两组均值"></a>
<a id="效应量"></a>

### 单因素与效应量

[单因素教程](anova-oneway.md)提供经典与 Welch 公式、可独立运行的 CPU 示例，
并说明如何解读检验结果与效应量。

<a id="双因素-anova"></a>

### 两个交叉因素

[双因素教程](anova-twoway.md)解释平衡设计、主效应、交互作用与平方和分解，并提供
完整的析因设计示例。

<a id="事后比较"></a>
<a id="tukey-hsd"></a>
<a id="bonferroni-两两-welch-检验"></a>

### 两两比较

[事后比较教程](anova-posthoc.md)包括 Tukey 与 Bonferroni 示例、均值差方向、
校正后的判定与同时置信区间。

<a id="路径"></a>
<a id="参数"></a>
<a id="参数-1"></a>
<a id="输出"></a>
<a id="输出-1"></a>

### 签名、字段与错误

[完整 API 参考](../reference/anova-api.md)列出所有函数、参数、默认值、结果字段、
输入要求与后端限制。

<a id="cpu-与-gpu-示例"></a>
<a id="可选的-gpu-执行"></a>
<a id="cupy"></a>
<a id="torch-cuda"></a>

### 可选 GPU 示例

[单因素 GPU 示例](anova-oneway.md#gpu-execution)将同一份数据转换为 CuPy 或
Torch CUDA 数组，演示如何在 GPU 上进行分析。

<a id="外部验证"></a>
<a id="与其他实现比较"></a>

### 与其他实现对照

[单因素教程](anova-oneway.md#comparing-implementations)说明需要对齐的方差假设与设置。
贡献者材料见[验证参考](../../../dev/reviews/pr168-model-validation-provenance.md#anova)。

## 参考文献

- Fisher, R. A. (1925). *Statistical Methods for Research Workers*.
- Welch, B. L. (1951). On the comparison of several mean values.
- Tukey, J. W. (1949). Comparing individual means in the analysis of variance.
- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences*.
