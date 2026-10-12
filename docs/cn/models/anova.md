# ANOVA：比较组均值

> 语言：中文\
> 最后更新：2026-10-12\
> 切换：[English](../../en/models/anova.md)

<a id="anova"></a>
<a id="概览"></a>

## ANOVA 回答什么问题？

假设要比较几种处理下的连续测量结果。即使各组总体均值相同，抽到的样本均值也会
有所不同。方差分析（ANOVA）考察的是：相对于组内观测值的自然波动，组间均值差异
是否大到难以用随机误差解释。

对于有 $k$ 个水平的单因素设计，总体检验的原假设为

$$
H_0: \mu_1=\mu_2=\cdots=\mu_k.
$$

在模型假设成立时，较小的 P 值提供反对这个联合原假设的证据。它不能直接指出哪一对
均值不同，不能衡量差异大小，也不能证明因果关系。不拒绝原假设同样不能证明均值相等：
样本过少或测量噪声较大时，有实际意义的差异也可能难以检出。

## 为什么比较方差能判断均值差异？

经典模型把每个观测值写成组均值与随机误差之和。组内变异估计误差方差；组间变异
既包含这种噪声，也可能包含总体均值不同带来的信号。两类平方和分别除以各自自由度，
再取比值，就得到 F 统计量。原假设下，过大的比值不容易仅由噪声解释。
[单因素教程](anova-oneway.md)给出具体分解，并说明如何结合效应量解释检验结果。

假设针对的是采样设计与误差：观测相互独立，通常的 F 模型要求组内误差近似正态，
经典 ANOVA 还要求各组总体方差相等。正态性并不是要求把所有组混在一起后呈现正态分布。
Welch ANOVA 放宽等方差假设，但不能解决观测相关、重复测量、严重离群值或结果变量
不适合该模型的问题。

## 根据设计选择分析方法

| 问题与设计 | 从这里开始 | 重要边界 |
|---|---|---|
| 一个因素定义的独立组，在共同误差方差下是否均值相等？ | [经典单因素 ANOVA](anova-oneway.md#anova-basic)，`f_oneway` | 各组样本量可以不同；仍需独立性与总体等方差假设 |
| 独立组方差不等时，是否均值相等？ | [Welch ANOVA](anova-oneway.md#anova-welch)，`f_welch` | 使用各组方差估计，分母自由度通常为小数 |
| 两个交叉因素如何影响结果？一个因素的作用是否取决于另一个因素？ | [平衡双因素 ANOVA](anova-twoway.md)，`f_twoway` | 所有因素组合需要相同样本量；事先判断模型是否需要交互项 |
| 共同方差模型下，具体哪些组均值不同？ | [Tukey HSD](anova-posthoc.md#anova-tukey)，`tukey_hsd` | 全部两两比较共用合并方差；样本量不等时采用 Tukey–Kramer 缩放 |
| 不要求等方差时，具体哪些组均值不同？ | [Bonferroni 两两 Welch 检验](anova-posthoc.md#anova-bonferroni)，`bonferroni` | 全部两两检验组成一个比较族；返回的 P 值是原始值 |
| 样本中的分组差异有多大？ | [效应量](anova-oneway.md#effect-sizes)，`cohens_f`、`partial_eta_squared` | 效应大小与统计显著性回答不同问题 |

应根据研究问题与采样设计选择模型，不应尝试多个方法后只保留最小的 P 值。
解释检验前，先查看组样本量、分布与方差差异。同一受试者出现在多个组中时，
这些独立组函数不会建模其观测之间的相关性。

## 从总体检验走向有用结论

1. 明确要比较的总体均值或因素效应，在查看结果前选定显著性水平。
2. 检查设计与假设，再查看总体统计量、P 值与自由度。双因素模型中，应先考察交互作用，
   再概括对另一个因素取平均后的主效应。
3. 同时报告观测均值与合适的效应量。统计上可检出的差异，实际意义未必大。
4. 需要作两两结论时，明确比较族并采用合适的同时推断方法。看过 P 值后只挑有希望的
   几对，并不能消除多重比较问题。

总体检验显著，并不意味着后续可以直接使用未校正的两两检验。反过来，预先规划的
同时两两比较，也不必看成总体检验显著后才被允许进行的分析。
[事后比较教程](anova-posthoc.md)解释比较目标与置信区间；
[多重检验原理](multiple-testing.md)进一步解释族错误率与错误发现率。

## 支持范围

模块包含**七个公开函数**，不是七种 ANOVA：三个总体检验、两种两两比较程序和两个
效应量辅助函数。[完整 API 参考](../reference/anova-api.md)列出全部七个函数与五种
结果类型，包括 `PairwiseComparison`。

支持的设计是独立组单因素 ANOVA、Welch 单因素 ANOVA 和平衡双因素 ANOVA。
`f_twoway` 没有 Type I/II/III 平方和选择参数，单元格样本量不等时会报错。
这些函数不提供重复测量 ANOVA、ANCOVA、MANOVA、任意对比或更多因素的 ANOVA。
省略交互项会拟合加性模型，将省略的交互变异并入残差，并非仅仅隐藏输出。

所有基于观测数据的函数都会拒绝 NaN 与无穷输入，而不会静默删除观测值。
即使输入全部有限，常数数据仍可能产生未定义或无穷大的统计量。
各函数的[输入与退化情形说明](../reference/anova-api.md)区分了非法输入与未定义统计量。

<a id="后端与执行边界"></a>
<a id="strict-与-approximate"></a>
<a id="限制与失败行为"></a>
<a id="torch-输入是否必须指定-backendtorch"></a>
<a id="为什么返回的-p-值可能是-python-标量"></a>
<a id="为什么拒绝不平衡双因素设计"></a>
<a id="faq"></a>

### 数组、设备与数值限制

分组归约支持 NumPy、CuPy 和 Torch。结果由 Python 标量与比较记录组成，
不是已拟合估计器。某些最终的标量分布计算在 CPU 上执行；Tukey 的学生化极差
分布计算使用 SciPy。函数参数 `backend="torch"` 选择 Torch 后端，本身并不要求 CUDA。
双因素嵌套单元格应显式选择后端，不要依赖输入类型推断。dtype、转换与设备限制见
[API 参考](../reference/anova-api.md)；估计器所用的另一套 `device` 接口见
[设备指南](../guides/device-and-memory.md)。

ANOVA 没有供用户选择的 strict/approximate 统计模式。把标量计算移到 CPU 不会改变
检验的方差假设，也不会把 Welch 推断变成经典 ANOVA。

## 继续阅读完整示例

本页原有的详细内容已按下列主题组织。旧章节链接仍可使用，并可从对应位置进入教程
或完整参考。

<a id="单因素-anova"></a>
<a id="welch-anova"></a>
<a id="anova-basic"></a>
<a id="在-cpu-上比较两组均值"></a>
<a id="效应量"></a>

### 单因素与效应量

[单因素教程](anova-oneway.md)包括经典与 Welch 公式、自包含 CPU 示例、结果解释与
效应量注意事项。

<a id="双因素-anova"></a>

### 两个交叉因素

[双因素教程](anova-twoway.md)解释平衡单元格、主效应、交互作用、平方和，并提供
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
输入限制与后端边界。

<a id="cpu-与-gpu-示例"></a>
<a id="可选的-gpu-执行"></a>
<a id="cupy"></a>
<a id="torch-cuda"></a>

### 可选 GPU 示例

[单因素 GPU 示例](anova-oneway.md#gpu-execution)通过显式 CuPy 或 Torch CUDA 数组
执行同一个统计问题。

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
