# 多重检验与 P 值合并

> 语言：中文  
> 最后更新：2026-10-14\
> 切换：[English](../../../en/models/testing/multiple-testing.md)

本页解释为什么多次检验需要额外考虑错误率、如何选择方法，以及应到哪篇教程深入学习。
完整的函数签名、返回形状、别名与后端选项见 [API 参考](../../guides/multiple-testing-combine-pvalues.md)。

## 为什么每次都用 5% 阈值还不够

假设所有原假设都成立，每次检验恰有 5% 的概率错误地拒绝原假设。
若各次检验**相互独立**，一次都不犯错的概率可以相乘，因此

$$P(\text{至少发生一次错误拒绝})=1-(1-0.05)^m.$$

检验五次时，这个概率约为 22.6%；检验二十次时，约为 64.2%。
原来的 5% 阈值约束的是每次检验，并没有把整组检验的错误风险限制在 5%。
若检验之间存在依赖关系，上面的乘积公式未必成立，但分别使用 5% 阈值，
通常仍不能保证整组检验的族错误率不超过 5%。

多重检验校正考虑了“多次获得假阳性结果”的机会。它不能修复原始检验校准不当、
模型假设无效，或看到 P 值后才将部分已完成的检验排除在校正假设族之外的问题。

## 先分清两个问题

需要对一组假设逐项作出判断时，使用 `adjust_pvalues`；需要把多个检验的证据汇总为一个总体原假设的检验时，使用 `combine_pvalues`。合并后的 P 值显著，并不能确定具体哪些原假设不成立，也不等于控制了逐项发现的 FDR。

例如，“这五个指标中，具体哪些发生了变化？”需要逐项校正后的判断；
“是否有证据表明至少一个指标发生了变化？”则可以通过总体检验回答。
把多个研究针对同一科学原假设的证据合并起来，也是总体检验的用途之一。
校正和合并都不估计效应大小，也不能说明发现的效应是否具有实际意义。

## 先选择要控制的错误率

对于逐项发现，设 $V$ 为错误拒绝的个数，$R$ 为全部拒绝的个数：

$$\mathrm{FWER}=P(V\ge1),\qquad
\mathrm{FDR}=E\left[\frac{V}{\max(R,1)}\right].$$

- **将族错误率 FWER 控制在 5%：** 预先确定的假设族中，出现至少一次错误拒绝的
  概率不超过 5%。当一次错误结论也可能带来较大代价时，这通常是需要考虑的标准。
- **将错误发现率 FDR 控制在 5%：** 在全部拒绝中，错误拒绝所占比例的期望不超过 5%；
  没有拒绝时，该比例定义为零。这里的期望是对重复获取的数据集而言，
  **不保证每次得到的发现列表中错误比例都不超过 5%**，也不是说每个发现都有 5% 的
  概率是错的。如果筛选阶段能够容忍部分假线索，并会进一步验证，FDR 往往更合适。

两种保证都要求单项 P 值有效：在真实原假设下，对 $0\le t\le1$，应有
$P(p_i\le t)\le t$。一个数落在 `[0,1]` 内，并不足以说明它是校准正确的 P 值。
此外，还必须满足所选方法的依赖条件。

## 选择教程

应在查看结果前确定假设族和错误率标准，不要先只保留观测到的较小 P 值，
再对这个子集进行校正。

| 想回答的问题 | 错误率目标与适用条件 | 教程 |
|---|---|---|
| 希望逐项拒绝假设，同时限制出现任何错误结论的风险？ | FWER；Holm/Bonferroni 允许任意依赖；Hochberg 要求独立或 Simes 条件 | [FWER：Bonferroni、Holm、Hochberg](multiple-testing-fwer.md) |
| 哪些候选项应进入发现列表？ | FDR；BH 要求独立或适当正依赖（PRDS）；BY 允许任意依赖 | [FDR：BH 与 BY](multiple-testing-fdr.md) |
| 汇总证据是否反对“所有分量原假设均成立”的交集原假设？ | 一个总体检验；校准取决于方法和原假设下的联合分布 | [P 值合并：Fisher、Stouffer、Cauchy/ACAT](pvalue-combination.md) |

它们回答不同的科学问题。应先确定错误率标准并检查假设，而不是选择拒绝更多项的方法。

## 可直接运行的 CPU 示例

下面五个示意性 P 值组成一个预先确定的假设族。单项 P 值有效时，Holm 在任意依赖
关系下控制 FWER。在选定的 5% 水平下，将校正 P 值与 `alpha` 比较（等于阈值也拒绝）。
[FWER 教程](multiple-testing-fwer.md)解释具体停止规则。

<!-- api-example: multiple-testing-learner -->
```python
import numpy as np
from statgpu.inference import adjust_pvalues
```

准备预先确定的假设族。

```python
p = np.array([0.001, 0.01, 0.03, 0.05, 0.50])
```

计算并查看拒绝判断。

```python
reject, adjusted = adjust_pvalues(p, method="holm", alpha=0.05, backend="numpy")
print(reject.tolist())
print(np.round(adjusted, 3))
```
<!-- example-end: multiple-testing-learner -->

拒绝掩码为 `[True, True, False, False, False]`，校正 P 值为
`[0.005, 0.04, 0.09, 0.1, 0.5]`，保留原假设的输入顺序。按该假设族与错误率标准，
拒绝前两个原假设。校正 P 值不是重要性排名，未拒绝也不等于证明原假设成立。

## 用 `axis` 确定假设族

下面的独立示例对照两种选择：把四项检验作为一个假设族，或把每行两项检验
分别作为一个假设族。`axis=None` 合并考虑所有元素，`axis=1` 逐行处理，
`axis=0` 逐列处理；负轴编号遵循通常的数组约定。
应根据科学问题预先确定假设族，而不是看到结果后选择拒绝更多假设的轴。
以下 Fisher 计算要求每行内的 P 值在原假设下相互独立、连续且服从均匀分布。

<!-- api-example: multiple-testing-families -->
```python
import numpy as np
from statgpu.inference import adjust_pvalues, combine_pvalues

p_matrix = np.array([[0.01, 0.04], [0.20, 0.80]])
reject_all, adjusted_all = adjust_pvalues(p_matrix, method="holm", axis=None, backend="numpy")
reject_rows, adjusted_rows = adjust_pvalues(p_matrix, method="holm", axis=1, backend="numpy")
print(reject_all.tolist())
print(np.round(adjusted_all, 4).tolist())
print(reject_rows.tolist())
print(np.round(adjusted_rows, 4).tolist())
row_stat, row_p_global = combine_pvalues(p_matrix, method="fisher", axis=1, backend="numpy")
print(np.round(row_p_global, 6).tolist())
```
<!-- example-end: multiple-testing-families -->

```text
[[True, False], [False, False]]
[[0.04, 0.12], [0.4, 0.8]]
[[True, True], [False, False]]
[[0.02, 0.04], [0.4, 0.8]]
[0.00353, 0.453213]
```

逐行校正拒绝第一行的两个假设；把四项合成一个假设族时，只拒绝第一项。
各行分别控制 FWER/FDR，**不会**自动控制所有元素合在一起的相应错误率。
如果发现结论覆盖全部四项检验，就应对整个假设族进行校正。

合并会移除指定的轴，因此 `row_p_global.shape == (2,)`。
在 5% 水平下，第一行有证据反对该行的交集原假设，第二行则没有；
这不能确定哪一个分量原假设不成立。使用 `axis=None` 时，只返回一个统计量
和一个总体 P 值。如果要在许多行级总体检验之间作出发现结论，还需对这些
总体 P 值采用适当的多重检验方法。

## 继续学习与查阅接口

- [FWER 教程](multiple-testing-fwer.md)：Bonferroni、Holm、Hochberg 的公式与判断
- [FDR 教程](multiple-testing-fdr.md)：BH、BY 的逐步上升规则及发现列表的解释
- [P 值合并教程](pvalue-combination.md)：总体原假设、校准、权重与方向
- [完整 API 参考](../../guides/multiple-testing-combine-pvalues.md)：参数、别名、形状、后端、缺失值、端点处理与数值限制
- [估计器辅助方法](../../reference/estimator-api.md)：估计器层面返回字典的接口

分析大量相关检验前，应先明确是在检验总体假设还是识别个别发现，并确定依赖条件；
应用领域名称本身不能决定应该使用哪一种方法。
