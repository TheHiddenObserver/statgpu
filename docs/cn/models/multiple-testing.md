# 多重检验与 P 值合并

> 语言：中文  
> 最后更新：2026-10-05  
> 切换：[English](../../en/models/multiple-testing.md)

## 先分清两个问题

需要对一组假设逐项作出判断时，使用 `adjust_pvalues`；需要把多个检验的证据汇总为一个总体原假设的检验时，使用 `combine_pvalues`。合并后的 P 值显著，并不能确定具体哪些原假设不成立，也不等于控制了逐项发现的 FDR。

设 V 为错误拒绝的个数，R 为全部拒绝的个数。族错误率 FWER 是 `P(V >= 1)`，错误发现率 FDR 是 `E[V / max(R, 1)]`。这些保证以单项 P 值有效、且满足所选方法的依赖条件为前提；增加检验数量不会修复无效的原始检验。

## 先选择要控制的错误率

- **Holm 或 Bonferroni：** 在任意依赖关系下控制 FWER。对同一假设族，Holm 的检验功效不会低于 Bonferroni。
- **BH：** 在相互独立或满足特定正依赖条件时控制 FDR，例如对真实原假设满足 PRDS 条件；不能对所有依赖结构作此保证。
- **BY：** 通过额外的调和因子，在任意依赖关系下控制 FDR。
- **Hochberg：** 在独立或能够保证 Simes 不等式成立的依赖条件下控制 FWER；仅知道两两相关系数非负，并不足以普遍保证适用。

应在查看结果前确定假设族。对矩阵设置 `axis=1` 时，每行分别构成一个假设族，并不同时控制所有行的总体错误率。[R 的 p.adjust 文档](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/p.adjust.html)介绍了这些方法及其依赖条件。

## 校正公式

将 P 值排序为 $p_{(1)}\le\cdots\le p_{(m)}$，令 $c_m=\sum_{j=1}^m1/j$。下列计算完成后，再恢复输入顺序：

$$
\begin{aligned}
\tilde p^{\rm Bonf}_{(i)} &= \min(1,mp_{(i)}),\\
\tilde p^{\rm Holm}_{(i)} &= \min\{1,\max_{j\le i}(m-j+1)p_{(j)}\},\\
\tilde p^{\rm BH}_{(i)} &= \min\{1,\min_{j\ge i}(m/j)p_{(j)}\},\\
\tilde p^{\rm BY}_{(i)} &= \min\{1,c_m\min_{j\ge i}(m/j)p_{(j)}\},\\
\tilde p^{\rm Hochberg}_{(i)} &= \min\{1,\min_{j\ge i}(m-j+1)p_{(j)}\}.
\end{aligned}
$$

返回的拒绝规则是 `adjusted <= alpha`。Holm 逐步下降检验在第一次比较不通过时停止拒绝；未拒绝原假设不代表已经证明它成立。

## 合并公式与适用条件

**Fisher：**

$$T_F=-2\sum_{i=1}^m\log p_i,\qquad p_{\rm global}=P(\chi^2_{2m}\ge T_F).$$

通常的卡方参考分布要求原假设下的 P 值相互独立、连续且服从均匀分布。此接口不估计依赖关系的修正。`weights` 必须为 `None`；提供权重会抛出 `ValueError`。

**Stouffer：** 对非负且总和为正的权重，

$$T_Z=\frac{\sum_i w_i\Phi^{-1}(1-p_i)}{\sqrt{\sum_iw_i^2}},\qquad p_{\rm global}=1-\Phi(T_Z).$$

分母采用原假设下标准正态分数相互独立的假设。若分数相关，需要考虑协方差的校准，此接口并未实现。研究有方向的效应时，应使用方向一致的单侧 P 值；传入双侧 P 值不会自动恢复效应正负号。[SciPy 的合并检验文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.combine_pvalues.html)说明了独立检验的校准条件。

**Cauchy/ACAT：** 实现先将权重归一化为 $a_i=w_i/\sum_jw_j$，未提供时等权，再计算

$$T_C=\sum_i a_i\tan\{\pi(1/2-p_i)\},\qquad p_{\rm global}=1/2-\arctan(T_C)/\pi.$$

存在依赖关系时，这是在一定正则条件下的尾部近似，并不意味着统计量精确服从 Cauchy 分布，也不对任意联合分布、任意显著性水平提供统一保证。理论范围见 [Liu–Xie 论文](https://arxiv.org/abs/1808.09011)。它用于汇总证据，不能代替逐项发现的多重检验校正。

## 可直接运行的 CPU 示例

这里使用示意性的 P 值；Fisher 结果的解释以相关检验满足适当的独立性假设为前提。

<!-- api-example: multiple-testing-learner -->
```python
import numpy as np
from statgpu.inference import adjust_pvalues, combine_pvalues

p = np.array([0.001, 0.01, 0.03, 0.05, 0.50])
reject, adjusted = adjust_pvalues(p, method="holm", alpha=0.05, backend="numpy")
print(reject.tolist())
print(np.round(adjusted, 3))
statistic, global_p = combine_pvalues(p, method="fisher", backend="numpy")
print(round(float(statistic), 4), round(float(global_p), 6))
```

第一行输出为 `[True, True, False, False, False]`，校正值为 `[0.005, 0.04, 0.09, 0.1, 0.5]`。应按预先确定的假设族和错误率解释拒绝掩码，而不是把校正 P 值当成“重要性”排名。

## 实际使用限制

输入应为 `[0,1]` 内的有限 P 值，`alpha` 应为严格介于 0 和 1 之间的有限数。当前 `alpha=NaN` 不会报错，而会返回全为假的拒绝掩码；若显著性水平由计算得到，应先检查其有效性。无效的 P 值会被拒绝。

合并检验会截断端点概率，分布尾部计算也可能受到浮点相消影响；返回零不代表真实概率精确等于零。[完整 API 指南](../guides/multiple-testing-combine-pvalues.md)说明了签名、别名、权重、轴、后端和数值限制。估计器辅助方法返回字典，与模块函数的元组不同，见[共享估计器方法](../reference/estimator-api.md)。

使用 Cauchy/Stouffer 时，还应在调用前[验证并缩放较大的权重](../guides/multiple-testing-combine-pvalues.md#validate-and-rescale-combination-weights)：即使每个权重都有限，当前归一化求和仍可能溢出，产生错误的有限结果或 NaN。

分析大量相关检验前，应先明确是在检验总体假设还是识别个别发现，并确定依赖条件；“全基因组关联分析”等应用名称本身不能决定应该使用哪一种方法。
