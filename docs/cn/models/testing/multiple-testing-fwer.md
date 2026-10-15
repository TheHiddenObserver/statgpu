# 族错误率控制：Bonferroni、Holm 与 Hochberg

> 语言：中文  
> 最后更新：2026-10-15  
> 切换：[English](../../../en/models/testing/multiple-testing-fwer.md)

如果一组检验中即使出现一次错误结论也可能带来较大代价，就需要控制
“至少发生一次错误拒绝”的概率。族错误率定义为

$$\mathrm{FWER}=P(V\ge1),$$

其中 $V$ 是预先确定的假设族中错误拒绝的个数。将 FWER 控制在 5%，
约束的是重复获取数据时发生上述事件的概率；并不是说某一次拒绝
有 5% 的概率属于错误拒绝。可先阅读[多重检验概览](multiple-testing.md)，
区分逐项发现与总体检验，并确定假设族。

## 在查看结果前选择方法

三种方法都要求单项 P 值有效：对每个真实原假设，
$P(p_i\le t)\le t$，其中 $0\le t\le1$。一个数落在 `[0,1]` 内，
并不足以说明它是有效的 P 值。校正不能修复无效的原始检验，
也不能因为某些已完成检验的 P 值较大，就将它们排除在假设族之外。

- Bonferroni 对全部 $m$ 项检验使用相同的 $\alpha/m$ 阈值，
  在任意依赖关系下控制 FWER。
- Holm 在较小的 P 值依次通过比较后逐步放宽阈值，
  同样在任意依赖关系下控制 FWER。对同一假设族，
  Holm 的检验功效不会低于 Bonferroni。
- Hochberg 使用逐步上升规则，在独立或能够保证 Simes 不等式成立的
  依赖条件下控制 FWER；仅知道两两相关系数非负，并不足以普遍保证适用。

如果筛选阶段可以容忍部分假线索，可考虑
[BH 或 BY 的 FDR 控制](multiple-testing-fdr.md)。不能仅因为某种方法
恰好拒绝了更多假设，就忽略其适用条件。
[R 的 p.adjust 文档](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/p.adjust.html)
介绍了这些方法及其依赖条件。

## 校正公式

### 用 Holm 逐项作出判断

在预先选定的 5% 水平下控制族错误率。Holm 先将 P 值从小到大排序，
再将第 $i$ 项与 $\alpha/(m-i+1)$ 比较：

| 排序位置 | 排序后的 P 值 | 5% 水平下的 Holm 阈值 | 判断 |
|---|---|---|---|
| 1 | 0.001 | 0.05 / 5 = 0.0100 | 拒绝，继续比较 |
| 2 | 0.010 | 0.05 / 4 = 0.0125 | 拒绝，继续比较 |
| 3 | 0.030 | 0.05 / 3 ≈ 0.0167 | 不拒绝，停止 |
| 4 | 0.050 | 0.05 / 2 = 0.0250 | 已停止，不拒绝 |
| 5 | 0.500 | 0.05 / 1 = 0.0500 | 已停止，不拒绝 |

这是**逐步下降**规则：从第 1 项开始，只要
$p_{(i)} \leq \alpha / (m - i + 1)$，就拒绝该项并继续比较。
首次不满足条件时立即停止，该项及后续假设均不拒绝。等于阈值时也通过比较；
若全部比较都通过，则拒绝全部假设。接口通过校正 P 值表达同一规则。

将 P 值排序为 $p_{(1)}\le\cdots\le p_{(m)}$，对应的校正值为

$$\tilde p^{\rm Holm}_{(i)}=\min\{1,\max_{j\le i}(m-j+1)p_{(j)}\}.$$

累积最大值保留了停止规则。结果会恢复到原来的假设顺序，
以 `adjusted <= alpha` 作为拒绝规则。

下面的独立 CPU 示例用五个检验组成一个预先确定的假设族。
按顺序运行三段代码即可。

**计算步骤**

1. 用 `axis` 确定假设族：`None` 将全部元素合为一族；整数轴则沿该轴
   分别校正每个切片。设族内检验数为 $m$。输入须来自有效检验，
   P 值有限且位于 `[0,1]`，并选择有限的 $0<\alpha<1$。
2. 将族内 P 值按升序排列，保留原始位置，并赋予排序位置 $i=1,\ldots,m$。
3. 乘以剩余检验数对应的因子，得到 $r_i=(m-i+1)p_{(i)}$。
4. 从第 1 项到第 $m$ 项取累积最大值 $q_i=\max_{j\le i}r_j$，
   再将大于 1 的值截为 1。
5. 恢复原始位置和输入形状，返回校正值及 `adjusted <= alpha` 的拒绝掩码；
   等于阈值时也拒绝。

<!-- api-example: multiple-testing-fwer -->
```python
import numpy as np
from statgpu.inference import adjust_pvalues
```

为每个假设提供一个 P 值，数组顺序对应这五个假设的顺序。

```python
p = np.array([0.001, 0.01, 0.03, 0.05, 0.50])
```

在预先选定的 5% 族错误率水平下应用 Holm。

```python
reject, adjusted = adjust_pvalues(p, method="holm", alpha=0.05, backend="numpy")
print(reject.tolist())
print(np.round(adjusted, 3))
```
<!-- example-end: multiple-testing-fwer -->

拒绝掩码为 `[True, True, False, False, False]`，校正值为
`[0.005, 0.04, 0.09, 0.1, 0.5]`。按所设定的假设族与错误率，拒绝前两个原假设。
校正 P 值不是重要性排名，未拒绝也不等于证明原假设成立。

### Bonferroni：对整个假设族使用同一阈值

如果希望控制任何一次错误拒绝的风险，又不对依赖结构作出假设，
Bonferroni 是一个简单的基准方法。对每项检验使用 $\alpha/m$ 阈值，
则 $m$ 项错误拒绝概率上界之和不超过 $\alpha$。
由概率的并集上界，至少发生一次错误拒绝的概率也不超过 $\alpha$。

对于每个原始 P 值，校正值为

$$\tilde p^{\rm Bonf}_i=\min(1,mp_i).$$

与 Holm 不同，其他检验通过比较后，这个阈值也不会放宽。
下面的独立示例使用同一组示意性的检验。

**计算步骤**

1. 用 `axis` 确定假设族（`None` 合并全部元素，整数轴分别处理各切片），
   并计算族内检验数 $m$。P 值须来自有效检验、有限且位于 `[0,1]`，
   显著性水平须有限且满足 $0<\alpha<1$。
2. 将每个原始 P 值乘以 $m$，得到 $r_i=mp_i$。
   这一数学过程不需要排序、排序位置或累积运算。
3. 将大于 1 的结果截为 1，保持原始假设顺序和输入形状。
4. 返回校正值及 `adjusted <= alpha` 的拒绝掩码，包含等于阈值的情况。

<!-- api-example: multiple-testing-bonferroni -->
```python
import numpy as np
from statgpu.inference import adjust_pvalues
```

为每个假设提供一个 P 值，数组顺序对应这五个假设的顺序。

```python
p = np.array([0.001, 0.01, 0.03, 0.05, 0.50])
```

在相同的 5% 族错误率水平下应用 Bonferroni。

```python
reject_bonferroni, adjusted_bonferroni = adjust_pvalues(
    p, method="bonferroni", alpha=0.05, backend="numpy",
)
print(reject_bonferroni.tolist())
print(np.round(adjusted_bonferroni, 3))
```
<!-- example-end: multiple-testing-bonferroni -->

拒绝掩码为 `[True, True, False, False, False]`，校正值为
`[0.005, 0.05, 0.15, 0.25, 1.0]`。等于阈值时也拒绝，
因此第二项在 5% 水平下通过比较。此例中 Holm 拒绝相同的两项，
但若干假设的校正值更小；两种方法在本例中得出相同判断，并不代表它们等价。

### Hochberg 的逐步上升判断

在前述独立性或 Simes 条件下，从最大的 P 值开始，找到满足
$p_{(k)} \leq \alpha / (m - k + 1)$ 的最大排序位置 $k$，
拒绝第 1 项到第 $k$ 项的全部假设，等于阈值时也通过比较。
若没有任何位置通过，就不拒绝任何假设。例如，排序后的 P 值为
`[0.0625, 0.0625, 0.5]`、`alpha=0.125` 时，第 3 项不通过，
第 2 项恰好等于阈值，因此拒绝前两项假设。

相应的校正值为

$$\tilde p^{\rm Hochberg}_{(i)}=\min\{1,\min_{j\ge i}(m-j+1)p_{(j)}\}.$$

反向累积最小值保留了逐步上升的判断。虽然逐项阈值与 Holm 相同，
比较方向和停止规则却不同。下面的例子中，Holm 会因为
$0.0625>0.125/3$ 而在第 1 项停止；Hochberg 则因第 2 项通过比较，
可以拒绝前两项假设。

**计算步骤**

1. 用 `axis` 确定假设族（`None` 合并全部元素，整数轴分别处理各切片）。
   族内 $m$ 个 P 值须来自有效检验、有限且位于 `[0,1]`，并选择有限的
   $0<\alpha<1$；同时检查上文的依赖条件。
2. 将 P 值按升序排列，保留原始位置，并赋予排序位置 $i=1,\ldots,m$。
3. 计算 $r_i=(m-i+1)p_{(i)}$，其位置因子与 Holm 相同。
4. 从第 $m$ 项倒推到第 1 项取累积最小值 $q_i=\min_{j\ge i}r_j$，
   再将大于 1 的值截为 1。
5. 恢复原始位置和输入形状，返回校正值及 `adjusted <= alpha` 的拒绝掩码。
   实现逐步上升判断的是反向累积最小值，而不是 Holm 的正向累积最大值。

<!-- api-example: multiple-testing-hochberg -->
```python
import numpy as np
from statgpu.inference import adjust_pvalues
```

建立上面阈值边界示例中的三个 P 值。

```python
p_hochberg = np.array([0.0625, 0.0625, 0.5])
```

在本阈值相等示例所选的 12.5% 水平下应用 Hochberg。

```python
reject_hochberg, adjusted_hochberg = adjust_pvalues(
    p_hochberg, method="hochberg", alpha=0.125, backend="numpy",
)
print(reject_hochberg.tolist())
print(adjusted_hochberg.tolist())
```
<!-- example-end: multiple-testing-hochberg -->

拒绝掩码为 `[True, True, False]`，校正值为
`[0.125, 0.125, 0.5]`。前两项的校正值都恰好等于所选水平，
因此均被拒绝。这个例子展示逐步上升的判断，不会消除 Hochberg 的依赖条件要求。

## 结合假设族解释判断

每个校正值都可以直接与所选水平比较，无须自行排序再恢复原顺序。
在相关假设成立时，拒绝提供了反对该原假设的证据，但不估计效应大小或实际意义。
未拒绝也不等于证明原假设成立。

逐行分别校正不会同时控制所有行的总体错误率。
请结合[假设族选择示例](multiple-testing.md#用-axis-确定假设族)，
在调用函数前确定哪些检验应放在一起。

完整的函数签名、别名、返回形状、轴、后端选择、缺失值处理与数值限制见
[API 指南](../../guides/multiple-testing-combine-pvalues.md)。

