# 错误发现率控制：BH 与 BY

> 语言：中文  
> 最后更新：2026-10-14  
> 切换：[English](../../../en/models/testing/multiple-testing-fdr.md)

筛选大量假设时，往往可以接受部分假线索，再对有希望的信号进一步验证。
错误发现率（FDR）控制的目标，是被拒绝的假设中错误发现比例的期望。
对于这种筛选目标，它有时比限制任何一次错误拒绝风险的
[FWER 方法](multiple-testing-fwer.md)更合适。

可先阅读[多重检验概览](multiple-testing.md)，
判断需要逐项发现，还是只做一个总体检验。

## FDR 控制保证什么

设 $V$ 为错误拒绝的个数，$R$ 为全部拒绝的个数：

$$\mathrm{FDR}=E\left[\frac{V}{\max(R,1)}\right].$$

没有拒绝时，该比例定义为零。在方法假设成立时，5% 的 FDR 水平约束的是：
对重复获取的数据集，错误发现比例的期望不超过 5%。
这**不保证每次得到的发现列表中错误比例都不超过 5%**，
也不是说每个发现都有 5% 的概率是错的。
某一次结果中的错误发现比例可能更高。

BH 和 BY 都要求单项 P 值有效：在真实原假设下，对 $0\le t\le1$，
应有 $P(p_i\le t)\le t$。一个数落在 `[0,1]` 内，并不足以说明它是有效的 P 值。
应在查看结果前确定假设族和错误率标准，不要先只保留观测到的较小 P 值，
再对这个子集进行校正。

- BH 在相互独立或满足特定正依赖条件时控制 FDR，
  例如对真实原假设满足 PRDS 条件。它按排序位置设置阈值，
  允许将一组较小的 P 值共同选入发现列表；
  不能对所有依赖结构保证 FDR 控制。
- BY 通过额外的调和因子，在任意依赖关系下控制 FDR。
  更广泛的保护通常意味着比 BH 更少的发现。

[R 的 p.adjust 文档](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/p.adjust.html)
介绍了这些方法及其依赖条件。

## 逐步理解 FDR 方法

### 对照 BH 的 FDR 控制

如果目标是控制 FDR，而且依赖条件适用，BH 将第 $i$ 项与 $i\alpha/m$ 比较。
找到满足 $p_{(k)}\le k\alpha/m$ 的**最大**排序位置 $k$，再拒绝第 1 项到第 $k$ 项。
如果没有任何位置通过，就不拒绝任何原假设。

| 排序位置 | 排序后的 P 值 | 5% 水平下的 BH 阈值 | 比较结果 |
|---|---|---|---|
| 1 | 0.001 | 0.010 | 通过 |
| 2 | 0.010 | 0.020 | 通过 |
| 3 | 0.030 | 0.030 | 通过，且是通过的最大位置 |
| 4 | 0.050 | 0.040 | 不通过 |
| 5 | 0.500 | 0.050 | 不通过 |

将 P 值排序为 $p_{(1)}\le\cdots\le p_{(m)}$，校正值为

$$\tilde p^{\rm BH}_{(i)}=\min\{1,\min_{j\ge i}(m/j)p_{(j)}\}.$$

反向累积最小值保留了逐步上升的判断。结果恢复到原来的假设顺序，
以 `adjusted <= alpha` 作为拒绝规则。下面是一个独立的 CPU 示例，
按顺序运行各段代码即可。

<!-- api-example: multiple-testing-fdr -->
```python
import numpy as np
from statgpu.inference import adjust_pvalues
```

为每个假设提供一个 P 值，数组顺序对应这五个假设的顺序。

```python
p = np.array([0.001, 0.01, 0.03, 0.05, 0.50])
```

在预先选定的 5% FDR 水平下应用 BH。

```python
reject_bh, adjusted_bh = adjust_pvalues(p, method="bh", alpha=0.05, backend="numpy")
print(reject_bh.tolist())
print(np.round(adjusted_bh, 4))
```

BH 返回 `[True, True, True, False, False]`，校正值为
`[0.005, 0.025, 0.05, 0.0625, 0.5]`。第三项被选中，是因为这里采用了 FDR 标准，
而不是 Holm 所控制的、更严格的 FWER；这并不意味着任何场景都应选择 BH。

BH 是**逐步上升**方法，不能从最小 P 值开始比较，并在首次不通过时停止。
例如，对 `[0.02, 0.03, 0.04]`，取 `m=3`、`alpha=0.05`，
第一次比较不通过（`0.02 > 0.05 / 3`），但最大位置通过（`0.04 <= 0.05`），
因此 BH 会拒绝全部三个原假设。

```python
p_step_up = np.array([0.02, 0.03, 0.04])
reject_step_up, adjusted_step_up = adjust_pvalues(
    p_step_up, method="bh", alpha=0.05, backend="numpy",
)
print(reject_step_up.tolist())  # [True, True, True]
```
<!-- example-end: multiple-testing-fdr -->

第二个示例的校正值为 `[0.04, 0.04, 0.04]`。
即使第一次逐项比较不通过，三个假设仍会一起被选中。
较后位置可能具有更宽松的阈值，这正是 BH 逐步上升规则的关键。

### BY：用调和因子保护任意依赖关系

如果不能证明 BH 的依赖条件成立，BY 会增加一项保守校正，
而不是直接忽略依赖关系。定义

$$c_m=\sum_{j=1}^m\frac1j.$$

BY 在 $\alpha/c_m$ 水平上应用 BH 的逐步上升比较：
找到满足 $p_{(k)}\le k\alpha/(mc_m)$ 的最大排序位置 $k$，
然后拒绝第 1 项到第 $k$ 项。若没有任何位置通过，就不拒绝任何假设。
其校正值为

$$\tilde p^{\rm BY}_{(i)}=\min\{1,c_m\min_{j\ge i}(m/j)p_{(j)}\}.$$

与 BH 一样，反向累积最小值表达了逐步上升的判断。
这里 $m=5$，$c_m\approx2.2833$，因此阈值比 BH 更小。
下面的独立示例使用同样的五个示意性检验。

<!-- api-example: multiple-testing-by -->
```python
import numpy as np
from statgpu.inference import adjust_pvalues
```

为每个假设提供一个 P 值，数组顺序对应这五个假设的顺序。

```python
p = np.array([0.001, 0.01, 0.03, 0.05, 0.50])
```

在相同的 5% FDR 水平下应用 BY。

```python
reject_by, adjusted_by = adjust_pvalues(p, method="by", alpha=0.05, backend="numpy")
print(reject_by.tolist())
print(np.round(adjusted_by, 6))
```
<!-- example-end: multiple-testing-by -->

拒绝掩码为 `[True, False, False, False, False]`，校正值为
`[0.011417, 0.057083, 0.114167, 0.142708, 1.0]`。
BH 拒绝三项，而 BY 只拒绝第一项。这反映了在本假设族上取得更广泛依赖保护的代价，
并不说明 BY 给出了更准确的效应大小估计；两种方法都不估计效应大小。

## 谨慎解释发现列表

校正 P 值不是重要性排名，也不是原假设成立的概率。
未拒绝原假设不代表它已被证明成立，统计上的发现也可能没有很大的实际意义。

各行分别控制 FDR，**不会**自动控制所有行合在一起的 FDR。
参见[假设族选择示例](multiple-testing.md#用-axis-确定假设族)。
如果每行先产生一个总体合并 P 值，再从多行中报告发现，
也需要对这些总体 P 值采用适当的多重检验方法。

完整的函数签名、别名、返回形状、轴、后端选择、缺失值处理与数值限制见
[API 指南](../../guides/multiple-testing-combine-pvalues.md)。

