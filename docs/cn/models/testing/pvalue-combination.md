# P 值合并：Fisher、Stouffer 与 Cauchy

> 语言：中文  
> 最后更新：2026-10-15  
> 切换：[English](../../../en/models/testing/pvalue-combination.md)

多个检验可以共同回答一个科学问题：所有分量原假设同时成立的交集原假设，
是否与数据相符？P 值合并为这个问题给出一个统计量和一个总体 P 值。
在方法的校准条件成立时，也可以用它合并多个研究针对同一科学原假设的证据。

总体 P 值较小，不能确定具体哪些分量原假设不成立，也不能控制逐项发现的 FDR、
估计效应大小或说明实际意义。可先阅读[多重检验概览](multiple-testing.md)，
区分这一目标与使用 [FWER](multiple-testing-fwer.md) 或
[FDR](multiple-testing-fdr.md) 校正逐项作出判断的目标。

## 合并前先理解参考分布

变换公式只是合并方法的一部分。总体 P 值还需要一个适用于这些检验的
原假设参考分布。各项数字落在 `[0,1]` 内，并不足以满足这个要求。
下面 Fisher 和 Stouffer 的精确参考分布要求原假设下的 P 值相互独立、
连续且服从均匀分布；离散检验或相关证据需要额外论证。
应在查看哪些 P 值较小之前，确定要合并的检验和权重。

## 确定每次合并的假设族

下面的步骤分别用于每个包含 $m$ 个 P 值的假设族。`axis=None` 将全部元素视为
一个假设族；整数 `axis` 则沿指定轴，对每个切片单独进行一次总体检验。
输入会转换为 float64，且必须是 `[0,1]` 内的有限概率。
空的合并假设族、NaN 或无穷值会报错；缺失值不会被自动忽略。

各方法都返回 `(statistic, pvalue)`，每个假设族对应一对结果。
`axis=None` 时返回标量或零维结果；否则返回形状为输入形状去掉合并轴后的数组。
合并无需排序，也不返回逐项拒绝掩码。

## 合并公式与适用条件

### Fisher：累加对数证据

较小的 P 值会产生较大的正值 $-2\log p_i$。累加这些值，等价于用 P 值的乘积
作为反对总体原假设的证据；它并不是对 P 值取平均。

$$T_F=-2\sum_{i=1}^m\log p_i,\qquad p_{\rm global}=P(\chi^2_{2m}\ge T_F).$$

如果原假设下的 P 值连续且均匀分布，$-2\log p_i$ 服从自由度为 2 的卡方分布。
独立性使这些自由度可以相加，得到上面的 $2m$ 自由度参考分布。
这种精确校准不能直接套用到离散或相关检验。此接口不估计依赖关系的修正。
这里的 Fisher 不使用权重。

**Fisher 的计算步骤**

1. 按上述轴规则选定并验证一个假设族，以其大小作为 $m$；
   向 Fisher 传入 `weights` 会报错。
2. 将 P 值截断到 $[\mathrm{tiny},1]$，其中 `tiny` 是 float64 最小的正正规数，
   再将每项变换为 $-2\log p_i$。这一数值处理使精确的零贡献有限值，
   而不是数学极限 $+\infty$。
3. 累加变换后的贡献，得到 $T_F$。
4. 用 `chi2.sf(T_F, df=2*m)` 计算卡方分布的上尾概率，
   返回该假设族的 $T_F$ 和总体 P 值。

### 用 Fisher 回答总体问题

下面的独立 CPU 示例回答这样一个问题：合并证据是否与五个原假设同时成立的
交集原假设不相符？这种解释要求原假设下的 P 值相互独立、连续且服从均匀分布。

Fisher 累加各项 $-2\log p_i$：P 值越小，正向贡献越大，因此较大的总和提供
反对总体原假设的证据。它通过 P 值的对数合并证据，而不是对 P 值取平均。

<!-- api-example: pvalue-combination-fisher -->
```python
import numpy as np
from statgpu.inference import combine_pvalues
```

为每个假设提供一个 P 值，数组顺序对应这五个假设的顺序。

```python
p = np.array([0.001, 0.01, 0.03, 0.05, 0.50])
```

使用 Fisher 的独立原假设参考分布合并证据。

```python
statistic, global_p = combine_pvalues(p, method="fisher", backend="numpy")
print(round(float(statistic), 4), round(float(global_p), 6))
```
<!-- example-end: pvalue-combination-fisher -->

统计量约为 `37.4167`，总体 P 值约为 `0.000048`。在上述假设下，它提供了
反对交集原假设的证据，但不能确定具体哪些分量原假设不成立。

### Stouffer：累加标准化证据

先将每个 P 值转换为 $z_i=\Phi^{-1}(1-p_i)$，其中 $\Phi$ 是标准正态分布的 CDF。
较小的 P 值对应较大的正分数，大于 0.5 的 P 值则贡献负分数。
对非负且总和为正的权重，

$$T_Z=\frac{\sum_i w_i\Phi^{-1}(1-p_i)}{\sqrt{\sum_iw_i^2}},\qquad p_{\rm global}=1-\Phi(T_Z).$$

当原假设下的 P 值相互独立、连续且服从均匀分布时，转换后的分数是独立标准正态变量，
加权和的方差为 $\sum_iw_i^2$；除以其平方根，就恢复了标准正态参考分布。
若分数相关，方差还包含协方差项；此接口不估计这些项，也没有相应的分母校正。
等权时，统计量简化为 $\sum_i z_i/\sqrt m$。
研究有方向的效应时，应使用方向一致的单侧 P 值；传入双侧 P 值不会自动恢复效应正负号。
[SciPy 的合并检验文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.combine_pvalues.html)说明了独立检验的校准条件。

**Stouffer 的计算步骤**

1. 按上述轴规则选定并验证一个假设族。未提供权重时使用等权；否则提供
   与 P 值一一对应的 $m$ 个有限非负权重，且总和为正。
   同一个权重向量用于所有假设族。
2. 将权重归一化为 $a_i=w_i/\sum_jw_j$。较大权重应按下文建议在调用前缩放；
   即使各项有限，求和仍可能溢出。
3. 将 P 值截断到 $[\varepsilon,1-\varepsilon]$，其中 $\varepsilon$ 是
   float64 机器精度，再用截断后的值计算 $z_i=\Phi^{-1}(1-p_i)$。
   因此，端点和更极端的输入值会被改变。
4. 计算 $T_Z=\sum_i a_i z_i/\sqrt{\sum_i a_i^2}$。
   这个分母使用独立分数的方差，不包含协方差项。
5. 用 `norm.sf(T_Z)` 计算正态分布的上尾概率，将计算出的 P 值截断到 `[0,1]`，
   返回该假设族的 $T_Z$ 和总体 P 值。

如果正态分数所表达的证据和预先指定的相对权重符合科学问题，
可考虑使用 Stouffer。下面的独立示例给第一项检验两倍于其他各项的权重。
这只是示意性的选择，实际使用时应在观察结果前说明依据。

<!-- api-example: pvalue-combination-stouffer -->
```python
import numpy as np
from statgpu.inference import combine_pvalues
```

为每个假设提供一个 P 值，数组顺序对应这五个假设的顺序。

```python
p = np.array([0.001, 0.01, 0.03, 0.05, 0.50])
```

设置预先指定的相对权重，再合并正态分数。

```python
weights = np.array([2.0, 1.0, 1.0, 1.0, 1.0])
statistic_stouffer, global_p_stouffer = combine_pvalues(
    p, method="stouffer", weights=weights, backend="numpy",
)
print(round(float(statistic_stouffer), 4), round(float(global_p_stouffer), 6))
```
<!-- example-end: pvalue-combination-stouffer -->

标准化统计量约为 `4.2541`，总体 P 值约为 `0.000010`。
在前述独立性和校准条件成立时，它提供了反对交集原假设的证据。
更大的权重增加了第一项检验的影响，但并不使该分量成为经过逐项校正的发现，
也不估计一个共同效应。

### Cauchy/ACAT：用重尾变换突出小 P 值

正切变换会使很小的 P 值贡献很大的正值：在零附近，
$\tan\{\pi(1/2-p_i)\}\approx1/(\pi p_i)$。
因此，少数小 P 值就可能贡献大部分证据，不过接近 1 的 P 值也会贡献很大的负值。
实现先将权重归一化为 $a_i=w_i/\sum_jw_j$，未提供时等权，再计算

$$T_C=\sum_i a_i\tan\{\pi(1/2-p_i)\},\qquad p_{\rm global}=1/2-\arctan(T_C)/\pi.$$

若原假设下的 P 值相互独立、连续且服从均匀分布，权重固定、非负且归一化，
则在精确算术下，该统计量服从标准 Cauchy 分布。
存在依赖关系时，Cauchy 尾部是在一定正则条件下的近似。
例如，[Liu–Xie 论文](https://arxiv.org/abs/1808.09011)中的固定维度结果，
使用两两联合正态的原假设统计量所产生的双侧 P 值，讨论的是极限尾部性质；
它不是对任意联合分布、任意显著性水平都成立的精确校准。
它用于汇总证据，不能代替逐项发现的多重检验校正。

**Cauchy/ACAT 的计算步骤**

1. 按上述轴规则选定并验证一个假设族。未提供权重时使用等权；否则提供
   与 P 值一一对应的 $m$ 个有限非负权重，且总和为正。
   同一个权重向量用于所有假设族。
2. 将权重归一化为 $a_i=w_i/\sum_jw_j$；如果输入权重较大，
   应先缩放以免求和溢出。
3. 以 float64 机器精度 $\varepsilon$ 将 P 值截断到
   $[\varepsilon,1-\varepsilon]$，再将每个截断后的值变换为
   $\tan\{\pi(1/2-p_i)\}$。这使端点的无穷极限变为有限的数值贡献。
4. 累加加权贡献，得到 $T_C$。
5. 用 $1/2-\arctan(T_C)/\pi$ 计算 Cauchy 上尾概率，将计算出的 P 值
   截断到 `[0,1]`，返回该假设族的 $T_C$ 和总体 P 值。
   上文关于依赖关系的适用条件仍然成立。

如果预期较小的分量 P 值会提供主要证据，而且其参考分布或尾部近似有充分依据，
可以考虑 Cauchy。下面的独立示例使用相同的预设相对权重。

<!-- api-example: pvalue-combination-cauchy -->
```python
import numpy as np
from statgpu.inference import combine_pvalues
```

为每个假设提供一个 P 值，数组顺序对应这五个假设的顺序。

```python
p = np.array([0.001, 0.01, 0.03, 0.05, 0.50])
```

设置预先指定的相对权重，再应用正切变换。

```python
weights = np.array([2.0, 1.0, 1.0, 1.0, 1.0])
statistic_cauchy, global_p_cauchy = combine_pvalues(
    p, method="cauchy", weights=weights, backend="numpy",
)
print(round(float(statistic_cauchy), 4), round(float(global_p_cauchy), 6))
```
<!-- example-end: pvalue-combination-cauchy -->

统计量约为 `114.2218`，总体 P 值约为 `0.002787`。
这些结果针对总体原假设，并依赖前述校准条件。本例中得到较小的数值，
并不能证明它对相关数据在任何显著性水平下都精确有效。

### 选择合并方法及权重

确定要检验总体原假设后，应先检查校准条件，再选择统计量。
Fisher 对独立检验提供不带权重的对数证据计算；Stouffer 在独立性假设下，
通过加权正态分数调整各项证据的相对影响。
若有理由采用尾部近似，Cauchy/ACAT 可用于相关检验；它并不是对未知依赖结构的通用修正。

Stouffer 和 Cauchy 使用相对权重：全部权重乘以同一个正常数，数学结果不变。
接口会对两者的权重归一化；这一缩放在 Stouffer 的分子与分母中抵消，
Cauchy 的权重则相加为 1。零权重使该项不再贡献统计量，但对应 P 值仍须是有效输入。
函数不会从数据中学习权重。应预先指定权重，或说明其与原假设检验统计量的独立性，
不要在看到较小 P 值后才增加它们的权重。这里的 Fisher 不支持权重。

合并双侧 P 值不能恢复效应方向，总体 P 值较小也不等于估计出了一个共同效应。
如果科学问题需要效应大小及其不确定性，仅靠 P 值合并还不够。

## 同时关注检验问题与数值限制

分析大量相关检验前，应先明确是在检验总体假设还是识别个别发现，
并确定依赖条件；“全基因组关联分析”等应用名称本身不能决定使用哪一种方法。
如果多个假设族分别产生总体 P 值，再从这些假设族中报告发现，
还需对总体 P 值采用适当的多重检验方法。
参见[假设族选择示例](multiple-testing.md#用-axis-确定假设族)。

公式描述的是数学方法；截断和浮点运算会限制实际计算结果。
Fisher 和 Stouffer 调用生存函数（`sf`）计算上尾，而不是从 1 中减去已计算的 CDF。
即使真实上尾概率仍为正，CDF 也可能舍入为 1；`sf` 最终同样可能下溢为零。
Cauchy 直接相减计算尾部，也可能丢失很小的概率。
因此，返回零或一可能只是数值饱和。将计算出的 P 值截断到 `[0,1]` 并不能修复 NaN：
即使输入的每项权重都有限，权重归一化中的溢出仍可能使 Stouffer 返回 NaN，
或使 Cauchy 返回错误结果。

完整的函数签名、别名、返回形状、轴、后端、缺失值处理、端点截断和数值限制见
[API 指南](../../guides/multiple-testing-combine-pvalues.md)。
使用较大权重前，尤其应阅读其中的
[权重验证与缩放建议](../../guides/multiple-testing-combine-pvalues.md#validate-and-rescale-combination-weights)。
即使统计假设适用，也需要注意这些数值问题。

