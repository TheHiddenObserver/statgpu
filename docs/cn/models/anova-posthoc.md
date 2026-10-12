# 事后比较：Tukey 与 Bonferroni 两两 Welch 检验

> 语言：中文\
> 最后更新：2026-10-12\
> 切换：[English](../../en/models/anova-posthoc.md)

<a id="the-question"></a>

## 从“有均值不同”到“哪些均值不同”

[整体方差分析](anova-oneway.md)检验各组总体均值是否全部相等，不直接指出具体差异。
两两比较检验更明确的原假设 $H_{0,ij}:\mu_i-\mu_j=0$，
并使用响应的原始单位估计每一对均值差。

$k$ 个组的全部两两比较构成 $m=k(k-1)/2$ 项比较的检验族。
本页两个函数都针对这个完整比较族处理多重性。
整体检验显著，并不意味着之后可以直接采用未校正的两两检验。
另一方面，这两个函数不需要传入 ANOVA 结果；事先规划好的同时两两比较可以直接运行。

应在查看结果之前确定组别、比较族与显著性水平。
[多重检验](multiple-testing.md)进一步解释检验族错误率、错误发现率及其他校正方法。
本页集中说明如何计算和解释均值比较。

<a id="choose-the-comparison"></a>

## 让比较方法与方差假设相匹配

| 问题或假设 | 使用方法 | 结果含义 |
|---|---|---|
| 比较全部组对，总体方差相等的假设合理 | `tukey_hsd` | 使用合并方差的 Tukey HSD；样本量不等时使用 Tukey–Kramer 尺度 |
| 比较全部组对，总体方差可能不同 | `bonferroni` | 分别进行 Welch t 检验，采用 Bonferroni 检验族阈值 |
| 特定对比、相关观测或重复测量 | 使用针对该问题设计的方法 | 这两个函数不接受对比矩阵、配对样本或重复测量结构 |

两种方法都假设组内和组间观测独立，并使用正态误差模型。
Welch 允许方差不等，但不能消除离群值或相关性的影响。
它的 t 参考分布是近似的，因此 Bonferroni 的错误率控制也取决于原始检验的校准质量。
在平衡、正态、等方差模型下，Tukey 的全部区间具有名义联合覆盖率；
不等样本量扩展在相应假设下较为保守。
见 [NIST 的 Tukey 说明](https://www.itl.nist.gov/div898/handbook/prc/section4/prc471.htm)。

特别要注意：先使用不等方差 Welch ANOVA，再使用合并方差 Tukey 比较，
会改变方差假设，应选择与前提一致的两两比较。
这里的 `tukey_hsd` 没有 `equal_var=False` 参数，也不提供 Games–Howell 比较。

<a id="read-comparisons"></a>

## 同时看方向、不确定性与判定

每个 `PairwiseComparison` 都提供从零开始的 `group_i`、`group_j`，
满足 `i < j`，对应输入顺序。差值始终定义为

$$\texttt{mean_diff}=\bar y_i-\bar y_j.$$

负值表示较早传入组的样本均值更小。`ci_lower`、`ci_upper` 使用同一方向。
如果反向描述比较，应将差值变号，并把区间 $[L,U]$ 改为 $[-U,-L]$。

- 除数值边界舍入情况外，区间不包含零与相应同时比较水平下的拒绝判定一致。
- 区间包含零表示该组对证据不足，不等于证明均值相等或实际等效。
- 同时 95% 区间组描述反复抽样时，整个比较族的区间共同覆盖真值的性质。
  不能把它解释成看过数据后某个固定参数有 95% 的概率落入区间。

**这两个函数的 `pvalue` 字段含义不同：**

| 函数 | 返回的 `comparison.pvalue` | 返回的 `comparison.reject` |
|---|---|---|
| `tukey_hsd` | 已针对全部组对校正的 p 值 | `pvalue < alpha` |
| `bonferroni` | 未校正的双侧 Welch t 检验 p 值 | `pvalue < alpha / m` |

导出报告时务必保留这一区别。不要把 Bonferroni 返回的 p 值标成“已校正”，
也不要直接与 `alpha` 比较；同样，不要针对同一个比较族再次校正 Tukey 的 p 值。

<a id="tukey-method"></a>

## Tukey：合并噪声估计与学生化极差

设 $MSE=SS_W/(N-k)$ 为合并组内均方。对于样本量为 $n_i,n_j$ 的一对组，
学生化极差所用尺度为

$$s_{q,ij}=\sqrt{\frac{MSE}{2}\left(\frac1{n_i}+\frac1{n_j}\right)},\qquad
q_{ij}=\frac{|\bar y_i-\bar y_j|}{s_{q,ij}}.$$

校正后的 p 值来自组数为 $k$、自由度为 $N-k$ 的学生化极差分布右尾。
同时置信区间为

$$(\bar y_i-\bar y_j)\;\pm\;q_{1-\alpha;k,N-k}\,s_{q,ij}.$$

样本量相等时，这是 Tukey HSD；不等时使用每一对的尺度，即 Tukey–Kramer 扩展。
两者仍然采用一个共同方差估计。请注意，极差分布使用组数 $k$，不是组对数 $m$。

<a id="anova-tukey"></a>

## 完整的 Tukey CPU 示例

在新的会话中依次运行这三个代码块。模拟数据的各组总体标准差都为一。

<!-- example: anova-tukey -->
```python
import numpy as np
from statgpu.anova import tukey_hsd
```

每组生成 30 个独立观测。输出中的组索引对应列表顺序，实际研究中应自行保留组标签。

```python
rng = np.random.default_rng(42)
groups = [rng.normal(mean, 1.0, 30) for mean in (0.0, 0.2, 0.9)]
```

将检验族显著性水平设为 5%，输出所有三对比较，包括没有被拒绝的组对。
这里报告的 p 值已经过校正。

```python
posthoc = tukey_hsd(*groups, alpha=0.05, backend="numpy")
for c in posthoc.comparisons:
    print(f"{c.group_i}-{c.group_j}: diff={c.mean_diff:.3f}, "
          f"adjusted_p={c.pvalue:.3g}, "
          f"CI=[{c.ci_lower:.3f}, {c.ci_upper:.3f}], reject={c.reject}")
```
<!-- example-end: anova-tukey -->

结果约为

| 组对 | 均值差 | 校正后的 p 值 | 同时置信区间 | 拒绝 |
|---|---:|---:|---|---|
| 0 − 1 | -0.297 | 0.288 | [-0.764, 0.170] | False |
| 0 − 2 | -0.782 | 0.000394 | [-1.248, -0.315] | True |
| 1 − 2 | -0.485 | 0.0399 | [-0.951, -0.018] | True |

组 2 的样本均值高于组 0 和组 1，相应同时区间不包含零。
0 − 1 的区间包含零，因此在指定检验族水平下不拒绝该组对的原假设。
估计值使用响应的原始单位，p 值不衡量实际重要性。
`posthoc.df_within` 为 `87`，`posthoc.mse` 约为 `0.574286`。

<a id="bonferroni-method"></a>

## Bonferroni：每一对分别使用 Welch 不确定性

对于不等方差，此函数计算

$$SE_{ij}=\sqrt{s_i^2/n_i+s_j^2/n_j},\qquad
t_{ij}=\frac{\bar y_i-\bar y_j}{SE_{ij}},$$

$$\nu_{ij}=\frac{(s_i^2/n_i+s_j^2/n_j)^2}
{(s_i^2/n_i)^2/(n_i-1)+(s_j^2/n_j)^2/(n_j-1)}.$$

先求出每对比较的原始双侧 t 检验 p 值，仅在 $p_{ij}<\alpha/m$ 时拒绝。
区间使用更严格的双侧临界值：

$$(\bar y_i-\bar y_j)\;\pm\;t_{1-\alpha/(2m),\nu_{ij}}\,SE_{ij}.$$

因此，校正体现在判定阈值和置信区间中，而存储的 p 值仍然是原始值。
如果报告需要校正后的值，可对同一个完整比较族计算 $\min(1,m p_{ij})$。
[NIST 的 Bonferroni 说明](https://www.itl.nist.gov/div898/handbook/prc/section4/prc473.htm)
介绍了同时覆盖率原理；本接口具体使用每一对的 Welch 不确定性，而不是合并误差方差。

<a id="anova-bonferroni"></a>

## 完整的不等方差两两比较示例

本例独立于前面的 Tukey 示例，也不复用单因素页面的 Welch 数据。
特意设置了一对原始 p 值小于 0.05、却不能通过校正阈值的组别。

<!-- example: anova-bonferroni -->
```python
import numpy as np
from statgpu.anova import bonferroni
```

生成三个样本量和标准差不同的组。本例中间组的总体均值为 `10.25`。

```python
rng = np.random.default_rng(2026)
unequal_groups = [
    rng.normal(10.0, 0.7, 20),
    rng.normal(10.25, 1.5, 28),
    rng.normal(13.0, 3.0, 36),
]
```

比较族包含三对，因此原始 p 值阈值为 `0.05 / 3`，约 `0.01667`。
直接输出函数返回的拒绝判定。

```python
bonf = bonferroni(*unequal_groups, alpha=0.05, backend="numpy")
print(f"comparisons={bonf.n_comparisons}, "
      f"cutoff={bonf.alpha / bonf.n_comparisons:.5f}")
for c in bonf.comparisons:
    print(f"{c.group_i}-{c.group_j}: diff={c.mean_diff:.3f}, "
          f"raw_p={c.pvalue:.3g}, "
          f"CI=[{c.ci_lower:.3f}, {c.ci_upper:.3f}], reject={c.reject}")
```

结果约为

| 组对 | 均值差 | 原始 p 值 | Bonferroni 区间 | 拒绝 |
|---|---:|---:|---|---|
| 0 − 1 | -0.697 | 0.0220 | [-1.428, 0.034] | False |
| 0 − 2 | -3.551 | 6.38e-07 | [-5.041, -2.061] | True |
| 1 − 2 | -2.854 | 5.04e-05 | [-4.445, -1.263] | True |

第一对的原始 p 值低于 0.05，却高于 0.01667，所以 `reject=False`，
其同时区间也包含零。如果报告要求校正后的 p 值，可单独计算展示值，不修改返回对象。

```python
adjusted_pvalues = np.minimum(
    1.0, bonf.n_comparisons * np.array([c.pvalue for c in bonf.comparisons])
)
print([f"{p:.3g}" for p in adjusted_pvalues])
```
<!-- example-end: anova-bonferroni -->

校正后的展示值为 `['0.066', '1.91e-06', '0.000151']`。
将它们与 0.05 比较，得到相同的 `[False, True, True]` 判定。
不能先删除不显著或不符合预期的比较，再缩小 $m$。

<a id="parameters-and-pitfalls"></a>

## 参数与常见误区

- 两个函数都接受分别传入的 `*groups`、`alpha=0.05`、`backend="auto"`
  和 `dtype=None`。`alpha` 必须有限且严格位于零与一之间，表示整个两两比较族的水平，
  不是单个组对的水平。
- 至少需要两个组，每组至少有两个有限观测。允许组间样本量不等。
  NaN 和无穷值会触发 `ValueError`，不会自动删除缺失观测。
- 输入会被展平，应为每组提供一个向量，不能期待矩阵自动表示多项检验或配对关系。
- 两个函数都比较所有输入组对，没有自定义组对或只与对照组比较的选项。
  多个响应或分别分析的子组可能构成比单次调用更大的比较族。
- 即使样本量不相等，Tukey 仍要求总体方差相等。
  Bonferroni 校正本身不能修复不恰当的原始检验，也不能弥补看过结果后才选定比较族的问题。
- 标准误为零时会得到退化区间；Bonferroni 对等均值组对返回原始 p 值一，
  对不等均值组对返回零。Tukey 在合并 MSE 为零时也分别处理相同和不同均值。
  不要把常数观测解释成总体效应已被无误差地证实。
- 组内归约计算支持 NumPy、CuPy 和 Torch。Tukey 的学生化极差分布函数与分位数
  使用 CPU 上的 SciPy，Bonferroni 的标量 t 分布计算也使用 CPU。
  组内统计量在 GPU 上计算，不代表每个步骤都在 GPU 上执行。

完整的 `TukeyResult`、`PosthocResult`、`PairwiseComparison` 字段、
输入校验及执行选项见 [ANOVA API 参考](../reference/anova-api.md)。
[多重检验教程](multiple-testing.md)介绍一般性的错误率标准和校正选择，
不局限于这两个 ANOVA 辅助函数。

<a id="references"></a>

## 参考资料与后续阅读

- [NIST：Tukey 方法](https://www.itl.nist.gov/div898/handbook/prc/section4/prc471.htm)。
- [NIST：Bonferroni 方法](https://www.itl.nist.gov/div898/handbook/prc/section4/prc473.htm)。
- [SciPy Tukey HSD 参考](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.tukey_hsd.html)，
  其中列出了 Tukey 与 Kramer 的原始文献。跨实现比较时应对齐方差假设；
  SciPy 与 statgpu 的 API 选项不同。
- [单因素与 Welch 检验](anova-oneway.md)、[平衡双因素方差分析](anova-twoway.md)
  和 [ANOVA 总览](anova.md)。
