# 事后比较：Tukey 与 Bonferroni 两两 Welch 检验

> 语言：中文\
> 最后更新：2026-10-12\
> 切换：[English](../../../en/models/testing/anova-posthoc.md)

<a id="the-question"></a>

## 从“有均值不同”到“哪些均值不同”

[方差分析的总体检验](anova.md)回答各组总体均值是否全部相等。
要进一步判断具体哪两组不同，就需要两两比较：对每一对组检验
$H_{0,ij}:\mu_i-\mu_j=0$，并以响应变量的原始单位估计均值差。

$k$ 个组共有 $m=k(k-1)/2$ 对，本页的两个函数都将这些比较作为一个完整的检验族，
统一考虑多重比较带来的错误风险。即使总体检验显著，后续两两检验仍需考虑校正。
如果已经事先规划好两两分析，也可以直接运行这两个函数，无需先传入 ANOVA 结果。

应在查看结果之前确定组别、比较族与显著性水平。
[多重检验](multiple-testing.md)进一步解释检验族错误率、错误发现率及其他校正方法。
本页集中说明如何计算和解释均值比较。

<a id="choose-the-comparison"></a>

## 让比较方法与方差假设相匹配

| 问题或假设 | 使用方法 | 结果含义 |
|---|---|---|
| 比较全部组对，总体方差相等的假设合理 | `tukey_hsd` | 使用合并方差的 Tukey HSD；样本量不等时采用 Tukey–Kramer 调整 |
| 比较全部组对，总体方差可能不同 | `bonferroni` | 每一对分别进行 Welch t 检验，按 Bonferroni 方法调整判定阈值 |
| 特定对比、相关观测或重复测量 | 使用针对该问题设计的方法 | 这两个函数不接受对比矩阵、配对样本或重复测量结构 |

两种方法都假设组内和组间观测独立，并使用正态误差模型。
Welch 允许方差不等，但仍会受到离群值或观测相关性的影响。
它使用近似 t 分布，所以 Bonferroni 校正能否达到预期的错误率控制，也取决于原始检验是否可靠。
在平衡、正态、等方差模型下，Tukey 方法得到的一组区间具有设定的联合覆盖率；
样本量不等时，在相应假设下通常较为保守。
见 [NIST 的 Tukey 说明](https://www.itl.nist.gov/div898/handbook/prc/section4/prc471.htm)。

尤其要注意两步分析的方差假设是否一致：如果总体检验选择了不要求等方差的
Welch ANOVA，后续便不宜直接改用要求等方差的 Tukey 比较。
这里的 `tukey_hsd` 没有 `equal_var=False` 参数，也不提供 Games–Howell 比较。

<a id="read-comparisons"></a>

## 同时看方向、不确定性与判定

每项比较以一个 `PairwiseComparison` 对象返回。`group_i`、`group_j` 是按输入顺序
从零开始编号的组索引，满足 `i < j`。均值差始终定义为

$$\texttt{mean_diff}=\bar y_i-\bar y_j.$$

负值表示先传入的那一组样本均值更小。区间端点 `ci_lower`、`ci_upper` 使用同一方向。
如果反向描述比较，应将差值变号，并把区间 $[L,U]$ 改为 $[-U,-L]$。

- 除临界值附近的数值舍入影响外，区间不包含零，对应在校正后的检验水平下拒绝均值相等的原假设。
- 区间包含零表示该组对证据不足，不等于证明均值相等或实际等效。
- “同时 95% 置信区间”关注整组区间：如果反复抽样并按同一方法计算，
  目标是使整组区间同时覆盖所有真实均值差的比例不低于 95%；能否达到这一目标，
  仍取决于相应的模型假设与分布近似是否适用。
  它不是说，看过这一次数据后，某个固定参数有 95% 的概率落在已算出的区间内。

**这两个函数的 `pvalue` 字段含义不同：**

| 函数 | 返回的 `comparison.pvalue` | 返回的 `comparison.reject` |
|---|---|---|
| `tukey_hsd` | 已针对全部组对校正的 P 值 | `pvalue < alpha` |
| `bonferroni` | 未校正的双侧 Welch t 检验 P 值 | `pvalue < alpha / m` |

写报告时务必保留这一区别：Bonferroni 返回的是原始 P 值，不能标成“已校正”，
也不能直接与 `alpha` 比较；Tukey 返回的 P 值已经校正，不应对同一个比较族再次校正。

<a id="tukey-method"></a>

## Tukey：合并噪声估计与学生化极差

设 $MSE=SS_W/(N-k)$ 为合并组内均方。对于样本量为 $n_i,n_j$ 的一对组，
先计算用于标准化均值差的尺度，再得到学生化极差统计量：

$$s_{q,ij}=\sqrt{\frac{MSE}{2}\left(\frac1{n_i}+\frac1{n_j}\right)},\qquad
q_{ij}=\frac{|\bar y_i-\bar y_j|}{s_{q,ij}}.$$

校正后的 P 值是该统计量在组数为 $k$、自由度为 $N-k$ 的学生化极差分布中的右尾概率。
同时置信区间为

$$(\bar y_i-\bar y_j)\;\pm\;q_{1-\alpha;k,N-k}\,s_{q,ij}.$$

样本量相等时，这就是 Tukey HSD；不等时，尺度会随每一对的样本量调整，称为 Tukey–Kramer 扩展。
两者都使用共同的方差估计。计算极差分布时使用的是组数 $k$，不是组对数 $m$。

<a id="anova-tukey"></a>

## 完整的 Tukey CPU 示例

在新的 Python 会话中依次运行以下三个代码块。模拟数据的各组总体标准差均为 1。

<!-- example: anova-tukey -->
```python
import numpy as np
from statgpu.anova import tukey_hsd
```

每组生成 30 个独立观测。输出用列表中的位置标识组别，实际分析时请保留索引与组标签的对应关系。

```python
rng = np.random.default_rng(42)
groups = [rng.normal(mean, 1.0, 30) for mean in (0.0, 0.2, 0.9)]
```

将整个检验族的显著性水平设为 5%，输出全部三对结果，也保留未发现显著差异的比较。
这里报告的 P 值已经过校正。

```python
posthoc = tukey_hsd(*groups, alpha=0.05, backend="numpy")
for c in posthoc.comparisons:
    print(f"{c.group_i}-{c.group_j}: diff={c.mean_diff:.3f}, "
          f"adjusted_p={c.pvalue:.3g}, "
          f"CI=[{c.ci_lower:.3f}, {c.ci_upper:.3f}], reject={c.reject}")
```
<!-- example-end: anova-tukey -->

结果约为

| 组对 | 均值差 | 校正后的 P 值 | 同时置信区间 | 拒绝原假设 |
|---|---:|---:|---|---|
| 0 − 1 | -0.297 | 0.288 | [-0.764, 0.170] | False |
| 0 − 2 | -0.782 | 0.000394 | [-1.248, -0.315] | True |
| 1 − 2 | -0.485 | 0.0399 | [-0.951, -0.018] | True |

组 2 的样本均值高于组 0 和组 1，相应同时区间不包含零。
0 − 1 的区间包含零，因此在为整个比较族设定的显著性水平下，不拒绝这一对均值相等的原假设。
均值差及其区间使用响应变量的原始单位；差异是否重要，需要看这些数值及应用背景，不能只看 P 值。
`posthoc.df_within` 为 `87`，`posthoc.mse` 约为 `0.574286`。

<a id="bonferroni-method"></a>

<a id="bonferroni每一对分别使用-welch-不确定性"></a>

## Bonferroni：分别估计每对均值差的不确定性

`bonferroni` 不合并各组方差，而是为每一对分别计算标准误、t 统计量和近似自由度：

$$SE_{ij}=\sqrt{s_i^2/n_i+s_j^2/n_j},\qquad
t_{ij}=\frac{\bar y_i-\bar y_j}{SE_{ij}},$$

$$\nu_{ij}=\frac{(s_i^2/n_i+s_j^2/n_j)^2}
{(s_i^2/n_i)^2/(n_i-1)+(s_j^2/n_j)^2/(n_j-1)}.$$

先求出每对比较的原始双侧 t 检验 P 值，仅在 $p_{ij}<\alpha/m$ 时拒绝原假设。
区间使用更严格的双侧临界值：

$$(\bar y_i-\bar y_j)\;\pm\;t_{1-\alpha/(2m),\nu_{ij}}\,SE_{ij}.$$

因此，校正体现在判定阈值和置信区间中，返回对象中的 P 值仍然是原始值。
如果报告需要校正后的值，可对同一个完整比较族计算 $\min(1,m p_{ij})$。
[NIST 的 Bonferroni 说明](https://www.itl.nist.gov/div898/handbook/prc/section4/prc473.htm)
介绍了同时覆盖率的原理；本函数具体使用各对的 Welch 标准误与自由度，不使用合并误差方差。

<a id="anova-bonferroni"></a>

## 完整的不等方差两两比较示例

本例可以独立运行，数据与前面的 Tukey 示例及 [Welch ANOVA 示例](anova-welch.md#anova-welch)不同。
其中一对的原始 P 值小于 0.05，却达不到校正后的显著性要求，正好用来说明校正的作用。

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

比较族包含三对，因此应将原始 P 值与 `0.05 / 3`（约 `0.01667`）比较。
函数已经按此阈值计算 `reject`，可以直接读取。

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

| 组对 | 均值差 | 原始 P 值 | Bonferroni 区间 | 拒绝原假设 |
|---|---:|---:|---|---|
| 0 − 1 | -0.697 | 0.0220 | [-1.428, 0.034] | False |
| 0 − 2 | -3.551 | 6.38e-07 | [-5.041, -2.061] | True |
| 1 − 2 | -2.854 | 5.04e-05 | [-4.445, -1.263] | True |

第一对的原始 P 值低于 0.05，却高于 0.01667，所以 `reject=False`，相应的同时区间也包含零。
如果报告需要校正后的 P 值，可以按下面的方法另行计算，保留返回对象中的原始值。

```python
adjusted_pvalues = np.minimum(
    1.0, bonf.n_comparisons * np.array([c.pvalue for c in bonf.comparisons])
)
print([f"{p:.3g}" for p in adjusted_pvalues])
```
<!-- example-end: anova-bonferroni -->

用于报告的校正 P 值为 `['0.066', '1.91e-06', '0.000151']`。
将它们与 0.05 比较，得到相同的 `[False, True, True]` 判定。
不能先删除不显著或不符合预期的比较，再缩小 $m$。

<a id="parameters-and-pitfalls"></a>

## 参数与常见误区

- 两个函数都通过 `*groups` 接收各组数据，另有 `alpha=0.05`、`backend="auto"`
  和 `dtype=None` 三个参数。`alpha` 必须为有限值且严格位于 0 与 1 之间，
  表示整个两两比较族的显著性水平，不是单对比较的水平。
- 至少需要两个组，每组至少有两个有限观测。允许组间样本量不等。
  NaN 和无穷值会触发 `ValueError`，不会自动删除缺失观测。
- 输入会被展平，请为每组提供一个向量；矩阵输入不会自动表示多项检验或配对关系。
- 两个函数都比较所有输入组对，没有自定义组对或只与对照组比较的选项。
  如果同时分析多个响应变量或多个子组，实际需要控制的比较族可能比单次调用覆盖的范围更大。
- 即使样本量不相等，Tukey 仍要求总体方差相等。
  Bonferroni 校正本身不能修复不恰当的原始检验，也不能弥补看过结果后才选定比较族的问题。
- 标准误为零时，区间会缩为一个点。Bonferroni 对均值相等的组对返回原始 P 值 1，
  对均值不同的组对返回 0。Tukey 在合并 MSE 为零时也分别处理相同和不同均值。
  常数观测造成的这种结果，不能解释为总体效应已经得到毫无误差的证实。
- 各组统计量的计算支持 NumPy、CuPy 和 Torch。Tukey 的学生化极差分布函数与分位数
  使用 CPU 上的 SciPy，Bonferroni 的标量 t 分布计算也使用 CPU。
  组内统计量在 GPU 上计算，不代表每个步骤都在 GPU 上执行。

完整的 `TukeyResult`、`PosthocResult`、`PairwiseComparison` 字段、
输入要求及后端选项见 [ANOVA API 参考](../../reference/anova-api.md)。
[多重检验教程](multiple-testing.md)介绍一般性的错误率标准和校正选择，
不局限于这两个 ANOVA 辅助函数。

<a id="references"></a>

## 参考资料与后续阅读

- [NIST：Tukey 方法](https://www.itl.nist.gov/div898/handbook/prc/section4/prc471.htm)。
- [NIST：Bonferroni 方法](https://www.itl.nist.gov/div898/handbook/prc/section4/prc473.htm)。
- [SciPy Tukey HSD 参考](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.tukey_hsd.html)，
  其中列出了 Tukey 与 Kramer 的原始文献。跨实现比较时应对齐方差假设；
  SciPy 与 statgpu 的 API 选项不同。
- [经典单因素检验](anova-oneway.md)、[Welch 检验](anova-welch.md)、[平衡双因素方差分析](anova-twoway.md)
  和 [ANOVA 总览](anova.md)。
