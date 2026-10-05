# P 值校正与合并 API

> 语言：中文  
> 最后更新：2026-10-05  
> 切换：[English](../../en/guides/multiple-testing-combine-pvalues.md)

以下函数可从 `statgpu.inference` 或 `statgpu` 导入。选择假设族、区分 FWER 与 FDR、理解公式及依赖条件时，请先阅读[多重检验](../models/multiple-testing.md)。

## 完整签名

```text
adjust_pvalues(pvalues, method="bh", alpha=0.05, axis=None, backend="auto")
multipletests(pvalues, alpha=0.05, method="bh", axis=None, backend="auto")
combine_pvalues(pvalues, method="fisher", weights=None, axis=None, backend="auto")
```

`multipletests` 调用 `adjust_pvalues`，但两者的第二、第三个位置参数顺序不同。建议显式写出 `method=` 和 `alpha=`。与 statsmodels 的同名函数不同，此别名只返回两个值，也没有 `is_sorted` 或 `returnsorted` 参数。

## 输入与返回值

| 参数 | 含义与限制 |
|---|---|
| `pvalues` | `[0,1]` 内的有限数值 P 值；保留假设的原始顺序。 |
| `method` | 下表中的校正或合并方法。名称忽略首尾空格和大小写。 |
| `alpha=0.05` | 拒绝掩码使用的显著性水平，须为 `(0,1)` 内的有限数。当前 NaN 不会报错，而会返回全为假的判断；应先检查计算得到的水平。它不影响校正后的数值。 |
| `axis=None` | 所有元素属于同一个假设族或总体检验。整数表示沿该轴分别计算，也接受负轴索引。 |
| `weights=None` | 仅用于合并。Cauchy/Stouffer 默认等权；提供时应为有限、非负、总和为正的向量，长度与合并轴一致。批处理中各组共用这个向量，实现内部会归一化。Fisher 不接受权重。 |
| `backend="auto"` | 从 P 值数组推断 `numpy`、`cupy` 或 `torch`；普通列表使用 NumPy。这是数组库选择，与估计器的 `device` 参数不同。 |

`adjust_pvalues` 和 `multipletests` 返回 `(reject, adjusted)`，两项都保留输入形状，`axis=None` 时也如此。`reject` 为布尔数组，`adjusted` 为 float64。标量只接受 `axis=None`。展平后的空输入可返回空校正数组；批处理时应使用非空的假设轴。

`combine_pvalues` 返回 `(statistic, pvalue)`：`axis=None` 时为标量或零维结果，否则返回去掉指定轴的数组。空合并组会报错。结果使用所选后端的 float64 数组或标量。显式选择 Torch 数组库时可以处理 Torch CPU 或 CUDA 张量，这不同于估计器要求 CUDA 的 `device="torch"`。CPU 示例不代表已经验证物理 GPU。

估计器的同名方法返回字典，且校正轴的默认值不同，见[共享估计器辅助方法](../reference/estimator-api.md)。

## 方法与别名

| 标准方法名 | 接受的别名 |
|---|---|
| `bh` | `fdr_bh`, `benjamini-hochberg`, `benjamini_hochberg` |
| `by` | `fdr_by`, `benjamini-yekutieli`, `benjamini_yekutieli` |
| `holm` | `holm-bonferroni`, `holm_bonferroni` |
| `bonferroni` | `bonf` |
| `hochberg` | `fdr_hochberg`, `step_up`, `stepup` |
| `fisher` | `fisher-combination`, `fisher_combination` |
| `cauchy` | `cauchy-combination`, `cauchy_combination`, `acat` |
| `stouffer` | `z-test`, `ztest`, `weighted_z` |

历史别名 `fdr_hochberg` 实际调用的是控制 **FWER 的 Hochberg**，不是控制 FDR 的 BH。此实现没有 Tippett 方法。Bonferroni/Holm 允许任意依赖；BH 要求独立或适当的正依赖；BY 允许任意依赖；Hochberg 要求独立或满足相应的 Simes 依赖条件。所有方法都要求边际 P 值有效。

Fisher 使用独立均匀 P 值对应的卡方参考分布；Stouffer 使用独立正态分数的方差，不估计检验之间的协方差。Cauchy/ACAT 对归一化权重的正切统计量使用一定正则条件下的 Cauchy 尾部近似，并不对任意依赖结构和任意有限显著性水平作统一保证。详见[合并公式及参考文献](../models/multiple-testing.md#合并公式与适用条件)。

## 可直接运行的轴与权重示例

Stouffer 计算要求原假设下的分数独立且方向一致。这里每行对应一个总体检验；代码不会再对所得的两个总体 P 值进行多重校正。

<!-- api-example: multiple-testing-axis -->
```python
import numpy as np
from statgpu import adjust_pvalues, combine_pvalues, multipletests

p = np.array([[0.01, 0.04, 0.6], [0.02, 0.2, 0.7]])
alpha = 0.05
if not np.isfinite(alpha) or not 0 < alpha < 1:
    raise ValueError("alpha must be finite and in (0, 1)")
reject, adjusted = adjust_pvalues(p, method="holm", alpha=alpha, axis=1, backend="numpy")
reject_alias, adjusted_alias = multipletests(p, method="holm", alpha=alpha, axis=1, backend="numpy")
np.testing.assert_array_equal(reject, reject_alias)
np.testing.assert_allclose(adjusted, adjusted_alias)
statistic, global_p = combine_pvalues(p, method="stouffer", weights=[1, 2, 1], axis=1, backend="numpy")
assert adjusted.shape == p.shape
assert statistic.shape == global_p.shape == (2,)
```

## 数值限制

- Fisher 在取对数前，将低于 float64 最小正正规数的概率截断到该值。因此输入精确零时，统计量仍为有限数，而不是无穷大。
- Cauchy 和 Stouffer 将概率截断到 `[eps, 1-eps]`，其中 `eps` 是 float64 的机器精度；精确端点及更极端的概率会被改变。
- Fisher 和 Stouffer 还调用[分布函数](distribution-api.md)，其生存函数和分位数尾部可能受相消或饱和影响。Cauchy 的直接尾部相减也可能丢失很小的概率。
- 返回零或一可能只是数值极限。极端尾部精度重要时，应使用已验证的稳定尾部方法；仅把相同数据转到 GPU 不能消除这些限制。

结果有限、或少量例子与其他库一致，都不能证明方法满足统计校准。逐项检验应按所选错误率解释；合并检验则针对一个总体原假设。

## 历史计时记录

原有计时表保存在[开发者历史记录](../../../dev/references/multiple-testing-historical-benchmarks.md)中。它们不能证明当前版本的加速比，也不是通用的 CPU/GPU 分界；性能重要时，应测量实际工作负载。
