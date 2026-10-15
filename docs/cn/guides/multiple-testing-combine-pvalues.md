# P 值校正与合并 API

> 语言：中文  
> 最后更新：2026-10-15  
> 切换：[English](../../en/guides/multiple-testing-combine-pvalues.md)

以下函数可从 `statgpu.inference` 或 `statgpu` 导入。选择假设族、区分 FWER 与 FDR、理解公式及依赖条件时，请先阅读[多重检验](../models/testing/multiple-testing.md)。

详细教程：[FWER](../models/testing/multiple-testing-fwer.md)、[FDR](../models/testing/multiple-testing-fdr.md)和 [P 值合并](../models/testing/pvalue-combination.md)。

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
| `backend="auto"` | 数组库选择：`"auto"`、`"numpy"`、`"cupy"` 或 `"torch"`；也接受历史别名 `"cpu"`、`"cuda"` 和 `"gpu"`。见[后端选择与转换](#backend-selection-and-conversion)。 |

`adjust_pvalues` 和 `multipletests` 返回 `(reject, adjusted)`，两项都保留输入形状，`axis=None` 时也如此。`reject` 为布尔数组，`adjusted` 为 float64。标量只接受 `axis=None`。展平后的空输入可返回空校正数组；批处理时应使用非空的假设轴。

`combine_pvalues` 返回 `(statistic, pvalue)`：`axis=None` 时为标量或零维结果，否则返回去掉指定轴的数组。空合并组会报错。结果使用所选后端的 float64 数组或标量。

估计器的同名方法返回字典，且校正轴的默认值不同，见[共享估计器辅助方法](../reference/estimator-api.md)。

没有 `nan_policy` 选项：非有限 P 值会引发 `ValueError`，不会静默忽略缺失值。

<a id="backend-selection-and-conversion"></a>

## 后端选择与转换

以下规则适用于上述三个模块级函数。后端名称忽略首尾空格和大小写，因此 `"AUTO"` 与 `"auto"` 等价。

| 后端 | 选择规则与要求 |
|---|---|
| `"auto"` | 从 `pvalues` 推断数组库：Torch 张量选择 Torch，CuPy 数组选择 CuPy，NumPy 数组、列表或标量选择 NumPy。`weights` 不参与数组库或设备选择。 |
| `"numpy"` 或历史别名 `"cpu"` | 使用 CPU 上的 NumPy。 |
| `"cupy"` 或历史别名 `"cuda"` / `"gpu"` | 使用 CuPy，要求已安装 CuPy 且 CUDA 可用。指定后端不可用时会报错，不会改用 NumPy。 |
| `"torch"` | 要求已安装 PyTorch。`pvalues` 为 Torch 张量时，保留其 CPU 或具体 CUDA 设备。其他输入在 CUDA 可用时使用当前 CUDA 设备，否则使用 CPU；未安装 PyTorch 时会报错，不会改用 NumPy。 |

P 值和传入的合并权重会转换为所选数组库的 float64 数组。无论通过 `"auto"` 还是显式 `"torch"` 选择 Torch，CPU 上的 `pvalues` 张量都会留在 CPU，即使 CUDA 可用；CUDA 张量则保留其确切的设备编号，例如 `cuda:1`，不受当前 CUDA 设备影响。Float32 P 值也在同一设备上升为 float64。秩、校正因子、默认或传入的合并权重以及全部输出，都与转换后的 P 值位于同一设备；权重不会另行决定设备。

这些 float64 计算支持 Torch CPU 和 CUDA 设备。`pvalues` 位于 MPS 或其他不支持的 Torch 设备时，会引发 `ValueError`，不会静默移到 CPU 或 CUDA。调用前请显式将此类输入移到 CPU 或受支持的 CUDA 设备。

这些函数没有单独的 `device` 参数。若需要在 CPU 上进行 Torch 计算并返回 Torch 结果，请传入 CPU Torch 张量作为 `pvalues`；若需要指定 CUDA GPU，请先将该张量放到相应编号的设备上。`pvalues` 不是 Torch 张量且显式指定 `backend="torch"` 时，CUDA 可用则转换到当前具体 CUDA 设备，否则转换到 CPU。显式选择其他数组库可能涉及设备之间的数据传输；若需要 CPU 计算和 NumPy 输出，请使用 `backend="numpy"`。

对于这些模块级函数，`statgpu.set_device(...)` 不会覆盖上述数组库或 Torch 设备选择规则。`model.combine_pvalues(...)` 等估计器方法还会考虑估计器与全局设备设置；其 `backend="auto"` 可能先选择解析得到的 GPU 后端并转换输入，再传给模块级函数。估计器方法显式指定 `backend="torch"` 时要求 CUDA，并先将输入转换到 CUDA；模块级函数保留输入设备的规则不会跳过这一步转换。相应规则见[共享估计器辅助方法](../reference/estimator-api.md)和[设备选择](device-and-memory.md)。

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

Fisher 使用独立均匀 P 值对应的卡方参考分布；Stouffer 使用独立正态分数的方差，不估计检验之间的协方差。Cauchy/ACAT 对归一化权重的正切统计量使用一定正则条件下的 Cauchy 尾部近似，并不对任意依赖结构和任意有限显著性水平作统一保证。详见[合并公式及参考文献](../models/testing/pvalue-combination.md)。

<a id="a-complete-axis-and-weight-example"></a>

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

<a id="validate-and-rescale-combination-weights"></a>

## 验证并缩放合并权重

Cauchy 和 Stouffer 只取决于相对权重，因此把全部权重乘以同一个正常数，结果应保持不变。目前，即使每个权重都有限，求和仍可能溢出：当 `p=[0.01, 0.1]`、`weights=[1e308, 1e308]` 时，Cauchy 会返回 `0.5`，而正确结果约为 `0.018222`；Stouffer 则返回 NaN。调用前应先检查权重，再除以最大权重。这样可以保留比例并避免该溢出，但不会修复极端概率尾部的精度，也不会改变方法的依赖假设。

<!-- safety-example: scaled-combination-weights -->
```python
import numpy as np
from statgpu.inference import combine_pvalues


def relative_weights(weights):
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 1 or weights.size == 0:
        raise ValueError("weights must be a nonempty vector")
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("weights must be finite and nonnegative")
    largest = weights.max()
    if largest <= 0:
        raise ValueError("at least one weight must be positive")
    return weights / largest


p = np.array([0.01, 0.1])
weights = relative_weights([1e308, 1e308])
for method in ("cauchy", "stouffer"):
    statistic, combined = combine_pvalues(
        p, method=method, weights=weights, backend="numpy",
    )
    print(method, round(float(statistic), 6), round(float(combined), 6))
```

输出为 `cauchy 17.4491 0.018222` 和 `stouffer 2.55117 0.005368`，与单位等权的结果一致。权重顺序须与 P 值一致；API 会检查权重个数是否等于合并轴长度。

## 历史计时记录

原有计时表保存在[开发者历史记录](../../../dev/references/multiple-testing-historical-benchmarks.md)中。它们不能证明当前版本的加速比，也不是通用的 CPU/GPU 分界；性能重要时，应测量实际工作负载。
