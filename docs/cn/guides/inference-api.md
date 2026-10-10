# 推断 API 参考

> 语言：中文  
> 最后更新：2026-10-05  
> 模块：`statgpu.inference`  
> 切换：[English](../../en/guides/inference-api.md)

`statgpu.inference` 汇总可复用的统计工具，包括概率分布、多重检验、p 值合并、排列检验与自助法（bootstrap）。

本页只作为**模块入口**。分布函数的详细行为统一放在 [分布 API](distribution-api.md)；模型系数推断的方法选择统一放在 [推断模式](inference-modes.md)。

## 快速参考

```python
from statgpu.inference import (
    norm,
    t,
    adjust_pvalues,
    combine_pvalues,
    permutation_test,
    bootstrap_statistic,
)
```

| API | 用途 | 详细文档 |
|---|---|---|
| `norm`、`t`、`chi2`、`poisson` 等分布对象 | CDF/SF/PPF/PDF/PMF/随机采样 | [分布 API](distribution-api.md) |
| `get_distribution(...)` | 动态选择概率分布与计算后端 | [分布 API](distribution-api.md) |
| `adjust_pvalues(...)` | 多重检验校正 | [多重检验](multiple-testing-combine-pvalues.md) |
| `combine_pvalues(...)` | 合并多个 p 值提供的证据 | [多重检验](multiple-testing-combine-pvalues.md) |
| `permutation_test(...)` | 基于排列的假设检验 | 本页 |
| `bootstrap_statistic(...)` | 对用户给定的统计量执行通用自助法 | 本页 |

## 分布函数

分布对象提供与 SciPy 风格接近的 `cdf`、`sf`、`ppf`、`isf`、`pdf`/`pmf` 与 `rvs` 方法。

```python
from statgpu.inference import norm, t

p = norm.cdf(1.96)
q = t.ppf(0.975, df=10)
```

后端选择、可用分布、反函数精度、R 风格兼容别名以及历史名称，见 [分布 API](distribution-api.md)。

## 多重检验与 p 值合并

```python
import numpy as np
from statgpu.inference import adjust_pvalues, combine_pvalues

pvals = np.array([0.001, 0.01, 0.03, 0.05, 0.5])

reject, pvals_bh = adjust_pvalues(pvals, method="bh")
stat, p_global = combine_pvalues(pvals, method="fisher")
```

可用的校正方法、合并方法及其统计解释见 [多重检验](multiple-testing-combine-pvalues.md)。

## 排列检验

`permutation_test` 按照接口定义对输入数据反复排列，并重新计算用户提供的统计量，从而构造排列参考分布。

<!-- api-example: module-permutation -->
```python
import numpy as np
from statgpu.inference import permutation_test

rng = np.random.default_rng(42)
X = rng.standard_normal((100, 5))
y = X @ np.ones(5) + rng.standard_normal(100)

result = permutation_test(
    lambda X_, y_: np.corrcoef(X_[:, 0], y_)[0, 1],
    X,
    y,
    n_resamples=999,
    random_state=42,
)

print(result.pvalue)
```

统计量与排列方案应当和应用中的原假设相匹配；通用排列检验工具无法替用户判断具体数据是否满足可交换性假设。

## 通用自助法

`bootstrap_statistic` 对调用者提供的统计量执行自助法重抽样。

<!-- api-example: module-bootstrap -->
```python
import numpy as np
from statgpu.inference import bootstrap_statistic

rng = np.random.default_rng(42)
data = rng.standard_normal(1000)

result = bootstrap_statistic(
    np.mean,
    data,
    n_resamples=9999,
    random_state=42,
)

print(result.observed)
print(result.confidence_interval)
```

这个通用工具与模型专属的推断模式并不是同一件事。例如，惩罚 Gaussian 模型的残差自助法有自己的统计定义。如果目标是已拟合模型的系数推断，应先查看 [推断模式](inference-modes.md)，不要假定通用自助法会自动复现某个模型专属的推断程序。

## 文档导航

可以按问题选择页面：

- **如何在 NumPy/CuPy/Torch 上计算概率分布？** → [分布 API](distribution-api.md)
- **如何校正或合并多个 p 值？** → [多重检验](multiple-testing-combine-pvalues.md)
- **如何运行通用排列检验或自助法？** → 本页
- **回归估计器应该选择哪种推断方法？** → [推断模式](inference-modes.md)
- **惩罚 GLM 系数推断的统计目标是什么？** → [惩罚 GLM 推断](penalized-glm-inference.md)

## 估计器包装方法

[估计器共享 API](../reference/estimator-api.md)列出继承辅助方法的全部参数与返回字段，并说明它们与独立函数的差异，包括 p 值结果的字典/元组形式及额外向量化控制。

## 重采样函数签名

独立函数要求显式传入数据，`backend="auto"` 根据这些数组选择后端，返回 `BootstrapResult` 或 `PermutationTestResult`，不是估计器或元组。[共享参考中的重采样部分](../reference/estimator-api.md#bootstrap_statistic)列出各项参数和结果字段；以下两个额外控制项只供独立函数使用。

```python
bootstrap_statistic(
    statistic, *arrays, n_resamples=200, strategy="iid", strata=None,
    clusters=None, block_size=None, confidence_level=0.95,
    random_state=None, statistic_name="statistic", backend="auto",
    force_vectorized=False, statistic_hint=None,
)
permutation_test(
    statistic, X, y, n_resamples=1000, strategy="iid", strata=None,
    groups=None, alternative="two-sided", random_state=None,
    statistic_name="statistic", backend="auto", force_vectorized=False,
    statistic_hint=None,
)
```

- 在批量计算路径下，`force_vectorized=True` 会拒绝不兼容的首次试调用，但后续批次仍可能退回逐次调用。回调应处理所有批次大小，包括最后只有一行的批次；该标志不保证绝不回退。由于正式逐次计算前可能先作批量试调用，回调应避免副作用。
- `statistic_hint="mean"` 加速一个一维数组的 bootstrap 均值。矩阵数据请省略该提示。`"pearson_corr"` 加速置换相关系数，此时 X 须为 `(n,)` 或 `(n, 1)`。原始样本回调必须计算同一统计量，提示不会自动核实两者是否等价。
- 当前整群 bootstrap 仅在群组等大小时保留完整群组。不等大小群组可能被截断，因此不能把所得区间用于整群推断。组内置换则是在各组内重排响应，没有这种截断行为。
- `observed` 和每个重采样统计量都必须有限。百分位区间的含义取决于所选统计量和重采样方案；增加 `n_resamples` 既不能证明方案有效，也不能消除估计偏差。

例如，对矩阵的总体均值作 bootstrap 时，省略快速计算提示，就能按整行重采样：

<!-- api-example: module-matrix-mean -->
```python
import numpy as np
from statgpu.inference import bootstrap_statistic

rows = np.arange(12.0).reshape(6, 2)
result = bootstrap_statistic(
    np.mean, rows, n_resamples=99, random_state=7, backend="numpy",
)
print(result.observed, result.samples.shape)
```

输出为 `5.5 (99,)`。每次重采样都有放回地抽取完整行，因此同一行的两个测量值始终成对保留。

<a id="validate-resampling-labels-before-calling"></a>

## 调用前验证重采样标签

使用 `strata`、`clusters` 或 `groups` 时，每行都应有一个非缺失标签。当前实现并不总会拒绝 NaN 标签：按相等关系分组时，这些行会被遗漏，部分批次元素可能没有初始化。因此，调用可能返回数值有限却无效的统计量，也可能触发索引错误。p 值或区间有限，并不能证明标签有效。

若使用数值编码，可以在重采样前显式验证标签：

<!-- safety-example: validated-resampling-labels -->
```python
import numpy as np
from statgpu.inference import permutation_test


def validated_labels(labels, n):
    labels = np.asarray(labels)
    if labels.ndim != 1 or len(labels) != n:
        raise ValueError("Provide one group label per observation")
    if labels.dtype.kind not in "biuf" or not np.isfinite(labels).all():
        raise ValueError("Group labels must be finite numeric codes")
    return labels


X = np.arange(6.0)
y = np.array([0.2, 1.3, 0.7, 2.5, 3.0, 2.8])
groups = validated_labels([0, 0, 0, 1, 1, 1], len(y))
result = permutation_test(
    lambda X_, y_: np.corrcoef(X_, y_)[0, 1], X, y,
    strategy="grouped", groups=groups, n_resamples=99,
    random_state=7, backend="numpy",
)
print(result.observed, result.pvalue)
```

相关系数约为 `0.903`，置换 p 值介于 0 和 1 之间。该检验要求零假设下响应在**指定的各组内部可交换**。辅助函数会拒绝 NaN、无穷编码和不正确的标签形状，但不会判断分组是否有统计意义。应从数据来源补齐缺失的组别，或明确采用合适的缺失数据处理方案，不要仅为运行计算而把所有未知身份归入一个人为群组。bootstrap 的分层和群组标签同样需要检查；不等大小群组的限制仍然适用。

`alternative="two-sided"` 统计满足 `abs(resampled) >= abs(observed)` 的次数，并作加一修正。因此应选择以零为原假设参照点、绝对值能够衡量极端程度的统计量，例如检验零相关时的相关系数。工具不会自动把统计量中心化，也不会为不对称的零假设分布构造等尾检验。若事先指定了单向备择假设，应相应使用 `"greater"` 或 `"less"`。
