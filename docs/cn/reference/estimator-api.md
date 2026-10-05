# 估计器共享 API

> 语言：中文  
> 最后更新：2026-10-05  
> 切换：[English](../../en/reference/estimator-api.md)

本页说明继承自 `BaseEstimator` 的方法。类上有这些方法，不代表相应模型一定提供系数 p 值或缓存训练数组。模型覆盖的方法及其拟合结果以模型专属参考为准。这些辅助方法不会自动重拟合模型、判断重采样方案是否科学合理，也不会自动修正变量选择的不确定性。

<a id="parameter-management"></a>

## 参数管理

| 方法 | 参数与默认值 | 返回值与行为 |
|---|---|---|
| `get_params(deep=True)` | `deep=True` 时，以 `name__parameter` 包含嵌套估计器配置 | 返回构造参数字典，包括运行时加入的公开参数；不包括拟合系数和任意附加属性。 |
| `set_params(**params)` | 使用 `get_params` 中的构造参数名；嵌套配置使用 `__` | 返回 `self`。基类实现验证更新并重建未拟合状态；未知参数抛出 `ValueError`。空调用不改变对象。更新后应重新拟合；模型覆盖的方法可能不同，尤其是 GAM，复用前应查阅其限制。 |

直接修改属性不等于保证执行 `set_params`/重新拟合的完整过程。`sample_weight` 等仅由 `fit` 接受的参数不是构造配置。`get_params` 也不是整个已拟合对象的序列化副本。

<a id="inference-helpers"></a>

## 推断辅助方法

以下是通过 `model.method(...)` 调用的**估计器方法**。它们与 `statgpu.inference` 中同名函数的签名和返回形式存在区别。

### adjust_pvalues

```python
model.adjust_pvalues(pvalues=None, method="bh", alpha=0.05, axis=0, backend="auto")
```

- `pvalues`：位于 `[0, 1]` 的有限 p 值；`None` 使用 `model._pvalues`。没有相应结果时，应显式传入 p 值，或使用受支持的推断方式重新拟合；否则抛出 `RuntimeError`。
- `method`：`"bh"`、`"by"`、`"holm"`、`"bonferroni"` 或 `"hochberg"`；别名见[多重检验指南](../guides/multiple-testing-combine-pvalues.md)。
- `alpha`：位于 `(0, 1)` 的显著性水平。
- `axis=0`：沿第一个轴校正；`None` 将全部元素视为同一个检验族。多目标系数数组尤其需要明确检验族。
- `backend`：`"auto"`、`"numpy"`、`"cupy"` 或 `"torch"`；估计器解析为 GPU 设备时，`auto` 显式选择对应 CuPy/Torch；解析为 CPU 时，当前仍由传入数组决定后端。若必须返回 NumPy，请设 `backend="numpy"`。估计器辅助方法中显式指定 `backend="torch"` 要求 Torch CUDA，即使估计器配置为 CPU 也是如此。

返回字典，键为 `method`、`alpha`、`axis`、`backend`、`pvalues`、`pvalues_adjusted` 和布尔数组 `reject`。校正值和拒绝决定保持输入形状。`backend` 字段记录辅助方法传下去的选项，可能仍为 `"auto"`，不一定是最终数组库名。没有显式转换时，`pvalues` 可能保留原列表或数组类型；它不是模块函数返回的 `(reject, adjusted)` 元组。原始边际 p 值必须有效，并满足所选方法的依赖结构假设。

### combine_pvalues

```python
model.combine_pvalues(pvalues=None, method="fisher", weights=None, axis=None, backend="auto")
```

`pvalues` 和 `backend` 含义同上。`method` 为 `"fisher"`、`"cauchy"` 或 `"stouffer"`，别名见多重检验指南。`weights=None` 使用方法的等权约定；加权方法的权重应为有限非负数、与归约轴对齐且总和为正。Fisher 必须使用 `weights=None`，传入权重会抛出 `ValueError`。`axis=None` 展平全部元素，整数则指定归约轴。

返回字典：`method`、`axis`、`backend`、`pvalues`、`weights`、`statistic` 和 `pvalue`。展平时后两项为标量，否则具有归约后的形状。模块函数则返回 `(statistic, pvalue)`。Fisher/Stouffer 的校准仍有相应依赖结构假设；合并 p 值不能使无效检验变得有效。

### bootstrap_statistic

```python
model.bootstrap_statistic(
    statistic, *arrays, n_resamples=200, strategy="iid", strata=None,
    clusters=None, block_size=None, confidence_level=0.95,
    random_state=None, statistic_name="statistic", backend="auto",
)
```

| 参数 | 含义与限制 |
|---|---|
| `statistic` | 接收对齐数组并返回有限标量的可调用对象；须支持所选数组后端。 |
| `*arrays` | 一个或多个非空数组，第一轴长度一致。单数组传入 `data`，不要包装成 `(data,)`。省略时尝试使用缓存 `_X_design` 和 `_y`，没有缓存则报错。设计缓存可能含截距、公式列或乘以权重平方根的行，而响应缓存未必采用同样的权重转换。定义加权统计量时请显式传入数组与权重，不能把缓存当作保证有效的原始训练数据对。 |
| `n_resamples=200` | 正整数，重采样次数。增加次数减少蒙特卡洛波动，不消除模型偏差。 |
| `strategy="iid"` | `iid`：有放回抽行；`stratified`：各层内重采样；`cluster`：等大小群组的整群重采样（限制见下文）；`block`：抽取连续块。所有数组使用相同的行索引。 |
| `strata=None`、`clusters=None` | 每行一个非缺失标签；分别为分层或整群策略所必需。调用前请检查标签，当前 NaN 标签可能造成不完整重采样，而不是明确报错。 |
| `block_size=None` | 分块策略必需的正整数；大于 n 时按 n 处理。应保留有意义的观测顺序。 |
| `confidence_level=0.95` | `(0, 1)` 内的百分位区间置信水平。 |
| `random_state=None` | 整数种子，或不固定种子。可复现性针对同一后端和过程，不保证不同数组库产生相同样本。 |
| `statistic_name="statistic"` | 报告标签，不负责选择或改变统计量。 |
| `backend="auto"` | 遵循上文 GPU 与 CPU-auto 的区别；要求 NumPy 时显式指定 `numpy`。显式 CuPy/Torch 请求要求相应 GPU 后端可用。 |

返回 `BootstrapResult`，包含 `observed`（原始标量）、`samples`（长度为 `n_resamples` 的后端数组）、`confidence_interval`（上下界二元组）、`confidence_level`、`n_resamples`、`random_state`、`statistic_name`、`strategy` 和 `metadata`。`to_dict()` 把样本转换为列表；`to_dataframe()` 需要 pandas，返回 `sample_index` 和 `statistic` 两列。原始统计量属性名为 `observed`，不是 `statistic`。

**不等大小群组的限制。** 当前 `cluster` 实现会持续抽取群组，直到累计行数达到 n，再截断最后一个群组以保留 n 行。这可能拆开群组，不能作为不等大小群组的有效整群 bootstrap。此时不要使用它给出的区间，应改用经过验证、保留完整群组的重采样程序。本辅助方法仅适用于群组确实等大小的情形；为凑齐大小而删行或补行会改变统计问题。

可运行的数值标签验证方法与缺失标签限制见[标签验证](../guides/inference-api.md#validate-resampling-labels-before-calling)。不要把身份不明且未必相关的观测归入一个人为群组。

区间取重采样分布的 `(1-confidence_level)/2` 与 `(1+confidence_level)/2` 分位数。可交换性和重采样单位的选择由调用者负责。回调可能先收到带前导批次维度的数组，再退回逐次调用；应避免副作用，若返回批量结果则明确沿正确的轴计算。这一通用方法不是 ElasticNet 的残差系数 bootstrap 推断模式。

### permutation_test

```python
model.permutation_test(
    statistic, X, y, n_resamples=1000, strategy="iid", strata=None,
    groups=None, alternative="two-sided", random_state=None,
    statistic_name="statistic", backend="auto",
)
```

`statistic(X, y)` 必须返回有限标量。`X` 与一维 `y` 的非零行数须一致。置换保持 `X` 不变，只重排响应：`iid` 在全体观测内，`stratified` 在长度为 n 的 `strata` 各层内，`grouped` 在长度为 n 的 `groups` 各组内。这里的 grouped 是**组内置换**，不是整组互换。`n_resamples` 为正整数；`alternative` 为 `"two-sided"`、`"greater"` 或 `"less"`。随机种子、报告标签、后端及非缺失标签检查与 bootstrap 相同。

返回 `PermutationTestResult`：`observed`、长度为 `n_resamples` 的后端 `samples`、`pvalue`、`n_resamples`、`random_state`、`statistic_name`、`strategy`、`alternative` 和 `metadata`。`to_dict()`、`to_dataframe()` 的行为与 bootstrap 结果一致。蒙特卡洛尾概率使用加一修正；双侧比较使用统计量绝对值，因此应选择以零为原假设参照点、绝对值能够衡量极端程度的统计量；工具不会自动中心化或构造等尾检验。只有在零假设下响应允许相应置换时，检验才有合理解释。

## 可运行的辅助方法示例

<!-- api-example: estimator-helpers -->
```python
import numpy as np
from statgpu import LinearRegression

model = LinearRegression(device="cpu", compute_inference=False)
adjusted = model.adjust_pvalues([0.01, 0.04, 0.5], method="holm")
print(adjusted["reject"].tolist())
boot = model.bootstrap_statistic(
    np.mean, np.arange(1.0, 11.0), n_resamples=99, random_state=7,
)
print(boot.observed, len(boot.samples))
```

输出为 `[True, False, False]` 和 `5.5 99`。显式传入 p 值与数据后，这些调用不要求模型已经拟合；该示例没有对模型系数作推断。

## 与模块函数的区别

独立的 `bootstrap_statistic` 与 `permutation_test` 函数还接受 `force_vectorized=False` 和 `statistic_hint=None`，估计器包装方法不接受这两项。IID、分层重采样、分块 bootstrap、等大小整群 bootstrap 和组内置换均有批量计算方式；这些方式下，`force_vectorized=True` 会拒绝不兼容的首次批量试调用，但后续批次不兼容时仍可能退回逐次调用，不能保证所有重采样都采用向量化执行。向量化回调应能处理任意批次大小，包括最后只有一行的批次，并为每行返回一个结果。不等大小整群 bootstrap 当前即使设置该标志也逐次调用，不能把它视为通用的向量化保证。`statistic_hint="mean"` 只适用于一维数组的 bootstrap 均值。矩阵数据请省略该提示并提供返回标量的统计量；当前均值快速计算只沿最后一轴求均值，可能触发广播错误。置换相关系数可用 `"pearson_corr"`，此时 X 须为 `(n,)` 或 `(n, 1)`。提示会选择内置的重采样计算，不会检查原始统计量回调是否与之等价。独立函数的 `backend="auto"` 根据数组推断，且数据必须显式提供。导入与示例见[模块指南](../guides/inference-api.md)。
