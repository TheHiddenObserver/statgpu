# Model-X Knockoff 特征选择

> 语言：中文  
> 最后更新：2026-10-13  
> 页面定位：Model-X 教程  
> 切换：[English](../../en/models/knockoff-model-x.md)

## 在能够刻画特征分布时选择变量

Model-X knockoff 关注：给定其他特征之后，哪些特征仍为 y 提供信息？
第 j 个特征的零假设是，给定其余特征后，y 与 X_j 条件独立。每个原始特征
都与一个人造对应变量竞争；正的 W 更支持原特征。目标是错误发现率（FDR）
控制，不是预测误差，也不是逐特征 p 值。

响应可以是非线性、离散或非高斯的。相应地，你需要根据联合特征分布论证
knockoff 构造的有效性。本教程先解释这一要求，再用已知的独立高斯特征分布
和非线性响应演示完整流程。

[Fixed-X knockoff](knockoff-fixed-x.md) 则依赖另一组假设：有效的配对设计
和高斯线性响应模型。仅凭样本量不能决定采用哪一种方法。选型比较见
[总览](knockoff.md)；全部函数、参数、方法和结果字段见
[特征选择 API 参考](../reference/feature-selection-api.md)。

## 什么样的 Model-X knockoff 才有效？

构造需要满足两个条件：

1. 任意交换一组特征对中的 X_j 与 Xk_j，(X, Xk) 的联合分布保持不变。
   这称为配对交换性。
2. 给定 X 后，Xk 与 y 条件独立。构造 knockoff 时不应利用响应信息。

这些是分布要求，不是对一份样本的 Gram 矩阵做相等检验。形状、样本协方差
或设备一致，不能证明这些条件。还需要兼容的反对称统计量和有效阈值规则。

下面的 CPU 示例使用已知特征分布：X 的每一行、每一个坐标都是独立标准高斯
变量。再从相同分布独立抽取一个矩阵作为 Xk，在这里就是有效构造：交换任意
特征对后，所有元素仍为相互独立的标准高斯变量。响应仅由 X 和独立噪声生成，
因此第二个条件也成立。

独立重抽样不是相关特征或未知分布的通用 knockoff 生成器。即便独立复制整个
相关特征向量，也未必满足逐坐标交换后的分布不变。不要用这个模拟替换观测
数据，也不要用各自的样本矩分别标准化 X 和 Xk。分析真实数据时，必须根据
实际特征分布论证构造的有效性。

## 首次调用前：确定 q，并了解阈值限制

预先确定检验的特征集合、q 和统计量。Knockoff+ 的阈值计数比最低为 1/p；
q=0.20 要求至少五个 W 为正且达到阈值的特征。20 个特征只是让非空选择
成为可能，并不保证一定选中变量。不要在看到结果后为了选中更多特征而改变 q。

调用筛选函数或拟合选择器前，检查 `np.isfinite(q) and 0 < q < 1`。
当前验证会漏过 `q=np.nan`，在两种阈值规则下都可能返回无效的空选择、
`threshold=inf`、`estimated_fdr=0.0` 或全为 False 的掩码。

<a id="tied-statistic-limitation"></a>

当前阈值实现可能错误处理绝对值并列的 W：它只对并列组的一部分计数，
随后却选中整个阈值组。对于 W=[8,8,-8]、q=0.5，报告的比值可能为 0.5，
而完整 knockoff+ 比值为 (1+1)/2=1。不能将这种输出解释为满足名义 FDR
控制；偏移量为 0 的 `knockoff` 同样受影响。增加抽样、重新拟合或更换设备
都不能修复这个计数问题。下面的例子没有绝对值并列的 W，详见
[通用阈值说明](knockoff.md#统计量并列时的限制)。

## 已知高斯特征分布的完整 CPU 示例

按顺序运行以下代码。模拟中已知特征分布，并显式提供 Xk，因此不涉及协方差
估计。两个矩阵的形状都是 `(400, 20)`，y 包含 400 个响应。正弦项使响应具有
非线性，线性部分则让所选的边际统计量具有检出信号的能力。

<!-- learner-example: knockoff-model-x -->
```python
import numpy as np
from statgpu import model_x_knockoff_filter

n, p = 400, 20
q = 0.20
if not (np.isfinite(q) and 0 < q < 1):
    raise ValueError("q must be finite and strictly between 0 and 1")
```

### 从已知分布生成 X 和独立的 knockoff

先抽取两个矩阵，再生成 y。每个元素都服从同一个标准高斯分布；两个矩阵
都不使用样本估计做中心化或缩放。

```python
rng = np.random.default_rng(73)
X = rng.normal(size=(n, p))
Xk = rng.normal(size=(n, p))
```

### 生成非线性响应

只有第 0–5 列参与响应生成，噪声与两个矩阵均独立。本例使用高斯噪声是为了
方便模拟，并不是 Model-X 对响应的要求。

```python
beta = np.zeros(p)
beta[:6] = [3, -2.5, 2, -1.5, 1.2, -1]
y = 2.0 + X @ beta + 2 * np.sin(X[:, 0]) + rng.normal(scale=1.0, size=n)
```

### 使用外部矩阵进行一次选择

这里特意设置 `modelx_draws=7`，说明显式提供 Xk 时的行为：只提供了一个
矩阵，函数就只对它计算一次，不会另行生成七份 knockoff。实际次数见元信息。

```python
result = model_x_knockoff_filter(
    X, y, Xk=Xk, q=q, method="corr_diff",
    fdr_control="knockoff_plus", backend="numpy", modelx_draws=7,
)
print("Selected columns:", result.selected_features.tolist())
print("Threshold:", round(result.threshold, 3))
print("Threshold ratio:", round(result.estimated_fdr, 3))
print("Knockoff source:", result.metadata["xk_source"])
print("Actual draws:", result.metadata["n_modelx_draws"])
```

### 查看完整阈值计数

计数时使用未舍入的 W 和 T。必须计入每个阈值组的全部成员；显示的小数阈值
只供阅读。

```python
positive_count = int(np.count_nonzero(result.W >= result.threshold))
negative_count = int(np.count_nonzero(result.W <= -result.threshold))
print("Positive / negative counts:", positive_count, negative_count)
print("W[0] / W[17]:", round(result.W[0], 3), round(result.W[17], 3))
assert np.unique(np.abs(result.W)).size == p
```
<!-- example-end: knockoff-model-x -->

## 解读 W、阈值和元信息

输出为 `Selected columns: [0, 1, 2, 3, 4, 5]`，阈值为 `173.092`，
阈值计数比为 `0.167`，来源为 `provided`，实际抽样次数为 `1`。正、负向
计数为 `6 0`；W[0] 与 W[17] 约为 `1541.746` 和 `-90.694`。
前者支持原始信号，后者更支持其 knockoff。

这一统计量为 W_j=|X_jᵀ(y−ȳ)|−|Xk_jᵀ(y−ȳ)|，是响应中心化后内积绝对值
之差，不是归一化相关系数；其尺度受输入尺度与样本量影响。六个较大的正 W
满足 knockoff+ 比值 (1+0)/6。这次运行恰好选中了全部模拟信号，但目标 q
不保证完美恢复，也不控制每份样本实际发生的错误发现比例，更不能用一次
成功模拟证明方法有效。

Knockoff+ 在正的 |W| 候选阈值中，选择使完整比值
(1+#{W≤−T})/max(1,#{W≥T})≤q 的最小值。空选择表示没有合格阈值；此时
`threshold=inf` 和 `estimated_fdr=0.0` 是结果报告约定，不能证明所有特征
均无效。`estimated_fdr` 是阈值规则的估计量，不是已知的实际错误比例或 p 值。
偏移量为 0 的 `knockoff` 在相应理论下控制的是不同的修正 FDR。

元信息确认的是 API 如何使用 Xk，不是它在统计上是否有效。本例的有效性
来自已知的独立特征分布及 y 的生成方式，而不是 `provided` 标签或没有并列。

## 用 KnockoffSelector 应用选择结果

继续使用相同的 X、y、Xk 和 q。统一选择器默认采用 fixed-X，因此这里必须
显式设置 `knockoff_type="model_x"`。Xk 传给 `fit`，而不是构造函数。

<!-- example-requires: knockoff-model-x -->
<!-- learner-example: knockoff-model-x-selector -->
```python
from statgpu import KnockoffSelector

selector = KnockoffSelector(
    knockoff_type="model_x", q=q, method="corr_diff",
    fdr_control="knockoff_plus", backend="numpy", modelx_draws=7,
)
assert selector.fit(X, y, Xk=Xk) is selector
support = selector.get_support()
selected = selector.transform(X[:10])
np.testing.assert_array_equal(selector.selected_features_, result.selected_features)
print("Support indices:", np.flatnonzero(support).tolist())
print("Transformed shape:", selected.shape)
```
<!-- example-end: knockoff-model-x-selector -->

掩码对应索引为 `[0, 1, 2, 3, 4, 5]`，变换后形状为 `(10, 6)`。
诊断信息保存在 `selector.result_` 中，尚未拟合预测模型。这些行只用于演示
掩码，不是留出评估。评价预测时，应在每个训练折内做选择，再用选中的训练列
拟合单独的模型，并将掩码应用于始终未参与选择的评估行。重新拟合失败可能
留下上一次成功的选择；使用新数据的结果前，应换一个选择器并成功完成拟合。

## 自动生成 Xk 时，有什么不同？

当 `Xk=None` 时，原生 Model-X 流程会标准化观测特征、估计正则化的高斯
协方差，再构造高斯 knockoff。`modelx_covariance_shrinkage` 和
`modelx_s_scale` 会影响这一构造。估计得到的高斯矩不能为任意特征分布提供
精确交换性保证，也不会自动恢复上述已知分布构造。仅增加样本量不能证明
模型正确。

自动生成时，`modelx_draws` 控制重复构造和统计量计算的次数。先对各次的 W
取平均，再对平均值做一次阈值选择，并不是对各次选中集合投票。设置为 `None`
时，当前 `corr_diff` 默认使用三次，OLS/Lasso 系数差默认使用五次。若显式
提供 Xk，无论请求的次数多大，都只计算一次。实际行为见
`metadata["n_modelx_draws"]` 和 `metadata["xk_source"]`。

平均可能降低蒙特卡洛波动，也会增加计算成本；但它本身不能证明平均统计量
具有 FDR 控制定理，不能修复无效的特征模型，也不能修复阈值并列计数。

### 兼容构造与元信息

`compat_mode="knockpy"` 会改变构造和统计量约定，也可能涉及 CPU 计算及
可选依赖。使用内置的自动 Model-X 构造时，应查看
`metadata["modelx_covariance_estimator"]` 和
`metadata["modelx_smatrix_source"]`。后者区分 `knockpy` 与
`equicorrelated_fallback`；请求的 `modelx_smatrix_method` 标签不能证明
实际执行了该方法。缺少依赖，或 S 矩阵调用发生异常（包括请求了无效方法），
都可能触发替代构造。协方差方法也可能被替代，例如 sklearn 不可用时的
`mle_fallback_no_sklearn`。这些标签不证明 knockoff 有效，也不证明与其他
软件包数值一致。各类替代行为详见
[兼容设置说明](../reference/feature-selection-api.md#兼容模式的实际方法与回退)。
可选的 `knockpy_sampler` 仅在 `compat_mode="knockpy"` 且 `Xk=None` 时参与
分派，其他路径会忽略它。这些分派目标目前不受支持，会抛出
`NotImplementedError`；使用内置构造时应保持为 `None`。

## 有意识地选择统计量与计算方式

`corr_diff` 计算开销小，但可能漏掉与响应边际相关较弱的特征，例如纯对称
二次效应。允许非线性响应，不意味着每一种统计量对每一种备择关系都有足够
检验功效。OLS/Lasso 系数差使用联合拟合，但不能验证估计的特征分布。
在查看发现结果前确定统计量及 `lasso_fast_profile`。较快设置可能改变交叉
验证折数、候选惩罚强度、迭代预算和容差，进而改变 W 及所选特征，而不只是
改变运行时间。

### Lasso 缓存限制

设置了种子的 `lasso_coef_diff` 在原地修改输入或复用早先输入内存后，可能
沿用过期统计量。新建选择器不能隔离这一缓存。需要可重复地分析更改后的数据时，
应使用新的 Python 进程。对于显式提供的 float64 NumPy X/y/Xk，也可以每次
复制全部三个输入，同时保留所有历史输入，且不再修改它们；自动生成 Xk 时
优先使用独立进程。`random_state=None` 可停用基于种子的复用，但无法再依靠
固定种子重复结果。详见[完整 Lasso 警告](../reference/feature-selection-api.md#repeated-lasso-statistic-calls)。

### Torch 构造与 Lasso 拟合的设备行为不同

`backend="torch"` 选择计算库，不是请求 CUDA。原生自动 Model-X 构造遵循
X.device，包括 CPU 和非默认 CUDA 设备；原生统计量的 X/y/Xk 应位于同一设备。
但原生 Torch `lasso_coef_diff` 调参与拟合即使面对 CPU 输入或外部 Xk，
仍会请求 CUDA，也不保证使用输入所在的 GPU 编号。显式设置
`lasso_cv_impl="statgpu"` 不能消除这一限制。在 Torch CPU 上，可按统计问题
选择 `corr_diff` 或 `ols_coef_diff`；CPU Lasso 则使用 NumPy 输入和
`backend="numpy"`。详见
[Torch 设备位置](../reference/feature-selection-api.md#torch-device-placement)与
[Lasso 设备路由](../reference/feature-selection-api.md#torch-lasso-device-routing)。

<a id="reproducibility-of-generated-torch-model-x"></a>

### 自动 Torch Model-X 构造的可重复性

原生自动构造（`compat_mode="statgpu"`、`Xk=None`）中，整数 `random_state`
可使 Torch 构造在相同后端、数据类型（dtype）、设备和软件环境下重复，同时不推进
全局 Torch 随机数状态。统一筛选函数和选择器同样适用。这既不是跨后端或跨 GPU 相等的
保证，也不保证实际 FDR 控制；统计量的可重复性需要单独检查，尤其是 Lasso。

当 `random_state=None` 时，原生 Torch/CuPy 构造会为每次抽样使用种子 0，
重复抽样因而使用相同的构造噪声。需要可重复但分别设种子的抽样时，应传入
整数。NumPy 构造使用 `None` 时仍不固定种子。提供经过外部验证的 Model-X Xk 可跳过随机构造，其随机性由
调用者控制，就像上述 CPU 示例。外部验证必须考虑交换性，以及给定 X 后
与 y 的条件独立性，不能只检查形状或设备。Fixed-X 有自己的匹配设计及响应
假设。这些种子规则与上述 Lasso 缓存限制是不同问题。详见
[种子行为说明](../reference/feature-selection-api.md#torch-model-x-构造的种子与可重复性)。

## 参考资料与后续阅读

- Candès, E., Fan, Y., Janson, L., & Lv, J. (2018). Panning for gold: Model-X
  knockoffs for high-dimensional controlled variable selection. *JRSS B*,
  80(3), 551–577。[论文](https://doi.org/10.1111/rssb.12265)。
- [Fixed-X 教程](knockoff-fixed-x.md)：匹配设计几何关系与高斯线性响应模型。
- [完整特征选择 API](../reference/feature-selection-api.md)。
- [可选贡献者验证指南](../../../dev/references/model-validation.md#knockoff)。
