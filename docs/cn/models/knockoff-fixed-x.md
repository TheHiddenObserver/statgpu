# Fixed-X Knockoff 特征选择

> 语言：中文  
> 最后更新：2026-10-13  
> 页面定位：Fixed-X 教程  
> 切换：[English](../../en/models/knockoff-fixed-x.md)

## 在高斯线性响应模型下选择变量

Fixed-X knockoff 将设计矩阵 X 视为固定，对每个观测预测变量构造一个人造对照，
比较两者对响应的重要性。目标是控制错误发现率（FDR），即所选特征中实际无效
特征比例的期望值；空选择时该比例按零计。较大的正统计量 W 表示原特征优于
其对照，但 W 不是回归系数，也不是 p 值。

使用这一方法，需要可信的匹配设计和高斯线性响应模型。
[Model-X knockoff](knockoff-model-x.md) 采用不同的特征分布假设，可以允许
非线性响应。两种方法的选择见[总览](knockoff.md)；完整函数签名、参数、
选择器方法和结果字段见[特征选择 API 参考](../reference/feature-selection-api.md)。

## 什么样的 fixed-X 配对才有效？

对于统计量实际使用的设计，记 Σ=XᵀX。匹配配对需要满足 XkᵀXk=Σ、
XᵀXk=Σ−S，其中 S 是非负对角矩阵，形成的联合 Gram 矩阵应为半正定。
形状相同或列范数一致还不够，交叉 Gram 条件同样重要。

截距对应投影 P=I−11ᵀ/n。需要在这一投影后检查配对：将上述 Gram 分别换成
XᵀPX、XkᵀPXk 和 XᵀPXk。若还要调整其他干扰变量，投影后的几何关系也
必须兼容。这里使用的[响应模型假设](#response-model-assumptions)为

y=b1+Xβ+ε，其中误差相互独立、方差相同，且 ε∼N(0,σ²I)。

此外，还需要兼容的统计量和阈值规则。内置构造通常需要满列秩和 n≥2p，但
这些维度条件本身不能解决[中心化限制](#generated-fixed-x-centering-limitation)。
下面的例子显式提供一个可以直接验证几何关系的配对。

## 首次调用前：确定 q，并了解阈值限制

在查看发现结果之前，确定检验的特征集合、q 和统计量。Knockoff+ 的阈值
计数比最低为 1/p：q=0.20 要求合格阈值至少对应五个正向发现。即使有
20 个预测变量，也不保证一定选中特征。

调用过滤函数或拟合选择器前，检查 `np.isfinite(q) and 0 < q < 1`。
当前验证会漏过 `q=np.nan`，可能返回无效的空选择、`threshold=inf`、
`estimated_fdr=0.0` 和全为 False 的选择掩码。这一提醒适用于两种阈值规则。

<a id="tied-statistic-limitation"></a>

统计量绝对值并列时需要特别小心。当前实现可能只对并列组的一部分计数，
随后却选中整个阈值组。对于 W=[8,8,-8]、q=0.5，报告的比值可能为 0.5，
而完整 knockoff+ 比值实际为 (1+1)/2=1。不能将这样的结果解释为满足名义
FDR 控制。偏移量为 0 的 `knockoff` 也受影响；更换设备、重新拟合或增加
抽样次数都不能修复这一问题。下面的例子没有绝对值并列的 W，详见
[通用阈值说明](knockoff.md#统计量并列时的限制)。

<a id="centered-pair-example"></a>
<a id="a-complete-cpu-example"></a>

## 使用外部配对的完整 CPU 示例

这个受控模拟在生成 y 之前一起构造 X 与 Xk。QR 正交化的第一列留给截距，
其余列分成两组中心化且相互正交的变量。因此 XᵀX=XkᵀXk=I、XᵀXk=0，
满足 S=I 时的匹配配对条件，截距投影后仍然成立。这一特殊构造需要 n≥2p+1。

这是特殊的模拟设计，不是修复任意观测 X 的方法。分析自己的数据时，需要
为原观测设计取得并验证匹配 X/Xk，包括截距或干扰变量投影后的条件。
不要用这段 QR 代码替换已经观测到的预测变量，也不要只对自动生成的 Xk
再做中心化：前者可能改变分析问题，后者可能破坏 Gram 约束。提供 `Xk`
可以跳过自动构造。

按顺序运行以下代码。两个矩阵的形状都是 `(240, 20)`，y 包含 240 个响应。
预先设定六个非零系数及 `corr_diff` 统计量；响应包含截距和标准差为 0.5 的
独立高斯噪声。

<!-- learner-example: knockoff-selection -->
```python
import numpy as np
from statgpu import fixed_x_knockoff_filter
```

### 预先确定设计与目标错误率

先设置维度和错误率，再生成响应。

```python
n, p = 240, 20
q = 0.20
if not (np.isfinite(q) and 0 < q < 1):
    raise ValueError("q must be finite and strictly between 0 and 1")
```

### 构造并验证外部配对

将常数方向留给截距，再取两组正交列。

```python
rng = np.random.default_rng(42)
Q, _ = np.linalg.qr(np.column_stack([np.ones(n), rng.normal(size=(n, 2*p))]))
X, Xk = Q[:, 1:p+1], Q[:, p+1:2*p+1]
```

在使用之前，检查中心化和全部三个 Gram 约束。由于两个矩阵都已中心化，
投影前后的 Gram 矩阵相同。

```python
np.testing.assert_allclose(X.mean(axis=0), 0, atol=1e-14)
np.testing.assert_allclose(Xk.mean(axis=0), 0, atol=1e-14)
np.testing.assert_allclose(X.T @ X, np.eye(p), atol=1e-14)
np.testing.assert_allclose(Xk.T @ Xk, np.eye(p), atol=1e-14)
np.testing.assert_allclose(X.T @ Xk, 0, atol=1e-14)
```

### 生成响应并选择特征

仅前六个原始列影响模拟响应。配对生成后，再用独立同方差高斯噪声生成 y。

```python
beta = np.zeros(p)
beta[:6] = [8, -7, 6, -5, 4, -3]
y = 2.0 + X @ beta + rng.normal(scale=0.5, size=n)
```

显式传入两个矩阵，跳过自动构造。

```python
result = fixed_x_knockoff_filter(
    X, y, Xk=Xk, q=q, method="corr_diff",
    fdr_control="knockoff_plus", backend="numpy",
)
```

查看原始列索引和阈值相关结果。

```python
print("Selected columns:", result.selected_features.tolist())
print("Threshold:", round(result.threshold, 3))
print("Threshold ratio:", round(result.estimated_fdr, 3))
```
<!-- example-end: knockoff-selection -->

## 解读选择结果与不确定性

这个种子的输出为 `Selected columns: [0, 1, 2, 3, 4, 5, 15]`，
阈值为 `1.019`，阈值计数比为 `0.143`。索引从零开始，对应原始列。
模拟真值已知：第 0–5 列是信号，第 15 列是错误发现。目标 q 既不保证每个
发现都正确，也不限制每一份样本中实际发生的错误发现比例。

对于 `corr_diff`，W_j=|X_jᵀ(y−ȳ)|−|Xk_jᵀ(y−ȳ)|。
例如 `result.W[0]` 约为 6.386，支持原始信号；`result.W[11]` 约为 −0.921，
更支持其 knockoff。统计量绝对值没有并列。在未舍入的所选阈值处，七个 W
满足 W≥T，没有 W 满足 W≤−T，所以 knockoff+ 比值为 (1+0)/7。
`estimated_fdr` 报告这一阈值计数比，不是本次实际的无效特征比例，也不是
逐特征 p 值。一次模拟结果不能从经验上证明 FDR 控制。

一般而言，knockoff+ 在正的 |W| 候选阈值中，选取使完整比值
(1+#{W≤−T})/max(1,#{W≥T}) 不超过 q 的最小值。若没有合格阈值，选择为空，
`threshold` 为无穷大，`estimated_fdr=0.0` 是实现的空结果约定，不能证明
所有预测变量都无效。应保留预先确定的 q，不要为了获得发现而放宽目标。
`fdr_control="knockoff"` 去掉分子中的 1，在相应理论下控制的是不同的
修正 FDR。

## 将同一选择用于选择器流程

继续使用 CPU 示例中的 X、y、Xk 和 q。`Xk` 应传给 `fit`，而不是构造函数。
选择器保存结果并应用列掩码，不会拟合响应预测模型。

<!-- example-requires: knockoff-selection -->
<!-- learner-example: knockoff-fixed-x-selector -->
```python
from statgpu import FixedXKnockoffSelector

selector = FixedXKnockoffSelector(
    q=q, method="corr_diff", fdr_control="knockoff_plus", backend="numpy",
)
assert selector.fit(X, y, Xk=Xk) is selector
support = selector.get_support()
selected = selector.transform(X[:10])
np.testing.assert_array_equal(selector.selected_features_, result.selected_features)
print("Support indices:", np.flatnonzero(support).tolist())
print("Transformed shape:", selected.shape)
```
<!-- example-end: knockoff-fixed-x-selector -->

掩码对应索引为 `[0, 1, 2, 3, 4, 5, 15]`，变换后形状为 `(10, 7)`。
通过 `selector.result_` 查看诊断信息。这十行仅演示列选择，不是留出评估。
如果要评价预测，应在每个训练折内做选择，再用选中的训练列拟合另一个估计器，
并将相同掩码应用于始终未参与选择的评估数据。

重新拟合失败后，选择器可能仍保留上一次成功的选择。发生拟合错误时，不要
将这些状态当作新数据的结果；应使用新的选择器，并完成一次成功拟合。
详见[选择器说明](../reference/feature-selection-api.md#选择器方法)。

## 在查看发现前选定统计量

- `corr_diff` 比较响应中心化后的内积，简单且计算开销小；边际关系较弱时，
  检验功效可能不足。
- `ols_coef_diff` 比较原特征与 knockoff 联合回归的系数绝对值。应考虑联合
  设计及其数值条件。
- `lasso_coef_diff` 使用正则化联合拟合。交叉验证实现和数值设置会改变 W。
  初次比较可保持 `lasso_fast_profile="off"`；较快设置可能改变折数、候选
  惩罚强度、迭代预算和容差，进而改变 W 及入选特征。

使用整数种子的 `lasso_coef_diff` 调用，在原地修改输入或复用早先数组内存后，
可能沿用过期统计量。仅创建新选择器不足以隔离这一风险。需要可重复地分析
更改后的数据时，应每次使用新的 Python 进程。对于显式提供的 float64 NumPy
X/y/Xk，也可以每次复制全部三个输入，同时让所有历史输入始终存活且不再改变。
`random_state=None` 可停用基于种子的复用，但不再提供种子可重复性。
详见[完整 Lasso 警告与复制示例](../reference/feature-selection-api.md#repeated-lasso-statistic-calls)。

<a id="generated-fixed-x-centering-limitation"></a>

## 为什么需要谨慎使用自动构造？

自动 fixed-X 构造会中心化 X，却可能生成列均值非零的 Xk。
`corr_diff`、`ols_coef_diff` 和非 knockpy 风格的原生 Lasso 统计量会中心化 y。
因此，原始 Gram 矩阵相等的关系可能在 P=I−11ᵀ/n 投影后消失：XᵀPX 与
XkᵀPXk 可以不同。即使满秩、n>2p 且没有阈值并列，通常依赖的无效特征
得分配对交换性论证仍可能失效。仅增加 n 不能修复这一问题。

例如，输入 X=[−1,1]ᵀ 的 Gram 为 2，标准化工作设计为
X_work=[−1,1]ᵀ/√2。构造种子 7
生成的 knockoff 约为 [0.707,0.707]ᵀ。未投影的工作 Gram 都为 1，但投影后
分别为 1 和 0。对于任意非常数中心化响应，不论响应符号如何，相关差统计量
都会偏向 X。这说明无效特征的符号对称性丢失，并不是测得了 FDR 超标：
只有一个特征时 knockoff+ 不会选中任何特征。

有效且中心化的外部配对可以避开这个几何缺陷。需要检查截距或干扰变量投影后
的全部匹配条件。直接对生成的 Xk 再做中心化，可能破坏所需的 Gram 关系。
[QR 示例](#centered-pair-example)是需要 n≥2p+1 的特殊模拟，不能代替观测 X，
也不是任意 X 的通用修复方法。形状检查、有效的原始 Gram 以及没有并列，
本身都不足以证明 FDR 控制。

## 同一示例的可选 GPU 写法

以下代码复用 CPU 示例的外部配对和 q，需要可用的 CUDA 后端。选择其中
一种即可；两者都不改变统计构造。

### CuPy

```python
import cupy as cp

X_gpu, y_gpu, Xk_gpu = cp.asarray(X), cp.asarray(y), cp.asarray(Xk)
res_gpu = fixed_x_knockoff_filter(
    X_gpu, y_gpu, Xk=Xk_gpu, q=q, method="corr_diff",
    fdr_control="knockoff_plus", backend="cupy",
)
```

### Torch CUDA

```python
import torch

X_torch, y_torch, Xk_torch = [torch.from_numpy(a).to("cuda") for a in (X, y, Xk)]
res_torch = fixed_x_knockoff_filter(
    X_torch, y_torch, Xk=Xk_torch, q=q, method="corr_diff",
    fdr_control="knockoff_plus", backend="torch",
)
```

`backend="torch"` 选择计算库，不同于请求 CUDA 的估计器 `device="torch"`。
原生构造遵循 X 所在设备；X/y/Xk 应位于同一设备。但这一行为不适用于原生
Torch `lasso_coef_diff` 的调参与拟合：即使输入在 CPU 或显式提供 Xk，
它仍请求 CUDA，也不保证使用输入所在的非默认 GPU。若要在 Torch CPU 上
计算统计量，可在适用时选择 `corr_diff` 或 `ols_coef_diff`；CPU Lasso 则
使用 NumPy 输入和 `backend="numpy"`。详见
[Torch 设备位置](../reference/feature-selection-api.md#torch-device-placement)与
[Torch Lasso 设备路由](../reference/feature-selection-api.md#torch-lasso-device-routing)。
兼容设置还可能涉及 CPU 计算，单凭后端标签不能说明每一步都在 GPU 上执行。

<a id="response-model-assumptions"></a>

## 统计适用范围与参考资料

Fixed-X 的有限样本结论要求高斯线性响应、独立同方差正态误差、有效匹配设计、
兼容统计量及适当的截距处理。输入有限的 y 不代表这些假设自动成立。
输出不包含系数协方差、置信区间或预测方法。若响应关系不限于线性，但有
可信的特征分布构造，请阅读 [Model-X 教程](knockoff-model-x.md)。

- Barber, R. F., & Candès, E. J. (2015). Controlling the false discovery rate via
  knockoffs. *Annals of Statistics*, 43(5), 2055–2085.
  [论文](https://doi.org/10.1214/15-AOS1337)。
- [完整特征选择 API](../reference/feature-selection-api.md)。
- [可选贡献者验证指南](../../../dev/references/model-validation.md#knockoff)。
