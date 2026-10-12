# 协方差估计

> 语言：中文
> 最后更新：2026-10-12
> 切换：[English](../../en/models/covariance.md)

## 概览

多个测量值怎样一起变化？什么样的组合才算偏离常态？协方差矩阵概括变量的共同变化，
可用于描述数据、构造多变量距离，或为其他模型提供协方差估计。这里没有单独的响应变量：
数据矩阵的每一行是一次观测，每一列是一个测量变量。

本页介绍各方法共有的概念，帮助你读懂矩阵、理解距离并选择估计器。
选定方法后，可直接阅读对应的独立教程；每篇都有完整的数据准备、拟合和结果解释。
需要核对构造参数、方法、输出、输入限制或设备规则时，查阅
[完整协方差 API](../reference/covariance-api.md)。

<a id="covariance-correlation-precision"></a>

<a id="目标函数"></a>

## 三种矩阵，三种含义

设随机向量 $x$ 的均值为 $\mu$，总体协方差为

$$
\Sigma=\mathbb E[(x-\mu)(x-\mu)^\top].
$$

对角线是各变量的方差。非对角元素为正，表示两个变量往往相对各自均值同向变化；
为负则表示反向变化。协方差刻画线性共同变化，协方差为零并不排除非线性依赖。

相关矩阵去除了各变量的边际尺度：

$$
R_{jk}=\frac{\Sigma_{jk}}{\sqrt{\Sigma_{jj}\Sigma_{kk}}}.
$$

两变量的方差均为正时，相关系数无量纲，取值在 $-1$ 到 $1$ 之间。
常量变量的相关系数没有定义。相比之下，不知道测量单位，就无法单凭“协方差为 10”
判断关系有多强。

当 $\Sigma$ 可逆时，精度矩阵为 $\Theta=\Sigma^{-1}$。在多元高斯模型中，
$\Theta_{jk}=0$ 表示给定其余变量后，变量 $j$ 与 $k$ 条件独立，
这与边际相关为零是不同的概念。偏相关系数是
$-\Theta_{jk}/\sqrt{\Theta_{jj}\Theta_{kk}}$，精度矩阵的元素本身并不是相关系数。
条件关联不能证明因果关系；上述条件独立解释也不能直接推广到任意非高斯分布。

<a id="units-and-standardization"></a>

## 单位与标准化会改变问题

用米测量的变量，其方差单位是平方米；两个变量的协方差具有两种单位的乘积。
把一列乘以 100，其方差就会乘以 10,000。因此，较大的矩阵元素可能只是测量单位不同，
未必意味着关系更强。

逐列标准化可以让变量具有可比较的边际尺度，适合某些稀疏图模型或向缩放单位矩阵收缩的分析。
但标准化也改变了所估计的协方差。只在训练行上估计均值和尺度，再将同一变换用于留出数据；
不要分别标准化训练集和测试集。常量列需要单独处理。若 $D$ 的对角线是训练数据的尺度，
标准化后的协方差估计 $\widehat\Sigma_z$ 可用 $D\widehat\Sigma_zD$ 换回原单位。

`assume_centered=True` 只是跳过均值估计，并不会标准化各列。
只有当零确实是希望使用的已知中心时，才采用这个选项。

<a id="sample-size-and-stability"></a>

<a id="限制与失败行为"></a>

## 为什么变量数会影响稳定性

有 $n$ 次观测、$p$ 个变量时，协方差包含 $p(p+1)/2$ 个不同的待估计量。
中心化后的经验协方差秩至多为 $\min(p,n-1)$：当 $p\ge n$ 时必然奇异；
即使尚未达到这个比例，变量之间很强的相关性也可能使矩阵不稳定。

求逆会放大小特征值方向上的误差。因此，逐个元素看起来合理的协方差矩阵，
未必能给出可靠的距离或精度估计。增加观测、减少冗余变量或采用合适的正则化都有帮助。
求逆时的数值保护无法补回数据中不存在的信息。即使稳定化后得到了精度矩阵，
保存的协方差矩阵仍可能奇异，高斯对数似然评分也可能不是有限值。

<a id="choose-an-estimator"></a>

<a id="稳健与稀疏估计的替代方案"></a>

## 先选问题，再选估计器

| 想回答的问题 | 独立教程 | 主要取舍 |
|---|---|---|
| 数据直接呈现什么协方差？ | [EmpiricalCovariance](covariance-empirical.md) | 简单基准，容易受异常值和病态矩阵影响 |
| 应该向等方差、零协方差的目标收缩多少？ | [ShrunkCovariance](covariance-shrunk.md) | 由用户选择偏差与稳定性的取舍 |
| 能否根据数据解析地选择收缩强度？ | [LedoitWolf](covariance-ledoit-wolf.md) | 向缩放单位矩阵收缩 |
| 高斯小样本下怎样选择解析收缩规则？ | [OAS](covariance-oas.md) | 以高斯假设下的协方差估计为出发点 |
| 少量观测受到污染时，大多数数据的中心和分散程度是什么？ | [MinCovDet](covariance-min-cov-det.md) | 搜索稳健子集，要求观测数大于变量数 |
| 哪些条件关系适合用稀疏结构表示？ | [GraphicalLasso](covariance-graphical-lasso.md) | 对精度矩阵的非对角元素施加 L1 惩罚 |
| 哪个 Graphical Lasso 惩罚值对留出数据评分更好？ | [GraphicalLassoCV](covariance-graphical-lasso-cv.md) | 为同一模型选参并重拟合，需要多次拟合 |

收缩、稳健性和稀疏性解决的是不同问题。收缩通过向目标矩阵靠拢来提高稳定性，
并不能阻止极端异常值影响经验协方差。稳健估计限制污染的影响，
但不能替代高维问题中的正则化。稀疏精度矩阵刻画条件依赖结构，
并不意味着协方差矩阵也稀疏，更不等于检验某条边是否具有科学意义。

<a id="joint-distance"></a>

## 共同示例：异常的是组合，而不只是数值大小

下面两个新观测到原点的欧氏距离相同，但第二个点偏离了数据中的正相关方向。
把变量之间的关系纳入距离计算后，它应该显得更不典型。

<!-- example: covariance-geometry -->
```python
import numpy as np
from statgpu.covariance import EmpiricalCovariance

rng = np.random.default_rng(42)
z = rng.normal(size=(400, 2))
X = np.column_stack((z[:, 0], 0.8 * z[:, 0] + 0.6 * z[:, 1]))
model = EmpiricalCovariance(device="cpu").fit(X)
```

先查看协方差矩阵、转换后的相关矩阵和精度矩阵：

```python
variances = np.diag(model.covariance_)
correlation = model.covariance_ / np.sqrt(np.outer(variances, variances))
print(np.round(model.covariance_, 3))
print(np.round(correlation, 3))
print(np.round(model.precision_, 3))
```

```text
[[1.007 0.898]
 [0.898 1.126]]
[[1.    0.843]
 [0.843 1.   ]]
[[ 3.434 -2.737]
 [-2.737  3.069]]
```

再计算两个新观测的距离：

```python
queries = np.array([[2.0, 2.0], [2.0, -2.0]])
distance_squared = model.mahalanobis(queries)
print(np.round(distance_squared, 3))
print(round(model.score(X), 3))
```

```text
[ 4.227 47.515]
-2.28
```
<!-- example-end: covariance-geometry -->

平方 Mahalanobis 距离为

$$
d^2(x)=(x-\widehat\mu)^\top\widehat\Theta(x-\widehat\mu).
$$

`(2, -2)` 与观测到的正相关关系明显更不一致。`mahalanobis` 和 `predict`
返回的都是**平方距离**，不是普通距离、标签、概率或 p 值。选择阈值还需要额外的假设和校准；
拟合协方差本身不会自动完成异常值检验。

`score` 使用拟合得到的中心、协方差和精度矩阵，计算每行观测的平均高斯对数似然。
这里的 `-2.28` 是训练评分，不是准确率。比较模型时，应使用相同单位下的同一组留出数据，
评分越大越好。改变特征尺度会改变密度的单位，未经调整的评分不能直接比较。

<a id="inference-boundaries"></a>

<a id="协方差精度矩阵与推断语义"></a>

## 拟合能说明什么，不能说明什么

这些估计器描述观测变量之间的变异。它们的协方差矩阵不自动等于回归系数估计的协方差矩阵。
如需回归标准误，应使用相应下游模型明确支持的推断方法；
[回归推断](../guides/inference-api.md) 和[面板协方差](../panel/covariance.md)
讨论的是不同的估计目标。

这里的拟合不会自动为协方差元素、稳健支持集成员或选中的图边给出置信区间、p 值或显著性检验。
交叉验证选择预测评分较好的惩罚强度，而不是显著性水平。若自行做重抽样分析，
需要明确抽样单位，以及是否在每次重抽样中重新进行中心化、标准化、调参和模型选择。
只检查拟合矩阵并不能反映这些步骤带来的不确定性。

<a id="可选的-gpu-执行"></a>

<a id="cpu-and-gpu-examples"></a>

## 输入与设备的实际选择

可使用数值矩阵、嵌套列表或数值型 pandas 数据，但不会按 DataFrame 列名自动对齐特征。
查询时保持训练列顺序，并事先处理缺失值和无穷值。一维拟合输入按单个特征处理；
查询形状及全部限制见 [API 参考](../reference/covariance-api.md)。

下面的可选代码复用上面的 `X`，分别要求对应的 CUDA 运行环境。
它们是 CPU 拟合的替代选择，不是运行前面示例的前提。

### CuPy

```python
import cupy as cp
from statgpu.covariance import LedoitWolf

model_cupy = LedoitWolf(device="cuda").fit(cp.asarray(X))
print(model_cupy.covariance_.shape)
```

### Torch CUDA

```python
import torch
from statgpu.covariance import LedoitWolf

X_torch = torch.as_tensor(X, device="cuda", dtype=torch.float64)
model_torch = LedoitWolf(device="torch").fit(X_torch)
print(model_torch.covariance_.shape)
```

显式请求 GPU 时，若相应运行环境不可用则报错。
自动选择、原生数组的设备保留规则见[设备与数据类型约定](../reference/covariance-api.md#device-and-dtype)。
即使全局设置后来变化，已拟合模型的查询仍使用拟合时的后端和设备。
数组输出、Torch 输入转换（包括支持的 FP8）以及 CPU 控制计算也在那里说明，
不应假定整个过程都驻留在 GPU 上。

## 与其他估计结果比较

比较实现前，要统一观测、列顺序、单位、中心化方式和协方差归一化。
经验协方差使用 `1/n`，而许多样本协方差函数默认使用 `1/(n-1)`。
还应按方法对齐收缩目标与强度，或 Graphical Lasso 的惩罚与收敛控制。
MCD 实现的子集搜索和校正细节可能不同，名称相同并不保证数值完全相同。
各教程列出了相应统计来源。

<a id="旧章节链接"></a>

## 接下来怎么读

无需按顺序读完七篇教程；根据当前问题选择一条路线即可。

<a id="经验协方差"></a>
<a id="收缩估计"></a>
<a id="covariance-basic"></a>
<a id="在-cpu-上估计协方差矩阵"></a>

- **从基线理解稳定性**：先运行[经验协方差](covariance-empirical.md)，再用
  [固定收缩](covariance-shrunk.md) 观察特征值的变化；需要自动选择收缩强度时，
  比较 [Ledoit–Wolf](covariance-ledoit-wolf.md) 和 [OAS](covariance-oas.md)。

<a id="最小协方差行列式"></a>
<a id="covariance-robust"></a>
<a id="稳健支持子集"></a>

- **判断污染的影响**：用 [MCD 教程](covariance-min-cov-det.md) 比较普通估计与稳健估计，
  并理解为什么未进入 `support_` 的观测不一定有误。

<a id="graphical-lasso"></a>
<a id="covariance-sparse"></a>
<a id="通过交叉验证选择稀疏精度矩阵"></a>

- **研究稀疏条件关系**：先读 [Graphical Lasso](covariance-graphical-lasso.md) 的目标函数
  与结果解释，再用[交叉验证教程](covariance-graphical-lasso-cv.md) 学习选参和留出评估。

<a id="cpu-与-gpu-示例"></a>
<a id="strict-与-approximate"></a>
<a id="估计算法"></a>
<a id="公共参数"></a>
<a id="后端与执行边界"></a>
<a id="外部验证"></a>
<a id="拟合属性与输出"></a>
<a id="api-与执行细节"></a>

已经选好方法时，可直接在[完整 API](../reference/covariance-api.md) 中查找参数、拟合属性
和执行限制。这些估计器没有统一的严格／近似开关，各自使用对应的估计算法。

## FAQ

### 当 $p$ 接近 $n$ 时应该使用哪个估计器？

可以考虑 [LedoitWolf](covariance-ledoit-wolf.md) 或 [OAS](covariance-oas.md)，
同时判断缩放单位矩阵这个目标是否适合变量及其单位。

### `MinCovDet` 是否会删除观测值？

它用支持掩码估计稳健中心和协方差，不会修改原始数据。
应结合实际问题解释支持集和距离。

### 为什么 Graphical Lasso 的精度矩阵稀疏，而协方差矩阵可能稠密？

惩罚针对精度矩阵的非对角元素；矩阵求逆通常不会保留零元素的位置。

### Torch CUDA 张量能否使用 `device="cuda"`？

可以，前提是 CuPy CUDA 可用，因为这个参数显式请求 CuPy 计算。
`device="torch"` 才是请求 Torch CUDA。输入类型不能覆盖显式后端选择，
详见[完整 API](../reference/covariance-api.md)。

## 路径

```python
from statgpu.covariance import (
    EmpiricalCovariance, ShrunkCovariance, LedoitWolf, OAS,
    MinCovDet, GraphicalLasso, GraphicalLassoCV,
)
```

## 参考文献

- Ledoit, O., & Wolf, M. (2004). A well-conditioned estimator for
  large-dimensional covariance matrices.
- Chen, Y., Wiesel, A., Eldar, Y. C., & Hero, A. O. (2010). Shrinkage algorithms
  for MMSE covariance estimation.
- Rousseeuw, P. J., & Van Driessen, K. (1999). A fast algorithm for the minimum
  covariance determinant estimator.
- Friedman, J., Hastie, T., & Tibshirani, R. (2008). Sparse inverse covariance
  estimation with the graphical lasso.

- [贡献者验证参考](../../../dev/reviews/pr168-model-validation-provenance.md#covariance).
