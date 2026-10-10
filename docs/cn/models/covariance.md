# 协方差估计

> 语言：中文  
> 最后更新：2026-10-10\
> 切换：[English](../../en/models/covariance.md)

## 概览

协方差描述变量如何共同变化；它的逆矩阵称为精度矩阵，在高斯模型中刻画条件关系。
样本充分时可用经验协方差作为基准，需要更稳定的估计时可用收缩方法，存在多元离群点
时可考虑稳健协方差，需要稀疏精度矩阵时可用 Graphical Lasso。第一个示例使用
`LedoitWolf`，由数据确定收缩强度。

`statgpu.covariance` 提供七个协方差或精度矩阵估计器：

- `EmpiricalCovariance`
- `LedoitWolf`
- `OAS`
- `ShrunkCovariance`
- `MinCovDet`
- `GraphicalLasso`
- `GraphicalLassoCV`

公共估计器提供 NumPy、CuPy 和 Torch 执行路径。

## 目标函数

### 经验协方差

对中心化观测矩阵 $X\in\mathbb R^{n\times p}$，

$$
\hat S = \frac{1}{n}X^\top X.
$$

除非设置 `assume_centered=True`，拟合前会估计并减去列均值。

### 收缩估计

`LedoitWolf`、`OAS` 和 `ShrunkCovariance` 使用

$$
\hat\Sigma=(1-\alpha)\hat S+\alpha\mu I,
\qquad
\mu=\frac{\operatorname{tr}(\hat S)}{p}.
$$

`LedoitWolf` 与 `OAS` 解析估计 $\alpha$；`ShrunkCovariance` 使用用户给定的
`shrinkage`。

### 最小协方差行列式

`MinCovDet` 搜索协方差行列式较小的集中子集，执行 FAST-MCD 集中迭代，并根据稳健 Mahalanobis 距离重加权。该方法用于存在多元离群点时的稳健
协方差估计。

### Graphical Lasso

`GraphicalLasso` 通过

$$
\max_{\Theta\succ0}
\left\{
\log\det(\Theta)-\operatorname{tr}(S\Theta)
-\alpha\lVert\Theta\rVert_{1,\mathrm{off}}
\right\}
$$

估计稀疏精度矩阵 $\Theta$。精度矩阵对角线不接受 L1 惩罚。
`GraphicalLassoCV` 通过交叉验证选择 alpha，并在全数据上重新拟合最终模型。

<a id="covariance-basic"></a>

## 在 CPU 上估计协方差矩阵

按顺序运行以下代码。这里没有响应向量，`X` 的每行是一条观测，每列是一个测量变量。
先导入估计器。

<!-- example: covariance-basic -->
```python
import numpy as np
from statgpu.covariance import LedoitWolf
```

生成 10 个独立标准正态变量的 500 条观测，`X` 的形状为 `(500, 10)`。
该模拟的总体协方差为单位矩阵。

```python
rng = np.random.default_rng(42)
X = rng.normal(size=(500, 10))
```

拟合收缩估计。默认会估计并减去各列均值，不需要手动中心化 `X`。

```python
lw = LedoitWolf(device="cpu").fit(X)
```

查看矩阵大小、收缩强度以及这些观测上的高斯模型评分。

```python
print(lw.covariance_.shape, lw.shrinkage_)
print(lw.score(X))
```
<!-- example-end: covariance-basic -->

协方差矩阵的形状为 `(10, 10)`：对角元素估计方差，非对角元素估计协方差。
本例 `shrinkage_` 为 `1.0`，即完全采用缩放后的单位矩阵目标；这与简单模拟相符，
并不意味着所有数据都会得到该值。评分约为 `-14.175`，是高斯对数似然评分，
不是准确率。比较泛化表现时，应在相同的留出观测上评价候选模型。

拟合及 `score`、`predict`、`mahalanobis` 均可接收数值型 pandas `DataFrame`
和嵌套列表等普通类数组输入。拟合时，一维数值输入（包括 pandas `Series`）
按单个特征处理。列名不会自动对齐特征，评估时请保持与训练时相同的列顺序。
拟合或评估前应先处理缺失值和无穷值。

## 估计算法

- `EmpiricalCovariance` 直接计算样本协方差，并通过求逆获得精度矩阵；只有当精确
  求逆失败或产生非有限结果时才使用数值稳定化。
- `LedoitWolf` 与 `OAS` 计算闭式收缩强度，再求收缩协方差的逆。
- `ShrunkCovariance` 使用固定收缩强度执行相同的直接计算路径。
- `MinCovDet` 使用多个初始子集、集中迭代（concentration steps）、一致性校正和重加权。
- `GraphicalLasso` 使用分块坐标更新与软阈值内层回归。
- `GraphicalLassoCV` 在各交叉验证折上拟合候选 alpha，再进行最终重拟合。

## 公共参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `assume_centered` | `False` | 将输入视为已中心化 |
| `device` | `"auto"` | `"cpu"`、`"cuda"`（CuPy）、`"torch"` 或 `"auto"` |
| `n_jobs` | `None` | 未实现处保留为 API 兼容参数 |

估计器特有参数包括：

| 估计器 | 参数 |
|---|---|
| `ShrunkCovariance` | `shrinkage` |
| `MinCovDet` | `support_fraction`、`random_state` |
| `GraphicalLasso` | `alpha`、`max_iter`、`tol` |
| `GraphicalLassoCV` | `alphas`、`cv`、`max_iter`、`tol`、`random_state` |

具体接受的类型和范围以类 docstring 为准。

## 拟合属性与输出

公共拟合属性包括：

| 属性 | 说明 |
|---|---|
| `covariance_` | 估计的协方差矩阵 |
| `precision_` | 估计的逆协方差或稀疏精度矩阵 |
| `location_` | 估计均值；假设已中心化时为零 |
| `n_samples_` | 拟合样本数 |
| `n_features_` | 特征数 |

额外属性包括：

- 收缩估计器的 `shrinkage_`；
- `MinCovDet` 的 `support_`、`raw_location_`、`raw_covariance_` 和稳健距离；
- 迭代稀疏估计器的 `n_iter_`；
- `GraphicalLassoCV` 的 `alpha_`、交叉验证分数和最终重拟合状态。

若类公开这些方法，`score(X)` 评估拟合的高斯协方差模型，
`mahalanobis(X)` 返回拟合位置和精度矩阵下的平方 Mahalanobis 距离。

## 稳健与稀疏估计的替代方案

这两个示例解决的问题与收缩估计不同，都复用[前面的 CPU 示例](#covariance-basic)中的 `X`。

### 稳健支持子集

离群点使普通协方差不可靠时，可使用 `MinCovDet`。本例没有人为加入离群点，
只演示 API 用法，并非已知哪些观测应该剔除。

<!-- example-requires: covariance-basic -->
<!-- example: covariance-robust -->
```python
from statgpu.covariance import MinCovDet

mcd = MinCovDet(random_state=42, device="cpu").fit(X)
print(mcd.support_.sum())
```
<!-- example-end: covariance-robust -->

`support_` 是与训练行对应的布尔掩码，求和得到最终稳健支持子集的行数。
未进入子集不能证明该观测有误。

### 通过交叉验证选择稀疏精度矩阵

使用 `GraphicalLassoCV` 选择 L1 惩罚强度并重新拟合稀疏精度模型。
在高斯模型中，精度矩阵的零非对角元素表示条件独立，不等于边际协方差为零，
也不代表因果关系。

<!-- example-requires: covariance-basic -->
<!-- example: covariance-sparse -->
```python
from statgpu.covariance import GraphicalLassoCV

glcv = GraphicalLassoCV(alphas=4, cv=5, device="cpu").fit(X)
print(glcv.alpha_)
```
<!-- example-end: covariance-sparse -->

`alpha_` 是选出的惩罚强度，不是显著性水平。`covariance_` 和 `precision_`
来自在 `X` 全部观测上进行的最终拟合。

<a id="cpu-与-gpu-示例"></a>

## 可选的 GPU 执行

复用[前面的 CPU 示例](#covariance-basic)中的 `X` 与 `LedoitWolf`。
根据可用的 CUDA 运行环境选择以下一段。

### CuPy

```python
import cupy as cp

X_cupy = cp.asarray(X)
model_cupy = LedoitWolf(device="cuda").fit(X_cupy)
print(model_cupy.covariance_.shape)
```

### Torch CUDA

```python
import torch

X_torch = torch.as_tensor(X, device="cuda", dtype=torch.float64)
model_torch = LedoitWolf(device="torch").fit(X_torch)
print(model_torch.covariance_.shape)
```

设备设置决定计算后端，不由输入数组类型覆盖：

- `device="cpu"` 将 NumPy、CuPy 或 Torch 输入转换为 NumPy CPU 数组。
- `device="cuda"` 将输入转换为 CuPy CUDA 数组。即使输入是 Torch 张量，也要求
  可用的 CuPy CUDA 运行时。
- `device="torch"` 将输入转换为 Torch CUDA 张量。Torch CPU 输入会移至 CUDA，
  不能用 CPU 计算代替显式请求的 Torch GPU 计算。
- `device="auto"` 继承 `statgpu.set_device(...)` 的全局设置。只有两者都为自动
  选择时，原生 CuPy/Torch 输入才保留原后端和设备，包括 Torch CPU 输入。其他输入
  按可用性依次选择 CuPy CUDA、Torch CUDA、NumPy。显式全局设置优先于原生输入
  选择；显式估计器设置又优先于全局设置。

选择 NumPy 计算时，受支持的实数 Torch 输入会脱离自动求导图，统一转换为
`float64` 并传至 CPU，原始输入不会被修改。这包括 `bfloat16` 数值以及稠密、
非量化张量的实数视图。这些估计器不保证兼容自动求导。

七个估计器采用相同策略。`GraphicalLassoCV` 的各折拟合、评分和最终重拟合沿用
开始拟合时选定的后端与设备。拟合数组保留在该后端。即使之后更改全局设置，
`score`、`mahalanobis` 和 `predict` 仍将新输入转换到已拟合数组所在的后端和设备。
`score` 返回 Python 浮点数；`mahalanobis` 和 `predict` 返回 NumPy 数组。

估计器的设备参数不接受 `"cuda:1"` 等带序号的值。选择与原生 GPU 输入相同的后端
时，会保留输入的 CUDA 设备序号；从 CPU 移至 GPU 的输入使用所选库当前的 CUDA
设备。评分使用已拟合数组所在的设备序号。全局配置详见[设备指南](../guides/device-and-memory.md)。

## 协方差、精度矩阵与推断语义

这些类估计协方差或精度矩阵，通常不提供回归系数标准误或回归 p 值。协方差估计
本身的不确定性应使用与估计器和应用相匹配的方法评估，例如重采样，或在推断方法
与适用条件明确的下游模型中处理。

奇异或近奇异经验协方差可能需要数值稳定化才能计算精度矩阵。稳定化是数值保护，
并不意味着秩亏协方差在所有方向都变成完全可识别。

## 后端与执行边界

中心化、协方差更新、矩阵乘法、线性代数、FAST-MCD 集中迭代 和
Graphical Lasso 坐标更新在实现支持时保留在所选后端。部分流程控制和标量分布
计算使用 CPU，因此不能假定整个流程常驻 GPU。

空特征维度以及 NaN/Inf 的输入验证会在中心化或求逆之前执行，避免把非法数据
误报为奇异协方差问题。

## strict 与 approximate

协方差估计器没有共享的全局 strict/approximate 开关。每个估计器使用其文档化
算法。求逆稳定化、稳健子集搜索和 CV 选择是对应算法的显式组成部分，不是静默
后端回退。

## 限制与失败行为

- 当 $p$ 相对 $n$ 较大时，`EmpiricalCovariance` 可能条件数很差，收缩估计通常
  更稳定。
- `LedoitWolf` 和 `OAS` 收缩到尺度单位阵，不适合要求其他结构目标的场景。
- `MinCovDet` 比直接协方差估计更昂贵，并要求足够样本构成有意义的支持子集。
- `GraphicalLasso` 假定精度矩阵具有稀疏表示，不合适的 alpha 或容差可能导致
  不收敛。
- `GraphicalLassoCV` 的成本会乘以交叉验证折数和候选 alpha 数。
- 显式 GPU 请求在对应运行时不可用时会报错，不会静默在 CPU 上执行。

<a id="外部验证"></a>

## 与其他估计结果比较

比较协方差估计时，使用相同观测值、特征顺序、中心化约定和估计器。对齐协方差的
归一化方式（上述经验估计使用 `1/n`），以及固定的收缩强度或 Graphical Lasso
惩罚强度。迭代拟合还应对齐收敛容差和迭代次数上限。比较高斯模型分数时，使用
相同的留出观测，不要把训练分数当成泛化表现。

贡献者可参阅[验证参考](../../../dev/reviews/pr168-model-validation-provenance.md#covariance)。

## FAQ

### 当 $p$ 接近 $n$ 时应该使用哪个估计器？

`LedoitWolf` 或 `OAS` 等收缩估计通常比无正则经验协方差更稳定。

### `MinCovDet` 是否会删除观测值？

它识别稳健支持并返回稳健估计。应检查 `support_` 和稳健距离，而不是假设每个观测
对最终估计的权重相同。

### 为什么 Graphical Lasso 的精度矩阵稀疏，而协方差矩阵可能稠密？

L1 惩罚施加在精度矩阵非对角元素上；稀疏精度矩阵的逆不必稀疏。

### Torch CUDA 张量能否使用 `device="cuda"`？

可以，前提是 CuPy CUDA 可用：张量会转换为 CuPy 数组，并由 CuPy 完成计算。
若要使用 Torch CUDA 计算，请选择 `device="torch"`。输入类型不会覆盖显式后端请求。

## 路径

```python
from statgpu.covariance import (
    EmpiricalCovariance,
    LedoitWolf,
    OAS,
    ShrunkCovariance,
    MinCovDet,
    GraphicalLasso,
    GraphicalLassoCV,
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
