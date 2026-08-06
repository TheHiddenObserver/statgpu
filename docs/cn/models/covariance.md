# 协方差估计

> 语言：中文  
> 最后更新：2026-07-24  
> 切换：[English](../../en/models/covariance.md)

## 概览

`statgpu.covariance` 提供七个协方差或精度矩阵估计器：

- `EmpiricalCovariance`
- `LedoitWolf`
- `OAS`
- `ShrunkCovariance`
- `MinCovDet`
- `GraphicalLasso`
- `GraphicalLassoCV`

公共估计器提供 NumPy、CuPy 和 Torch 执行路径。这里的后端支持表示公共路径存在；
数值与性能结论仅适用于相应测试记录中的具体估计器、后端、硬件和 commit。

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

`MinCovDet` 搜索协方差行列式较小的集中子集，执行 FAST-MCD concentration
steps，并根据稳健 Mahalanobis 距离重加权。该方法用于存在多元离群点时的稳健
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

## 估计算法

- `EmpiricalCovariance` 直接计算样本协方差，并通过求逆获得精度矩阵；只有当精确
  求逆失败或产生非有限结果时才使用数值稳定化。
- `LedoitWolf` 与 `OAS` 计算闭式收缩强度，再求收缩协方差的逆。
- `ShrunkCovariance` 使用固定收缩强度执行相同的直接计算路径。
- `MinCovDet` 使用多个初始子集、concentration steps、一致性校正和重加权。
- `GraphicalLasso` 使用分块坐标更新与软阈值内层回归。
- `GraphicalLassoCV` 在 fold 和候选 alpha 上拟合，再进行最终 refit。

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
| `GraphicalLassoCV` | `alphas`、`cv`、`max_iter`、`tol` |

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
- `GraphicalLassoCV` 的 `alpha_`、CV 分数和最终 refit 状态。

若类公开这些方法，`score(X)` 评估拟合的高斯协方差模型，
`mahalanobis(X)` 返回拟合位置和精度矩阵下的平方 Mahalanobis 距离。

## CPU 与 GPU 示例

### NumPy

```python
import numpy as np
from statgpu.covariance import LedoitWolf, MinCovDet, GraphicalLassoCV

rng = np.random.default_rng(42)
X = rng.normal(size=(500, 10))

lw = LedoitWolf(device="cpu").fit(X)
print(lw.covariance_.shape, lw.shrinkage_)
print(lw.score(X))

mcd = MinCovDet(random_state=42, device="cpu").fit(X)
print(mcd.support_.sum())

glcv = GraphicalLassoCV(alphas=4, cv=5, device="cpu").fit(X)
print(glcv.alpha_)
```

### CuPy

```python
import cupy as cp
from statgpu.covariance import LedoitWolf

X_cupy = cp.random.randn(500, 10, dtype=cp.float64)
model_cupy = LedoitWolf(device="cuda").fit(X_cupy)
print(model_cupy.covariance_.shape)
```

### Torch CUDA

```python
import torch
from statgpu.covariance import LedoitWolf

X_torch = torch.randn(500, 10, device="cuda", dtype=torch.float64)
model_torch = LedoitWolf(device="torch").fit(X_torch)
print(model_torch.covariance_.shape)
```

`device="cuda"` 选择 CuPy。Torch 张量应使用 `device="torch"`；两个显式 GPU
设备值不能互换。

## 协方差、精度矩阵与推断语义

这些类估计协方差或精度矩阵，通常不提供回归系数标准误或回归 p 值。协方差估计
本身的不确定性应使用与估计器和应用相匹配的方法评估，例如重采样，或在具有明确
推断合同的下游模型中处理。

奇异或近奇异经验协方差可能需要数值稳定化才能计算精度矩阵。稳定化是数值保护，
并不意味着秩亏协方差在所有方向都变成完全可识别。

## 后端与执行边界

中心化、协方差更新、矩阵乘法、线性代数、FAST-MCD concentration steps 和
Graphical Lasso 坐标更新在实现支持时保留在所选后端。小型整数索引元数据、随机
子集 bookkeeping、收敛标量和卡方分布标量计算可能跨到 CPU。

空特征维度以及 NaN/Inf 的输入验证会在中心化或求逆之前执行，避免把非法数据
误报为奇异协方差问题。

## strict 与 approximate

协方差估计器没有共享的全局 strict/approximate 开关。每个估计器使用其文档化
算法。求逆稳定化、稳健子集搜索和 CV 选择是对应算法的显式组成部分，不是静默
后端 fallback。

## 限制与失败行为

- 当 $p$ 相对 $n$ 较大时，`EmpiricalCovariance` 可能条件数很差，收缩估计通常
  更稳定。
- `LedoitWolf` 和 `OAS` 收缩到尺度单位阵，不适合要求其他结构目标的场景。
- `MinCovDet` 比直接协方差估计更昂贵，并要求足够样本构成有意义的支持子集。
- `GraphicalLasso` 假定精度矩阵具有稀疏表示，不合适的 alpha 或容差可能导致
  不收敛。
- `GraphicalLassoCV` 的成本会乘以 fold 数和候选 alpha 数。
- 显式 GPU 请求在对应运行时不可用时会报错，不会静默在 CPU 上执行。

## 外部验证

维护测试覆盖非有限输入验证、后端保持的拟合数组、与科学 Python 协方差估计器的
参考比较、稳健支持语义、稀疏精度收敛和 CV refit 行为。硬件相关的准确性和性能
证据应记录在相应维护测试或 benchmark artifact 中。

## FAQ

### 当 $p$ 接近 $n$ 时应该使用哪个估计器？

`LedoitWolf` 或 `OAS` 等收缩估计通常比无正则经验协方差更稳定。

### `MinCovDet` 是否会删除观测值？

它识别稳健支持并返回稳健估计。应检查 `support_` 和稳健距离，而不是假设每个观测
对最终估计的权重相同。

### 为什么 Graphical Lasso 的精度矩阵稀疏，而协方差矩阵可能稠密？

L1 惩罚施加在精度矩阵非对角元素上；稀疏精度矩阵的逆不必稀疏。

### Torch CUDA 张量能否使用 `device="cuda"`？

不能。`device="cuda"` 表示 CuPy 后端；Torch 执行应使用 `device="torch"`。

## 参考文献

- Ledoit, O., & Wolf, M. (2004). A well-conditioned estimator for
  large-dimensional covariance matrices.
- Chen, Y., Wiesel, A., Eldar, Y. C., & Hero, A. O. (2010). Shrinkage algorithms
  for MMSE covariance estimation.
- Rousseeuw, P. J., & Van Driessen, K. (1999). A fast algorithm for the minimum
  covariance determinant estimator.
- Friedman, J., Hastie, T., & Tibshirani, R. (2008). Sparse inverse covariance
  estimation with the graphical lasso.
