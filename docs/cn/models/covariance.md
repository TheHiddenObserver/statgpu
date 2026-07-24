# 协方差估计

> 语言：中文  
> 最后更新：2026-07-24  
> 切换：[English](../../en/models/covariance.md)

## 概览

`statgpu.covariance` 提供：

- `EmpiricalCovariance`
- `LedoitWolf`
- `OAS`
- `ShrunkCovariance`
- `MinCovDet`
- `GraphicalLasso`
- `GraphicalLassoCV`

这些估计器提供 NumPy、CuPy 与 Torch 执行路径。这里的“后端支持”表示公开路径
存在；数值与性能结论仍应限定到实际测试的估计器、后端、硬件和 commit。

## 核心定义

中心化观测的经验协方差为

$$
\hat S = \frac{1}{n}X^\top X.
$$

收缩估计器使用

$$
\hat\Sigma = (1-\alpha)\hat S + \alpha\mu I,
\qquad
\mu = \frac{\operatorname{tr}(\hat S)}{p}.
$$

`LedoitWolf` 与 `OAS` 解析估计收缩强度；`ShrunkCovariance` 使用用户指定的
`shrinkage`。

`GraphicalLasso` 求解

$$
\max_{\Theta\succ 0}
\left\{
\log\det(\Theta)-\operatorname{tr}(S\Theta)
-\alpha\lVert\Theta\rVert_{1,\mathrm{off}}
\right\}.
$$

`MinCovDet` 使用 FAST-MCD concentration step 并进行重加权。

## 公共参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `assume_centered` | `False` | 数据已中心化时跳过均值估计 |
| `device` | `"auto"` | `"cpu"`、`"cuda"`（CuPy）、`"torch"` 或 `"auto"` |
| `n_jobs` | `None` | 未实现并行处保留用于 API 兼容 |

估计器特有参数包括 `shrinkage`、`support_fraction`、`random_state`、
`alpha`、`alphas`、`cv`、`max_iter` 与 `tol`。

## 拟合属性

公共输出包括：

- `covariance_`
- `precision_`
- `location_`
- `n_samples_`
- `n_features_`

收缩估计器提供 `shrinkage_`；稳健与稀疏估计器还会提供 support 或 convergence
相关属性。

## 示例

### NumPy

```python
import numpy as np
from statgpu.covariance import LedoitWolf

X = np.random.randn(500, 10)
model = LedoitWolf(device="cpu").fit(X)
print(model.covariance_.shape)
print(model.score(X))
```

### CuPy

```python
import cupy as cp
from statgpu.covariance import LedoitWolf

X_cupy = cp.random.randn(500, 10, dtype=cp.float64)
model_cupy = LedoitWolf(device="cuda").fit(X_cupy)
```

### Torch CUDA

```python
import torch
from statgpu.covariance import LedoitWolf

X_torch = torch.randn(500, 10, device="cuda", dtype=torch.float64)
model_torch = LedoitWolf(device="torch").fit(X_torch)
```

`device="cuda"` 选择 CuPy；Torch tensor 应使用 `device="torch"`。两个显式 GPU
设备值不可互换。

## 执行边界

中心化、协方差更新、线性代数、FAST-MCD concentration step 与 Graphical Lasso
坐标更新在支持范围内保留在所选数值后端。少量整数索引元数据、收敛标量以及后端
缺失的标量卡方分布计算可能在 CPU 上完成。

空特征维度与 NaN/Inf 输入会在中心化或求逆前验证，避免把非法输入误报为协方差
奇异问题。

## 验证说明

本页不维护全局 GPU “待完成”或“全部完成”状态。物理 GPU 结果与 benchmark
证据应记录在对应维护测试、release 记录和硬件特定 artifact 中。

## 参考文献

- Ledoit, O., & Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices.
- Chen, Y., Wiesel, A., Eldar, Y. C., & Hero, A. O. (2010). Shrinkage algorithms for MMSE covariance estimation.
- Rousseeuw, P. J., & Van Driessen, K. (1999). A fast algorithm for the minimum covariance determinant estimator.
- Friedman, J., Hastie, T., & Tibshirani, R. (2008). Sparse inverse covariance estimation with the graphical lasso.
