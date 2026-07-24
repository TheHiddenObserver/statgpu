# 核方法

> 语言：中文  
> 最后更新：2026-07-24  
> 切换：[English](../../../en/models/kernel-methods.md)

## 概览

核方法模块提供：

- `KernelRidge`
- `KernelRidgeCV`
- `KernelPCA`
- `Nystroem`
- `pairwise_kernels`
- RBF、多项式、线性、Laplacian、sigmoid、余弦与 chi-squared 核

公开实现会在所选估计器和 kernel 支持范围内提供 NumPy、CuPy 与 Torch 执行路径。

## 核岭回归

给定训练 kernel matrix $K$，核岭回归求解

$$
(K+\alpha I)c = y,
$$

并通过

$$
\hat y_{\mathrm{test}} = K_{\mathrm{test}}c
$$

进行预测。

`KernelRidgeCV` 在交叉验证 fold 上评估 alpha grid，并使用选定 alpha 重新拟合。
具体 batch 与 decomposition 策略依赖后端。

## Kernel PCA 与 Nystroem

`KernelPCA` 对中心化 kernel matrix 做特征分解以构造非线性主成分。
`Nystroem` 抽取 landmark 并构造显式低秩特征，使成本与 landmark 数量相关，
避免直接保存完整 $n\times n$ kernel matrix。

## 内置 Kernel

| Kernel | 定义 |
|---|---|
| RBF | $\exp(-\gamma\lVert x-y\rVert_2^2)$ |
| Polynomial | $(\gamma x^\top y+c_0)^d$ |
| Linear | $x^\top y$ |
| Laplacian | $\exp(-\gamma\lVert x-y\rVert_1)$ |
| Sigmoid | $\tanh(\gamma x^\top y+c_0)$ |
| Cosine | $x^\top y/(\lVert x\rVert\lVert y\rVert)$ |
| Chi-squared | $\exp\{-\gamma\sum_j (x_j-y_j)^2/(x_j+y_j)\}$ |

Chi-squared kernel 要求输入非负。

## 示例

### NumPy

```python
import numpy as np
from statgpu.nonparametric.kernel_methods import KernelRidge

X = np.random.randn(500, 10)
y = X[:, 0] - 0.5 * X[:, 1] + 0.1 * np.random.randn(500)
model = KernelRidge(alpha=1.0, kernel="rbf", device="cpu").fit(X, y)
```

### CuPy

```python
import cupy as cp
from statgpu.nonparametric.kernel_methods import KernelRidgeCV

X = cp.random.randn(500, 10, dtype=cp.float64)
y = X[:, 0] - 0.5 * X[:, 1]
model = KernelRidgeCV(kernel="rbf", cv=5, device="cuda").fit(X, y)
```

### Torch CUDA

```python
import torch
from statgpu.nonparametric.kernel_methods import KernelRidgeCV

X = torch.randn(500, 10, device="cuda", dtype=torch.float64)
y = X[:, 0] - 0.5 * X[:, 1]
model = KernelRidgeCV(kernel="rbf", cv=5, device="torch").fit(X, y)
```

`device="cuda"` 选择 CuPy；`device="torch"` 选择 Torch。

## 推断与验证

核方法目前不提供 coefficient-level 标准误或假设检验。模型质量通过预测指标、
embedding 性质和交叉验证结果评估。

本页不维护全局物理 GPU 完成标记。硬件特定的精度与性能结论应记录在注明具体
后端、环境与 commit 的维护测试和 benchmark artifact 中。
