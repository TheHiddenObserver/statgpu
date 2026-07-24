# Kernel Methods

> Language: English  
> Last updated: 2026-07-24  
> Switch: [Chinese](../../cn/models/kernel-methods.md)

## Overview

The kernel-methods module provides:

- `KernelRidge`
- `KernelRidgeCV`
- `KernelPCA`
- `Nystroem`
- `pairwise_kernels`
- RBF, polynomial, linear, Laplacian, sigmoid, cosine, and chi-squared kernels

The public implementations expose NumPy, CuPy, and Torch execution paths where
supported by the selected estimator and kernel.

## Kernel Ridge Regression

Given a training kernel matrix $K$, kernel ridge regression solves

$$
(K+\alpha I)c = y
$$

and predicts with

$$
\hat y_{\mathrm{test}} = K_{\mathrm{test}}c.
$$

`KernelRidgeCV` evaluates an alpha grid across cross-validation folds and refits the
selected model. Its exact batching and decomposition strategy is backend-dependent.

## Kernel PCA and Nystroem

`KernelPCA` eigendecomposes the centered kernel matrix to construct nonlinear
components. `Nystroem` samples landmark points and forms an explicit low-rank feature
map with cost proportional to the number of landmarks rather than the full
$n\times n$ kernel matrix.

## Built-In Kernels

| Kernel | Definition |
|---|---|
| RBF | $\exp(-\gamma\lVert x-y\rVert_2^2)$ |
| Polynomial | $(\gamma x^\top y+c_0)^d$ |
| Linear | $x^\top y$ |
| Laplacian | $\exp(-\gamma\lVert x-y\rVert_1)$ |
| Sigmoid | $\tanh(\gamma x^\top y+c_0)$ |
| Cosine | $x^\top y/(\lVert x\rVert\lVert y\rVert)$ |
| Chi-squared | $\exp\{-\gamma\sum_j (x_j-y_j)^2/(x_j+y_j)\}$ |

The chi-squared kernel requires non-negative inputs.

## Examples

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

`device="cuda"` selects CuPy; `device="torch"` selects Torch.

## Inference and Validation

Kernel methods do not currently expose coefficient-level standard errors or
hypothesis tests. Model quality is evaluated through prediction metrics, embedding
properties, and cross-validation results.

This page does not maintain a global physical-GPU completion flag. Hardware-specific
accuracy and performance claims belong to the maintained tests and benchmark artifacts
that record the exact backend, environment, and commit.
