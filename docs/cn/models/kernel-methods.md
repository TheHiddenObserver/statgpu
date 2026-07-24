# 核方法

> 语言：中文  
> 最后更新：2026-07-24  
> 切换：[English](../../en/models/kernel-methods.md)

## 概览

核方法模块提供：

- `KernelRidge`
- `KernelRidgeCV`
- `KernelPCA`
- `Nystroem`
- `pairwise_kernels`
- RBF、多项式、线性、Laplacian、sigmoid、cosine 和 chi-squared 核

公共实现会在所选估计器和核支持的范围内提供 NumPy、CuPy 和 Torch 执行路径。

## 路径

```text
statgpu.nonparametric.kernel_methods.KernelRidge
statgpu.nonparametric.kernel_methods.KernelRidgeCV
statgpu.nonparametric.kernel_methods.KernelPCA
statgpu.nonparametric.kernel_methods.Nystroem
statgpu.nonparametric.kernel_methods.pairwise_kernels
```

各个核函数也可以从 `statgpu.nonparametric.kernel_methods` 直接导入。

## 核岭回归

给定训练核矩阵 $K$，核岭回归求解对偶线性系统

$$
(K+\alpha I)c=y.
$$

等价的对偶目标为

$$
\min_c \lVert y-Kc\rVert_2^2+\alpha\lVert c\rVert_2^2.
$$

测试样本预测为

$$
\hat y_{test}=K(X_{test},X_{train})c.
$$

`KernelRidge` 直接求解正则化线性系统。实现收到兼容的响应矩阵时，可支持多输出响应。

## 核岭交叉验证

`KernelRidgeCV` 在 CV folds 上评估一组正则化参数，并在完整数据上使用选出的值
重新拟合。后端特定实现可能复用核矩阵特征分解，或向量化全部 alpha，而不是为每个
候选值独立求解线性系统。

选出的 alpha 存在 `alpha_` 中。CV 诊断记录在 `cv_results_`；具体字段以所选路径
拟合后的对象为准。

## Kernel PCA

对中心化核矩阵 $\widetilde K$，Kernel PCA 执行

$$
\widetilde K = V\Lambda V^\top.
$$

领先特征向量定义非线性主成分。变换新数据时，需要计算测试集到训练集的核矩阵，
应用训练期中心化量，并投影到保留的成分。

## Nystroem 近似

Nystroem 选择 $m$ 个 landmark，并构造显式近似特征映射。若 landmark 核矩阵为

$$
K_{mm}=V\Lambda V^\top,
$$

则变换特征具有形式

$$
Z=K_{nm}V\Lambda^{-1/2}.
$$

当 $m\ll n$ 时，这会用 $n\times m$ 特征矩阵替代完整的 $n\times n$ 核表示。

## 内置核

| 核 | 定义 |
|---|---|
| RBF | $\exp(-\gamma\lVert x-y\rVert_2^2)$ |
| Polynomial | $(\gamma x^\top y+c_0)^d$ |
| Linear | $x^\top y$ |
| Laplacian | $\exp(-\gamma\lVert x-y\rVert_1)$ |
| Sigmoid | $\tanh(\gamma x^\top y+c_0)$ |
| Cosine | $x^\top y/(\lVert x\rVert\lVert y\rVert)$ |
| Chi-squared | $\exp\{-\gamma\sum_j (x_j-y_j)^2/(x_j+y_j)\}$ |

chi-squared 核要求输入特征非负。估计器允许时也可传入 callable kernel；该函数必须
返回请求后端上的数组，并满足预期的两两核矩阵形状。

## 参数

### KernelRidge

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `alpha` | `1.0` | Ridge 正则化强度 |
| `kernel` | `"rbf"` | 内置核名称或 callable |
| `gamma` | `None` | 适用核的系数 |
| `degree` | `3` | 多项式次数 |
| `coef0` | `1` | 多项式或 sigmoid 截距项 |
| `kernel_params` | `None` | callable kernel 的额外参数 |
| `device` | `"auto"` | `"cpu"`、`"cuda"`（CuPy）、`"torch"` 或 `"auto"` |
| `n_jobs` | `None` | 未实现并行处保留的参数 |

### KernelRidgeCV

除核相关参数外：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `alphas` | `None` | 候选正则化强度 |
| `cv` | `5` | CV fold 数 |
| `random_state` | `None` | 适用时的 fold 随机状态 |

### KernelPCA

常用参数包括 `n_components`、`kernel`、`gamma`、`degree`、`coef0`、`alpha`、
`eigen_solver` 和 `device`。

### Nystroem

常用参数包括 `kernel`、`n_components`、`gamma`、`degree`、`coef0`、
`random_state` 和 `device`。

确切别名、callable 合同和参数验证以类 docstring 为准。

## 拟合属性与输出

### KernelRidge

常见拟合状态包括训练样本、对偶系数、解析后的核参数和拟合特征数。
`predict(X)` 在维护的估计器路径中保持后端类型，`score(X, y)` 返回决定系数。

### KernelRidgeCV

除最终 refit 状态外，模型还暴露 `alpha_`、实现支持时的 `best_score_`，以及
`cv_results_`。

### KernelPCA

拟合输出包括保留的特征值/特征向量或归一化对偶成分、训练核中心化量，以及
`fit_transform` 或 `transform` 生成的成分。

### Nystroem

拟合状态包括 landmark 索引或 components，以及归一化矩阵。`transform(X)` 返回
显式近似特征映射。

## CPU 与 GPU 示例

### NumPy

```python
import numpy as np
from statgpu.nonparametric.kernel_methods import (
    KernelRidge,
    KernelRidgeCV,
    KernelPCA,
    Nystroem,
)

rng = np.random.default_rng(42)
X = rng.normal(size=(500, 10))
y = X[:, 0] - 0.5 * X[:, 1] + rng.normal(scale=0.1, size=500)

kr = KernelRidge(alpha=1.0, kernel="rbf", device="cpu").fit(X, y)
print(kr.score(X, y))

kr_cv = KernelRidgeCV(kernel="rbf", cv=5, device="cpu").fit(X, y)
print(kr_cv.alpha_)

kpca = KernelPCA(n_components=3, kernel="rbf", device="cpu")
X_kpca = kpca.fit_transform(X)

nystroem = Nystroem(kernel="rbf", n_components=50, random_state=42)
X_features = nystroem.fit_transform(X)
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

`device="cuda"` 选择 CuPy，`device="torch"` 选择 Torch。

## 后端与执行边界

两两核矩阵构造、正则化求解、特征分解、投影和变换后的特征数组在实现支持时保留
在所选后端。小型随机索引元数据、CV fold 索引、参数 bookkeeping 和标量 score
可以位于 CPU。显式设备请求不会静默选择其他后端。

维护中的验证路径会在 `KernelPCA` 和 `Nystroem` 的拟合与变换阶段拒绝 NaN/Inf。
核特有定义域检查（例如 chi-squared 核要求非负输入）会显式失败。

## 推断语义

核方法当前不提供系数级标准误、假设检验或置信区间。模型质量通过预测分数、
交叉验证损失、嵌入性质、重构或近似诊断以及应用相关验证进行评估。

该模块没有 strict/approximate inference 模式。Nystroem 是显式低秩核近似，
不是精确核估计器的静默 fallback。

## 复杂度与性能说明

- 精确核方法构造 $n\times n$ 训练核矩阵，内存复杂度为二次量级。
- 直接核岭求解和稠密特征分解对训练样本数具有三次最坏计算复杂度。
- `KernelRidgeCV` 可以在 alpha 间复用分解，但 CV 仍会乘以 fold 数。
- `Nystroem` 将核存储降低到 $O(nm)$，另加 landmark 线性代数。
- GPU 收益依赖样本量、dtype、核、同步和可用显存；小问题可能 CPU 更快。

## 限制与失败行为

- 核矩阵可能病态；必要时增大 `alpha` 或调整核尺度。
- RBF 等核对 `gamma` 敏感。
- Chi-squared 核要求非负输入。
- 大规模稠密精确核方法可能耗尽设备内存。
- 用户自定义核负责满足所选估计器要求的后端、dtype、形状和对称性合同。
- 大 fold 和 alpha 网格会使 `KernelRidgeCV` 成本很高。

## 外部验证

维护测试覆盖 NumPy/Torch 一致性、非有限输入验证、秩亏核保护、CV alpha 选择与
refit、核定义域错误以及输出后端保持。准确性和性能结论仅适用于相应测试或
benchmark artifact 记录的具体估计器、后端、硬件和 commit。

## FAQ

### 应使用 KernelRidge，还是 Nystroem 加线性模型？

当完整训练核矩阵能放入内存且精确核表示重要时使用 Kernel Ridge；当需要可控低秩
特征近似以扩展规模或复用特征时使用 Nystroem。

### 为什么 GPU 核方法可能比 CPU 慢？

核构造和线性代数需要足够大，才能摊薄设备 launch、同步和内存传输成本。

### `device="auto"` 会覆盖显式请求吗？

不会。`"auto"` 本身表示自动选择；显式 `"cuda"` 或 `"torch"` 在对应后端不可用时
会报错。

### 不同拟合的 KernelPCA 成分是否可直接比较？

特征向量符号，以及重根或近重根特征空间内的基不唯一。需要时应比较表示的子空间
或下游量，而不是逐列直接比较。

## 参考文献

- Schölkopf, B., Smola, A., & Müller, K.-R. (1998). Nonlinear component analysis
  as a kernel eigenvalue problem.
- Williams, C. K. I., & Seeger, M. (2001). Using the Nystroem method to speed up
  kernel machines.
- Shawe-Taylor, J., & Cristianini, N. (2004). *Kernel Methods for Pattern Analysis*.
