# 核主成分分析

> 语言：中文  
> 最后更新：2026-10-13  
> 切换：[English](../../../en/models/nonparametric/kernel-pca.md)

## 不使用响应，提取非线性坐标

`KernelPCA` 根据观测之间的核相似度，用少量坐标表示数据。普通 PCA 寻找输入特征
的线性组合；核 PCA 寻找核所定义特征空间中的变异方向。它适合探索非线性结构，
或为后续模型构造表示，不需要响应变量。

输出是嵌入坐标，不是响应预测，也不是原始特征的斜率。监督回归使用
[KernelRidge](kernel-ridge.md)；若希望用较少代表节点构造近似核特征，可选择
[Nystroem](kernel-nystroem.md)。如果线性坐标已经足够，且不值得构造稠密训练核，
应先考虑普通 PCA。

<a id="theory"></a>

## 中心化、特征分解与投影

学习坐标系分为两步：先将训练核中心化，再找出主要的变异方向。这两步都通过观测
之间的相似度完成，无需显式构造核特征向量。

设训练核为 $K_{ij}=k(x_i,x_j)$，并令
$H=I-\mathbf 1\mathbf 1^\top/n$。特征空间中心化后有

$$
\widetilde K=HKH
=K-\mathbf 1\bar k_{\mathrm{col}}^\top
  -\bar k_{\mathrm{row}}\mathbf 1^\top
  +\bar k\mathbf 1\mathbf 1^\top,
$$

其中横线分别表示训练核的列均值、行均值及总体均值。这是核中心化，并不等于简单
减去各原始特征的均值。对中心化核进行特征分解：

$$
\widetilde K=V\Lambda V^\top.
$$

只保留大于 `1e-12` 的特征值及其单位范数特征向量，最多 `n_components` 个。
投影系数与训练坐标分别为

$$
A=V\Lambda^{-1/2},\qquad
Z_{\mathrm{train}}=\widetilde K A=V\Lambda^{1/2}.
$$

`alphas_` 保存 $A$，其列是缩放后的特征向量，并非单位范数向量。`lambdas_` 保存
中心化核保留的特征值，不是原始特征的解释方差比例。不定核的负方向会被丢弃；
不能将其直接视作半正定 RKHS 中的普通 PCA。

对于新观测，先计算 $K_*=K(X_{\mathrm{new}},X_{\mathrm{train}})$。每条查询使用
**训练核**的列均值和总体均值，以及自身到训练行的平均相似度进行中心化：

$$
\widetilde K_*=
K_*-\mathbf 1\bar k_{\mathrm{col}}^\top
-\bar k_{*,\mathrm{row}}\mathbf 1^\top
+\bar k\mathbf 1\mathbf 1^\top,\qquad
Z_{\mathrm{new}}=\widetilde K_*A.
$$

因此，新数据必须复用已拟合变换器，而不能再拟合第二个。分别拟合会学习不同的
中心化统计量和坐标系。

<a id="example"></a>

## 完整的 CPU 嵌入示例

本示例可独立运行，从 180 条训练观测学习坐标系，再原样应用到具有相同两列输入
特征的 60 条留出观测。拟合不使用 `y`。

<!-- example: kernel-pca-cpu -->
```python
import numpy as np
from statgpu.nonparametric.kernel_methods import KernelPCA
```

### 准备训练行与留出行

这里使用与回归教程相同的模拟数据。生成响应是为了保持数据一致，学习嵌入只使用
输入特征。
```python
rng = np.random.default_rng(42)
X = rng.uniform(-2, 2, size=(240, 2))
# 与回归教程生成相同的数据，但不使用响应拟合。
y = np.sin(1.7 * X[:, 0]) + 0.4 * X[:, 1] ** 2 + rng.normal(0, 0.1, 240)
indices = rng.permutation(len(X))
train, test = indices[:180], indices[180:]
X_train, X_test = X[train], X[test]
```

### 学习坐标系

已拟合对象保留训练时的中心化信息和投影，因此可以将测试行放入同一坐标系。
本例使用默认的 RBF 核 $k(x,x')=\exp(-\gamma\lVert x-x'\rVert^2)$。
`gamma=0.7` 控制距离尺度：`gamma` 越大，相似度随距离衰减越快，越强调局部关系。

```python
kpca = KernelPCA(n_components=3, gamma=0.7, device="cpu")
train_coordinates = kpca.fit_transform(X_train)
test_coordinates = kpca.transform(X_test)
```

### 查看坐标与保留方向

绘图或开展下游分析前，先检查输出。
```python
assert np.isfinite(kpca.lambdas_).all()
assert np.isfinite(kpca.alphas_).all()
assert np.isfinite(train_coordinates).all()
assert np.isfinite(test_coordinates).all()
print("Training / test coordinate shapes:",
      train_coordinates.shape, test_coordinates.shape)
print("Retained centered-kernel eigenvalues:", kpca.lambdas_)
print("First two test coordinates:\n", test_coordinates[:2])
```
<!-- example-end: kernel-pca-cpu -->

输出形状为 `(180, 3)` 和 `(60, 3)`。两个原始特征可以产生三个非线性坐标，因为核
特征空间的维度不必等于输入维度。第一列对应最大的保留中心化核特征值。可绘制前
两列观察嵌入，或把全部保留列用于后续分析；这些操作都不会把坐标变成响应预测。

若用交叉验证调整下游模型或预处理，应在每个训练折内拟合完整的“预处理、KernelPCA、
下游模型”流程。先在所有行上学习嵌入，再评价某个验证折，会泄漏该验证折输入分布
的信息。

## 选择维度与相似度

先考虑希望坐标保留哪种相似关系。对于 RBF 核，`gamma` 控制距离尺度：较大值强调
局部相似度，较小值使观测之间更相似。应合理选择特征单位和预处理，并针对用途
验证嵌入。

再决定保留多少个方向。`n_components` 是**上限**，不保证返回这么多列。仅保留
大于 `1e-12` 的特征值；低秩中心化核可能返回更少列，无正方向时会报错。

另外两个参数影响分解过程。`alpha` 是特征分解前加入的有限、非负对角偏移量，
分解后会从特征值中扣除。它**不是**监督预测器的岭惩罚，也不用于逆变换正则化。
`eigen_solver="auto"` 和 `"dense"` 目前都使用稠密对称特征分解。

[完整核方法 API](../../reference/kernel-methods-api.md) 列出了所有构造参数、拟合
字段、方法签名、核选择和形状。其中，`fit(X, y=None)` 忽略 `y`，`predict(X)`
是 `transform(X)` 的别名。没有模型专用评分，也没有 `inverse_transform` 方法；
该 API 不负责重构输入或提供系数推断。

## 谨慎解释不同拟合之间的差别

特征向量的符号不唯一：两个同样正确的拟合可能得到符号相反的分量。重复或近重复
特征值还允许对应特征子空间内的旋转。需要比较时，应关注表示的子空间、点对几何
关系或下游量，而不是强求每列坐标逐项相等。与其他实现比较前，先对齐核设置、
训练中心化、保留方向及特征向量归一化。

图上看似分离的嵌入，本身不能证明预测有效、聚类有实际意义，或已恢复原始变量。
应评价与应用相关的性质；若目标是预测，就评价留出数据上的下游表现。

<a id="failed-refit"></a>

## 重新拟合失败后，应弃用该实例

`fit` 或 `fit_transform` 抛出异常后，应创建新的 `KernelPCA`，用合适数据成功拟合
后再变换查询点，不要继续使用表面上保留的旧拟合。

失败的重新拟合可能覆盖训练核中心化信息，同时留下旧特征向量和已拟合标记。
随后的 `transform` 或 `predict` 可能返回**有限但错误的坐标**，即使查询的是原训练
行也如此。只检查有限性无法识别这种混合拟合状态。

有限常数数据就可能触发该问题，因为其中心化核没有正特征值方向。新实例也会拒绝
相同的退化数据；重建对象不能使零秩嵌入有效，需要重新考虑数据或分析目标。修改
参数后也应重新拟合，因为 `set_params` 可能保留先前的拟合数组。

## 输入、数值与执行边界

输入须为有限数值，查询时保持特征顺序与数量一致。一维 `X` 表示单个特征的多行
观测。拟合与变换会拒绝 NaN/Inf 输入，卡方核还要求非负特征。

有限输入不保证核有限，例如多项式核仍可能溢出。除变换输出外，还应检查适用的
核参数与拟合数组，丢弃无效结果。对于带巨大共同偏移的 RBF 坐标，应对训练与查询
减去同一个从训练数据得到的偏移，减少平方距离中的消减误差。

训练核需要平方级内存，稠密特征分解最坏为立方级算术成本。分解通常使用所选后端；
线性代数异常会触发 NumPy CPU 特征分解，并且没有单独的回退标志。即使数值运算
在 GPU 上进行，保存的 `X_fit_`、`alphas_`、`lambdas_` 仍为 NumPy 数组。

显式请求不可用的 `device="cuda"` 或 `"torch"` 会报错。不过，KernelPCA 当前在
Torch CUDA 可用性检查通过后，仍可能让 NumPy 或 Torch CPU 输入留在 CPU 上。
应通过返回特征的 Torch `.device` / `.is_cuda` 或 CuPy `.device` 检查实际位置；
所选后端及主机上的拟合数组都不能证明输出位置。若必须使用 CUDA，应在使用前拒绝
CPU 输出。NumPy 输入配合 `device="cpu"` 是可预期的 CPU 路径。
[设备指南](../../guides/device-and-memory.md#current-smoothing-and-spline-exceptions)
进一步说明了各核方法的设备放置限制。

## 延伸阅读

- [核方法总览](kernel-methods.md)
- [Nystroem：可复用的近似特征](kernel-nystroem.md)
- Schölkopf, B., Smola, A., & Müller, K.-R. (1998). Nonlinear component analysis
  as a kernel eigenvalue problem.
