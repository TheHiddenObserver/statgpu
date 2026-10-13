# Nystroem 核近似

> 语言：中文  
> 最后更新：2026-10-12  
> 切换：[English](../../en/models/kernel-nystroem.md)

## 用可复用特征替代完整核

`Nystroem` 从部分训练观测中选出代表节点（landmarks），学习近似核特征映射。
完整的 $n\times n$ 训练核过大时，可用 $m\ll n$ 个节点得到 $n\times m$ 特征矩阵，
交给后续模型复用。

若能负担精确核并希望直接进行回归，使用 [KernelRidge](kernel-ridge.md)；若希望
得到基于主要核主成分的中心化非线性坐标系，使用 [KernelPCA](kernel-pca.md)。
Nystroem 输出的是近似核特征，本身不预测响应。其他选择见
[核方法总览](kernel-methods.md)。

<a id="theory"></a>

## 代表节点如何定义映射

从训练数据中无放回选取 $m=\min(\texttt{n\_components},n)$ 条观测。
令 $K_{mm}$ 为节点之间的核，$K_{nm}$ 为全部输入行到节点的相似度。
对于半正定节点核，设

$$
K_{mm}=V\Lambda V^\top,
$$

则对称逆平方根方向的特征形式为

$$
Z=K_{nm}V\Lambda^{-1/2}V^\top.
$$

忽略数值下限，并在需要时使用相应伪逆，$ZZ^\top$ 以
$K_{nm}K_{mm}^{\dagger}K_{mn}$ 的形式近似完整核。节点为观测提供了一组更少的
相似度比较对象，以此构造数据表示。

实现实际使用 **NumPy SVD**，不是直接使用有符号特征值：

$$
K_{mm}=U\Sigma V^\top,\qquad
N=U\operatorname{diag}\!\left(
\frac{1}{\sqrt{\max(\sigma_j,10^{-12})}}\right)V^\top,\qquad
Z=K_{nm}N.
$$

`normalization_` 保存 $N$。虽然名为 `eigenvalues_`，该属性保存的是**设置下限后的
奇异值**，而不是有符号特征值。较小奇异值会设为下限而非删除，因此保留全部 $m$
个节点并输出 $m$ 列。对于半正定核，这对应带小值下限的对称逆平方根方向。
不定核不意味着存在精确欧氏特征表示：任何 $ZZ^\top$ 都是半正定的，无法精确
复现真正不定的核。

<a id="example"></a>

## 完整的 CPU 映射与预测示例

本示例可独立运行。只从 180 条训练行选择 40 个代表节点，用同一对象映射训练行
和留出行，再对映射特征拟合无权重线性岭预测器。小规模 NumPy 求解明确展示下游
模型，不需要额外模型库。`downstream_alpha` 对应求和平方损失，且不单独拟合截距；
它与节点数、核尺度是不同的控制项。

<!-- example: kernel-nystroem-cpu -->
```python
import numpy as np
from statgpu.nonparametric.kernel_methods import Nystroem, pairwise_kernels
```

### 准备训练行与留出行

留出输入不参与代表节点选择。
```python
rng = np.random.default_rng(42)
X = rng.uniform(-2, 2, size=(240, 2))
y = np.sin(1.7 * X[:, 0]) + 0.4 * X[:, 1] ** 2 + rng.normal(0, 0.1, 240)
indices = rng.permutation(len(X))
train, test = indices[:180], indices[180:]
X_train, X_test = X[train], X[test]
y_train, y_test = y[train], y[test]
```

### 拟合并复用特征映射

测试行使用训练时选定的节点和归一化矩阵。
```python
nystroem = Nystroem(
    n_components=40, gamma=0.7, random_state=42, device="cpu",
)
train_features = nystroem.fit_transform(X_train)
test_features = nystroem.transform(X_test)
```

### 检查学习到的映射

同时检查归一化矩阵与变换后的特征。
```python
assert np.isfinite(nystroem.normalization_).all()
assert np.isfinite(nystroem.eigenvalues_).all()
assert np.isfinite(train_features).all()
assert np.isfinite(test_features).all()
print("Training / test feature shapes:", train_features.shape, test_features.shape)
```

### 拟合下游预测器

利用训练特征矩阵学习响应权重。
```python
downstream_alpha = 1.0
weights = np.linalg.solve(
    train_features.T @ train_features
    + downstream_alpha * np.eye(train_features.shape[1]),
    train_features.T @ y_train,
)
prediction = test_features @ weights
assert np.isfinite(prediction).all()
print("Prediction shape:", prediction.shape)
print("Held-out prediction MSE:", np.mean((y_test - prediction) ** 2))
```

### 在小数据上比较核近似

该可选诊断复用相同测试特征，并有意构造稠密核。
```python
# 仅用于小数据诊断：这里显式构造稠密测试核。
K_test = pairwise_kernels(X_test, metric="rbf", gamma=0.7, xp=np)
approximation_error = np.linalg.norm(test_features @ test_features.T - K_test)
relative_kernel_error = approximation_error / np.linalg.norm(K_test)
print("Held-out relative kernel approximation error:", relative_kernel_error)
```
<!-- example-end: kernel-nystroem-cpu -->

映射输出为 `(180, 40)` 训练特征和 `(60, 40)` 留出特征。只有使用 `y_train` 拟合
`weights` 后，才得到 `(60,)` 响应预测。`Nystroem.predict(X)` 只是
`transform(X)` 的别名，会返回 40 列特征，而非上述响应预测。该类没有模型专用
评分，也没有逆变换。

<a id="approximation-check"></a>

## 区分核近似误差和预测误差

示例中的小数据诊断使用相对 Frobenius 误差：

$$
\frac{\lVert Z_{\mathrm{test}}Z_{\mathrm{test}}^\top-K_{\mathrm{test,test}}\rVert_F}
     {\lVert K_{\mathrm{test,test}}\rVert_F}.
$$

值越小，表示学习到的映射在这些行上越接近精确 RBF 核，但**不一定**使响应预测
误差更小。预测还取决于目标、下游模型、正则化和数据分布。预测 MSE 使用 `y_test`，
回答的是另一类问题。这些是评价诊断，不是系数级标准误或置信区间。

该诊断会显式分配稠密测试核，只适用于小数据。如果使用 Nystroem 正是为了避免
完整核开销，就不应在同等大规模上做此诊断。不要反复利用留出诊断选择节点或核
设置；模型选择应限制在训练数据中。

## 选择节点，并在各训练折内学习映射

- `n_components` 在近似容量、内存和计算之间取舍，实际节点数不超过训练行数。
  节点越多，并不自动保证下游预测越好。
- `random_state` 控制节点采样。固定它可以复现结果；应用需要时，可在训练过程
  内评估对不同节点选择的敏感性。
- `gamma` 控制 RBF 距离尺度，默认值为 `1 / n_features`。特征单位和缩放会改变
  相似度；其他核有各自的参数与定义域限制。
- 下游模型的正则化也应在同一训练过程中选择。核近似较好，不能替代对下游模型
  的验证。

交叉验证时，缩放或填补**以及 Nystroem 映射都要在每个训练折内单独拟合**，然后用
该折的映射变换验证行，并训练、评价下游模型。先在完整数据中选节点再做 CV，会
泄漏验证输入的信息。选定设置后，用全部外层训练行重新拟合预处理、节点映射和
下游模型，再原样应用到留出数据。必要时使用分组或时间分折。
[外部循环示例](kernel-ridge-cv.md#fold-local-preprocessing) 展示折内预处理；
Nystroem 流程还必须在该循环内部学习映射。

以上仅列教学精选控制项。[完整核方法 API](../reference/kernel-methods-api.md)
列出所有参数、方法、拟合数组、核函数与输出形状。

## 数值与执行边界

`fit` 和 `transform` 要求有限数值输入，会拒绝 NaN/Inf。一维 `X` 表示一个特征的
多条观测，查询时应保持特征顺序和数量。卡方核要求非负输入；自定义核必须满足
成对矩阵形状与后端要求，而且还要接受 NumPy 节点输入。

有限输入不能保证核有限。多项式溢出或无效核参数可能使方法成功返回，但归一化
矩阵、奇异值或特征非有限。按示例检查 `normalization_`、`eigenvalues_` 和输出；
丢弃非有限拟合，修正缩放或设置后用新模型拟合。增大下游正则化不能修复已经溢出
的核。有限性只是有用映射的必要条件，不是充分条件。对于有巨大共同坐标偏移的
RBF 数据，对训练与查询使用同一个从训练数据得到的偏移量，以减少距离消减误差；
这不是所有核通用的预处理规则。修改参数后应重新拟合，否则先前拟合数组可能
仍然存在。

表示需要 $O(nm)$ 特征/核存储，加上 $O(m^2)$ 节点存储及稠密节点线性代数，而非
完整的 $n\times n$ 训练核。归一化矩阵也是稠密的，应用它同样有成本。
Nystroem 是显式低秩近似，不是精确估计器的静默回退，也不是近似推断模式。

显式请求不可用后端会报错。节点会复制到 NumPy，节点核构造与 SVD 在 CPU 上运行。查询到节点的核和输出
使用所选数组库，但存在当前设备放置限制：`device="torch"` 会拒绝不可用的 CUDA，
通过检查后，NumPy 或 Torch CPU 输入仍可能产生 CPU 张量。应通过 Torch
`.device` / `.is_cuda` 或 CuPy `.device` 检查输出；若要求 CUDA，就拒绝 CPU
输出。`components_`、`component_indices_`、`normalization_`、`eigenvalues_`
都是主机 NumPy 数组，不能据此判断查询运算在哪个设备完成。NumPy 输入配合
`device="cpu"` 可获得可预期的 CPU 路径，详见
[设备指南](../guides/device-and-memory.md#current-smoothing-and-spline-exceptions)。

与其他实现比较时，要对齐节点选择、核设置、特征方向和 `1e-12` 奇异值下限。
CPU 比较不能证明 GPU 精度或性能。

## 延伸阅读

- [KernelRidge：精确核回归](kernel-ridge.md)
- [核方法总览](kernel-methods.md)与[完整 API](../reference/kernel-methods-api.md)
- Williams, C. K. I., & Seeger, M. (2001). Using the Nystroem method to speed up
  kernel machines.
