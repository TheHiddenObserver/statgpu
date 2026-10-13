# 核方法

> 语言：中文  
> 最后更新：2026-10-13
> 切换：[English](../../en/models/kernel-methods.md)

核方法利用观测之间的相似性来预测响应或学习新的数据表示。本页介绍共用概念，
帮助你选择适合问题的方法；随后可进入[方法教程](#方法教程)，通过独立的 CPU
示例学习具体用法。

## 先按分析问题选择工具

先确定分析目标：预测响应、提取非线性坐标，还是构造较小的特征矩阵。
这几类方法都以核函数衡量观测之间的相似性，无需手工列出所有非线性交互项。

- **预测数值响应**：使用 `KernelRidge`。`KernelRidgeCV` 在训练折内选择正则化
  强度，再用传入的全部训练数据重新拟合。若相近观测应有相近响应，但难以指定曲线
  形式，RBF 核是可考虑的选择。
- **没有响应变量，想提取非线性坐标**：使用 `KernelPCA`。输出可用于可视化或后续
  模型，是数据表示，不是响应预测。
- **需要较小且可复用的核特征矩阵**：使用 `Nystroem`。它选取部分训练观测作为
  代表节点，近似完整核；当精确训练核太大时，可把这些特征交给下游模型。

若线性关系已足够，或希望解释原始特征的斜率，应先考虑[线性模型](linear-regression.md)。
若非线性主要来自各特征独立作用的相加，[GAM](semiparametric.md) 通常更易解释。
精确核方法能更灵活地表示交互，但训练核的存储量随样本数平方增长。RBF 相似度
依赖距离，须合理选择特征单位与尺度；需要从数据学习的预处理必须在每个训练折内
拟合，不能提前使用验证观测。

## 从相似度到 Gram 矩阵

对观测 $x_i$ 和 $x_j$，核函数 $k(x_i,x_j)$ 给出两者的相似度。正半定核可以解释为某个特征空间中的内积，这个空间可能非常高维：

$$
k(x_i,x_j)=\langle\phi(x_i),\phi(x_j)\rangle.
$$

不必显式构造映射 $\phi$，算法可以直接使用这些两两内积，这就是核技巧。上述解释要求核对称且正半定；接口接受自定义函数或 sigmoid 核，并不保证它对当前数据和参数满足这个条件。

对 $n$ 个训练观测，Gram 矩阵定义为 $K_{ij}=k(x_i,x_j)$，形状为 $(n,n)$。对 $q$ 个新观测，新数据到训练数据的核矩阵形状为 $(q,n)$。它的列对应训练观测，而非原始特征；预测时应复用拟合对象和训练行顺序。相似度通常既不是概率，也不是相关系数。

### 三个点的小例子

RBF 核为 $k(x,y)=\exp(-\gamma\lVert x-y\rVert_2^2)$。取一维点 0、1、2，并令 $\gamma=\log 2$；距离为 1、2 时，相似度分别为 $1/2$、$1/16$：

$$
K=\begin{pmatrix}
1 & 1/2 & 1/16\\
1/2 & 1 & 1/2\\
1/16 & 1/2 & 1
\end{pmatrix}.
$$

<!-- example: kernel-gram-cpu -->
```python
import numpy as np
from statgpu.nonparametric.kernel_methods import pairwise_kernels

X = np.array([[0.0], [1.0], [2.0]])
```

计算训练 Gram 矩阵，并与手算结果核对。

```python
K = pairwise_kernels(X, metric="rbf", gamma=np.log(2.0), xp=np)
expected = np.array([[1.0, 0.5, 0.0625],
                     [0.5, 1.0, 0.5],
                     [0.0625, 0.5, 1.0]])
np.testing.assert_allclose(K, expected)
```

对新行计算它们与同一组三个训练观测之间的相似度。

```python
query = np.array([[0.0], [2.0]])
K_query = pairwise_kernels(query, X, metric="rbf", gamma=np.log(2.0), xp=np)
np.testing.assert_allclose(K_query, expected[[0, 2]])
print(K.shape, K_query.shape)
```
<!-- example-end: kernel-gram-cpu -->

两个矩阵的形状分别为 `(3, 3)` 和 `(2, 3)`。每个点到自身的距离为零，因此对角线为一。第一个与第三个训练点之间的相似度低于相邻点。这里只计算了相似度，还没有拟合响应模型或学习降维坐标。

## 特征单位与核尺度

对 RBF，`gamma` 越大，相似度随距离衰减得越快。`gamma` 太小时，不同行可能看起来几乎一样；太大时，训练核矩阵可能接近对角矩阵，新行与所有训练行的相似度都很低。这两个极端都不保证预测有效。多项式核的 `degree` 和 `coef0` 控制另一种相似度，不能按 RBF 距离尺度来理解。

缩放一列会改变距离，也会改变核衡量的关系。应根据问题决定保留原始单位还是标准化到可比尺度。均值和尺度只能从训练行学习，再用于新数据；验证时要在每个训练折内重新学习预处理。KernelPCA 对 Gram 矩阵的中心化是在核特征空间内进行的，并不会标准化原始输入列。

对采用这一默认值的核，`gamma=None` 通常取 `1 / n_features`，并不是自动搜索带宽。Chi-squared 核的默认值不同，而且要求输入非负；直接标准化可能使输入不再满足这一要求。详见[核参数及传递规则](../reference/kernel-methods-api.md#kernel-parameter-forwarding)。

## 方法教程

<a id="kernel-cpu-workflow"></a>
<a id="训练集与留出集的-cpu-流程"></a>
<a id="准备并划分数据"></a>
<a id="拟合一个核岭模型"></a>
<a id="预测并解释结果"></a>

### [核岭回归：拟合与预测](kernel-ridge.md#example)

从正则化目标函数出发，完成训练到留出集预测的完整流程，理解核系数与原始特征
斜率的区别，并检查拟合结果是否满足求解方程。

<a id="kernel-cv-workflow"></a>
<a id="核岭交叉验证"></a>
<a id="在训练折内选择-alpha"></a>

### [核岭交叉验证：选择 alpha](kernel-ridge-cv.md#selection)

用验证 MSE 选择 `alpha`，理解 `best_score_` 报告的 R²，并判断何时需要
外部划分循环来处理预处理、分组观测或时间顺序。

<a id="kernel-pca"></a>
<a id="学习并复用非线性坐标"></a>

### [KernelPCA：非线性坐标](kernel-pca.md#example)

不使用响应变量，从中心化核学习坐标系，再将新观测投影到同一空间。教程说明
可保留的维数、符号歧义，以及重新拟合失败后的处理方式。

<a id="nystroem-近似"></a>
<a id="构造较小的核特征矩阵"></a>
<a id="可选检查小数据上的核近似"></a>

### [Nystroem：低秩核特征](kernel-nystroem.md#example)

从训练数据选取代表节点，构造较小的特征矩阵并交给下游预测器。小数据诊断
进一步说明核近似误差与预测误差为何是两个不同的问题。

<a id="可选-gpu-示例"></a>
<a id="cupy"></a>
<a id="torch-cuda"></a>

### [可选 GPU 示例](kernel-ridge-cv.md#gpu-examples)

在 CPU 交叉验证流程基础上改用 CuPy 或 Torch CUDA。使用 KernelPCA 与
Nystroem 前，还应了解下文的[执行边界](#后端与执行边界)：选择设备并不意味着
每一步都在该设备上执行。

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

卡方核要求输入特征非负。估计器允许时也可传入自定义核函数；该函数必须
返回请求后端上的数组，并满足预期的两两核矩阵形状。

<a id="完整估计器-api"></a>
<a id="构造参数"></a>
<a id="方法与形状"></a>
<a id="拟合属性与输出"></a>

## 查阅完整 API

[核方法 API 参考](../reference/kernel-methods-api.md) 集中列出四个估计器与八个核函数的完整签名、默认值、方法、拟合属性、数组形状、自定义核和后端约束。

每个类的构造参数、方法形状和拟合属性分别见 [KernelRidge](../reference/kernel-methods-api.md#kernelridge)、[KernelRidgeCV](../reference/kernel-methods-api.md#kernelridgecv)、[KernelPCA](../reference/kernel-methods-api.md#kernelpca) 和 [Nystroem](../reference/kernel-methods-api.md#nystroem)。

<a id="交叉验证结果的组织"></a>

候选数组、逐折表、折均值，以及 `best_score_` 与选择损失的区别，见
[CV 结果字段](../reference/kernel-methods-api.md#kernelridgecv)。

<a id="完整成对核函数-api"></a>

[成对核函数 API](../reference/kernel-methods-api.md#pairwise-kernels) 列出
`pairwise_kernels` 和七个核函数，并说明自定义核用法、`xp`、数据类型和参数传递规则。

<a id="进一步理解核方法"></a>
<a id="核岭回归"></a>

若想理解拟合与预测背后的数学，可从[核岭目标函数](kernel-ridge.md#theory)
开始，了解平方和损失、核系数及正则化线性系统之间的联系。

<a id="核主成分的计算"></a>

[核主成分推导](kernel-pca.md#theory) 说明训练核的中心化如何用于新数据投影。

<a id="nystroem-的特征映射"></a>

[Nystroem 推导](kernel-nystroem.md#theory) 说明代表节点核与 SVD 归一化
如何构成低秩特征映射。

## 后端与执行边界

四类估计器都接受 `device`，但核矩阵构造、分解和输出不一定全在同一设备上：

- **KernelRidge / KernelRidgeCV**：核矩阵构造和求解使用所选后端。
- **KernelPCA**：通常在所选后端分解；若发生线性代数异常，会改用 NumPy 在 CPU
  上进行特征分解，且没有单独的回退标记。保存的训练数据和投影系数也是 NumPy
  数组，因此不能假定整个拟合过程都在 GPU 上完成。
- **Nystroem**：将选定节点复制到 NumPy，在 CPU 上构造节点核矩阵并完成 SVD，
  随后使用所选数组库计算查询点到节点的核矩阵并返回特征，实际设备还受下述限制。
  自定义核也必须接受 NumPy 节点输入。节点索引、归一化矩阵和奇异值保存在 CPU
  内存中；选择 GPU 不会把节点分解移到 GPU。

显式请求不可用的设备会报错。

Torch CUDA 不可用时，KernelPCA 与 Nystroem 仍会拒绝 `device="torch"`。通过该检查并不保证 CUDA 执行：这些路径未始终将输入移到请求设备，因此 NumPy 或 Torch CPU 输入在拟合和变换时仍可能留在 CPU。

请检查 `fit_transform`、`transform` 和 `predict` 实际返回的特征：Torch 查看 `.device`/`.is_cuda`，CuPy 查看 `.device`，不能仅依赖设备配置或选中的后端/数组库。两者的公开拟合数组保存为 NumPy，不能说明数值计算所在设备。此限制还影响
Nystroem 的查询核与输出，应与有意安排在 CPU 的节点 SVD 区分。

必须使用 CUDA 时，应在使用结果前拒绝 CPU 输出。需要明确的 CPU 路径时，请传入 NumPy 数组并设 `device="cpu"`。详见[设备说明](../guides/device-and-memory.md#current-smoothing-and-spline-exceptions)。

`KernelPCA` 和 `Nystroem` 在拟合和变换时拒绝输入数组中的 NaN/Inf。这项输入检查
不保证计算得到的核矩阵或学习数组也是有限值；详见[结果有限性检查](#检查核拟合结果是否有限)。
卡方核的输入必须非负。

<a id="推断语义"></a>

## 如何评价模型

核方法当前不提供系数级标准误、假设检验或置信区间。模型质量通过预测分数、
交叉验证损失、嵌入性质、核矩阵重构或近似诊断以及应用相关验证进行评估。KernelPCA 不提供将坐标还原成原始输入行的逆变换。

该模块没有严格/近似推断模式；Nystroem 是显式的低秩核近似，
而不是精确核估计器的静默回退。

## 复杂度与性能说明

- 精确核方法构造 $n\times n$ 训练核矩阵，内存复杂度为二次量级。
- 直接核岭求解和稠密特征分解对训练样本数具有三次最坏计算复杂度。
- `KernelRidgeCV` 可以在 alpha 间复用分解，但 CV 仍会增加各折的计算量。
- `Nystroem` 将核存储降低到 $O(nm)$，另加代表节点上的线性代数运算。
- GPU 加速效果取决于样本量、数据类型、核函数、同步开销和可用显存；小问题可能 CPU 更快。

例如，float64 的 $n=10{,}000$ 方阵仅数据就约占 800 MB，还不含分解临时数组和副本。Nystroem 取 $m=100$ 时，$(n,m)$ 特征数组约为 8 MB，但仍需代表节点矩阵及 $O(nm^2+m^3)$ 的稠密特征归一化与 SVD 运算；下游模型另有成本。这种存储节省需要与近似精度权衡，不能只凭列数更少就判断预测质量更好。

<a id="限制与失败行为"></a>

## 数值问题与使用注意事项

### 重新拟合失败后

如果 `KernelPCA.fit` 或 `fit_transform` 抛出异常，应丢弃该实例。请新建
`KernelPCA`，成功拟合后再变换查询点，不要继续使用看似仍然保留的旧拟合。

原因是失败的重新拟合可能已经改写训练核的中心化信息，却仍保留旧特征向量和
已拟合标记。此后调用 `transform` 或 `predict`，即使查询原训练数据，也可能
得到有限但错误的坐标；仅检查输出是否有限无法识别这种新旧状态混用。

有限的常数数据就可能触发该问题，因为中心化核没有正特征值方向。新实例也会
拒绝同样的退化数据；需要调整数据或分析方法，重新创建对象本身不能让零秩核
产生有效嵌入。

### 检查核拟合结果是否有限

即使训练数据都是有限值，多项式核的计算仍可能溢出。当前 `KernelRidge` 和
`Nystroem` 可能正常返回拟合对象，但学习到的数组以及预测或特征均包含 NaN；
NaN `gamma` 等无效核参数也可能产生这种结果。线性代数警告或没有抛出异常，都
不能可靠证明拟合成功。

应使用有限且适用于所选核的参数，并在使用前检查拟合得到的数组和查询输出。
对于 NumPy CPU 拟合，KernelRidge 检查 `np.isfinite(model.dual_coef_).all()`；
Nystroem 检查 `np.isfinite(model.normalization_).all()` 和
`np.isfinite(model.eigenvalues_).all()`。预测或变换特征也要检查；设备数组使用
对应后端的有限值检查。出现非有限结果时，应丢弃该拟合，检查特征单位、尺度与
核参数，然后用新实例重新拟合。核已经溢出时，增大 `alpha` 无法修复问题。
缩放会改变多项式相似度，因此预处理的选择和验证应在训练折内完成。数值有限只是
必要条件，不能单独证明核矩阵条件良好或模型适合分析目的。

### RBF 坐标偏移过大时

RBF 核遇到很大的共同坐标偏移时，请用训练数据确定一个偏移量，并同时从训练和查询特征中减去它。当前平方距离计算可能丢失相近点的差异：`1e9` 附近的坐标可能生成几乎全为 1 的核矩阵，显著改变预测。中心化保持预期的 RBF 核不变，但并非所有核的通用预处理规则，尤其不能直接套用于卡方核。

### 其他使用注意事项

核矩阵病态时，可尝试增大 `alpha` 或调整核尺度；RBF 等距离核尤其需要检查
`gamma` 是否合适。自定义核还须满足所选估计器对后端、数据类型、矩阵形状和
对称性的要求。大样本的内存成本与交叉验证开销见上文的复杂度说明。

<a id="验证方法"></a>

## 与其他实现核对结果

核岭回归可直接验证正则化线性系统，或与 scikit-learn 对照，但须使用相同核、alpha、数据及不加权目标。交叉验证应检查各候选 MSE 是否有限，并确认最终重拟合沿用选定 alpha。比较 Nystroem 归一化矩阵或特征时，应对齐代表节点和 SVD 奇异值下限。CPU 上的检查不能说明 GPU 上的精度或性能，应在实际使用的设备上验证。

## FAQ

### 应使用 KernelRidge，还是 Nystroem 加线性模型？

当完整训练核矩阵能放入内存且需要精确核表示时使用 `KernelRidge`；当需要可控低秩
特征近似以扩展规模或复用特征时使用 Nystroem。

### 为什么 GPU 核方法可能比 CPU 慢？

核矩阵构造和线性代数运算的规模需要足够大，才能摊薄设备任务启动、同步和内存传输成本。

### `device="auto"` 会覆盖显式请求吗？

不会。`"auto"` 本身表示自动选择；显式 `"cuda"` 或 `"torch"` 在对应后端不可用时
会报错。但 KernelPCA/Nystroem 通过检查后仍可能让 CPU 输入留在 CPU，须按上文检查实际返回的特征。

### 不同拟合的 KernelPCA 成分是否可直接比较？

特征向量符号，以及重根或近重根特征空间内的基不唯一。需要时应比较表示的子空间
或下游量，而不是逐列直接比较。

## 参考文献

- Schölkopf, B., Smola, A., & Müller, K.-R. (1998). Nonlinear component analysis
  as a kernel eigenvalue problem.
- Williams, C. K. I., & Seeger, M. (2001). Using the Nystroem method to speed up
  kernel machines.
- Shawe-Taylor, J., & Cristianini, N. (2004). *Kernel Methods for Pattern Analysis*.

