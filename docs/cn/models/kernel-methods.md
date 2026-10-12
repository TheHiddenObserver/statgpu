# 核方法

> 语言：中文  
> 最后更新：2026-10-12
> 切换：[English](../../en/models/kernel-methods.md)

## 先按分析问题选择工具

当响应与特征之间的关系难以用直线描述时，核方法可以从观测之间的相似性出发建模。
核函数把两条观测映射为一个相似度，模型再据此预测响应或提取新的数据坐标，
无需手工列出所有非线性交互项。

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

<a id="kernel-cpu-workflow"></a>

## 训练集与留出集的 CPU 流程

先用固定参数的核岭回归学习完整的拟合与预测流程，再在下文加入交叉验证或特征映射。示例先划分训练行与留出行，并固定 RBF 尺度 `gamma`。若要选择 `alpha`、`gamma` 或特征映射，应把选择及预处理限制在训练过程中，不要反复根据留出得分调整设置。

<!-- example: kernel-methods-cpu -->
```python
import numpy as np
from statgpu.nonparametric.kernel_methods import KernelRidge
```

### 准备并划分数据

`X` 为 `(240, 2)`，每行是一条观测，两列是连续特征；`y` 为 `(240,)`，每行一个连续响应。随机划分后，180 行用于训练，60 行用于测试。后续各节会明确复用本节的训练与测试数组。

```python
rng = np.random.default_rng(42)
X = rng.uniform(-2, 2, size=(240, 2))
y = np.sin(1.7 * X[:, 0]) + 0.4 * X[:, 1] ** 2 + rng.normal(0, 0.1, 240)
indices = rng.permutation(len(X))
train, test = indices[:180], indices[180:]
X_train, X_test = X[train], X[test]
y_train, y_test = y[train], y[test]
```

### 拟合一个核岭模型

`gamma` 控制 RBF 相似度随距离衰减的速度，`alpha` 控制正则化强度。本例先固定这两个参数，展示单次拟合；如何选择 `alpha` 见下文“核岭交叉验证”。

```python
kr = KernelRidge(alpha=1.0, gamma=0.7, device="cpu").fit(X_train, y_train)
```

### 预测并解释结果

复用刚拟合的 `kr`，对相同特征顺序的测试行预测响应。

```python
fixed_prediction = kr.predict(X_test)
print(fixed_prediction.shape)
print("Fixed-alpha held-out R2:", kr.score(X_test, y_test))
```

输出形状为 `(60,)`，每条留出观测对应一个响应预测。R²（决定系数）以留出响应的均值预测为基准：1 表示完全准确，0 表示与该常数预测相当，负值则更差。核系数不能当作原始特征的斜率。

## 核岭交叉验证

`KernelRidgeCV` 在训练数据内比较一组 `alpha`，然后用选中的值在传入的全部
训练数据上重新拟合。它只选择 `alpha`，不会同时搜索核类型或 `gamma`。
`cv` 只接受整数，采用打乱后的按行 K 折划分，不支持自定义、分组或时间顺序划分。
若分析需要这些划分，或需要在每个训练折中学习预处理，请在外部自行划分，
逐折拟合预处理和 `KernelRidge`，不要先在全部数据上拟合需要学习参数的预处理，再交给 `KernelRidgeCV`。

选择 `alpha` 时，以各折、各响应列等权平均后的验证 MSE 最小为准。
`alpha_` 保存选中的值；`best_score_` 则是该值的平均折内 R²，两者用途不同。

训练折核矩阵奇异时，显式的零 alpha 可能产生非有限 CV 得分，却仍被选中。请使用严格为正的候选值，并在解释选择结果前检查 `mean_mse` 和 `best_score_` 是否有限。能够返回拟合对象，并不保证搜索结果有效。自动网格使用正值。

<a id="kernel-cv-workflow"></a>

### 在训练折内选择 alpha

先运行[训练集与留出集的 CPU 流程](#kernel-cpu-workflow)。下面继续使用该节的 `np`、`X_train`、`y_train`、`X_test` 和 `y_test`，保持相同的 `gamma=0.7`，只在训练行中选择 `alpha`。

```python
from statgpu.nonparametric.kernel_methods import KernelRidgeCV

kr_cv = KernelRidgeCV(
    alphas=[0.01, 0.1, 1.0, 10.0], gamma=0.7,
    cv=5, random_state=42, device="cpu",
).fit(X_train, y_train)
```

使用结果前，确认搜索得分有限。然后评价留出响应；MSE 的单位是响应单位的平方，越低越好。

```python
assert np.isfinite(kr_cv.cv_results_["mean_mse"]).all()
assert np.isfinite(kr_cv.best_score_)
prediction = kr_cv.predict(X_test)
held_out_r2 = kr_cv.score(X_test, y_test)
held_out_mse = np.mean((y_test - prediction) ** 2)
print("Selected alpha:", kr_cv.alpha_)
print("CV model held-out R2 / MSE:", held_out_r2, held_out_mse)
```

`best_score_` 汇总训练数据上的交叉验证折，不是留出集结果。这里的留出 R² 和 MSE 用于说明评估方法，不应用这些测试行继续调参。

## Kernel PCA

`KernelPCA` 根据观测之间的核相似度学习非线性坐标，不需要响应变量。与普通 PCA
直接寻找原始特征的线性组合不同，核主成分描述核所定义的特征空间中的变化。
`n_components` 指定最多保留多少个坐标；实际维数还取决于中心化后的训练核中可用的正特征值方向。

### 学习并复用非线性坐标

复用[训练集与留出集的 CPU 流程](#kernel-cpu-workflow)中的 `X_train` 和 `X_test`。这里不使用 `y`；先用训练行学习坐标定义，再变换测试行。

```python
from statgpu.nonparametric.kernel_methods import KernelPCA

kpca = KernelPCA(n_components=3, gamma=0.7, device="cpu")
train_coordinates = kpca.fit_transform(X_train)
```

同一个已拟合对象保证新行沿用训练期的中心化与投影。

```python
test_coordinates = kpca.transform(X_test)
print(test_coordinates.shape)
```

`(60, 3)` 表示 60 行、三个非线性坐标，不是三个响应预测。

## Nystroem 近似

`Nystroem` 从训练数据中选取少量代表节点，用每条观测与这些节点的相似度构造
近似核特征。它适合完整核矩阵过大，或需要将同一组特征交给不同下游模型的场景。
`n_components` 控制节点数，也决定输出列数（最多为训练样本数）。

### 构造较小的核特征矩阵

复用[训练集与留出集的 CPU 流程](#kernel-cpu-workflow)中的 `np`、`X_train` 和 `X_test`。40 个代表节点只从训练行中选取；测试行通过同一个映射变换。

```python
from statgpu.nonparametric.kernel_methods import Nystroem

nystroem = Nystroem(
    n_components=40, gamma=0.7, random_state=42, device="cpu",
)
train_features = nystroem.fit_transform(X_train)
```

再变换测试行，得到可供后续模型使用的特征。

```python
test_features = nystroem.transform(X_test)
print(test_features.shape)
```

输出 `(60, 40)` 是近似核特征，单独使用不会预测 `y`。用于下游预测时，应在每个训练折内单独拟合映射。

### 可选：检查小数据上的核近似

继续使用上面得到的 `test_features`，与相同测试行的精确 RBF 核比较。此诊断显式构造稠密核，只适合小数据。

```python
from statgpu.nonparametric.kernel_methods import pairwise_kernels

K_test = pairwise_kernels(X_test, metric="rbf", gamma=0.7, xp=np)
approximation_error = np.linalg.norm(test_features @ test_features.T - K_test)
relative_kernel_error = approximation_error / np.linalg.norm(K_test)
print("Held-out relative kernel approximation error:", relative_kernel_error)
```
<!-- example-end: kernel-methods-cpu -->

误差越小，说明这些行上 `Z @ Z.T` 与精确核越接近，不代表响应预测一定更好。完整核无法容纳时，不应照搬此诊断。

## 可选 GPU 示例

先完成[在训练折内选择 alpha](#kernel-cv-workflow)。下面继续使用该节的 `KernelRidgeCV`、`X_train`、`y_train`、`X_test` 和已配置的 `kr_cv`，只更改设备。根据安装情况选择一个代码段；每个代码段都需要可工作的 CUDA 后端。

### CuPy

<!-- example-requires: kernel-methods-cpu -->
<!-- example: kernel-methods-cupy -->
```python
gpu_options = {**kr_cv.get_params(), "device": "cuda"}
kr_cupy = KernelRidgeCV(**gpu_options).fit(X_train, y_train)
prediction_cupy = kr_cupy.predict(X_test)
```
<!-- example-end: kernel-methods-cupy -->

### Torch CUDA

<!-- example-requires: kernel-methods-cpu -->
<!-- example: kernel-methods-torch -->
```python
gpu_options = {**kr_cv.get_params(), "device": "torch"}
kr_torch = KernelRidgeCV(**gpu_options).fit(X_train, y_train)
prediction_torch = kr_torch.predict(X_test)
```
<!-- example-end: kernel-methods-torch -->

`device="cuda"` 请求 CuPy CUDA，`device="torch"` 请求 Torch CUDA。KernelPCA/Nystroem 仅成功选择后端并不能证明输出位于 GPU，详见下文。

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

## 完整估计器 API

以下四个类均从 `statgpu.nonparametric.kernel_methods` 导入。构造函数及默认值为：

```text
KernelRidge(alpha=1.0, kernel='rbf', gamma=None, degree=3, coef0=1, kernel_params=None, device='auto', n_jobs=None)
KernelRidgeCV(alphas=None, cv=5, kernel='rbf', gamma=None, degree=3, coef0=1, kernel_params=None, random_state=None, device='auto', n_jobs=None)
KernelPCA(n_components=2, kernel='rbf', gamma=None, degree=3, coef0=1, alpha=1.0, eigen_solver='auto', device='auto', n_jobs=None)
Nystroem(kernel='rbf', n_components=100, gamma=None, degree=3, coef0=1, random_state=None, device='auto', n_jobs=None)
```

### 构造参数

| 参数 | 含义与限制 |
|---|---|
| `kernel` | 内置核名称或返回完整矩阵的可调用函数，见下方函数 API。搭配估计器专属参数时，请用规范的小写名称。 |
| `gamma` | RBF、Laplacian、多项式和 sigmoid 核的系数；`None` 使用 `1 / n_features`。卡方核默认 `1.0`。KernelRidge/CV 当前会忽略卡方核的此构造参数，应改用 `kernel_params={"gamma": value}`。 |
| `degree`、`coef0` | 多项式次数和加法常数；sigmoid 核也使用 `coef0`。只在适用的核上设置。 |
| `kernel_params` | 仅 KernelRidge/CV：传给内置核或自定义核的额外参数。适用的非默认构造参数可能覆盖字典中的同名值。自定义核所需参数请放在此字典中。 |
| `alpha` | KernelRidge：平方和目标中的有限非负惩罚强度。KernelPCA：特征分解时加入的有限非负对角平移量；报告特征值时会减去该量，并不表示逆变换的正则化。 |
| `alphas` | KernelRidgeCV：有限、非空、非负的候选集合；建议使用严格正值。`None` 根据训练核的特征值构造 100 个正的对数等距候选值。 |
| `cv` | KernelRidgeCV：2 到行数之间的整数；按行打乱后进行 K 折划分。不提供自定义或分组折参数。 |
| `n_components` | 正整数。KernelPCA 最多保留该数量的正特征值方向，实际输出可能更少；Nystroem 不放回选取 `min(n_components, n_training_rows)` 个代表节点。 |
| `eigen_solver` | KernelPCA：`"auto"` 或 `"dense"`；当前都使用稠密对称特征分解。 |
| `random_state` | CV 打乱行顺序或 Nystroem 选择代表节点所用的整数种子，也可为 `None`。 |
| `device` | `"cpu"`、`"cuda"`（CuPy）、`"torch"`（Torch CUDA）或 `"auto"`；KernelPCA/Nystroem 通过可用性检查后仍存在 Torch CPU 放置例外，详见后端边界。 |
| `n_jobs` | 共享估计器设置；这些估计器不通过此参数并行拟合。 |

### 方法与形状

`X` 应为有限数值矩阵 `(n,p)`，预测时须保持特征顺序。一维 `X` 表示单个特征的多行观测，不表示一个多维查询点。估计器将数值输入转为 float64。记查询行数为 `q`，响应列数为 `r`；KernelPCA 保留 `k <= min(n_components,n)` 个正方向，Nystroem 使用 `m=min(n_components,n)` 个代表节点。

| 类 | 方法调用 | 返回值与限制 |
|---|---|---|
| KernelRidge | `fit(X, y, sample_weight=None)`、`predict(X)`、`score(X, y)` | `fit` 返回 `self`；`y` 为 `(n,)` 或 `(n,r)`。当前忽略权重。单个响应的预测为 `(q,)`，包括训练时传单列矩阵的情况；多响应为 `(q,r)`。 |
| KernelRidgeCV | `fit(X, y)`、`predict(X)`、`score(X, y)` | 响应与预测形状同上；`fit` 返回 `self`。没有样本权重参数。预测和评分委托给 `estimator_`。 |
| KernelPCA、Nystroem | `fit(X, y=None)`、`transform(X)`、`fit_transform(X, y=None)`、`predict(X)` | `fit` 返回 `self`；不使用 `y`。其他方法返回特征数组（Torch 设备例外见下文）：KernelPCA 为 `(q,k)`，Nystroem 为 `(q,m)`；`predict` 是变换别名，不预测响应。 |

岭回归 `score` 返回 Python 浮点数：先按响应列计算 R²，再等权平均。常数响应在预测近乎完全一致时取 1，否则取 0。KernelPCA/Nystroem 没有模型专属的 `score` 或逆变换方法。四个类还提供 `get_params(deep=True)`、`set_params(**params)` 及[共享推断工具](../reference/estimator-api.md)。修改参数后必须重拟合；KernelPCA/Nystroem 可能一直保留旧拟合数组到下次拟合，不能依赖自动失效处理。这些工具不会自动提供核系数推断。

### 拟合属性与输出

| 类 | 字段与解释 |
|---|---|
| KernelRidge | `X_fit_`：后端数组 `(n,p)`；`dual_coef_`：后端数组 `(n,r)`，向量响应也保存为 `(n,1)`；`n_features_in_`：整数。这些是核系数，不是原始特征斜率。 |
| KernelRidgeCV | `alpha_`、`best_score_`、`cv_results_` 的含义见前文；`estimator_` 为最终 KernelRidge；`dual_coef_`、`X_fit_` 引用其数组。特征数请读取 `estimator_.n_features_in_`。CV 结果数组为 NumPy。 |
| KernelPCA | NumPy `lambdas_` `(k,)` 为保留的中心化核正特征值；NumPy `alphas_` `(n,k)` 为特征向量除以对应特征值平方根；NumPy `X_fit_` `(n,p)`；整数 `n_samples_`、`n_features_in_`。只保留超过数值阈值的特征值；没有正方向时抛出错误。 |
| Nystroem | NumPy `components_` `(m,p)`、`component_indices_` `(m,)`、`normalization_` `(m,m)`、`eigenvalues_` `(m,)`；整数 `n_features_in_`。历史名称 `eigenvalues_` 实际存放经过下限截断的奇异值。`m=min(n_components,n_training_rows)`。 |

### 交叉验证结果的组织

选择标准是验证 MSE，对各折和各响应列等权平均。`alpha_` 是选中的值；`best_score_` 是该值对应的平均折内 R²，不是用于选择的 MSE。`cv_results_` 包含 `alphas`、`mean_mse`、`mse_table`、`mean_r2`、`r2_table`、`best_alpha` 和 `best_score`。表格数组形状为 `(n_alphas, n_folds, n_targets)`，均值数组为 `(n_alphas, n_targets)`。

## 完整成对核函数 API

以下函数均从同一模块公开导出：

```text
pairwise_kernels(X, Y=None, metric='rbf', xp=None, **params)
rbf_kernel(X, Y=None, gamma=None, xp=None)
polynomial_kernel(X, Y=None, degree=3, gamma=None, coef0=1, xp=None)
linear_kernel(X, Y=None, xp=None)
laplacian_kernel(X, Y=None, gamma=None, xp=None)
sigmoid_kernel(X, Y=None, gamma=None, coef0=1, xp=None)
cosine_kernel(X, Y=None, xp=None)
chi2_kernel(X, Y=None, gamma=1.0, xp=None)
```

`X`、`Y` 应为兼容的有限后端数组，形状分别为 `(n,p)`、`(q,p)`；`Y=None` 表示 `Y=X`。返回成对矩阵 `(n,q)`，省略 `Y` 时为 `(n,n)`。对内置核，`xp=None` 使用 NumPy，不自动识别输入后端。应将 `xp` 设为对应的 NumPy/CuPy/Torch 模块；仅传模块不会把输入移到 GPU。这些底层函数没有统一的数据类型转换和输入验证规则，请自行准备有效数组。卡方核要求非负输入以及有限非负 `gamma`。

`pairwise_kernels` 会统一名称的大小写并去除两端空格；别名为 `gaussian` 对应 `rbf`、`poly` 对应 `polynomial`、`chi-squared` 对应 `chi2`。自定义函数接收完整 `X, Y` 数组和额外的 `params`，若签名允许，还会收到 `xp` 关键字。分派时会原样传入 `xp`，包括 `None`；自定义函数须自行确定默认模块，或由调用方显式传入 `xp=np`。按该形式调用时，还须处理 `Y=None`，返回完整成对矩阵，而非每次只计算一对样本的标量。未知核名称抛出 `ValueError`。

以下自定义核自行将省略的模块设为 NumPy，同时支持显式模块：

<!-- example: kernel-callable-cpu -->
```python
import numpy as np
from statgpu.nonparametric.kernel_methods import pairwise_kernels

def custom_linear(X, Y=None, xp=None):
    xp = np if xp is None else xp
    Y = X if Y is None else Y
    return xp.asarray(X) @ xp.asarray(Y).T

X = np.array([[-1.0], [0.5], [2.0]])
implicit = pairwise_kernels(X, metric=custom_linear)
explicit = pairwise_kernels(X, metric=custom_linear, xp=np)
np.testing.assert_allclose(implicit, X @ X.T)
np.testing.assert_allclose(explicit, implicit)
```
<!-- example-end: kernel-callable-cpu -->

余弦实现会在范数乘积的分母上加 `1e-10`，因此涉及零向量的结果为零，极小范数输入也不同于精确归一化。卡方核在 NumPy 上将零除零项视为零；CuPy/Torch 使用 `1e-10` 的分母下限，极小的非负特征可能因此产生不同结果。

实现参考：[KernelRidge](../../../statgpu/nonparametric/kernel_methods/_krr.py)、[KernelRidgeCV](../../../statgpu/nonparametric/kernel_methods/_krr_cv.py)、[KernelPCA](../../../statgpu/nonparametric/kernel_methods/_kpca.py)、[Nystroem](../../../statgpu/nonparametric/kernel_methods/_nystroem.py) 和[成对核函数](../../../statgpu/nonparametric/kernel_methods/_kernels.py)。

## 进一步理解核方法

### 核岭回归

给定训练核矩阵 $K$，核岭回归求解对偶线性系统

$$
(K+\alpha I)c=y.
$$

当核矩阵半正定时，对应的再生核希尔伯特空间（RKHS）惩罚目标为

$$
\min_c \lVert y-Kc\rVert_2^2+\alpha c^\top Kc.
$$

损失使用残差平方和，不除以样本量。$K$ 正定时，一阶条件给出上述线性系统；$K$ 半正定但奇异时，$\alpha>0$ 的同一系统选定一组实现最优拟合函数的系数。仅惩罚 $\lVert c\rVert^2$ 会得到另一种估计量。不定核没有这一凸 RKHS 解释。

测试样本预测为

$$
\hat y_{test}=K(X_{test},X_{train})c.
$$

`KernelRidge` 直接求解正则化线性系统。若求解抛出线性代数错误，会逐步增大对角扰动后重试；稳定化后的结果不一定精确满足原系统。建议使用尺度合适的正 `alpha`。兼容的响应矩阵可用于多输出回归。

`fit(X, y, sample_weight=None)` 当前会忽略传入的 `sample_weight`，并不拟合加权目标。请省略该参数，仅用于不加权的分析。如果需求只是排除某些行，可以在拟合前删除零权重行；这不能实现任意的加权拟合。

### 核主成分的计算

对中心化核矩阵 $\widetilde K$，Kernel PCA 执行

$$
\widetilde K = V\Lambda V^\top.
$$

最大特征值对应的特征向量定义非线性主成分。变换新数据时，需要计算测试集到训练集的核矩阵，
应用训练期中心化量，并投影到保留的成分。

### Nystroem 的特征映射

Nystroem 选择 $m$ 个代表节点，并构造显式近似特征映射。若代表节点核矩阵为

$$
K_{mm}=V\Lambda V^\top,
$$

则变换特征具有形式

$$
Z=K_{nm}V\Lambda^{-1/2}V^\top.
$$

对半正定核，这是输出所用的对称逆平方根形式。实现实际用 NumPy SVD 计算 $K_{mm}=U\Sigma V^\top$，归一化矩阵为 $U\max(\Sigma,10^{-12}I)^{-1/2}V^\top$。输出保留全部 $m$ 个选定节点；`eigenvalues_` 存放截断后的奇异值，即使核矩阵不定也如此。不定核并不对应精确的欧氏特征表示。

当 $m\ll n$ 时，这会用 $n\times m$ 特征矩阵替代完整的 $n\times n$ 核表示。

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

Torch CUDA 不可用时，KernelPCA 与 Nystroem 仍会拒绝 `device="torch"`。通过该检查并不保证 CUDA 执行：这些路径未始终将输入移到请求设备，因此 NumPy 或 Torch CPU 输入在拟合和变换时仍可能留在 CPU。请检查 `fit_transform`、`transform` 和 `predict` 实际返回的特征：Torch 查看 `.device`/`.is_cuda`，CuPy 查看 `.device`，不能仅依赖设备配置或选中的后端/数组库。两者的公开拟合数组按设计保留为 NumPy，不能证明数值计算所在设备。此限制还影响 Nystroem 的查询核与输出，须与有意安排在 CPU 的节点 SVD 区分。必须使用 CUDA 时，应在使用结果前拒绝 CPU 输出。需要明确的 CPU 路径时，请传入 NumPy 数组并设 `device="cpu"`。详见[设备说明](../guides/device-and-memory.md#current-smoothing-and-spline-exceptions)。

`KernelPCA` 和 `Nystroem` 在拟合和变换时拒绝输入数组中的 NaN/Inf。这项输入检查
不保证计算得到的核矩阵或学习数组也是有限值；详见[结果有限性检查](#检查核拟合结果是否有限)。
卡方核的输入必须非负。

<a id="推断语义"></a>

## 如何评价模型

核方法当前不提供系数级标准误、假设检验或置信区间。模型质量通过预测分数、
交叉验证损失、嵌入性质、重构或近似诊断以及应用相关验证进行评估。

该模块没有严格/近似推断模式；Nystroem 是显式的低秩核近似，
而不是精确核估计器的静默回退。

## 复杂度与性能说明

- 精确核方法构造 $n\times n$ 训练核矩阵，内存复杂度为二次量级。
- 直接核岭求解和稠密特征分解对训练样本数具有三次最坏计算复杂度。
- `KernelRidgeCV` 可以在 alpha 间复用分解，但 CV 仍会增加各折的计算量。
- `Nystroem` 将核存储降低到 $O(nm)$，另加代表节点上的线性代数运算。
- GPU 加速效果取决于样本量、数据类型、核函数、同步开销和可用显存；小问题可能 CPU 更快。

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
必要条件，不能单独证明模型条件良好或适合分析目的。

### RBF 坐标偏移过大时

RBF 核遇到很大的共同坐标偏移时，请用训练数据确定一个偏移量，并同时从训练和查询特征中减去它。当前平方距离计算可能丢失相近点的差异：`1e9` 附近的坐标可能生成几乎全为 1 的核矩阵，显著改变预测。中心化保持预期的 RBF 核不变，但并非所有核的通用预处理规则，尤其不能直接套用于卡方核。

### 其他使用注意事项

核矩阵病态时，可尝试增大 `alpha` 或调整核尺度；RBF 等距离核尤其需要检查
`gamma` 是否合适。自定义核还须满足所选估计器对后端、数据类型、矩阵形状和
对称性的要求。大样本的内存成本与交叉验证开销见上文的复杂度说明。

<a id="验证方法"></a>

## 与其他实现核对结果

核岭回归可直接验证正则化线性系统，或与 scikit-learn 对照，但须使用相同核、alpha、数据及不加权目标。交叉验证应检查各候选 MSE 是否有限，并确认最终重拟合沿用选定 alpha。比较 Nystroem 归一化矩阵或特征时，应对齐代表节点和 SVD 奇异值下限。CPU 检查不等于 GPU 精度或性能证据，应在实际使用的设备上验证。

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
