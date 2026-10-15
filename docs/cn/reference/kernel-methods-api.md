# 核方法 API

> 语言：中文  
> 最后更新：2026-10-13  
> 切换：[English](../../en/reference/kernel-methods-api.md)

本页列出 `statgpu.nonparametric.kernel_methods` 中四个估计器和八个成对核函数
的完整签名、参数、方法、拟合属性及执行限制。选择方法可先阅读
[核方法总览](../models/nonparametric/kernel-methods.md)；完整示例分别见
[KernelRidge](../models/nonparametric/kernel-ridge.md)、[KernelRidgeCV](../models/nonparametric/kernel-ridge-cv.md)、
[KernelPCA](../models/nonparametric/kernel-pca.md) 和 [Nystroem](../models/nonparametric/kernel-nystroem.md)。

<a id="complete-estimator-api"></a>

## 完整估计器 API

```text
KernelRidge(alpha=1.0, kernel='rbf', gamma=None, degree=3, coef0=1, kernel_params=None, device='auto', n_jobs=None)
KernelRidgeCV(alphas=None, cv=5, kernel='rbf', gamma=None, degree=3, coef0=1, kernel_params=None, random_state=None, device='auto', n_jobs=None)
KernelPCA(n_components=2, kernel='rbf', gamma=None, degree=3, coef0=1, alpha=1.0, eigen_solver='auto', device='auto', n_jobs=None)
Nystroem(kernel='rbf', n_components=100, gamma=None, degree=3, coef0=1, random_state=None, device='auto', n_jobs=None)
```

这里用字符串 `'auto'` 表示运行时默认值 `Device.AUTO`。四个类均接受字符串或 `statgpu.Device` 枚举值。标准导入方式为：

```python
from statgpu.nonparametric.kernel_methods import (
    KernelRidge, KernelRidgeCV, KernelPCA, Nystroem,
)
```

### 共有构造参数

| 参数 | 含义与限制 |
|---|---|
| `kernel='rbf'` | 内置核名称，或返回完整成对矩阵的可调用函数。参见[成对核分派](#pairwise-kernels)和[参数转发规则](#kernel-parameter-forwarding)。没有 `precomputed` 核模式。 |
| `gamma=None` | 核系数。RBF、Laplacian、多项式和 sigmoid 核通常在 `None` 时使用 `1 / n_features`；卡方核默认 `1.0`。估计器的转发规则有差异，尤其要注意卡方核和自定义核。 |
| `degree=3` | 多项式次数。应使用有限、有效的核参数；通用接口并未完整检查所有核参数。 |
| `coef0=1` | 多项式和 sigmoid 核的加法常数。 |
| `device='auto'` | `'cpu'` 为 NumPy，`'cuda'` 为 CuPy CUDA，`'torch'` 为 Torch CUDA。`'auto'` 遵循全局设备设置；全局也为自动时，依次优先 CuPy、Torch CUDA、NumPy。显式请求的设备不可用会报错。KernelPCA/Nystroem 还存在重要的 [CPU 执行与设备放置边界](#backend-and-execution-boundaries)。 |
| `n_jobs=None` | 共享估计器选项；这些估计器不使用它并行拟合。 |

### 共有输入、输出与生命周期约定

- `X` 应为有限数值矩阵 `(n,p)`，训练行数和特征数均非零。一维 `X` 会转成 `(n,1)`，表示一个特征的多行观测，而不是一个多维查询点。查询矩阵为 `(q,p)`，须保持训练特征顺序。
- 估计器将数值输入转为 float64。自定义核还须保持所需的后端、数据类型、形状和设备兼容性。这与底层函数的[数据类型规则](#pairwise-inputs-and-dtypes)不同。
- 监督学习的 `y` 为 `(n,)` 或 `(n,r)`，向量内部按单列保存。单响应岭回归的预测为 `(q,)`，即使训练时 `y` 为 `(n,1)`；多响应为 `(q,r)`。
- `fit` 返回 `self`。未拟合就调用预测或变换会抛出 `RuntimeError`。数值输入检查会拒绝 NaN/Inf，但这**不保证**计算得到的核矩阵、拟合数组或所有标量核参数有效。
- 四个类均有 `get_params(deep=True)` 和 `set_params(**params)`。前者返回构造配置，并按需包含嵌套设置；后者返回 `self`。未知构造参数名抛出 `ValueError`。修改参数后须重新拟合。KernelRidge/CV 使用基类的重建路径；KernelPCA/Nystroem 对自身参数的更新可能保留旧拟合数组，不可依赖自动失效处理。
- 四个类均继承 `adjust_pvalues`、`combine_pvalues`、`bootstrap_statistic` 和 `permutation_test`，完整签名、限制与返回值见[共享估计器 API](estimator-api.md)。这些工具不提供核系数推断，也不会自动建立适用于该模型的有效重采样过程。

<a id="kernelridge"></a>

### KernelRidge

[教程：拟合并评估非线性响应预测](../models/nonparametric/kernel-ridge.md)。

除共有参数外，`alpha=1.0` 是有限非负标量，`kernel_params=None` 是可选的核关键字参数字典。求解的正则化系统为

$$
(K+\alpha I)c=y.
$$

对于半正定核，它对应最小化 $\lVert y-Kc\rVert_2^2+\alpha c^\top Kc$：损失是残差**平方和**，不除以样本量，也不是惩罚 $\lVert c\rVert^2$。当 $K$ 正定时，一阶条件给出该系统；当 $K$ 半正定但奇异且 alpha 为正时，该系统选定一组表示最优拟合函数的系数。不定核没有这一凸 RKHS 解释。模型不单独拟合截距；预测为 $K(X_{query},X_{train})c$。

| 方法 | 返回值与限制 |
|---|---|
| `fit(X, y, sample_weight=None)` | 返回 `self`。拟合**不加权**目标，当前忽略已接受的 `sample_weight`，请省略。删除零权重行可排除观测，但不能实现任意加权拟合。 |
| `predict(X)` | 后端响应数组 `(q,)` 或 `(q,r)`。 |
| `score(X, y)` | Python 浮点数：各响应列 R² 的等权平均。向量转列后，响应形状须与预测一致。常数响应的残差平方和不超过 `1e-15` 时取 1，否则取 0。 |
| `get_params(deep=True)`、`set_params(**params)` | 参数管理规则见上文。 |

| 拟合属性 | 形状与含义 |
|---|---|
| `X_fit_` | 后端 float64 数组 `(n,p)`，预测时使用的训练设计。 |
| `dual_coef_` | 后端数组 `(n,r)`，单响应也保存为 `(n,1)`。这是核系数，不是原始特征斜率。 |
| `n_features_in_` | 输入特征数，整数。 |

直接求解若抛出线性代数异常，会逐步增大对角扰动后重试。稳定化仍失败时抛出 `ValueError`，提示增大 alpha。稳定化后的系数不一定精确满足原系统；建议选择与尺度匹配的正 alpha。未触发异常的非有限结果可能绕过此机制，因此还须进行[结果有限性检查](#check-for-nonfinite-kernel-fits)。

<a id="kernelridgecv"></a>

### KernelRidgeCV

[教程：在训练折内选择 alpha](../models/nonparametric/kernel-ridge-cv.md)。

| 额外参数 | 含义与限制 |
|---|---|
| `alphas=None` | 有限、非空、非负的候选数组；传入值会展平。建议使用严格正值。`None` 根据完整训练核的特征值构造 100 个正的对数等距候选值。 |
| `cv=5` | 满足 `2 <= cv <= n_samples` 的整数，不接受布尔值。打乱行后做 K 折划分；不接受自定义、分组或按时间顺序的划分器。 |
| `kernel_params=None` | 可选核参数字典，转发规则同 KernelRidge。 |
| `random_state=None` | NumPy `RandomState` 打乱行顺序所用的整数种子或 `None`。 |

| 方法 | 返回值与限制 |
|---|---|
| `fit(X, y)` | 返回 `self`；搜索 alpha，然后在全部输入训练行上重新拟合最终 KernelRidge。没有样本权重参数。 |
| `predict(X)` | 委托给 `estimator_`，返回后端 `(q,)` 或 `(q,r)` 预测。 |
| `score(X, y)` | 委托给 `estimator_`，返回上述 Python 浮点数 R²。 |
| `get_params(deep=True)`、`set_params(**params)` | 参数管理规则见上文。 |

选择标准为验证 MSE，对**各折和各响应列等权平均**，所有响应共用一个选中的 alpha。不调整核类型、gamma 或预处理。`best_score_` 是选中 alpha 对应的平均折内 R²，与用于选择的损失不同，也不是独立测试集分数。常数响应的折内 R² 在该折 MSE 不超过 `1e-15` 时取 1，否则取 0。

完整训练核只构造一次，每折的对称特征分解在各候选 alpha 间复用。自动网格根据完整训练核的谱调整，并且只包含正值；如需在不同数据集上保持固定、可复现的搜索范围，请显式传入 `alphas`。

| 拟合属性 | 形状与含义 |
|---|---|
| `alpha_` | Python 浮点数，选中的正则化强度。 |
| `best_score_` | Python 浮点数，选中 alpha 对应的平均折内 R²，再对响应列等权平均。 |
| `cv_results_` | 详见下表的字典，结果数组为 NumPy。 |
| `estimator_` | 最终拟合的 KernelRidge，沿用选中 alpha、核参数和设备。特征数读取 `estimator_.n_features_in_`。 |
| `dual_coef_`、`X_fit_` | 最终估计器后端数组的快捷引用，形状同 KernelRidge。 |

记候选数为 `a`、折数为 `f`、响应列数为 `r`，`cv_results_` 包含：

| 键 | 值 |
|---|---|
| `alphas` | 候选数组 `(a,)`。 |
| `mse_table`、`r2_table` | 各候选、各折、各响应的数组 `(a,f,r)`。 |
| `mean_mse`、`mean_r2` | 各折均值 `(a,r)`；向量响应也保留单列。 |
| `best_alpha`、`best_score` | 与 `alpha_`、`best_score_` 相同的值。 |

奇异折内核遇到显式的零 alpha 时，可能产生非有限 CV 分数，且该候选仍可能被选中。CV 中的特征值除法不使用 KernelRidge 的对角扰动重试。解释结果前，须确认 `cv_results_["mean_mse"]` 与 `best_score_` 有限；仅返回了估计器不代表搜索有效。正候选值可降低奇异风险，但不能修复非有限或尺度不合适的核。

如需分组/时间划分，或在每折内学习预处理，请使用外部划分循环，在各训练折内分别拟合预处理和 KernelRidge。不要先用全部数据学习预处理，再交给此内部 CV；最终评估数据应始终留在选择过程之外。

<a id="kernelpca"></a>

### KernelPCA

[教程：学习非线性坐标](../models/nonparametric/kernel-pca.md)。

| 额外参数 | 含义与限制 |
|---|---|
| `n_components=2` | 正整数，最多保留的正方向数；实际输出列数可能更少。 |
| `alpha=1.0` | 有限非负对角平移量，在特征分解前加上，报告特征值时减去。**不是**逆变换正则化参数。 |
| `eigen_solver='auto'` | `'auto'` 或 `'dense'`；当前均执行稠密对称特征分解。 |

| 方法 | 返回值与限制 |
|---|---|
| `fit(X, y=None)` | 返回 `self`，不使用 `y`。 |
| `transform(X)` | 后端坐标数组 `(q,k)`，`k <= min(n_components,n)` 为保留的正方向数。另见下方 Torch 放置限制。 |
| `fit_transform(X, y=None)` | 拟合后，对 `X` 使用同一变换和中心化路径，返回 `(n,k)`。 |
| `predict(X)` | `transform` 的别名，不预测响应。 |
| `get_params(deep=True)`、`set_params(**params)` | 参数管理规则见上文。 |

没有模型专属的 `score` 或 `inverse_transform` 方法。训练先构造中心化核 $\widetilde K$，分解 $\widetilde K+\alpha I$，再从特征值中减去 alpha，最多保留 `n_components` 个**严格大于 `1e-12`** 的特征值。没有正方向时抛出 `ValueError`。不定核的负方向会被丢弃；减去平移量后，alpha 不会将负方向变成有效的正成分。

| 拟合属性 | 形状与含义 |
|---|---|
| `lambdas_` | NumPy `(k,)`，未平移的中心化核所保留的正特征值。 |
| `alphas_` | NumPy `(n,k)`，特征向量除以对应特征值平方根；缩放后通常不是单位范数特征向量。 |
| `X_fit_` | NumPy `(n,p)`，即使选择 GPU 后端，训练数据也按此形式保存。 |
| `n_samples_`、`n_features_in_` | 训练行数和特征数，整数。 |

查询中心化使用**训练核**的列均值和总均值，以及各查询行的交叉核均值；不能独立中心化查询核。坐标等于中心化后的查询到训练核乘以 `alphas_`。特征向量符号以及重根或近重根子空间内的基不唯一，必要时应比较子空间或下游量。

如果 `fit` 或 `fit_transform` 抛出异常，应丢弃该实例，在新实例上成功拟合合适的数据。失败的重新拟合可能混用新中心化信息与旧特征向量，返回有限但错误的坐标。详见[失败重拟合处理](#after-a-failed-refit)和[后端边界](#backend-and-execution-boundaries)。

<a id="nystroem"></a>

### Nystroem

[教程：构造可复用的近似核特征](../models/nonparametric/kernel-nystroem.md)。

| 额外参数 | 含义与限制 |
|---|---|
| `n_components=100` | 正整数，请求的代表节点数；实际为 `m=min(n_components,n_training_rows)`。 |
| `random_state=None` | NumPy `RandomState` 不放回抽样所用的整数种子或 `None`。保留抽样顺序，不排序。 |

| 方法 | 返回值与限制 |
|---|---|
| `fit(X, y=None)` | 返回 `self`，不使用 `y`；选择代表节点并学习归一化矩阵。 |
| `transform(X)` | 后端特征数组 `(q,m)`，另见下方 Torch 放置限制。 |
| `fit_transform(X, y=None)` | 拟合并变换训练行，返回 `(n,m)`。 |
| `predict(X)` | `transform` 的别名，不预测响应。 |
| `get_params(deep=True)`、`set_params(**params)` | 参数管理规则见上文。 |

没有模型专属的 `score` 或 `inverse_transform` 方法。CPU 上的代表节点核若有 SVD $K_{mm}=U\Sigma V^\top$，实现使用

$$
N=U\max(\Sigma,10^{-12}I)^{-1/2}V^\top,\qquad Z=K_{qm}N.
$$

对于半正定代表节点核，这一归一化采用对称逆平方根来构造特征。全部 `m` 列均保留，较小奇异值是**截断到下限，而非删除**。不定核也执行 SVD，但不因此得到不定相似度的精确欧氏特征表示。

| 拟合属性 | 形状与含义 |
|---|---|
| `components_` | NumPy `(m,p)`，选中的代表节点。 |
| `component_indices_` | NumPy 整数数组 `(m,)`，节点在训练行中的索引。 |
| `normalization_` | NumPy `(m,m)`，SVD 归一化矩阵 `N`。 |
| `eigenvalues_` | NumPy `(m,)`，截断到 `1e-12` 下限的奇异值。虽然沿用此名称，但**不是带符号的特征值**。 |
| `n_features_in_` | 输入特征数，整数。 |

代表节点核的构造与 SVD 始终在 CPU 上进行，因此自定义核既须接受 NumPy 代表节点输入，也须支持选中的查询后端。特征映射只能从训练数据拟合；选择下游预测器时，须在每个训练折内重拟合映射。比较 `Z @ Z.T` 和精确核的诊断需要构造稠密对照核，仅适合小数据；更低的核近似误差不一定意味着更好的响应预测。

<a id="kernel-parameter-forwarding"></a>

## 核参数转发规则

各估计器对构造参数的转发并不相同：

| 估计器 | 转发规则 |
|---|---|
| KernelRidge / KernelRidgeCV | 从 `kernel_params` 的副本或 `{}` 开始。对规范化后的 `rbf`、`gaussian`、`polynomial`、`poly`、`laplacian`、`sigmoid`，非 `None` 的构造 `gamma` 覆盖字典值。对多项式别名，非默认 `degree != 3` 覆盖字典 `degree`；对多项式别名和 sigmoid，非默认 `coef0 != 1` 覆盖字典 `coef0`。构造默认值不会覆盖对应字典值。 |
| KernelRidge / KernelRidgeCV 使用卡方核 | 当前忽略构造 `gamma`，须用 `kernel_params={"gamma": value}`；否则使用函数默认值 `1.0`。 |
| KernelRidge / KernelRidgeCV 使用可调用核 | 不自动传入构造 `gamma`、`degree` 和 `coef0`。自定义核参数须放入 `kernel_params`。 |
| KernelPCA / Nystroem | 没有 `kernel_params` 参数。任何非 `None` 的 `gamma` 都会转发，包括卡方核和自定义核。精确匹配小写 `poly`/`polynomial` 时，转发 `degree` 和 `coef0`，包括默认值；精确匹配小写 `sigmoid` 时，转发 `coef0`。 |

成对核分派会规范化名称，但 KernelPCA/Nystroem 选择 degree/coef0 的逻辑不会。配置时请用规范小写名称；例如 `'Polynomial'` 虽可成功分派，却可能遗漏你设置的 degree/coef0。它们的 linear/cosine 核应保持 `gamma=None`；非 `None` 的 gamma 会被转发，而相应函数不接受此参数，因此抛出 `TypeError`。字典里的额外关键字也不会按内置函数签名过滤，可能同样引起 `TypeError`。

自定义核须返回预期后端上的完整 `(n,q)` 矩阵，保持数据类型和设备兼容，并满足所选方法所需的对称性、半正定性。自定义函数自身抛出的错误会继续向上传递。每次调用应返回两组观测之间的完整核矩阵；某些库则逐对调用自定义核，要求每次返回一个标量。

<a id="complete-pairwise-function-api"></a>

## 完整成对核函数 API

以下八个函数均从 `statgpu.nonparametric.kernel_methods` 公开导出：

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

<a id="pairwise-inputs-and-dtypes"></a>

### 输入、形状、数据类型与设备

对内置核，请准备兼容、有限的二维后端数组 `X` `(n,p)` 和 `Y` `(q,p)`，特征数须相同。`Y=None` 使用 `X`。返回值为 `(n,q)`，省略 `Y` 时为 `(n,n)`。与估计器不同，这些函数不将向量自动转为列，通常也不接受嵌套列表，请先转换。NumPy 卡方核路径还接受可转换为二维数组的输入。

对内置核，`xp=None` 使用 NumPy，**不会**自动识别输入后端。应将 `xp` 设为对应的 NumPy、CuPy 或 Torch 模块，并自行准备位于所需设备上的兼容数组。仅传 `xp` 不会把数组移到 GPU，也不会统一为 float64。这些底层函数没有统一的数据类型转换和有限值检查规则：

- NumPy RBF 使用 `np.result_type(X.dtype, Y.dtype, np.float32)`，在该提升规则下保留显式 float32/float64。Torch/CuPy RBF 保留兼容的浮点输入类型；只要一个输入不是浮点数，两者都提升为 float64。RBF 拒绝复数输入。
- NumPy Laplacian 使用 SciPy `cdist`，返回其浮点距离数组的数据类型，通常为 float64。
- NumPy 卡方核优先使用 scikit-learn；未安装时使用分块 NumPy 计算，累加精度至少为 float64。不能假定所有可选依赖和后端路径返回完全相同的数据类型。
- 其他核遵循后端运算及其类型提升规则。整数点积或幂运算可能在后续浮点计算前已溢出，应使用合适的浮点数组。Torch 矩阵操作数须具有兼容的数据类型和设备。

<a id="pairwise-kernels"></a>

### pairwise_kernels

`pairwise_kernels(X, Y=None, metric='rbf', xp=None, **params)` 分派到指定函数，返回完整成对矩阵。内置名称会去除首尾空格并转为小写。规范名称为 `rbf`、`polynomial`、`linear`、`laplacian`、`sigmoid`、`cosine`、`chi2`；相应别名为 `gaussian`、`poly`、`chi-squared`。未知名称（包括 `precomputed`）抛出 `ValueError`。`**params` 原样转发，不自动过滤。

当 `metric` 可调用时，原始 `X`、`Y` 原样传入，包括 `Y=None`。签名若接受 `xp` 或 `**kwargs`，则原样传入所给 `xp`，包括 `None`；若无法检查签名，也会尝试传入 `xp`，否则省略该关键字。自定义函数须自行决定默认模块、输入转换和省略 `Y` 时的处理，并返回完整矩阵。函数内部抛出错误时，不会再用另一种签名重试。

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

此 NumPy 示例同时处理省略 `xp` 和显式传入 `xp=np` 的情况。自定义 GPU 核
还须明确处理数据类型和数组所在设备。

<a id="rbf-kernel"></a>

### rbf_kernel

`rbf_kernel(X, Y=None, gamma=None, xp=None)` 为各样本对返回 $\exp(-\gamma\lVert x-y\rVert_2^2)$。`gamma=None` 使用 `1/p`。函数检查形状兼容性并拒绝复数输入，但未完整验证 gamma，应提供有限且有意义的系数。

平方距离由平方范数减去两倍点积计算，舍入产生的负值截断为零。很大的共同坐标偏移会使相近点的差异丢失：`1e9` 附近的坐标可能生成近乎全为 1 的核矩阵，显著改变预测。必要时，用**训练数据确定**偏移量，同时从训练和查询特征减去它。平移保持预期的 RBF 核不变，但不是所有核通用的预处理规则，尤其不能直接套用于卡方核。NumPy 按需分块计算，但仍返回完整稠密矩阵。

<a id="polynomial-kernel"></a>

### polynomial_kernel

`polynomial_kernel(X, Y=None, degree=3, gamma=None, coef0=1, xp=None)` 返回 $(\gamma x^\top y+\mathrm{coef0})^{\mathrm{degree}}$。`gamma=None` 使用 `1/p`。很大的有限输入、尺度或次数可导致溢出，无效参数也可能产生 NaN，却不抛出明确异常。缩放会改变此相似度，应在训练过程中选择。另见[结果有限性检查](#check-for-nonfinite-kernel-fits)。

<a id="linear-kernel"></a>

### linear_kernel

`linear_kernel(X, Y=None, xp=None)` 返回 $x^\top y$。不接受 gamma、degree 或 coef0，也不自动中心化或添加截距。矩阵乘法直接使用输入数组的运算规则，`xp` 不会转换输入。

<a id="laplacian-kernel"></a>

### laplacian_kernel

`laplacian_kernel(X, Y=None, gamma=None, xp=None)` 返回 $\exp(-\gamma\lVert x-y\rVert_1)$，`gamma=None` 使用 `1/p`。NumPy 使用 SciPy 的曼哈顿距离；CuPy/Torch 广播构造 `(n,q,p)` 差分数组，临时内存可能明显大于最终核矩阵。

<a id="sigmoid-kernel"></a>

### sigmoid_kernel

`sigmoid_kernel(X, Y=None, gamma=None, coef0=1, xp=None)` 返回 $\tanh(\gamma x^\top y+\mathrm{coef0})$，`gamma=None` 使用 `1/p`。不接受 degree。任意参数和数据下的 sigmoid 核不一定半正定，仍须注意岭回归/Nystroem 的半正定解释和 KernelPCA 只保留正方向的规则。

<a id="cosine-kernel"></a>

### cosine_kernel

`cosine_kernel(X, Y=None, xp=None)` 返回 $x^\top y/(\lVert x\rVert\lVert y\rVert+10^{-10})$。不接受 gamma、degree 或 coef0。固定分母增量使涉及零向量的结果为零，也使极小范数输入不同于精确余弦归一化。

<a id="chi2-kernel"></a>

### chi2_kernel

`chi2_kernel(X, Y=None, gamma=1.0, xp=None)` 返回

$$
\exp\left\{-\gamma\sum_j\frac{(x_j-y_j)^2}{x_j+y_j}\right\}.
$$

这是指数化卡方核，不是加性卡方核。要求输入特征非负、二维形状匹配、gamma 有限且非负，违反时抛出 `ValueError`。默认 gamma 为 `1.0`，不是 `1/p`。

NumPy 将零除零特征项视为零。CuPy/Torch 对每个分母使用 `1e-10` 下限，因此极小非负特征的结果可能不同，并使用 `(n,q,p)` 广播临时数组。不要把直方图特征中心化为负数。KernelRidge/CV 修改 gamma 时须通过 `kernel_params`，详见上文。

<a id="backend-and-execution-boundaries"></a>

## 后端与执行边界

四个估计器均接受 `device`，但核构造、分解、拟合数组和输出未必始终位于同一设备：

| 估计器 | 执行与存储 |
|---|---|
| KernelRidge / KernelRidgeCV | 在所选后端构造核并求解。岭回归拟合数组和预测为后端数组；CV 结果表复制为 NumPy。 |
| KernelPCA | 通常在所选数组后端分解。线性代数异常会触发 NumPy CPU 特征分解，**没有单独的回退标记**。训练数据、特征值和投影系数保存为 NumPy。 |
| Nystroem | 把选中的代表节点复制为 NumPy，在 CPU 上构造节点核并执行 SVD，再使用所选数组库计算查询到节点的核与返回特征，另受下述放置限制。公开拟合数组保留为 NumPy，自定义核须接受 NumPy 节点输入。 |

显式请求不可用的设备会报错。Torch CUDA 不可用时，KernelPCA 和 Nystroem 仍拒绝 `device="torch"`；但通过可用性检查不保证 CUDA 执行：请求设备的转换并未始终执行，NumPy 或 Torch CPU 输入可能在拟合与变换时继续留在 CPU。此限制也影响 Nystroem 的查询核与输出，与有意安排在 CPU 的节点 SVD 是两回事。

请检查实际 `fit_transform`、`transform` 和 `predict` 输出的 Torch `.device`/`.is_cuda`，或 CuPy `.device`。配置的设备和所选数组库不能证明实际放置，公开的 NumPy 拟合数组也不能证明计算发生在哪个设备。必须使用 CUDA 时，应在使用前拒绝 CPU 输出。需要明确 CPU 路径时，请传 NumPy 输入并设置 `device="cpu"`。另见[设备说明](../guides/device-and-memory.md#current-smoothing-and-spline-exceptions)。KernelPCA 和 Nystroem 均不承诺全程 GPU 执行。

<a id="numerical-failures"></a>
<a id="limitations-and-failure-modes"></a>
<a id="限制与失败行为"></a>

## 数值问题与失败处理

<a id="after-a-failed-refit"></a>

### 重新拟合失败后

如果 `KernelPCA.fit` 或 `fit_transform` 抛出异常，**丢弃该实例**。失败重拟合可能已经改写训练核中心化信息，却保留旧特征向量和已拟合标记。此后的 `transform` 或 `predict` 即使查询原训练数据，也可能返回有限但错误的坐标。有限值检查无法识别这种混合状态。

有限的常数数据就可能触发失败，因为中心化核没有正方向。新实例也会拒绝同样的退化数据；应重新考虑数据或分析方法，在新模型上成功拟合合适的数据后再变换查询点。

<a id="check-for-nonfinite-kernel-fits"></a>

### 检查核拟合结果是否有限

有限输入不保证计算后的核有限。多项式核可能溢出，当前 KernelRidge 和 Nystroem 可能正常返回，但拟合数组以及预测或特征中含有 NaN。NaN gamma 等无效标量核参数也可产生非有限结果。有无警告、是否抛出异常都不是可靠的拟合状态检查。

应使用有限、适用的核参数，使用结果前检查拟合数组和查询输出。NumPy CPU 上，KernelRidge 检查 `np.isfinite(model.dual_coef_).all()`；Nystroem 检查 `np.isfinite(model.normalization_).all()` 和 `np.isfinite(model.eigenvalues_).all()`。预测和变换特征也须检查，设备数组使用对应后端检查。CV 还须确认 `cv_results_["mean_mse"]` 与 `best_score_` 有限。

丢弃非有限拟合，检查特征单位、尺度和核参数，再用新实例拟合。增大 alpha 不能修复已经溢出的核。缩放会改变多项式相似度，因此预处理应在训练折内选择与验证。数值有限是必要条件，不足以证明模型条件良好或适用。对于有限但病态的核，可考虑更大的正 alpha 或更合适的核尺度；两者都不能代替修复 [RBF 坐标偏移问题](#rbf-kernel)。

<a id="inference-semantics"></a>
<a id="推断语义"></a>

## 推断与验证边界

这些估计器不提供系数级标准误、假设检验或置信区间，也没有严格/近似推断模式。Nystroem 是显式低秩核近似，不是精确估计器的静默回退。模型质量应通过预测分数、CV 损失、嵌入性质、近似诊断及应用相关验证评估。

核岭回归可直接验证正则化线性系统，或与 scikit-learn 对照，但须对齐核、alpha、数据和**不加权平方和**目标。对角扰动后的解可能不同于原系统。CV 应检查各候选 MSE 有限，并确认最终重拟合采用选中的 alpha。Nystroem 对照应对齐节点、节点顺序和奇异值下限；KernelPCA 若涉及符号或基不唯一，应比较子空间。CPU 检查不能证明 GPU 精度或速度，须在实际使用的设备上验证。

## 复杂度

精确核方法构造 `n x n` 训练核，内存为二次量级；直接求解或稠密特征分解最坏为三次算术复杂度。KernelRidgeCV 在 alpha 间复用各折分解，但增加折内计算及候选相关临时数组。Nystroem 用 `n x m` 特征替代主要表示，另有 `m x m` 节点线性代数。GPU 速度取决于问题规模、数据类型、核、同步、传输和可用内存，小问题可能 CPU 更快。设置 `device` 本身不代表加速。

## 实现与统计参考

实现：[KernelRidge](../../../statgpu/nonparametric/kernel_methods/_krr.py)、[KernelRidgeCV](../../../statgpu/nonparametric/kernel_methods/_krr_cv.py)、[KernelPCA](../../../statgpu/nonparametric/kernel_methods/_kpca.py)、[Nystroem](../../../statgpu/nonparametric/kernel_methods/_nystroem.py)、[成对核函数](../../../statgpu/nonparametric/kernel_methods/_kernels.py)和[共享估计器行为](../../../statgpu/_base.py)。

- Schölkopf, B., Smola, A., & Müller, K.-R. (1998). Nonlinear component analysis as a kernel eigenvalue problem.
- Williams, C. K. I., & Seeger, M. (2001). Using the Nystroem method to speed up kernel machines.
- Shawe-Taylor, J., & Cristianini, N. (2004). *Kernel Methods for Pattern Analysis*.
