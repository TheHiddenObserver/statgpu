# 协方差 API 参考

> 语言：中文
> 最后更新：2026-10-12
> 切换：[English](../../en/reference/covariance-api.md)

本页完整列出协方差估计器的公开 API。选择方法时，可先阅读[协方差概览](../models/covariance.md)。下列七个类均可从 `statgpu.covariance` 或 `statgpu` 导入。

| 类 | 方法指南 | 专属构造参数 |
|---|---|---|
| `EmpiricalCovariance` | [经验协方差](../models/covariance-empirical.md) | 无 |
| `LedoitWolf` | [Ledoit–Wolf 收缩](../models/covariance-ledoit-wolf.md) | 无 |
| `OAS` | [Oracle Approximating Shrinkage](../models/covariance-oas.md) | 无 |
| `ShrunkCovariance` | [固定强度收缩](../models/covariance-shrunk.md) | `shrinkage` |
| `MinCovDet` | [最小协方差行列式](../models/covariance-min-cov-det.md) | `support_fraction`、`random_state` |
| `GraphicalLasso` | [Graphical Lasso](../models/covariance-graphical-lasso.md) | `alpha`、`max_iter`、`tol` |
| `GraphicalLassoCV` | [交叉验证 Graphical Lasso](../models/covariance-graphical-lasso-cv.md) | `alphas`、`cv`、`max_iter`、`tol`、`random_state` |

<a id="constructors-and-common-parameters"></a>

## 构造函数与共享参数

以下签名按顺序列出全部构造参数。实际默认值 `Device.AUTO` 是枚举值，作用与传入 `"auto"` 相同。`LedoitWolf` 和 `OAS` 继承 `EmpiricalCovariance` 的构造函数。

```python
from statgpu import Device
from statgpu.covariance import (
    EmpiricalCovariance, LedoitWolf, OAS, ShrunkCovariance,
    MinCovDet, GraphicalLasso, GraphicalLassoCV,
)

EmpiricalCovariance(assume_centered=False, device=Device.AUTO, n_jobs=None)
LedoitWolf(assume_centered=False, device=Device.AUTO, n_jobs=None)
OAS(assume_centered=False, device=Device.AUTO, n_jobs=None)
ShrunkCovariance(shrinkage=0.1, assume_centered=False, device=Device.AUTO, n_jobs=None)
MinCovDet(support_fraction=None, random_state=None, assume_centered=False,
          device=Device.AUTO, n_jobs=None)
GraphicalLasso(alpha=0.01, max_iter=100, tol=1e-4, assume_centered=False,
               device=Device.AUTO, n_jobs=None)
GraphicalLassoCV(alphas=4, cv=5, max_iter=100, tol=1e-4, random_state=None,
                 assume_centered=False, device=Device.AUTO, n_jobs=None)
```

| 参数 | 类型与默认值 | 行为 |
|---|---|---|
| `assume_centered` | bool，`False` | 默认估计位置并在计算协方差前减去它。`True` 将位置固定为零，直接使用输入值，但不检查数据均值是否为零。对于 `MinCovDet`，子集位置和重加权后的位置也遵循此设置。 |
| `device` | str 或 `Device`，`Device.AUTO` | 支持 `"cpu"`、`"cuda"`、`"torch"`、`"auto"`；见[后端与数据类型规则](#device-and-dtype)。 |
| `n_jobs` | int 或 `None`，`None` | 接受此参数，`get_params` 也会返回它，但目前仅为预留参数，不用于并行拟合或并行执行交叉验证折。 |

这七个类**没有** `gpu_memory_cleanup` 构造参数或公开控制项。向构造函数传入它会抛出 `TypeError`，向 `set_params` 传入它会抛出 `ValueError`；不能根据其他 statgpu 估计器推断这里也支持它。这些类也不接受 `sample_weight`、公式、`store_precision` 或预计算协方差选项，`fit` 接收的是观测数据。

<a id="inputs-and-public-methods"></a>

## 输入与公开方法

<a id="data-shape-and-validation"></a>

### 数据形状与验证

输入应为有限、实值、稠密的数值数据。支持 NumPy 数组、CuPy 数组、受支持的稠密 Torch 张量、数值型 pandas `DataFrame`/`Series`，以及嵌套列表等普通类数组输入。pandas 可空数值类型在没有缺失值时也可使用。每行表示一个观测，每列表示一个特征。DataFrame 标签不用于特征对齐、重排或记录；查询数据必须保持训练时的列顺序。

| 调用 | 二维输入 | 一维输入 |
|---|---|---|
| `fit(X, y=None)` | `(n, p)`，`n >= 2`、`p >= 1` | `(n,)` 转为 `(n, 1)`，即单特征的 n 个观测 |
| `score(X, y=None)` | `(m, p)`，`m >= 1`，p 须与训练时一致 | `(m,)` 转为 `(m, 1)`，仅适用于单特征模型 |
| `mahalanobis(X)` | `(m, p)`，`m >= 1`，p 须与训练时一致 | `(p,)` 转为 `(1, p)`，即一个观测 |
| `predict(X)` | 与 `mahalanobis` 相同 | 与 `mahalanobis` 相同 |

查询一个观测时，建议明确传入一行，例如 `X_query.reshape(1, -1)`。查询单特征的多个观测时，使用 `X_query.reshape(-1, 1)`，尤其是在调用 `predict` 或 `mahalanobis` 时。标量、空数组、超过二维的数据以及特征数不符的查询会在验证时失败。`MinCovDet` 还要求 `n > p`；Graphical Lasso 交叉验证的每个训练折至少需要两行。

NaN、无穷值和 pandas 缺失值会在估计前引发 `ValueError`。虽然 `y` 不参与计算，显式传入的 `y` 仍会接受公开方法的有限值检查，其中的 NaN/Inf 也可能引发错误；请保留默认的 `None`。稀疏、量化、复数和非数值输入不属于本稠密实值 API 的支持范围，不应依赖隐式转换或假设它们都会抛出同一种异常。这些方法不自动插补缺失值、编码类别或标准化特征。

<a id="fit-score-predict-and-mahalanobis"></a>

### fit、score、predict 和 mahalanobis

七个类均提供以上准确签名对应的方法。`fit` 返回 `self`；其余三个方法要求对象已成功拟合，否则抛出 `RuntimeError`。如果输入和对象状态同时无效，公开输入检查可能先报错。

| 方法 | 返回值 | 定义 |
|---|---|---|
| `fit(X, y=None)` | 当前估计器对象 | 设置[共享拟合属性](#common-fitted-attributes)及类专属拟合属性。 |
| `score(X, y=None)` | Python `float` | 根据拟合的位置、协方差行列式和精度矩阵，计算查询观测的平均高斯对数似然。在相同查询数据上，值越大越好。 |
| `mahalanobis(X)` | NumPy `ndarray`，`float64`，形状 `(m,)` | 每行查询观测的马氏距离平方。 |
| `predict(X)` | NumPy `ndarray`，`float64`，形状 `(m,)` | 与 `mahalanobis(X)` 完全相同，包括一维输入的解释方式；不输出标签，也不应用阈值。 |

以 $\mu$ 表示拟合位置、P 表示精度矩阵，则

$$
d_i^2=(x_i-\mu)^\top P(x_i-\mu),\qquad
\operatorname{score}(X)=-\frac12\left[
p\log(2\pi)+\log\det(\hat\Sigma)+\frac1m\sum_{i=1}^m d_i^2
\right].
$$

`score` 是平均值，不是总和、准确率或决定系数，也不会改用查询数据自身的均值进行中心化。距离输出为平方值；需要通常的马氏距离时，请自行开平方。

多数类先直接求逆，仅在求逆失败或结果含非有限值时加入对角稳定项；此操作不改变 `covariance_`。协方差行列式的符号非正时，`score` 返回 `-inf`，即使稳定化已得到有限的 `precision_` 和距离。Graphical Lasso 另有[下文说明的伪逆特例](#graphicallasso)。数值稳定化不等于统计正则化，也不能保证奇异高斯模型具有合理解释。如果加入稳定项后仍无法得到有限逆矩阵，会抛出 `ValueError`；底层后端的其他线性代数异常也可能直接传出。

<a id="get_params-and-set_params"></a>

### get_params 和 set_params

- `get_params(deep=True)` 返回上述构造参数组成的字典，不含拟合属性。`deep=False` 不展开嵌套估计器；这七个类本身没有以估计器为值的构造参数。返回的可变对象（如 `alphas` 列表）不是独立的模型序列化副本。
- `set_params(**params)` 返回 `self`；空调用不改变对象，未知名称抛出 `ValueError`。多数统计参数在 `fit` 时才接受验证。
- 修改设置时，请用所需参数创建并拟合新估计器。不要依赖 `set_params` 清除已拟合状态，或应用修改后的 `max_iter`、`tol`、`cv`。

<a id="inherited-statistical-helpers"></a>

### 继承的统计辅助方法

其余四个公开方法是继承的通用辅助方法。完整参数、默认值、返回字段、后端行为和统计限制见[估计器共享 API](estimator-api.md#inference-helpers)：

| 方法 | 完整签名与返回值参考 |
|---|---|
| `adjust_pvalues` | [p 值校正](estimator-api.md#adjust_pvalues)，返回字典 |
| `combine_pvalues` | [p 值合并](estimator-api.md#combine_pvalues)，返回字典 |
| `bootstrap_statistic` | [标量 bootstrap](estimator-api.md#bootstrap_statistic)，返回 `BootstrapResult` |
| `permutation_test` | [置换检验](estimator-api.md#permutation_test)，返回 `PermutationTestResult` |

协方差拟合既不提供系数 p 值，也不提供这些通用方法在省略数据参数时使用的回归 X/y 缓存。因此，调用 `adjust_pvalues`/`combine_pvalues` 时须显式传入 p 值，调用 `bootstrap_statistic` 时须显式传入数组，否则即使已拟合协方差模型也会抛出 `RuntimeError`。`permutation_test` 要求显式提供统计量函数、X 和 y。这些辅助方法不会自动重新拟合协方差，也不会自动计算协方差元素的标准误。其后端规则独立于下文的已拟合模型查询规则，选择后端时请查阅相应参考。

<a id="common-fitted-attributes"></a>

## 共享拟合属性

成功拟合后，七个类均公开下列属性。其中 n、p 是训练数据的形状。“后端数组”指所选设备上的 NumPy `ndarray`、CuPy `ndarray` 或 Torch `Tensor`。

| 属性 | 类型与形状 | 含义 |
|---|---|---|
| `covariance_` | `float64` 后端数组，`(p, p)` | 估计的协方差矩阵，各估计器的具体定义见下文。 |
| `precision_` | `float64` 后端数组，`(p, p)` | 逆矩阵、稳定化后的逆矩阵，或 Graphical Lasso 特例中的伪逆。 |
| `location_` | `float64` 后端数组，`(p,)` | 拟合的位置；`assume_centered=True` 时为零。 |
| `n_samples_` | Python `int` | 训练行数，不是稳健支持集或某个交叉验证折的大小。 |
| `n_features_` | Python `int` | 训练列数。 |

这些类不提供拟合后的 `n_features_in_`、按特征名对齐、系数表或回归 p 值。重新拟合失败时，不保证清除之前成功拟合的结果；发生拟合异常后，不能把对象中仍存在的属性视为此次输入的结果。

## EmpiricalCovariance

没有共享清单之外的参数或拟合属性。对于中心化数据 $X_c$，`covariance_ = X_c.T @ X_c / n`，分母为 n 而非 n−1。`precision_` 遵循上文的求逆和稳定化规则。秩亏不一定在拟合时被拒绝，但可能使 `score(X)` 返回 `-inf`。

## LedoitWolf

没有专属构造参数。估计形式为

$$
\hat\Sigma=(1-\alpha)S+\alpha\mu I,\qquad \mu=\operatorname{tr}(S)/p,
$$

其中 S 与经验协方差使用相同的中心化方式和 `1/n` 归一化。解析计算的收缩强度被截断到 `[0, 1]`；分母非正时采用完全收缩。

| 专属属性 | 类型 | 含义 |
|---|---|---|
| `shrinkage_` | Python `float` | 根据数据选择的收缩强度，位于 `[0, 1]`。 |

收缩目标固定为缩放后的单位矩阵，不提供其他结构化目标或块大小参数。

## OAS

构造函数和收缩形式与 `LedoitWolf` 相同，但采用 Oracle Approximating Shrinkage 的强度公式。实现沿用 scikit-learn 常用约定，省略原公式的 `2/p` 修正项；解释见 [OAS 指南](../models/covariance-oas.md)。

| 专属属性 | 类型 | 含义 |
|---|---|---|
| `shrinkage_` | Python `float` | 根据数据选择、截断到 `[0, 1]` 的收缩强度；分母非正时采用完全收缩。 |

## ShrunkCovariance

| 专属参数 | 类型与默认值 | 验证与行为 |
|---|---|---|
| `shrinkage` | float，`0.1` | 上述缩放单位矩阵公式中的固定强度。非有限值或超出 `[0, 1]` 的值会在 `fit` 时引发 `ValueError`。 |

| 专属属性 | 类型 | 含义 |
|---|---|---|
| `shrinkage_` | 数值标量，通常为 `float` | 用户提供的 `shrinkage` 值，不进行优化。 |

零表示使用经验协方差，一表示使用缩放单位矩阵。采用默认的均值中心化时，所有特征均为零方差的数据在收缩后仍得到奇异的零协方差矩阵。

## MinCovDet

| 专属参数 | 类型与默认值 | 验证与行为 |
|---|---|---|
| `support_fraction` | float 或 `None`，`None` | `None` 使用 `h = min(ceil((n + p + 1) / 2), n)`。显式传入的有限比例须位于 `(0, 1]`，先计算 `ceil(fraction * n)`，再限制到 `[p + 1, n]`。无效数值比例会在拟合时引发 `ValueError`。 |
| `random_state` | int 或 `None`，`None` | NumPy 随机初始子集选择的种子。在相同数值环境下需要可复现选择时，请传整数。 |

要求 `n > p`，否则 `fit` 抛出 `ValueError`。仅满足此条件并不保证支持集满秩；无法找到行列式为正的支持集协方差时，拟合也会抛出 `ValueError`。

| 专属属性 | 类型与形状 | 含义 |
|---|---|---|
| `support_` | 布尔后端数组，`(n,)` | 重加权后的最终训练行支持集掩码；若使用下述保留原始结果的情况，则为原始选中子集的掩码。 |
| `raw_location_` | `float64` 后端数组，`(p,)` | 修正和重加权前的选中子集位置；假定输入已中心化时为零。 |
| `raw_covariance_` | `float64` 后端数组，`(p, p)` | 选中子集以 h 为分母的协方差，**尚未**进行一致性修正。 |
| `dist_` | `float64` 后端数组，`(n,)` | 训练行在最终位置及协方差/精度计算结果下的马氏距离平方。 |

原始协方差先接受高斯一致性修正，再用自由度为 p 的卡方分布第 97.5 百分位数进行重加权筛选。如果符合条件的行少于 `p + 1`，结果保留原始支持集和修正后的原始协方差；否则在重加权支持集上重新计算并修正协方差。因此，`support_fraction` 控制的是初始子集大小，不是最终 `True` 值所占比例。`support_` 也不是新观测的预测标签；评估新观测应使用距离平方。本类没有 `raw_support_` 属性或按内置阈值输出异常标签的方法。

## GraphicalLasso

| 专属参数 | 类型与默认值 | 验证与行为 |
|---|---|---|
| `alpha` | float，`0.01` | 精度矩阵非对角元素的有限非负 L1 惩罚强度；无效数值会在拟合时引发 `ValueError`。对角元素不受惩罚。 |
| `max_iter` | int，`100` | 外层迭代次数上限，必须是正整数；布尔值、非整数和非正数会在拟合时引发 `ValueError`。 |
| `tol` | float，`1e-4` | 外层迭代间协方差最大绝对变化的停止容差，须为有限正数；无效数值会在拟合时引发 `ValueError`。 |

| 专属属性 | 类型 | 含义 |
|---|---|---|
| `n_iter_` | Python `int` | 实际外层迭代次数；`alpha == 0` 或 `p == 1` 时为 1。 |

当 `alpha == 0` **或**只有一个特征时，协方差为经验估计，精度矩阵为其伪逆。其他情况使用块坐标更新及稳定化求逆。在正惩罚、多特征情形下，经验方差对角线上存在非正值会引发 `ValueError`。达到 `max_iter` 后会返回当前估计，不发出收敛警告，也不提供单独的收敛标志；`n_iter_ == max_iter` 不能证明已经收敛。没有公开的求解器、热启动、对偶间隙或目标函数历史控制项。

## GraphicalLassoCV

| 专属参数 | 类型与默认值 | 验证与行为 |
|---|---|---|
| `alphas` | 正整数或类数组，`4` | 整数 k 生成固定网格 `np.logspace(-2, 0, k)`，范围为 0.01 至 1；k=1 时只有 0.01。显式候选值转换为 `float64` 并展平，须为非空、有限、非负集合。 |
| `cv` | int，`5` | 整数折数，须满足 `2 <= cv <= n`；布尔值、非整数或超出范围的值引发 `ValueError`。不支持切分器对象或用户提供的切分索引迭代器。 |
| `max_iter` | int，`100` | 与 `GraphicalLasso` 相同的正整数外层迭代上限，用于每个折的拟合和最终重拟合。 |
| `tol` | float，`1e-4` | 与 `GraphicalLasso` 相同的有限正协方差变化容差。 |
| `random_state` | int 或 `None`，`None` | 使用 NumPy 随机打乱行顺序并构造折时的种子。 |

行顺序始终被随机打乱，再划分成大小近似相等的 `cv` 折。没有关闭打乱的选项，也不支持分层、分组或考虑时间顺序的切分。所有训练折至少需要两行，例如 `n=2, cv=2` 会通过折数检查，但在拟合某一折时失败。网格固定，不根据数据自适应生成或细化；显式候选项的顺序和重复值都会保留。

每个候选惩罚都在各训练折上拟合，并用对应测试折的平均高斯对数似然评分。选择分数是各折分数的等权平均，即使各折大小不同也是如此。平均分最高者胜出，完全相同时保留最早的候选项。如果所有平均分都是 `-inf`，仍会选择第一个候选项；应先检查分数，再判断此次选择是否有意义。某一折拟合失败会直接报错，不会用 `error_score` 替代后继续。最后，用选中的惩罚在全部训练行上重新拟合。

| 专属属性 | 类型与形状 | 含义 |
|---|---|---|
| `alpha_` | Python `float` | 从候选网格中选出的惩罚强度。 |
| `cv_results_` | `list[dict]`，长度等于候选项数量 | 按输入或网格顺序保存，每条记录恰有 `alpha`（float）、`mean_score`（float）和 `scores`（含 `cv` 个 float 的列表）三个键。 |

`covariance_`、`precision_` 和 `location_` 描述最终的全数据重拟合结果。本类不公开 `n_iter_` 或 `best_score_`，也不保留公开的最终估计器对象。特别注意，`cv_results_` 不是 scikit-learn 风格的数组字典，也没有拟合后的 `alphas_` 网格属性；可从记录中读取候选惩罚值。

<a id="backend-device-and-dtype"></a>
<a id="device-and-dtype"></a>

## 后端、设备与数据类型

以下规则适用于七个类的 `fit`、`score`、`predict` 和 `mahalanobis`；继承的通用统计辅助方法有其单独的后端规则。

| 估计器设置 | 计算方式 |
|---|---|
| `device="cpu"` | 使用 NumPy CPU，必要时将其他数组库的输入转换过来。 |
| `device="cuda"` | 使用 CuPy CUDA，即使输入为 Torch 张量也如此；要求可用的 CuPy CUDA 运行时。 |
| `device="torch"` | 使用 Torch CUDA；Torch CPU 输入会移至 CUDA，不能用 Torch CPU 执行来满足这个显式 GPU 请求。 |
| `device="auto"` | 继承 `statgpu.set_device(...)`。仅当全局设置也是自动时，原生 CuPy/Torch 输入才保留后端和设备，包括 Torch CPU；其他输入按可用性依次选择 CuPy CUDA、Torch CUDA、NumPy。 |

估计器上的显式设备设置优先于全局配置。自动估计器继承的显式全局 GPU 请求也不会静默回退：要求的运行时不可用时会报错，而不是改用 CPU 或其他 GPU 数组库。无效设备值引发 `ValueError`；估计器设备参数不接受 `"cuda:1"` 等带编号字符串。原生 GPU 输入在选中其后端时保留 GPU 编号；CPU 输入移到 GPU 时使用该数组库当前的 CUDA 设备。更多设置见[设备指南](../guides/device-and-memory.md)。

数值协方差数组统一使用 `float64`，不随受支持输入的浮点精度或整数类型变化；支持集掩码保留布尔类型。拟合数组留在所选后端和设备上，不会全部转换成 NumPy。查询输出例外：`score` 返回 Python 浮点数，`mahalanobis`/`predict` 返回 CPU 上的 NumPy 数组。

<a id="torch-normalization-and-float8"></a>

### Torch 类型转换与 float8

选择 NumPy 计算时，受支持的实值 Torch 输入先脱离自动微分图、移至 CPU，再转为 `float64`，随后才检查有限值；不会修改调用者的张量。支持情形包括 `bfloat16`、`float8_e5m2`，以及稠密非量化张量的实值视图，包括延迟取负视图。先传输后转换也允许从 MPS 张量进行 CPU 计算，因为 MPS 源设备本身不支持 `float64`。

选择 Torch 计算时，稠密 `float8_e5m2` 先在源设备上扩展为 `float64`，再检查有限值；原生 CUDA 数据始终留在 CUDA 上。这样不需要原始 float8 类型的有限值检查内核，但仍要求 Torch 运行时支持转换到 `float64`。此处理不会增加对其他 float8 格式、稀疏或量化张量、复数输入的支持，也不意味着估计器保证可自动微分。不能因为显式转到 CPU 可行，就推断任意源设备与数据类型组合都受支持。

<a id="fitted-device-ownership-and-execution-boundaries"></a>

### 已拟合模型的设备规则与执行边界

`score`、`mahalanobis` 和 `predict` 将查询输入转换到**拟合结果所在的**后端和设备，包括其 CUDA 编号，不会根据查询类型或拟合后改变的全局设备配置重新选择后端。`GraphicalLassoCV` 从拟合开始选定后端和设备后，所有折、评分和最终重拟合都沿用它们。

矩阵乘法、中心化、协方差更新、求逆、Graphical Lasso 坐标更新，以及 MCD 距离、集中步骤和重加权计算均使用所选数值后端。随机子集/折索引、MCD 卡方阈值、Python 控制流程和部分标量诊断使用 CPU。因此，API 不保证整个流程都驻留 GPU，也不保证异步返回结果。各类没有统一的严格/近似模式；后端选择、数值稳定化和统计估计器的选择是不同的问题。
