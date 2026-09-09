# 弹性网络（Elastic Net）

> 语言：中文
> 最后更新：2026-09-09
> 切换：[English](../../en/models/elastic-net.md)

## 它解决什么问题？

`ElasticNet` 把 [Lasso](lasso.md) 的 L1 惩罚和 [Ridge](ridge.md) 的 L2 惩罚组合在一起。当 L1 部分为正时，它特别适合这样一种情况：你既希望模型稀疏，又知道特征之间存在较强相关性，而纯 Lasso 的变量选择会因此不稳定。

它解决的是一个很实际的矛盾：

- Ridge 很擅长处理相关变量，但通常不会把系数压成 0；
- Lasso 可以删除变量，但在几个几乎可互换的特征中，可能任意保留一个、丢掉另一个；
- Elastic Net 同时加入 L1 与 L2，因此既能产生 0，又能让相关变量更平滑地共享信号。

## 一个直观例子

假设四个真正有用的特征组成两对高度相关变量，而且每一对里的两个变量都确实携带信号。纯 Lasso 可能把一对变量的信号分得很不均匀：

```text
              x1     x2     x3     x4
Lasso        1.62   0.71  -0.68  -1.24
Elastic Net  1.19   1.14  -0.95  -0.98
```

两种拟合都可能预测得很好，但 Elastic Net 更能反映“组内变量是近似可互换测量”的结构。

## 直觉

Elastic Net 同时施加两种偏好：

1. **L1 部分**：弱系数可能被直接压成 0；
2. **L2 部分**：大而不稳定的系数被平滑收缩，有助于相关变量更像一个组。

两个参数控制这种权衡：

- `alpha` 控制**总正则化强度**；
- `l1_ratio` 控制 **L1 惩罚在总惩罚中的占比**。

```text
l1_ratio = 0.0        0.5             1.0
              Ridge ←──── Elastic Net ────→ Lasso
```

在 statgpu 的目标函数尺度下，`l1_ratio=0` 会去掉 L1 项，留下与 Ridge 相同形式的 L2 惩罚；`l1_ratio=1` 则去掉 L2 项，留下 Lasso 惩罚。这里说的是**目标函数层面的退化关系**。公共 `ElasticNet` 封装类仍保留自己的求解器默认值、参数校验和推断分派；如果明确希望得到纯 Ridge 的 API、求解器和推断语义，应直接使用 `Ridge`，而不是依赖 `ElasticNet(l1_ratio=0)`。

## 什么时候使用？

Elastic Net 很适合：

- 希望做特征选择，但很多特征高度相关；
- 特征天然形成携带类似信息的组；
- 纯 Lasso 在近重复变量中频繁更换被选择成员；
- 候选变量很多，希望模型稀疏同时又更稳定；
- 愿意同时调 `alpha` 与 `l1_ratio`。

以下情况可优先考虑其他方法：

- 不需要精确 0，主要关心稳定预测——Ridge 更简单；
- 最看重更强的稀疏性，且相关性不强——Lasso 可能足够；
- 响应不适合高斯线性回归——使用相应的惩罚 GLM；
- 特征选择本身没有明确科学意义，例如特征只是任意编码或强混杂代理。

## 模型与目标函数

带截距 $b$ 时，Elastic Net 最小化

$$
\frac{1}{2n}\sum_{i=1}^{n}
\left(y_i-b-x_i^\top\beta\right)^2
+\alpha\lambda\lVert\beta\rVert_1
+\frac{\alpha}{2}(1-\lambda)\lVert\beta\rVert_2^2,
$$

其中 $\lambda$ 就是 `l1_ratio`。`alpha` 越大，总体收缩越强；`l1_ratio` 越接近 1 越像 Lasso，越接近 0 越像 Ridge。截距不受惩罚。需要注意：`l1_ratio=0` 时已经没有 L1 阈值项，因此目标函数本身不再鼓励精确稀疏。

## 最小可运行示例

```python
import numpy as np
from statgpu.linear_model import ElasticNet, Lasso

rng = np.random.default_rng(2)
n = 500
z1 = rng.normal(size=n)
z2 = rng.normal(size=n)
X = np.column_stack([
    z1 + 0.05 * rng.normal(size=n),
    z1 + 0.05 * rng.normal(size=n),
    z2 + 0.05 * rng.normal(size=n),
    z2 + 0.05 * rng.normal(size=n),
    rng.normal(size=n),
    rng.normal(size=n),
])

true_coef = np.array([1.2, 1.2, -1.0, -1.0, 0.0, 0.0])
y = 0.5 + X @ true_coef + rng.normal(scale=0.8, size=n)

lasso = Lasso(alpha=0.08, device="cpu", compute_inference=False).fit(X, y)
elastic = ElasticNet(
    alpha=0.08,
    l1_ratio=0.5,
    device="cpu",
    compute_inference=False,
).fit(X, y)

print("Lasso:      ", np.round(lasso.coef_, 2))
print("Elastic Net:", np.round(elastic.coef_, 2))
```

固定随机种子后，Lasso 往往会把两组相关变量的信号分得更不均匀，而 Elastic Net 会更平均地分配，同时让两个噪声变量保持在 0 或接近 0。

## 如何理解结果？

- 当 `l1_ratio>0` 时，`coef_[j] == 0` 表示当前组合惩罚把该变量从拟合的线性预测子中移除；当 `l1_ratio=0` 时目标函数是纯 L2，精确稀疏并不是预期行为。
- 非零系数仍经过收缩，不能当作未惩罚 OLS 系数。
- `intercept_` 不受惩罚。
- `predict(X_new)` 返回连续预测。
- `score(X, y)` 返回 $R^2$，并支持 `sample_weight=`。
- 当 L1 部分存在时，活跃集同时依赖 `alpha` 与 `l1_ratio`。

相关变量一起保持非零，是 Elastic Net 常见行为，但不等于它们分别具有独立的因果效应。

## 关键参数应该怎么选？

这里是正常分析流程的**精选参数表**。完整构造函数参数见[完整 API 参考](#完整-api-参考)。

| 参数 | 默认值 | 应该怎么理解 |
|---|---:|---|
| `alpha` | `1.0` | 总正则化强度；越大收缩越强，在 L1 部分为正时也可能删除更多变量。建议通过验证选择。 |
| `l1_ratio` | `0.5` | L1 惩罚占比；接近 1 更像 Lasso，接近 0 更像 Ridge。最好与 `alpha` 联合调参；取 0 时完全移除 L1 稀疏项。 |
| `fit_intercept` | `True` | 一般保持开启，除非理论上固定截距或设计矩阵已有截距。 |
| `device` | `"auto"` | 小问题使用 CPU 最简单；规模足够大时 GPU 更有意义。 |
| `solver` | `"fista"` | 当前公共 Elastic Net 接口的稳定默认值；改变它主要影响数值算法和性能。 |
| `stopping` | `"coef_delta"` | 更关心最优性诊断时可使用 `"kkt"`。 |
| `compute_inference` | `False` | 普通预测/选择时保持关闭；需要拟合后推断时再开启，并应同时理解其统计假设。 |

多数正则化分析流程应先标准化连续特征，因为 L1/L2 都直接作用于系数大小。

## 与 Ridge 和 Lasso 比较

| 性质 | Ridge | Lasso | **Elastic Net** |
|---|:---:|:---:|:---:|
| 平滑收缩 | 是 | 是 | 是 |
| 精确 0 | 通常否 | 是 | `l1_ratio>0` 时可以 |
| 相关变量稳定性 | 强 | 可能不稳定 | 比纯 Lasso 强 |
| 主要调参 | `alpha` | `alpha` | `alpha` + `l1_ratio` |
| 直观理解 | 稳定 | 选择 | 有 L1 时“选择 + 稳定” |

## CPU、GPU、公式接口、加权拟合与热启动

```python
from statgpu.linear_model import ElasticNet

model = ElasticNet(
    alpha=0.08,
    l1_ratio=0.5,
    device="cuda",
    solver="fista",
    compute_inference=False,
).fit(X, y)
```

`fit()` 支持 `sample_weight=`，并通过 `**kwargs` 转发共享的 `formula=` / `data=` 公式接口。单次拟合还可以通过 `initial_coef=` 进行**热启动（warm start）**：

```python
warm = ElasticNet(alpha=0.08, l1_ratio=0.5).fit(
    X,
    y,
    initial_coef=previous_coef,
)
```

## 进阶：求解器与优化

当 `l1_ratio>0` 时，Elastic Net 目标是非光滑的，因此近端方法是正常数值路径。`l1_ratio=0` 时数学目标已经变成光滑 L2，但公共 `ElasticNet` 封装类仍保留 Elastic Net 自身的求解器契约；若需要专门的 Ridge 求解器接口，应使用 `Ridge`。

| `solver` 值 | CPU | CuPy / Torch | 说明 |
|---|:---:|:---:|---|
| `fista`（默认） | 支持 | 支持 | 推荐的近端梯度路径 |
| `auto` | FISTA | FISTA | 当前平方误差 + Elastic Net 的自动分派 |
| `fista_bb` | 支持 | 支持 | 使用自适应谱步长 |
| `admm` | 支持 | 支持 | 替代拆分路径；仅支持均匀样本权重 |
| `coordinate_descent` | 支持 | 不支持 | 仅 CPU 的兼容坐标下降路径 |

`newton`、`lbfgs`、`irls`、`exact` 会被当前公共 Elastic Net 估计器接口拒绝。一次直接 `ElasticNet.fit` 中，`solver` 是直接拟合算法的正式选择参数；`cpu_solver` 仅作为旧版/共享路径的兼容控制保留，不会选择直接拟合算法。新代码应使用 `solver`。

在消去未惩罚截距之后——等价地，在中心化后的系数问题上——一阶 KKT 条件为

$$
\frac{1}{n}X_c^\top(X_c\hat\beta-y_c)
+\alpha(1-\lambda)\hat\beta
+\alpha\lambda\,\partial\lVert\hat\beta\rVert_1=0.
$$

若不用中心化记号，数据拟合项中应显式包含拟合截距 $\hat b\mathbf 1$。`stopping="kkt"` 只改变收敛判定方式，不改变统计模型。

## 进阶：推断

`ElasticNet` 默认只做估计。设置 `compute_inference=True` 后会运行拟合后推断，但不会改变已经得到的惩罚系数。

| `inference_method` | 用途 | 重要限制 |
|---|---|---|
| `debiased`（默认推断方法） | 使用共享逐节点框架进行一步纠偏系数推断 | 复用了 Lasso 家族的计算构造，但有效性仍依赖初始 Elastic Net 估计量、稀疏性/设计/噪声条件以及正则化与调参方式；Lasso 理论文献只是主要背景，并不是对任意 `l1_ratio` 的一揽子保证 |
| `post_selection_ols` | 在拟合阶段确定的 NumPy/CuPy/Torch 后端上做未惩罚活跃集 OLS/WLS 重拟合 | 启发式选择后诊断，不是一般的选择性推断保证 |
| `bootstrap` | 残差自助法替代路径 | 计算更昂贵，并依赖相应的自助法假设 |

`debiased` 实现会让 L1 和 Elastic Net 惩罚复用相同的逐节点一步校正框架。这种**软件复用**不应解读成“任何 Elastic Net 配置都自动继承某个特定纠偏 Lasso 定理的全部保证”；仍需针对实际的初始估计量、惩罚尺度和理论条件判断。

`post_selection_ols` 是与硬件无关的规范名称。旧 `cpu_ols` / `gpu_ols` 是处于弃用期的兼容别名，会发出 `FutureWarning` 并映射到 `post_selection_ols`；执行后端仍由独立的 `device` 控制。选择后重拟合会复用成功惩罚拟合记录的后端/设备，而不是重新根据原始输入判断后端。

`cov_type` 与 `hac_maxlags` 也是公开构造参数，在所选推断路径支持相应协方差估计时使用。

对于 `ElasticNetCV`，`compute_inference=True` 只在 `alpha` 和 `l1_ratio` 选择完成后的最终全数据重拟合上运行推断；各折模型仍只用于估计和评分。所报告的推断条件于这些已经选出的调参值，当前实现不会再额外校正交叉验证调参不确定性。

**拟合诊断量属于兼容性诊断。** 在可用时，`rsquared_adj`、`fvalue`、`f_pvalue`、`aic` 和 `bic` 使用惩罚拟合保存的状态以及普通参数个数/残差自由度约定；它们没有计入数据驱动活跃集选择、有效自由度或超参数调节过程，因此不应作为 penalty-aware 模型选择准则替代验证或交叉验证。

## 常见误区

- 不要只调 `alpha` 而把 `l1_ratio` 当作无关参数。
- 不要认为 `l1_ratio=0` 会让 `ElasticNet` 封装类在所有 API 层面都与 `Ridge` 完全相同；只是目标函数退化为 L2。
- 相关变量一起非零不等于分别具有独立因果效应。
- 不要忘记标准化。
- 不要根据训练 $R^2$ 选择超参数。
- 弱信号下，活跃集仍可能随数据微扰而变化。
- 数据驱动选择后，不要直接套用普通未惩罚推断而忽略选择过程。
- 不要把纠偏 Lasso 的参考文献理解成对任意 Elastic Net 调参配置的自动定理。
- 不要把当前 AIC/BIC/F 输出当成已经考虑选择过程或有效自由度的模型选择准则。

## 完整 API 参考

前面的参数表是教学用选择指南；这里列出当前 `ElasticNet` 封装类的完整构造函数和模型方法/属性清单。

### 构造函数

```python
ElasticNet(
    alpha=1.0,
    l1_ratio=0.5,
    fit_intercept=True,
    max_iter=1000,
    tol=1e-4,
    stopping="coef_delta",
    device="auto",
    n_jobs=None,
    solver="fista",
    cpu_solver="fista",
    lipschitz_L=None,
    gpu_memory_cleanup=False,
    compute_inference=False,
    inference_method="debiased",
    cov_type="nonrobust",
    hac_maxlags=None,
)
```

<!-- API-CONSTRUCTOR-START:ElasticNet -->
| 参数 | 默认值 | API 含义 |
|---|---:|---|
| `alpha` | `1.0` | 总正则化强度。 |
| `l1_ratio` | `0.5` | L1 惩罚占比；0 去掉 L1 项，1 去掉 L2 项。 |
| `fit_intercept` | `True` | 拟合不受惩罚的截距。 |
| `max_iter` | `1000` | 最大求解迭代数。 |
| `tol` | `1e-4` | 数值收敛容差。 |
| `stopping` | `"coef_delta"` | 兼容路径使用 `coef_delta` 或 `kkt`。 |
| `device` | `"auto"` | `auto`、`cpu`、`cuda`（CuPy）或 `torch`（Torch CUDA）。 |
| `n_jobs` | `None` | 所选路径使用并行时的并行度提示。 |
| `solver` | `"fista"` | 与后端无关的直接拟合求解器；一次 `ElasticNet.fit` 中由它决定算法。 |
| `cpu_solver` | `"fista"` | 旧版/共享行为的兼容控制；不会替代直接拟合的 `solver`。 |
| `lipschitz_L` | `None` | 兼容近端路径的预计算 Lipschitz 常数。 |
| `gpu_memory_cleanup` | `False` | 拟合后尽力释放缓存的 GPU 内存。 |
| `compute_inference` | `False` | 执行所选拟合后推断。 |
| `inference_method` | `"debiased"` | 拟合后推断路径：`debiased`（纠偏）、规范的 `post_selection_ols` 或 `bootstrap`。旧 `cpu_ols` / `gpu_ols` 会以 `FutureWarning` 提示并映射到 `post_selection_ols`。 |
| `cov_type` | `"nonrobust"` | 所选推断路径使用协方差估计时的约定。 |
| `hac_maxlags` | `None` | 所选推断方法支持 HAC 时的滞后阶数。 |
<!-- API-CONSTRUCTOR-END:ElasticNet -->

### `fit`

封装类的直接签名为：

```python
model.fit(
    X=None,
    y=None,
    sample_weight=None,
    initial_coef=None,
    **kwargs,
)
```

`**kwargs` 当前可转发共享拟合接口的 `formula` 与 `data`。

| 参数 | 含义 |
|---|---|
| `X` | 二维特征矩阵。 |
| `y` | 一维连续因变量。 |
| `sample_weight` | 可选的非负分析权重；部分求解器有额外限制。 |
| `initial_coef` | 可选的热启动系数向量，每个特征一个值。 |
| `formula` | 通过 `**kwargs` 转发的 Patsy 风格公式。 |
| `data` | 公式接口使用的 DataFrame。 |

`fit()` 返回 `self`。

### 预测、评分与报告方法

| 方法 | 签名 | 行为 |
|---|---|---|
| `predict` | `predict(X, return_cpu=True)` | 连续预测；`return_cpu=False` 可让 GPU 预测结果保留在实际后端。 |
| `score` | `score(X, y, sample_weight=None)` | 返回 $R^2$，支持加权。 |
| `summary` | `summary()` | 打印系数/推断摘要；要求模型已拟合且推断结果可用。 |
| `get_params` / `set_params` | scikit-learn 风格工具 | 查看或替换构造参数状态。 |

继承的模型上下文工具 `adjust_pvalues`、`combine_pvalues`、`bootstrap_statistic` 和 `permutation_test` 的完整签名、后端解析和拟合状态复用语义见[推断 API](../guides/inference-api.md)。

### 拟合后属性与诊断量

| 属性 | 含义 / 可用条件 |
|---|---|
| `coef_` | 惩罚系数；有 L1 部分时精确 0 定义当前活跃集。 |
| `intercept_` | 不受惩罚的截距。 |
| `n_iter_` | 所选数值路径的迭代次数。 |
| `n_features_in_` | 相应拟合路径发布时的特征数。 |
| `rsquared`, `rsquared_adj` | 所需状态可用时的 $R^2$ 与按普通参数个数计算的调整 $R^2$；不是选择/有效自由度修正。 |
| `fvalue`, `f_pvalue` | 使用普通参数个数/残差自由度的兼容性联合拟合诊断；不是选择感知的经典 F 推断。 |
| `llf`, `aic`, `bic` | 基于当前惩罚拟合、使用普通参数个数的高斯 plug-in 诊断；不是选择、调参或有效自由度感知的准则。 |
| `_bse` | 所选推断方法的标准误。 |
| `_tvalues` | 使用 t 型统计量的推断路径所保存的统计量。 |
| `_zvalues` | 纠偏推断的 z 型统计量。 |
| `_pvalues` | 推断成功时的系数 p 值。 |
| `_conf_int` | 推断成功时的系数置信区间。 |
| `_inference_result` | 结构化推断结果与元数据。 |

## 验证

当前维护的验证覆盖 Elastic Net 目标函数、求解器/KKT 行为、CPU/GPU 路径、在拟合确定的后端上执行的 `post_selection_ols`、拟合后推断、热启动，以及 `ElasticNetCV` 最终重拟合推断约定。

## 参考文献

- Zou, H., & Hastie, T. (2005). Regularization and variable selection via the Elastic Net. *Journal of the Royal Statistical Society: Series B*, 67(2), 301–320.
- Beck, A., & Teboulle, M. (2009). A fast iterative shrinkage-thresholding algorithm for linear inverse problems. *SIAM Journal on Imaging Sciences*, 2(1), 183–202.