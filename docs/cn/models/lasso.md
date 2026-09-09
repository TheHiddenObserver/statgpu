# Lasso 回归

> 语言：中文  
> 最后更新：2026-09-09  
> 切换：[English](../../en/models/lasso.md)

## 它解决什么问题？

`Lasso` 是加入 L1 惩罚的线性回归。它最有辨识度的特点是：正则化可以把一部分系数**精确压成 0**，因此模型可以同时完成预测和特征选择。

假设你有 100 个候选特征，但相信真正有用的只有少数几个。OLS 会给每个特征一个系数；Ridge 会把所有系数缩小，但通常不会让它们变成 0；Lasso 则可以直接返回一个很小的**活跃集（active set）**。

典型问题包括：

- 哪些特征可以移除，同时保留主要预测能力？
- 候选特征很多、绝大部分只是噪声时，能否减少过拟合？
- 能否得到一个更容易解释、存储或部署的稀疏模型？

## 一个直观例子

假设有 12 个测量变量，但真实响应只由其中三个产生：

```text
feature       0    1    2    3    4    5    6    7    8    9   10   11
true coef     0   2.0   0    0    0  -1.5   0    0    0   0.8   0    0
Lasso coef    0  ~2.0   0    0    0  ~-1.4  0    0    0  ~0.7   0    0
```

Lasso 的核心吸引力，就是不需要额外设置硬阈值做特征选择，也能得到稀疏线性表示。

## 直觉

Lasso 按绝对值惩罚系数：

$$
\lVert\beta\rVert_1=\sum_j|\beta_j|.
$$

这会产生**软阈值**效果：证据较弱的系数不仅会变小，还可能被直接压到 0。

```text
OLS   ：只要能改善拟合就保留
Ridge ：把所有系数缩小
Lasso ：收缩，同时把弱系数直接删除
```

代价是：当多个特征几乎携带相同信息时，纯 Lasso 的选择可能不稳定。它可能任意保留其中一个、丢掉另一个；这种场景通常更适合 [Elastic Net](elastic-net.md)。

## 什么时候使用？

Lasso 特别适合：

- 你相信真实信号是稀疏的；
- 候选特征相对样本量很多；
- 希望得到较小的活跃集；
- 存储、部署或下游建模能从删除特征中获益；
- 愿意通过验证来选择正则化强度。

以下情况可优先考虑其他方法：

- 大多数特征都可能有小但真实的作用——考虑 [Ridge](ridge.md)；
- 特征形成高度相关的组，希望组内变量一起保留或收缩——考虑 [Elastic Net](elastic-net.md)；
- 响应不适合高斯线性回归——使用相应的惩罚 GLM、生存模型等；
- 首要目标是严格的选择后推断——需要更谨慎的选择性推断设计。

## 模型与目标函数

带截距 $b$ 时，Lasso 最小化

$$
\frac{1}{2n}\sum_{i=1}^{n}
\left(y_i-b-x_i^\top\beta\right)^2
+\alpha\lVert\beta\rVert_1.
$$

`alpha` 越大，收缩越强，出现精确 0 的系数通常也越多；截距不受惩罚。

### 为什么 L1 会产生精确 0？

绝对值惩罚在 0 处有尖角。优化中的软阈值算子为

$$
\mathcal S_\lambda(z)
=
\operatorname{sign}(z)\max(|z|-\lambda,0).
$$

当 $|z|\le\lambda$ 时，结果就是精确的 0。这是 Lasso 产生稀疏解的直接计算机制。

## 最小可运行示例

```python
import numpy as np
from statgpu.linear_model import Lasso

rng = np.random.default_rng(1)
X = rng.normal(size=(500, 12))

true_coef = np.zeros(12)
true_coef[[1, 5, 9]] = [2.0, -1.5, 0.8]
y = 0.7 + X @ true_coef + rng.normal(scale=0.7, size=500)

model = Lasso(
    alpha=0.08,
    device="cpu",
    compute_inference=False,
).fit(X, y)

print("coefficients:", np.round(model.coef_, 3))
print("selected features:", np.flatnonzero(np.abs(model.coef_) > 1e-8))
print("R²:", round(model.score(X, y), 3))
```

固定随机种子和这个 `alpha` 后，明显非零的系数应在索引 `1`、`5`、`9`，大约为 `1.97`、`-1.41`、`0.73`。它们比生成系数更小，是因为收缩本来就是 Lasso 估计量的一部分。

## 如何理解结果？

- `coef_[j] == 0` 表示在当前 `alpha` 下，该特征没有进入拟合的线性预测子。
- 非零系数仍然经过收缩，不能当作未惩罚 OLS 系数。
- `intercept_` 不受 L1 惩罚。
- `predict(X_new)` 返回连续预测。
- `score(X, y)` 返回 $R^2$，并支持 `sample_weight=`。
- `n_iter_` 是所选数值求解路径的迭代次数。

一个特征在某个样本中被压成 0，并不证明总体中的真实作用必然为 0。

## 关键参数应该怎么选？

这里是正常工作流程的**精选参数表**。完整构造函数参数见[完整 API 参考](#完整-api-参考)。

| 参数 | 默认值 | 应该怎么理解 |
|---|---:|---|
| `alpha` | `1.0` | 最重要的统计选择；越大通常收缩越强、0 越多。预测/选择任务优先用 `LassoCV`。 |
| `fit_intercept` | `True` | 一般保持开启，除非理论上固定截距或设计矩阵已有截距。 |
| `device` | `"auto"` | 小问题优先 CPU；计算规模足够大时再考虑 GPU。 |
| `solver` | `"fista"` | 直接拟合算法的正式选择参数。CPU 坐标下降使用 `coordinate_descent`；其他场景按支持矩阵选择 FISTA 等近端求解器。 |
| `stopping` | `"coef_delta"` | 需要按最优性条件而不是系数变化判断收敛时可用 `"kkt"`。 |
| `compute_inference` | `True` | 只做预测/选择时可关闭；需要推断时应明确理解 `inference_method`，并阅读独立推断参考页。 |

### 正则化之前先标准化

L1 惩罚直接作用于系数大小。特征量纲不同会导致有效惩罚程度不同，因此多数 Lasso 分析流程应先标准化连续变量。

## 与相近方法比较

| 方法 | 相关变量组 | 精确 0？ | 典型选择理由 |
|---|---|:---:|---|
| OLS / `LinearRegression` | 无正则化 | 否 | 未惩罚估计 |
| [Ridge](ridge.md) | 倾向共享信号 | 否 | 稳定相关变量，不删除特征 |
| **Lasso** | 可能只选组内一个 | 是 | 稀疏预测 / 自动特征选择 |
| [Elastic Net](elastic-net.md) | 比 Lasso 更照顾组内变量 | 是 | 稀疏模型 + 相关变量稳定性 |
| Adaptive Lasso | 数据依赖的 L1 权重 | 是 | 在更强稀疏假设下减轻统一惩罚偏差 |

## CPU、GPU、公式接口与加权拟合

```python
from statgpu.linear_model import Lasso

model = Lasso(
    alpha=0.08,
    device="cuda",
    solver="fista",
    stopping="kkt",
    compute_inference=False,
).fit(X, y)
```

显式 `device="cuda"` / `"torch"` 使用相应 GPU 后端；不可用时会明确失败，而不是静默切换执行路径。

`fit()` 同时支持 `sample_weight=`，以及共享的 `formula=` / `data=` 公式接口。公式接口元数据会保留下来，用于后续 DataFrame 预测时重建一致的设计矩阵。

## 进阶：求解器支持

| `solver` 值 | CPU | CuPy / Torch | 含义 |
|---|:---:|:---:|---|
| `fista`（默认） | 支持 | 支持 | L1 目标的稳定近端梯度路径 |
| `auto` | FISTA | FISTA | 当前平方误差 + L1 的自动分派 |
| `fista_bb` | 支持 | 支持 | 使用 Barzilai-Borwein 步长的 FISTA |
| `admm` | 支持 | 支持 | 替代拆分路径；仅支持均匀样本权重 |
| `coordinate_descent` | 支持 | 不支持 | 仅 CPU 的直接拟合坐标下降路径 |

`solver` 是统一的、与后端无关的直接拟合算法选择参数。L1 目标拒绝 `newton`、`lbfgs`、`irls`、`exact`。

`cpu_solver` 仍作为旧版/CV 行为的兼容控制保留，但它不选择一次直接 `Lasso.fit` 的算法；直接拟合算法应使用 `solver`。CV 估计器具有独立的选择阶段与最终重拟合阶段，详见[交叉验证指南](../guides/cross-validation.md)。

`admm_rho` 控制 ADMM 惩罚参数；`lipschitz_L` 可为兼容的近端路径提供预计算 Lipschitz 常数。

## 进阶：Lasso 之后的推断

选择后推断比预先指定的 OLS 模型推断困难得多。statgpu 提供纠偏推断、OLS 风格的选择后诊断、残差自助法，以及可选的 max-|Z| 同时推断；它们的统计主张并不相同。

| `inference_method` | 用途 | 重要限制 |
|---|---|---|
| `debiased`（构造函数默认值） | 纠偏 Lasso 的逐系数推断 | 边际置信区间依赖高维纠偏假设；同时覆盖需要单独的同时推断程序 |
| `post_selection_ols` | 在拟合阶段确定的 NumPy/CuPy/Torch 后端上执行活跃集 OLS/WLS 诊断 | 选择后启发式区间，不是一般的选择性推断 |
| `bootstrap` | 残差自助法替代路径 | 计算更昂贵，也不是对选择不确定性的普适修正 |

`post_selection_ols` 是与硬件无关的规范名称。旧 `cpu_ols` / `gpu_ols` 是处于弃用期的兼容别名，会发出 `FutureWarning` 并映射到同一方法；执行设备仍由独立的 `device` 与后端路由决定。

逐节点 Lasso 构造、与纠偏斜率一致的截距参数化、边际 z 推断、真正包含截距的 max-|Z| 乘子自助法、数值后端与结果报告边界、多重检验区别和输出字段，都集中在 **[Lasso 推断](lasso-inference.md)**。

## 常见误区

- “被选择”不等于“总体真实非零”，更不等于因果。
- L1 不具备尺度不变性，不能忽略标准化。
- 高度相关变量之间的纯 Lasso 选择可能很不稳定。
- 不要根据训练 $R^2$ 选择 `alpha`。
- 不要在数据驱动选择后直接附上普通 OLS p 值，并把它当作预先指定模型的推断；见 [Lasso 推断](lasso-inference.md)。
- 数值收敛（例如 KKT 残差很小）不等于统计模型正确。

## 完整 API 参考

前面的参数表是教学用的选择指南；这里列出当前 `Lasso` 封装类的完整构造函数和模型方法/属性清单。

### 构造函数

```python
Lasso(
    alpha=1.0,
    fit_intercept=True,
    max_iter=1000,
    tol=1e-4,
    stopping="coef_delta",
    inference_method="debiased",
    n_bootstrap=200,
    bootstrap_random_state=None,
    enable_simultaneous_inference=False,
    simultaneous_method="maxz_bootstrap",
    simultaneous_alpha=0.05,
    simultaneous_n_bootstrap=1000,
    simultaneous_random_state=None,
    simultaneous_include_intercept=False,
    device="auto",
    n_jobs=None,
    compute_inference=True,
    solver="fista",
    cpu_solver="coordinate_descent",
    lipschitz_L=None,
    admm_rho=1.0,
    gpu_memory_cleanup=False,
)
```

<!-- API-CONSTRUCTOR-START:Lasso -->
| 参数 | 默认值 | API 含义 |
|---|---:|---|
| `alpha` | `1.0` | L1 惩罚强度。 |
| `fit_intercept` | `True` | 拟合不受惩罚的截距。 |
| `max_iter` | `1000` | 最大求解迭代数。 |
| `tol` | `1e-4` | 数值收敛容差。 |
| `stopping` | `"coef_delta"` | 兼容路径使用 `coef_delta` 或 `kkt`。 |
| `inference_method` | `"debiased"` | 拟合后推断路径：`debiased`（纠偏）、规范的 `post_selection_ols` 或 `bootstrap`。旧 `cpu_ols` / `gpu_ols` 会以 `FutureWarning` 提示并映射到 `post_selection_ols`。 |
| `n_bootstrap` | `200` | `inference_method="bootstrap"` 时残差自助法的抽样次数。 |
| `bootstrap_random_state` | `None` | 残差自助法随机种子。 |
| `enable_simultaneous_inference` | `False` | 在纠偏推断后启用 max-|Z| 同时置信区间。 |
| `simultaneous_method` | `"maxz_bootstrap"` | 同时推断校准方法；当前为 `maxz_bootstrap`。 |
| `simultaneous_alpha` | `0.05` | 同时置信区间的族错误率水平。 |
| `simultaneous_n_bootstrap` | `1000` | max-|Z| 乘子自助法抽样次数。 |
| `simultaneous_random_state` | `None` | 同时推断自助法随机种子。 |
| `simultaneous_include_intercept` | `False` | 将中心化纠偏截距纳入 max-|Z| 目标参数集合和最终同时置信区间。 |
| `device` | `"auto"` | `auto`、`cpu`、`cuda`（CuPy）或 `torch`（Torch CUDA）。 |
| `n_jobs` | `None` | 所选路径使用并行时的并行度提示。 |
| `compute_inference` | `True` | 执行所选拟合后推断。 |
| `solver` | `"fista"` | 与后端无关的直接拟合求解器；在 CPU/GPU 上由它选择直接拟合算法。 |
| `cpu_solver` | `"coordinate_descent"` | 旧版/CV 行为的兼容控制；不会替代一次直接 `Lasso.fit` 中的 `solver`。 |
| `lipschitz_L` | `None` | 兼容近端路径的预计算 Lipschitz 常数。 |
| `admm_rho` | `1.0` | ADMM 增广拉格朗日惩罚参数。 |
| `gpu_memory_cleanup` | `False` | 拟合后尽力释放缓存的 GPU 内存。 |
<!-- API-CONSTRUCTOR-END:Lasso -->

### `fit`

共享的惩罚线性模型拟合签名为：

```python
model.fit(
    X=None,
    y=None,
    sample_weight=None,
    formula=None,
    data=None,
)
```

| 参数 | 含义 |
|---|---|
| `X` | 二维特征矩阵。 |
| `y` | 一维连续因变量。 |
| `sample_weight` | 可选的非负分析权重；部分求解器还有额外限制。 |
| `formula` | 可选的 Patsy 风格公式。 |
| `data` | 公式接口使用的 DataFrame。 |

`fit()` 返回 `self`。

### 预测、评分与报告方法

| 方法 | 签名 | 行为 |
|---|---|---|
| `predict` | `predict(X, return_cpu=True)` | 连续预测；`return_cpu=False` 可让 GPU 预测结果保留在实际后端。 |
| `score` | `score(X, y, sample_weight=None)` | 返回 $R^2$，支持加权。 |
| `summary` | `summary()` | 打印系数/推断摘要；要求模型已拟合且推断结果可用。 |
| `get_params` / `set_params` | scikit-learn 风格工具 | 查看或替换构造参数状态。 |

继承的模型上下文工具 `adjust_pvalues`、`combine_pvalues`、`bootstrap_statistic` 和 `permutation_test` 的完整签名、后端解析和拟合状态复用语义见[推断 API](../guides/inference-api.md)。Lasso 专属的系数推断见 [Lasso 推断](lasso-inference.md)。

### 拟合后属性与诊断量

| 属性 | 含义 / 可用条件 |
|---|---|
| `coef_` | 惩罚系数；精确 0 定义当前拟合得到的活跃集。 |
| `intercept_` | 不受惩罚的截距。 |
| `n_iter_` | 所选数值路径的迭代次数。 |
| `n_features_in_` | 相应拟合路径发布时的特征数。 |
| `rsquared`, `rsquared_adj` | 所需状态可用时的 $R^2$ 与调整 $R^2$。 |
| `fvalue`, `f_pvalue` | 在定义时可用的联合拟合统计量与 p 值。 |
| `llf`, `aic`, `bic` | 所需结果状态可用时的高斯拟合诊断量。 |
| `_bse` | 所选推断方法产生的标准误。 |
| `_tvalues` | 某些兼容路径沿用的历史统计量字段；纠偏推断时具有 z 统计量语义。 |
| `_zvalues` | 结构化推断结果填充时的 z 统计量。 |
| `_pvalues` | 推断成功时的系数 p 值。 |
| `_conf_int` | 推断成功时的边际系数置信区间。 |
| `_conf_int_simultaneous` | 显式启用并成功校准时的同时置信区间。 |
| `_inference_result` | 结构化推断结果与元数据。 |

下划线开头的推断属性是当前版本既有的结果接口；其统计含义取决于 `inference_method`。

## 验证

当前维护的验证覆盖求解器收敛、CPU/GPU 一致性、KKT 停止准则、纠偏推断、残差自助法、在拟合确定的后端上执行的 `post_selection_ols`、同时推断以及需要时的实体 GPU 验证。相关入口包括 `dev/benchmarks/validate_post_selection_ols_gpu.py` 与 `dev/benchmarks/benchmark_lasso_inference_gpu_vs_cpu.py`。

## 参考文献

- Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *Journal of the Royal Statistical Society: Series B*, 58(1), 267–288.
- Bühlmann, P., & van de Geer, S. (2011). *Statistics for High-Dimensional Data*. Springer.
- Zhang, C.-H., & Zhang, S. S. (2014). Confidence intervals for low-dimensional parameters in high-dimensional linear models. *JRSS B*, 76(1), 217–242.
- Javanmard, A., & Montanari, A. (2014). Confidence intervals and hypothesis testing for high-dimensional regression. *JMLR*, 15, 2869–2909.