# 弹性网络（Elastic Net）

> 语言：中文
> 最后更新：2026-09-05
> 切换：[English](../../en/models/elastic-net.md)

## 它解决什么问题？

`ElasticNet` 把 [Lasso](lasso.md) 的 L1 惩罚和 [Ridge](ridge.md) 的 L2 惩罚组合在一起。它特别适合这样一种情况：你既希望模型稀疏，又知道特征之间存在较强相关性，而纯 Lasso 的变量选择会因此不稳定。

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
- `l1_ratio` 控制**正则化类型**。

```text
l1_ratio = 0.0        0.5             1.0
              Ridge ←──── Elastic Net ────→ Lasso
```

在 statgpu 的目标函数尺度下，`l1_ratio=0` 对应 Ridge-style penalty，`l1_ratio=1` 对应 Lasso-style penalty。

## 什么时候使用？

Elastic Net 很适合：

- 希望做特征选择，但很多特征高度相关；
- 特征天然形成携带类似信息的组；
- 纯 Lasso 在近重复变量中频繁更换被选择成员；
- 候选变量很多，希望稀疏同时又更稳定；
- 愿意同时调 `alpha` 与 `l1_ratio`。

优先考虑其他方法，当：

- 不需要精确 0，主要关心稳定预测——Ridge 更简单；
- 最看重最激进的稀疏性，且相关性不强——Lasso 可能足够；
- 响应不适合 Gaussian 线性回归——使用相应 penalized GLM；
- 特征选择本身没有科学意义，例如特征只是任意编码或强混杂代理。

## 模型与目标函数

带截距 $b$ 时，Elastic Net 最小化

$$
\frac{1}{2n}\sum_{i=1}^{n}
\left(y_i-b-x_i^\top\beta\right)^2
+\alpha\lambda\lVert\beta\rVert_1
+\frac{\alpha}{2}(1-\lambda)\lVert\beta\rVert_2^2,
$$

其中 $\lambda$ 就是 `l1_ratio`。`alpha` 越大总体收缩越强；`l1_ratio` 越接近 1 越像 Lasso，越接近 0 越像 Ridge。截距不受惩罚。

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

- `coef_[j] == 0` 表示在当前 `alpha` 与 `l1_ratio` 下，该变量没有进入拟合 predictor。
- 非零系数仍经过收缩，不能当作未惩罚 OLS 系数。
- `intercept_` 不受惩罚。
- `predict(X_new)` 返回连续预测。
- `score(X, y)` 返回 $R^2$，并支持 `sample_weight=`。
- active set 同时依赖 `alpha` 与 `l1_ratio`。

相关变量一起保持非零，是 Elastic Net 常见行为，但不等于它们分别具有独立因果效应。

## 关键参数应该怎么选？

这里是正常工作流的**精选参数表**。完整 constructor 见[完整 API 参考](#完整-api-参考)。

| 参数 | 默认值 | 应该怎么理解 |
|---|---:|---|
| `alpha` | `1.0` | 总正则化强度；越大收缩越强，也可能删除更多变量。建议用验证选择。 |
| `l1_ratio` | `0.5` | L1/L2 混合比例；接近 1 更像 Lasso，接近 0 更像 Ridge。最好与 `alpha` 联合调参。 |
| `fit_intercept` | `True` | 一般保持开启，除非理论固定截距或设计矩阵已有截距。 |
| `device` | `"auto"` | 小问题 CPU 最简单；规模足够大时 GPU 更有意义。 |
| `solver` | `"fista"` | 当前非光滑目标的稳定默认值，改变它主要是数值/性能选择。 |
| `stopping` | `"coef_delta"` | 更关心最优性诊断时可使用 `"kkt"`。 |
| `compute_inference` | `False` | 普通预测/选择保持关闭；需要支持的拟合后推断时再开启。 |

多数正则化 workflow 应先标准化连续特征，因为 L1/L2 都直接作用于系数大小。

## 与 Ridge 和 Lasso 比较

| 性质 | Ridge | Lasso | **Elastic Net** |
|---|:---:|:---:|:---:|
| 平滑收缩 | 是 | 是 | 是 |
| 精确 0 | 通常否 | 是 | 是 |
| 相关变量稳定性 | 强 | 可能不稳定 | 比纯 Lasso 强 |
| 主要调参 | `alpha` | `alpha` | `alpha` + `l1_ratio` |
| 心智模型 | 稳定 | 选择 | 选择 + 稳定 |

## CPU、GPU、Formula、加权拟合与 warm start

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

`fit()` 支持 `sample_weight=`，并通过 `**kwargs` 转发共享的 `formula=` / `data=` 接口。单次拟合还可以通过 `initial_coef=` warm start：

```python
warm = ElasticNet(alpha=0.08, l1_ratio=0.5).fit(
    X,
    y,
    initial_coef=previous_coef,
)
```

## 进阶：求解器与优化

| `solver` 值 | CPU | CuPy / Torch | 说明 |
|---|:---:|:---:|---|
| `fista`（默认） | 支持 | 支持 | 推荐近端路径 |
| `auto` | FISTA | FISTA | 当前 squared-error + Elastic Net 分发 |
| `fista_bb` | 支持 | 支持 | 自适应谱步长 |
| `admm` | 支持 | 支持 | 替代拆分路径；仅均匀样本权重 |
| `coordinate_descent` | 支持 | 不支持 | CPU-only 兼容路径 |

`newton`、`lbfgs`、`irls`、`exact` 会被当前非光滑 Elastic Net estimator surface 拒绝。`cpu_solver` 不会覆盖单次拟合的 `solver`。

KKT 条件为

$$
\frac{1}{n}X^\top(X\hat\beta-y)
+\alpha(1-\lambda)\hat\beta
+\alpha\lambda\,\partial\lVert\hat\beta\rVert_1=0.
$$

`stopping="kkt"` 只是改变收敛诊断，不改变统计模型。

## 进阶：推断

`ElasticNet` 默认仅估计。设置 `compute_inference=True` 后运行拟合后推断，而不会改变已经得到的 penalized coefficients。

| `inference_method` | 用途 | 重要限制 |
|---|---|---|
| `debiased`（默认 inference method） | bias-corrected coefficient inference | 依赖去偏假设；推断条件于选定正则化参数 |
| `cpu_ols` | 轻量 post-selection OLS-style 路径 | 启发式，不是一般 selective-inference 保证 |
| `bootstrap` | 重采样替代路径 | 计算更贵，并依赖相应 bootstrap 假设 |

`cov_type` 与 `hac_maxlags` 也是 public constructor controls，在所选 inference path 支持相应 covariance 时使用。

## 常见误区

- 不要只调 `alpha` 而把 `l1_ratio` 当作无关参数。
- 相关变量一起非零不等于分别具有独立因果效应。
- 不要忘记标准化。
- 不要根据训练 $R^2$ 选择超参数。
- 弱信号下 active set 仍可能随数据微扰变化。
- 数据驱动选择后不要直接套普通未惩罚推断而忽略 selection。

## 完整 API 参考

前面的参数表是教学用选择指南；这里是当前 `ElasticNet` wrapper 的完整 constructor 与 model-method inventory。

### Constructor

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
| `l1_ratio` | `0.5` | L1 混合比例；0 更像 Ridge，1 更像 Lasso。 |
| `fit_intercept` | `True` | 拟合不受惩罚的截距。 |
| `max_iter` | `1000` | 最大求解迭代数。 |
| `tol` | `1e-4` | 数值收敛容差。 |
| `stopping` | `"coef_delta"` | 兼容路径使用 `coef_delta` 或 `kkt`。 |
| `device` | `"auto"` | `auto`、`cpu`、`cuda`（CuPy）或 `torch`（Torch CUDA）。 |
| `n_jobs` | `None` | 所选路径使用并行时的并行度提示。 |
| `solver` | `"fista"` | 单模型求解器。 |
| `cpu_solver` | `"fista"` | 兼容共享路径使用的 CPU helper/dispatch 控制。 |
| `lipschitz_L` | `None` | 兼容近端路径的预计算 Lipschitz 常数。 |
| `gpu_memory_cleanup` | `False` | 拟合后尽力释放缓存 GPU 内存。 |
| `compute_inference` | `False` | 执行所选拟合后推断。 |
| `inference_method` | `"debiased"` | 拟合后推断方法。 |
| `cov_type` | `"nonrobust"` | 所选推断路径使用 covariance 时的约定。 |
| `hac_maxlags` | `None` | 所选推断方法支持 HAC 时的滞后阶数。 |
<!-- API-CONSTRUCTOR-END:ElasticNet -->

### `fit`

wrapper 的直接签名为：

```python
model.fit(
    X=None,
    y=None,
    sample_weight=None,
    initial_coef=None,
    **kwargs,
)
```

`**kwargs` 当前可转发共享 fit 的 `formula` 与 `data`。

| 参数 | 含义 |
|---|---|
| `X` | 二维特征矩阵。 |
| `y` | 一维连续因变量。 |
| `sample_weight` | 可选非负分析权重；部分 solver 有额外限制。 |
| `initial_coef` | 可选 warm-start 系数向量，每个特征一个值。 |
| `formula` | 通过 `**kwargs` 转发的 Patsy 风格 Formula。 |
| `data` | Formula 接口使用的 DataFrame。 |

`fit()` 返回 `self`。

### 预测、评分与报告方法

| 方法 | 签名 | 行为 |
|---|---|---|
| `predict` | `predict(X, return_cpu=True)` | 连续预测；`return_cpu=False` 可让 GPU 预测留在实际 backend。 |
| `score` | `score(X, y, sample_weight=None)` | 返回 $R^2$，支持加权。 |
| `summary` | `summary()` | 打印系数/推断摘要；要求已拟合且推断可用。 |
| `get_params` / `set_params` | sklearn 风格工具 | 查看或替换 constructor 状态。 |

已有 p 值时，共享 estimator base 还提供 p 值校正/合并工具，见[推断 API](../guides/inference-api.md)。

### 拟合后属性与诊断量

| 属性 | 含义 / 可用条件 |
|---|---|
| `coef_` | 惩罚系数；精确 0 定义当前 active set。 |
| `intercept_` | 不受惩罚的截距。 |
| `n_iter_` | 所选数值路径迭代次数。 |
| `n_features_in_` | 相应拟合路径发布时的特征数。 |
| `rsquared`, `rsquared_adj` | 所需状态可用时的 $R^2$ 与调整 $R^2$。 |
| `fvalue`, `f_pvalue` | 在定义时可用的联合拟合统计量与 p 值。 |
| `llf`, `aic`, `bic` | 所需 reporting state 可用时的 Gaussian fit diagnostics。 |
| `_bse` | 所选 inference method 的标准误。 |
| `_tvalues` | 使用 t-style 语义的推断路径统计量。 |
| `_zvalues` | debiased inference 的 z-style 统计量。 |
| `_pvalues` | 推断成功时的系数 p 值。 |
| `_conf_int` | 推断成功时的 coefficient interval。 |
| `_inference_result` | 结构化推断结果与 metadata。 |

## 验证

维护中的验证覆盖 Elastic Net 目标函数、solver/KKT 行为、CPU/GPU 路径、post-fit inference、warm start 和 `ElasticNetCV` final-refit inference contract。

## 参考文献

- Zou, H., & Hastie, T. (2005). Regularization and variable selection via the Elastic Net. *Journal of the Royal Statistical Society: Series B*, 67(2), 301–320.
- Beck, A., & Teboulle, M. (2009). A fast iterative shrinkage-thresholding algorithm for linear inverse problems. *SIAM Journal on Imaging Sciences*, 2(1), 183–202.
