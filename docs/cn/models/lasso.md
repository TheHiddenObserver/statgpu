# Lasso 回归

> 语言：中文
> 最后更新：2026-09-05
> 切换：[English](../../en/models/lasso.md)

## 它解决什么问题？

`Lasso` 是加入 L1 惩罚的线性回归。它最有辨识度的特点是：正则化可以把一部分系数**精确压成 0**，因此模型会同时完成预测和特征选择。

假设你有 100 个候选特征，但相信真正有用的只有少数几个。OLS 会给每个特征一个系数；Ridge 会把所有系数缩小，但通常不会让它们变成 0；Lasso 则可以直接返回一个很小的 active set。

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

Lasso 的核心吸引力，就是不需要额外的硬特征选择步骤，也能得到稀疏线性表示。

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
- 希望得到较小的 active set；
- 存储、部署或下游建模能从删除特征中获益；
- 愿意通过验证来选择正则化强度。

优先考虑其他方法，当：

- 大多数特征都可能有小但真实的作用——考虑 [Ridge](ridge.md)；
- 特征形成高度相关的组，希望组内变量一起保留/收缩——考虑 [Elastic Net](elastic-net.md)；
- 响应不适合 Gaussian 线性回归——使用相应 penalized GLM、生存模型等；
- 首要目标是严格 post-selection inference——需要更谨慎的选择后推断设计。

## 模型与目标函数

带截距 $b$ 时，Lasso 最小化

$$
\frac{1}{2n}\sum_{i=1}^{n}
\left(y_i-b-x_i^\top\beta\right)^2
+\alpha\lVert\beta\rVert_1.
$$

`alpha` 越大，收缩和精确 0 通常越多；截距不受惩罚。

### 为什么 L1 会产生精确 0？

绝对值惩罚在 0 处有尖角。优化中的 soft-thresholding 为

$$
\mathcal S_\lambda(z)
=
\operatorname{sign}(z)\max(|z|-\lambda,0).
$$

当 $|z|\le\lambda$ 时，结果就是精确的 0。这是 Lasso 稀疏性的直接计算机制。

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

固定随机种子和这个 `alpha` 后，明显非零的系数应在索引 `1`、`5`、`9`，大约为 `1.97`、`-1.41`、`0.73`。它们比生成系数更小，是因为收缩本来就是估计量的一部分。

## 如何理解结果？

- `coef_[j] == 0` 表示在当前 `alpha` 下，该特征没有进入拟合线性 predictor。
- 非零系数仍然经过收缩，不能当作未惩罚 OLS 系数。
- `intercept_` 不受 L1 惩罚。
- `predict(X_new)` 返回连续预测。
- `score(X, y)` 返回 $R^2$，并支持 `sample_weight=`。
- `n_iter_` 是所选数值路径的迭代次数。

一个特征在某个样本中被压成 0，并不证明总体中的真实作用必然为 0。

## 关键参数应该怎么选？

这里是正常工作流的**精选参数表**。完整 constructor 见[完整 API 参考](#完整-api-参考)。

| 参数 | 默认值 | 应该怎么理解 |
|---|---:|---|
| `alpha` | `1.0` | 最重要的统计选择；越大通常收缩越强、0 越多。预测/选择任务优先用 `LassoCV`。 |
| `fit_intercept` | `True` | 一般保持开启，除非理论固定截距或设计矩阵已有截距。 |
| `device` | `"auto"` | 小问题优先 CPU；工作负载足够大时再考虑 GPU。 |
| `solver` | `"fista"` | 单模型拟合的稳定默认路径，改变它主要是数值/性能选择。 |
| `stopping` | `"coef_delta"` | 需要按最优性而不是系数变化判断收敛时可用 `"kkt"`。 |
| `compute_inference` | `True` | 只做预测/选择时可关闭；需要推断时应显式理解 `inference_method` 的统计含义。 |

### 正则化之前先标准化

L1 直接惩罚系数大小。特征量纲差异会导致有效惩罚不同，因此多数 Lasso workflow 应先标准化连续变量。

## 与相近方法比较

| 方法 | 相关变量组 | 精确 0？ | 典型选择理由 |
|---|---|:---:|---|
| OLS / `LinearRegression` | 无正则化 | 否 | 未惩罚估计 |
| [Ridge](ridge.md) | 倾向共享信号 | 否 | 稳定相关变量，不删除特征 |
| **Lasso** | 可能只选组内一个 | 是 | 稀疏预测 / 自动特征选择 |
| [Elastic Net](elastic-net.md) | 比 Lasso 更照顾组 | 是 | 稀疏模型 + 相关变量 |
| Adaptive Lasso | 数据依赖 L1 权重 | 是 | 在更强稀疏假设下减轻统一惩罚偏差 |

## CPU、GPU、Formula 与加权拟合

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

显式 `device="cuda"` / `"torch"` 使用相应 GPU backend；不可用时应失败，而不是静默换执行路径。

`fit()` 同时支持 `sample_weight=`，以及共享的 `formula=` / `data=` 接口。Formula 元数据会用于后续 DataFrame prediction。

## 进阶：求解器支持

| `solver` 值 | CPU | CuPy / Torch | 含义 |
|---|:---:|:---:|---|
| `fista`（默认） | 支持 | 支持 | L1 目标的稳定近端梯度路径 |
| `auto` | FISTA | FISTA | 当前 squared-error + L1 自动分发 |
| `fista_bb` | 支持 | 支持 | Barzilai-Borwein 步长 FISTA |
| `admm` | 支持 | 支持 | 替代拆分路径；仅均匀样本权重 |
| `coordinate_descent` | 支持 | 不支持 | CPU-only 兼容路径 |

L1 目标拒绝 `newton`、`lbfgs`、`irls`、`exact`。`cpu_solver` 由 Lasso CV/path helper 使用，不会覆盖单次 `Lasso.fit` 的 `solver`。

`admm_rho` 控制 ADMM penalty parameter；`lipschitz_L` 可为兼容近端路径提供预计算 Lipschitz 常数。

## 进阶：Lasso 之后的推断

选择后推断比预先指定的 OLS 模型推断困难得多。不同 `inference_method` 代表不同统计主张：

| `inference_method` | 用途 | 重要限制 |
|---|---|---|
| `cpu_ols_inference` | 轻量 CPU post-selection diagnostic | 启发式 OLS-style 区间，不是严格 selective inference |
| `gpu_ols_inference` | 减少 GPU→CPU 大块传输 | 同样存在 selection validity 限制 |
| `debiased`（constructor 默认） | de-biased / de-sparsified 推断 | `_conf_int` 当前是单系数 marginal interval，且依赖高维去偏假设 |
| `bootstrap` | residual-bootstrap 替代路径 | 计算更贵，并依赖相应重采样假设 |

`n_bootstrap` 与 `bootstrap_random_state` 控制 bootstrap 路径。推断成功后可以得到 `_bse`、`_tvalues` 或 `_zvalues`、`_pvalues`、`_conf_int`。

可选 simultaneous interval：

```python
model = Lasso(
    alpha=0.08,
    inference_method="debiased",
    enable_simultaneous_inference=True,
    simultaneous_method="maxz_bootstrap",
    simultaneous_alpha=0.05,
    simultaneous_n_bootstrap=1000,
    simultaneous_random_state=7,
).fit(X, y)
```

Simultaneous inference 要求 `compute_inference=True`、`inference_method="debiased"` 和 `simultaneous_method="maxz_bootstrap"`。

## 常见误区

- “被选择”不等于“总体真实非零”，更不等于因果。
- L1 不具备尺度不变性，不能忽略标准化。
- 高度相关变量之间的纯 Lasso 选择可能很不稳定。
- 不要根据训练 $R^2$ 选择 `alpha`。
- 不要在数据驱动选择后直接附上普通 OLS p 值并当作预先指定模型推断。
- 数值收敛（例如 KKT 很小）不等于统计模型正确。

## 完整 API 参考

前面的参数表是教学用的选择指南；这里是当前 `Lasso` wrapper 的完整 constructor 和 model-method inventory。

### Constructor

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
| `inference_method` | `"debiased"` | 拟合后推断路径。 |
| `n_bootstrap` | `200` | `inference_method="bootstrap"` 时 residual-bootstrap 抽样次数。 |
| `bootstrap_random_state` | `None` | residual bootstrap 随机种子。 |
| `enable_simultaneous_inference` | `False` | 在 debiased inference 后启用 simultaneous max-|Z| 区间。 |
| `simultaneous_method` | `"maxz_bootstrap"` | simultaneous calibration 方法；当前为 `maxz_bootstrap`。 |
| `simultaneous_alpha` | `0.05` | simultaneous interval 的 family-wise error level。 |
| `simultaneous_n_bootstrap` | `1000` | max-|Z| multiplier-bootstrap 次数。 |
| `simultaneous_random_state` | `None` | simultaneous bootstrap 随机种子。 |
| `simultaneous_include_intercept` | `False` | 在支持时把截距纳入 simultaneous target family。 |
| `device` | `"auto"` | `auto`、`cpu`、`cuda`（CuPy）或 `torch`（Torch CUDA）。 |
| `n_jobs` | `None` | 所选路径使用并行时的并行度提示。 |
| `compute_inference` | `True` | 执行所选拟合后推断。 |
| `solver` | `"fista"` | 单模型求解器。 |
| `cpu_solver` | `"coordinate_descent"` | Lasso CV/path helper 的 CPU 选择，不会替代单次拟合的 `solver`。 |
| `lipschitz_L` | `None` | 兼容近端路径的预计算 Lipschitz 常数。 |
| `admm_rho` | `1.0` | ADMM augmented-Lagrangian penalty parameter。 |
| `gpu_memory_cleanup` | `False` | 拟合后尽力释放缓存 GPU 内存。 |
<!-- API-CONSTRUCTOR-END:Lasso -->

### `fit`

共享 penalized-linear fit 签名为：

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
| `sample_weight` | 可选非负分析权重；部分 solver 还有额外限制。 |
| `formula` | 可选 Patsy 风格 Formula。 |
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
| `coef_` | 惩罚系数；精确 0 定义当前 fitted active set。 |
| `intercept_` | 不受惩罚的截距。 |
| `n_iter_` | 所选数值路径迭代次数。 |
| `n_features_in_` | 相应拟合路径发布时的特征数。 |
| `rsquared`, `rsquared_adj` | 所需状态可用时的 $R^2$ 与调整 $R^2$。 |
| `fvalue`, `f_pvalue` | 在定义时可用的联合拟合统计量与 p 值。 |
| `llf`, `aic`, `bic` | 所需 reporting state 可用时的 Gaussian fit diagnostics。 |
| `_bse` | 所选 inference method 产生的标准误。 |
| `_tvalues` | 使用 t-style 语义的推断路径统计量。 |
| `_zvalues` | debiased inference 的 z-style 统计量。 |
| `_pvalues` | 推断成功时的系数 p 值。 |
| `_conf_int` | 推断成功时的 marginal coefficient interval。 |
| `_conf_int_simultaneous` | 显式启用并成功校准时的 simultaneous interval。 |
| `_inference_result` | 结构化推断结果与 metadata。 |

下划线推断属性在当前版本中是既有 reporting surface；其统计含义取决于 `inference_method`。

## 验证

维护中的验证覆盖 solver 收敛、CPU/GPU 一致性、KKT stopping、debiased inference、bootstrap、simultaneous inference 和 physical-GPU 行为。

## 参考文献

- Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *Journal of the Royal Statistical Society: Series B*, 58(1), 267–288.
- Bühlmann, P., & van de Geer, S. (2011). *Statistics for High-Dimensional Data*. Springer.
- Zhang, C.-H., & Zhang, S. S. (2014). Confidence intervals for low-dimensional parameters in high-dimensional linear models. *JRSS B*, 76(1), 217–242.
- Javanmard, A., & Montanari, A. (2014). Confidence intervals and hypothesis testing for high-dimensional regression. *JMLR*, 15, 2869–2909.
