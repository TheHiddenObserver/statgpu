# 岭回归（Ridge）

> 语言：中文
> 最后更新：2026-09-05
> 切换：[English](../../en/models/ridge.md)

## 它解决什么问题？

`Ridge` 是加入 L2 惩罚的线性回归。它适合普通最小二乘（OLS）能够拟合数据、但单个系数因为特征高度相关而非常不稳定的情况。

一个常见症状是**多重共线性**：两个特征携带几乎相同的信息时，OLS 的预测可能变化不大，但系数会在两者之间剧烈摇摆。Ridge 接受一定 regularization bias，换取更小、更稳定的系数。

典型问题包括：

- 能否让线性模型对高度相关的特征更稳定？
- 能否在不直接删除变量的前提下降低系数方差？
- 当 OLS 在噪声方向上过拟合时，能否改善预测？

## 一个直观例子

假设两个传感器测量几乎相同的物理量。两列 `X` 几乎重复，OLS 有很多近乎等价的方式把信号分配给它们：

```text
                 传感器 1   传感器 2
OLS 系数              0.32        2.08
Ridge 系数            1.09        1.09
```

两种模型的预测都可能不错，但 Ridge 的系数分配通常更稳定。

## 直觉

OLS 只问：

> 哪组系数能让训练数据上的预测误差最小？

Ridge 额外加了一个偏好：

> 如果两组模型拟合得差不多，优先选择系数更小的那一组。

L2 惩罚会连续地**收缩**系数，但通常不会把系数精确压成 0。因此：

- Ridge 主要用于**稳定化 / 收缩**；
- [Lasso](lasso.md) 可以产生精确 0，适合稀疏选择；
- [Elastic Net](elastic-net.md) 同时结合两者。

## 什么时候使用？

Ridge 很适合：

- 因变量连续，线性条件均值是合理近似；
- 多个特征高度相关；
- 特征很多，希望降低估计方差；
- 相比保留经典未惩罚 OLS 估计量，更关心稳定预测；
- 希望保留所有变量，而不是硬删除。

优先考虑其他方法，当：

- 明确需要稀疏模型——考虑 Lasso 或 Elastic Net；
- 因变量是二分类、计数或生存时间——使用相应 GLM / 生存模型；
- 主要关系是非线性的，但还没有通过合适特征表达；
- 目标是因果解释，但研究设计本身不能支持因果结论。

## 模型与目标函数

带截距的线性模型为

$$
y_i=b+x_i^\top\beta+\varepsilon_i.
$$

Ridge 最小化

$$
\frac{1}{2n}\sum_{i=1}^{n}
\left(y_i-b-x_i^\top\beta\right)^2
+\frac{\alpha}{2}\lVert\beta\rVert_2^2.
$$

其中 $\alpha\ge0$ 控制收缩强度，截距不受惩罚。`alpha=0` 时退化为 OLS。

中心化后的 normal equation 为

$$
\left(X_c^\top X_c+n\alpha I\right)\hat\beta=X_c^\top y_c.
$$

新增的 $n\alpha I$ 会抑制弱方向和近共线方向，因此可以改善数值与统计稳定性。

## 最小可运行示例

```python
import numpy as np
from statgpu.linear_model import LinearRegression, Ridge

rng = np.random.default_rng(0)
n = 400
shared = rng.normal(size=n)
X = np.column_stack([
    shared + 0.03 * rng.normal(size=n),
    shared + 0.03 * rng.normal(size=n),
    rng.normal(size=n),
])
y = (
    1.0
    + 1.2 * X[:, 0]
    + 1.2 * X[:, 1]
    + 0.5 * X[:, 2]
    + rng.normal(scale=0.8, size=n)
)

ols = LinearRegression(device="cpu", compute_inference=False).fit(X, y)
ridge = Ridge(alpha=0.2, device="cpu", compute_inference=False).fit(X, y)

print("OLS coefficients:  ", np.round(ols.coef_, 3))
print("Ridge coefficients:", np.round(ridge.coef_, 3))
print("Ridge R²:", round(ridge.score(X, y), 3))
```

固定随机种子后，前两个 OLS 系数大约是 `0.32` 和 `2.08`，Ridge 则接近 `1.09` 和 `1.09`。重点不是 Ridge 找到了唯一“真实”的分配，而是系数对几乎重复的列不再那么敏感。

## 如何理解结果？

- `coef_[j]` 是经过收缩后的系数；惩罚项改变了估计量本身。
- `intercept_` 是不受惩罚的截距。
- `predict(X_new)` 返回连续预测。
- `score(X, y)` 返回 $R^2$，并支持 `sample_weight=`。
- 系数变小不等于科学意义一定变弱，其中可能只是 regularization bias。

若需要系数不确定性，可设置 `compute_inference=True`。推断应理解为条件于当前选定的 `alpha`。

## 关键参数应该怎么选？

这里是**教学用的精选参数表**，并不是完整 API。完整 constructor、方法和拟合后属性见[完整 API 参考](#完整-api-参考)。

| 参数 | 默认值 | 应该怎么理解 |
|---|---:|---|
| `alpha` | `1.0` | 最重要的建模参数；越大收缩越强。预测任务优先用 `RidgeCV` 或验证程序选择。 |
| `fit_intercept` | `True` | 一般保持开启，除非理论固定截距为 0 或设计矩阵已有截距。 |
| `device` | `"auto"` | 小中型问题优先 CPU；规模足够大时再考虑 CUDA/Torch。 |
| `compute_inference` | `True` | 只需要预测/系数时可关闭，避免推断开销。 |
| `cov_type` | `"nonrobust"` | 异方差考虑 HC；有真实顺序与序列相关时考虑 HAC。 |
| `solver` | `"exact"` | 普通 Ridge 默认直接求解路径，通常不需要修改。 |

### 先考虑特征尺度

正则化直接作用于系数大小。多数 regularized workflow 应先标准化连续特征，除非你有意让原始单位参与 penalty 的含义。

## 与相近方法比较

| 方法 | 惩罚作用 | 精确 0？ | 更适合的场景 |
|---|---|:---:|---|
| OLS / `LinearRegression` | 不收缩 | 否 | 设计稳定，希望使用未惩罚估计量 |
| **Ridge** | L2 收缩 | 通常否 | 特征相关，希望稳定预测并保留所有变量 |
| [Lasso](lasso.md) | L1 收缩 | 是 | 稀疏模型 / 自动特征选择 |
| [Elastic Net](elastic-net.md) | L1 + L2 | 是 | 稀疏性 + 相关变量稳定性 |

## CPU、GPU、Formula 与加权拟合

```python
from statgpu.linear_model import Ridge

model = Ridge(
    alpha=0.2,
    device="cuda",
    compute_inference=False,
).fit(X, y)
```

分析权重：

```python
weighted = Ridge(alpha=0.2).fit(X, y, sample_weight=w)
```

`fit()` 也支持 `formula=` 与 `data=`；Formula 元数据会保留下来，以便 DataFrame 预测重建一致的设计矩阵。

## 进阶：求解器支持

| `solver` 值 | CPU | CuPy / Torch | 用途 |
|---|:---:|:---:|---|
| `exact`（默认） | 支持 | 支持 | 稠密 L2 直接求解 |
| `auto` | exact | Newton | 后端感知分发 |
| `fista` / `fista_bb` | 支持 | 支持 | 迭代对照 / 特殊工作负载 |
| `newton` / `lbfgs` | 支持 | 支持 | 光滑目标替代路径 |
| `admm` | 支持 | 支持 | 实验性拆分路径，仅均匀样本权重 |
| `irls` | 不支持 | 不支持 | squared-error 没有 IRLS contract |

通用迭代机制见[求解器算法指南](../guides/solver-algorithms.md)。

## 进阶：推断与目标函数尺度

支持 `nonrobust`、`hc0`、`hc1`、`hc2`、`hc3` 和 `hac`。推断启用后，可得到 `_bse`、`_tvalues`、`_pvalues`、`_conf_int`，以及在相应状态下的 `rsquared`、`rsquared_adj`、`fvalue`、`f_pvalue`、`llf`、`aic`、`bic`。

与 scikit-learn 比较 `alpha` 时注意目标函数尺度：

- 无权重：`sklearn_alpha = n_samples * statgpu_alpha`；
- 加权：`sklearn_alpha = sample_weight.sum() * statgpu_alpha`。

## 常见误区

- 不要只用训练 $R^2$ 选择 `alpha`。
- 不要把 Ridge 收缩误解为变量删除。
- 比较系数大小前先检查特征尺度。
- 不要直接复制其他库的 `alpha` 数值而忽略目标函数缩放。
- 正则化不能自动修复非线性、依赖结构或混杂。
- 调参之后的小 p 值不等于完成了 selection-aware inference。

## 完整 API 参考

这里是当前 `Ridge` wrapper 的完整 public API inventory。

### Constructor

```python
Ridge(
    alpha=1.0,
    fit_intercept=True,
    device="auto",
    n_jobs=None,
    gpu_memory_cleanup=False,
    compute_inference=True,
    cov_type="nonrobust",
    hac_maxlags=None,
    max_iter=1000,
    tol=1e-4,
    solver="exact",
    cpu_solver="fista",
    lipschitz_L=None,
)
```

<!-- API-CONSTRUCTOR-START:Ridge -->
| 参数 | 默认值 | API 含义 |
|---|---:|---|
| `alpha` | `1.0` | 平均损失尺度下的 L2 惩罚强度。 |
| `fit_intercept` | `True` | 拟合不受惩罚的截距。 |
| `device` | `"auto"` | `auto`、`cpu`、`cuda`（CuPy）或 `torch`（Torch CUDA）。 |
| `n_jobs` | `None` | 所选路径使用并行时的并行度提示。 |
| `gpu_memory_cleanup` | `False` | 拟合后尽力释放缓存 GPU 内存。 |
| `compute_inference` | `True` | 计算标准误、检验、区间与 summary 状态。 |
| `cov_type` | `"nonrobust"` | `nonrobust`、`hc0`、`hc1`、`hc2`、`hc3` 或 `hac`。 |
| `hac_maxlags` | `None` | HAC 最大滞后阶。 |
| `max_iter` | `1000` | 迭代路径最大迭代次数。 |
| `tol` | `1e-4` | 数值收敛容差。 |
| `solver` | `"exact"` | 估计器级求解器选择。 |
| `cpu_solver` | `"fista"` | 兼容共享路径使用的 CPU helper/dispatch 控制。 |
| `lipschitz_L` | `None` | 兼容迭代路径可使用的预计算 Lipschitz 常数。 |
<!-- API-CONSTRUCTOR-END:Ridge -->

### `fit`

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
| `X` | 数组接口二维特征矩阵。 |
| `y` | 一维连续因变量。 |
| `sample_weight` | 可选非负分析权重，总和必须有限且为正。 |
| `formula` | 可选 Patsy 风格 Formula，与 `data` 一起使用。 |
| `data` | Formula 接口使用的 DataFrame。 |

`fit()` 返回 `self`。

### 预测、评分与报告方法

| 方法 | 签名 | 行为 |
|---|---|---|
| `predict` | `predict(X, return_cpu=True)` | 连续预测；GPU 拟合后 `return_cpu=False` 可让结果保留在 CuPy/Torch backend。 |
| `score` | `score(X, y, sample_weight=None)` | 返回 $R^2$，支持加权。 |
| `summary` | `summary()` | 打印 Ridge 推断摘要；要求已拟合且推断可用。 |
| `get_params` / `set_params` | sklearn 风格工具 | 查看或替换 constructor 状态。 |

已有 p 值时，共享 estimator base 还提供 p 值校正/合并工具，见[推断 API](../guides/inference-api.md)。

### 拟合后属性与诊断量

| 属性 | 含义 / 可用条件 |
|---|---|
| `coef_` | 惩罚系数。 |
| `intercept_` | 不受惩罚的截距。 |
| `n_iter_` | 迭代次数；direct exact path 为一次 solve。 |
| `n_features_in_` | 相应拟合路径发布时的特征数。 |
| `rsquared`, `rsquared_adj` | $R^2$ 与调整 $R^2$。 |
| `fvalue`, `f_pvalue` | 在定义时可用的联合拟合统计量与 p 值。 |
| `llf`, `aic`, `bic` | 所需状态可用时的 Gaussian log-likelihood 与信息准则。 |
| `_bse` | 系数标准误。 |
| `_tvalues` | Ridge t-style 统计量。 |
| `_pvalues` | 系数 p 值。 |
| `_conf_int` | 系数置信区间。 |
| `_inference_result` | 结构化推断结果与 metadata。 |

下划线开头的推断数组属于当前版本既有 reporting 属性；需要长期 schema 稳定性的代码应优先使用高层 reporting 接口。

## 验证

维护中的测试覆盖平均损失闭式解、加权拟合、exact/FISTA、Formula 行对齐、推断、RidgeCV final-refit 与 backend-native inference contract。

## 参考文献

- Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression: Biased estimation for nonorthogonal problems. *Technometrics*, 12(1), 55–67.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.
