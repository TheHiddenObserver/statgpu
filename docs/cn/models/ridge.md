# 岭回归（Ridge）

> 语言：中文
> 最后更新：2026-09-05
> 切换：[English](../../en/models/ridge.md)

## 它解决什么问题？

`Ridge` 是加入 L2 惩罚的线性回归。它最适合处理这样一类问题：普通最小二乘（OLS）能够拟合数据，但由于若干特征高度相关，单个系数非常不稳定。

一个常见症状是**多重共线性**。两个特征如果携带几乎相同的信息，OLS 即使预测结果变化不大，也可能在这两个特征之间给出非常极端、非常敏感的系数分配。Ridge 接受少量偏差，换取更小、更稳定的系数。

它通常回答这些问题：

- 能否让线性模型对高度相关的特征更稳定？
- 能否在不直接删除变量的前提下降低系数方差？
- 当 OLS 在噪声方向上过拟合时，能否改善预测？

## 一个直观例子

假设有两个传感器测量几乎相同的物理量。它们都确实有用，但因为 `X` 中的两列几乎重复，OLS 有很多近乎等价的方法把信号分配给它们。

你可能看到类似：

```text
                 传感器 1   传感器 2
OLS 系数              0.32        2.08
Ridge 系数            1.09        1.09
```

两种模型的预测可能都不错，但 Ridge 往往更容易信任：它不需要让一个高度相关变量得到很大的正系数，再让另一个变量去补偿。

## 直觉

OLS 只问：

> 哪组系数能让训练数据上的预测误差最小？

Ridge 额外加了一个偏好：

> 如果两组模型拟合得差不多，优先选择系数更小的那一组。

从几何上看，L2 惩罚会阻止整个系数向量在任意方向上离零点太远。它会连续地**收缩**系数，但通常不会把系数精确压成 0。

因此要区分：

- Ridge 主要是**稳定化 / 收缩**方法；
- [Lasso](lasso.md) 会引入稀疏性，可以把部分系数压成精确的 0；
- [Elastic Net](elastic-net.md) 同时结合这两种行为。

## 什么时候使用？

以下情况很适合 Ridge：

- 因变量是连续变量，并且线性条件均值是合理近似；
- 多个特征高度相关；
- 特征很多，希望降低估计方差；
- 相比保持 OLS 的无偏性，更关心稳定预测；
- 希望所有变量仍留在模型中，而不是做硬特征选择。

以下情况应优先考虑其他方法：

- 明确需要一个很多系数为 0 的稀疏模型——优先考虑 Lasso 或 Elastic Net；
- 因变量是二分类、计数、生存时间等——使用相应的 GLM 或生存模型；
- 主要关系是非线性的，但尚未通过合适的特征变换表达；
- 目标是因果解释，但研究设计本身并不能支持因果结论。

## 模型与目标函数

带截距的线性模型为

$$
y_i=b+x_i^\top\beta+\varepsilon_i.
$$

Ridge 通过最小化

$$
\frac{1}{2n}\sum_{i=1}^{n}
\left(y_i-b-x_i^\top\beta\right)^2
+\frac{\alpha}{2}\lVert\beta\rVert_2^2
$$

来估计截距 $b$ 和系数向量 $\beta$。

其中：

- $n$ 是样本数；
- $x_i$ 是第 $i$ 个样本的特征向量；
- $\alpha\ge 0$ 控制收缩强度；
- 截距不受惩罚。

`alpha` 越大，模型越愿意牺牲一部分训练拟合来换取更小的系数。当 `alpha=0` 时，目标函数退化为 OLS。

### 为什么 L2 能改善共线性？

中心化后，Ridge 的 normal equation 为

$$
\left(X_c^\top X_c+n\alpha I\right)\hat\beta
=X_c^\top y_c.
$$

新增的 $n\alpha I$ 会抑制那些很弱、或者几乎共线的方向，因此即使 $X^\top X$ 条件很差，Ridge 也往往能保持更稳定。

## 最小可运行示例

下面这个例子故意构造两个几乎重复的特征，直接展示 Ridge 与 OLS 的区别。

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

ols = LinearRegression(
    device="cpu",
    compute_inference=False,
).fit(X, y)

ridge = Ridge(
    alpha=0.2,
    device="cpu",
    compute_inference=False,
).fit(X, y)

print("OLS coefficients:  ", np.round(ols.coef_, 3))
print("Ridge coefficients:", np.round(ridge.coef_, 3))
print("Ridge R²:", round(ridge.score(X, y), 3))
```

固定随机种子后，前两个 OLS 系数大约是 `0.32` 和 `2.08`，而 Ridge 大约会给出 `1.09` 和 `1.09`。这里重点不是 Ridge 找到了唯一“真实”的分配，而是：面对几乎重复的两列，系数分配明显不再那么敏感。

## 如何理解结果？

- `coef_[j]` 是**经过收缩后的**系数。它不是“OLS 系数加了一个显示修正”，惩罚项确实改变了估计量。
- `intercept_` 是截距，不受惩罚。
- `predict(X_new)` 返回连续因变量的预测值。
- `score(X, y)` 返回 $R^2$。
- 系数变小不代表变量的科学意义一定下降；其中一部分变化只是为了稳定性引入的 regularization bias。

如果你关心系数不确定性，可以设置 `compute_inference=True`。Ridge 支持标准误、检验统计量、p 值和置信区间，但这些推断都应理解为**在当前 `alpha` 已经选定的条件下**进行。

## 关键参数应该怎么选？

| 参数 | 默认值 | 应该怎么理解 |
|---|---:|---|
| `alpha` | `1.0` | 最重要的建模参数。越大，收缩越强。预测任务通常应使用 `RidgeCV` 或其他验证程序选择，而不是只看训练拟合。 |
| `fit_intercept` | `True` | 一般保持开启，除非理论上截距就是 0，或设计矩阵已经显式包含截距。 |
| `device` | `"auto"` | 小中型问题优先 CPU；数据规模足够大、能覆盖传输和初始化成本时再考虑 `"cuda"` 或 `"torch"`。 |
| `compute_inference` | `True` | 只需要预测或系数时可以关闭，避免额外推断开销。 |
| `cov_type` | `"nonrobust"` | 异方差时使用 HC；有真实时间顺序和序列相关时使用 HAC。 |
| `solver` | `"exact"` | 普通 Ridge 的默认直接求解路径。通常无需修改，除非做受控数值实验或特殊工作负载。 |

### 先考虑特征尺度

正则化直接作用于系数大小。如果一个特征用“米”，另一个用“微米”，同样的预测作用可能需要数量级完全不同的系数，于是它们会承受不同程度的惩罚。

因此，绝大多数 regularized workflow 都应先标准化连续特征，除非你有意让原始尺度本身参与 penalty 的含义。

## 与相近方法比较

| 方法 | 惩罚的作用 | 会产生精确 0 系数吗？ | 更适合的场景 |
|---|---|:---:|---|
| OLS / `LinearRegression` | 不收缩 | 否 | 设计稳定，希望使用未惩罚线性估计量 |
| **Ridge** | L2 收缩 | 通常不会 | 特征相关，希望稳定预测，同时保留所有变量 |
| [Lasso](lasso.md) | L1 收缩 | 会 | 希望得到稀疏模型 / 自动特征选择 |
| [Elastic Net](elastic-net.md) | L1 + L2 | 会 | 既想要稀疏性，又希望相关变量比纯 Lasso 更稳定 |

可以记成：

```text
OLS        ：只拟合
Ridge      ：拟合 + 收缩
Lasso      ：拟合 + 收缩 + 选择
Elastic Net：拟合 + 收缩 + 选择，同时更照顾相关变量组
```

## CPU、GPU 与加权拟合

同一个公开估计器可在支持时使用 NumPy CPU、CuPy CUDA 或 Torch CUDA。

```python
from statgpu.linear_model import Ridge

model = Ridge(
    alpha=0.2,
    device="cuda",
    compute_inference=False,
).fit(X, y)
```

分析权重可直接传入：

```python
weighted = Ridge(alpha=0.2).fit(X, y, sample_weight=w)
```

statgpu 会用 `sum(sample_weight)` 归一化加权拟合项。因此，把全部权重同时乘以同一个正数不会改变拟合结果。

## 进阶：求解器支持

`Ridge` 的公开默认值是模型专属的 `solver="exact"`。

| `solver` 值 | CPU | CuPy / Torch | 主要用途 |
|---|:---:|:---:|---|
| `exact`（默认） | 支持 | 支持 | 稠密 L2 直接求解；普通 Ridge 首选 |
| `auto` | exact | Newton | 后端感知自动分发 |
| `fista` / `fista_bb` | 支持 | 支持 | 迭代路径对照或受控优化实验 |
| `newton` / `lbfgs` | 支持 | 支持 | 光滑目标的替代路径 |
| `admm` | 支持 | 支持 | 实验性拆分路径；仅均匀样本权重 |
| `irls` | 不支持 | 不支持 | squared-error loss 没有 IRLS contract |

`coordinate_descent`、quantile coordinate descent 和 L-BFGS-B 都不是 Ridge 估计器选项。通用迭代公式见[求解器算法指南](../guides/solver-algorithms.md)。

## 进阶：推断与目标函数尺度

支持的协方差类型包括 `nonrobust`、`hc0`、`hc1`、`hc2`、`hc3` 和 `hac`。启用 `compute_inference=True` 后，公开报告接口包括 `_bse`、`_tvalues`、`_pvalues`、`_conf_int`，以及在定义时可用的 `rsquared`、`rsquared_adj`、`fvalue`、`aic`、`bic` 等诊断量。

加权推断使用与拟合一致的分析权重定义。协方差和参考分布相关数值工作先在实际执行的 NumPy/CuPy/Torch backend 上完成，随后公开 reporting 数组再 snapshot 为 NumPy。

### 与 scikit-learn 比较 `alpha`

statgpu 使用上面的平均损失目标，而 scikit-learn Ridge 使用未归一化残差平方和。比较系数时需要映射：

- 无权重：`sklearn_alpha = n_samples * statgpu_alpha`；
- 加权：`sklearn_alpha = sample_weight.sum() * statgpu_alpha`。

因此，两边直接使用相同数值的 `alpha`，比较的其实是不同目标函数。

## 常见误区

- **不要只根据训练 $R^2$ 选 `alpha`。** 训练拟合几乎总偏向更弱的正则化；真正需要验证的是 bias-variance trade-off。
- **不要把收缩误解为变量删除。** Ridge 通常会保留全部系数。
- **比较系数大小之前先看特征尺度。** regularization 场景下标准化通常非常重要。
- **不要把其他库的 `alpha` 数值直接复制过来。** 先确认目标函数归一化是否一致。
- **正则化不能修复模型设定错误。** 非线性、依赖结构、遗漏变量或混杂不会因为 Ridge 自动消失。
- **调完 `alpha` 后的小 p 值并不等于完整的模型选择推断。** 除非推断程序明确考虑 tuning，否则应把结果理解为条件于已选正则参数。

## API 与验证

导入：

```python
from statgpu.linear_model import Ridge
```

构造函数还提供 `max_iter`、`tol`、`cpu_solver`、`lipschitz_L`、`hac_maxlags`、`gpu_memory_cleanup` 等进阶参数。它们被有意放在本页后半段，因为大多数用户首先应该决定的是“Ridge 是否适合”和“`alpha` 怎么选”，而不是数值求解器细节。

维护中的测试会验证平均损失闭式解、共享 penalized-linear engine、加权拟合、exact/FISTA 一致性、formula 对齐、推断、RidgeCV final-refit 行为，以及 backend-native inference 和真实 GPU acceptance contract。

## 参考文献

- Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression: Biased estimation for nonorthogonal problems. *Technometrics*, 12(1), 55–67.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.
