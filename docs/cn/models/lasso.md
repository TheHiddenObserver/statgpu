# Lasso 回归

> 语言：中文
> 最后更新：2026-09-05
> 切换：[English](../../en/models/lasso.md)

## 它解决什么问题？

`Lasso` 是加入 L1 惩罚的线性回归。它最有辨识度的特点是：正则化可以把一部分系数**精确压成 0**，因此模型会同时完成预测和特征选择。

假设你有 100 个候选特征，但相信真正有用的只有少数几个。OLS 会给每个特征一个系数；Ridge 会把所有系数缩小，但通常不会让它们变成 0；Lasso 则可以直接返回一个很小的 active set。

它通常回答这些问题：

- 哪些特征可以从线性模型中移除，同时保留主要预测能力？
- 当候选特征很多、绝大部分只是噪声时，能否减少过拟合？
- 能否得到一个更容易解释、存储或部署的稀疏模型？

## 一个直观例子

想象一个数据集有 12 个测量变量，但真实响应实际上只由其中 3 个生成。

未正则化模型可能给很多噪声变量分配小系数，而 Lasso 更倾向于得到：

```text
特征          0    1    2    3    4    5    6    7    8    9   10   11
真实系数      0   2.0   0    0    0  -1.5   0    0    0   0.8   0    0
Lasso 系数    0  ~2.0   0    0    0  ~-1.4  0    0    0  ~0.7   0    0
```

这就是 Lasso 的核心吸引力：它不需要额外做一次硬特征筛选，就能把连续回归问题直接转化为稀疏表示。

## 直觉

Lasso 会按照系数绝对值的大小收取“代价”：

$$
\lVert\beta\rVert_1=\sum_j|\beta_j|.
$$

这会产生一种**软阈值（soft-thresholding）**效果。证据较弱的系数不仅会变小，还可能被一路拉到 0。

可以这样记：

```text
OLS   ：只保留能改善拟合的东西
Ridge ：所有系数一起收缩
Lasso ：收缩，并把弱系数直接删掉
```

稀疏性的代价是：当多个特征携带几乎相同的信息时，选择可能不稳定。纯 Lasso 可能保留相关变量组中的一个、丢掉另一个，而且这个决定会随样本轻微变化。[Elastic Net](elastic-net.md) 往往更适合这种场景。

## 什么时候使用？

Lasso 特别适合：

- 你相信真实信号本身比较稀疏；
- 候选特征数量相对样本量较大；
- 希望最终模型只保留少量 active features；
- 存储、部署或下游流程会从“删除特征”中直接受益；
- 愿意通过验证数据选择正则化强度。

以下情况应优先考虑其他方法：

- 大多数特征都可能有小但真实的作用——[Ridge](ridge.md) 往往更合适；
- 特征明显成组高度相关，希望同组变量一起保留/收缩——优先尝试 [Elastic Net](elastic-net.md)；
- 因变量并不适合 Gaussian 线性回归——使用相应的 penalized GLM、生存或其他模型族；
- 首要目标是严格的选择后推断，而不是预测/筛选——post-selection inference 需要额外假设和专门处理。

## 模型与目标函数

带截距 $b$ 时，Lasso 最小化

$$
\frac{1}{2n}\sum_{i=1}^{n}
\left(y_i-b-x_i^\top\beta\right)^2
+\alpha\lVert\beta\rVert_1.
$$

其中：

- $n$ 是样本数；
- $x_i$ 是第 $i$ 个样本的特征向量；
- $\beta$ 是系数向量；
- $\alpha\ge 0$ 控制正则化强度；
- 截距不参与 L1 惩罚。

`alpha` 越大，出现 0 系数的可能性越高。`alpha` 很小时接近未正则化线性拟合；过大时则可能把真正有用的信号也一起删除。

### 为什么 L1 会产生精确的 0？

绝对值惩罚在 0 点有一个尖角。优化时，小的系数更新会经过软阈值：

$$
\mathcal S_\lambda(z)
=
\operatorname{sign}(z)\max(|z|-\lambda,0).
$$

只要 $|z|\le\lambda$，结果就是精确的 0。这就是 Lasso 稀疏性的计算机制。

## 最小可运行示例

下面的数据有 12 个特征，但只有索引 `1`、`5`、`9` 真正影响响应。

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

使用固定随机种子和这个 `alpha` 后，明显非零的系数应位于 `1`、`5`、`9`，大约是 `1.97`、`-1.41` 和 `0.73`。它们会比生成数据时的真实系数略小，因为**收缩本身就是估计器的一部分**。

## 如何理解结果？

- `coef_[j] == 0` 表示在当前 `alpha` 下，Lasso 把这个变量从拟合线性预测器中移除了。
- 非零系数同样已经被**收缩**，不能把它直接当成未惩罚 OLS 的效应估计。
- `intercept_` 独立拟合，不属于 L1 penalty。
- `predict(X_new)` 返回连续响应预测值。
- `score(X, y)` 返回 $R^2$。
- `n_iter_` 表示当前数值路径的优化迭代次数。

变量选择是数据依赖的。某个特征在这次样本中被压成 0，并不能证明它在总体中的真实效应严格等于 0，尤其是在特征相关或样本较小时。

## 关键参数应该怎么选？

| 参数 | 默认值 | 应该怎么理解 |
|---|---:|---|
| `alpha` | `1.0` | 最重要的统计选择。越大，收缩越强、0 越多。预测型特征选择优先用 `LassoCV` 或其他验证程序。 |
| `fit_intercept` | `True` | 一般保持开启，除非理论上截距固定，或设计矩阵已经包含截距。 |
| `device` | `"auto"` | 小问题 CPU 最简单；优化规模足够大、能覆盖传输和初始化成本时再使用 GPU。 |
| `solver` | `"fista"` | 单模型拟合的稳定默认值。通常因为数值/性能原因才修改，而不是为了改变统计模型。 |
| `stopping` | `"coef_delta"` | 如果希望按照最优性条件而不是单纯系数移动判断收敛，可以使用 `"kkt"`。 |
| `compute_inference` | `True` | 只做预测/选择时建议关闭；如果要推断，应明确选择 `inference_method` 并理解后面的有效性边界。 |

### 正则化前先标准化

L1 直接惩罚系数大小。不同量纲的变量因此会受到实际上不同程度的惩罚。

大多数 Lasso workflow 都应先标准化连续特征。上面的合成数据天然处在近似相同尺度上，所以不需要额外处理。

## 与相近方法比较

| 方法 | 对相关变量组的行为 | 会产生精确 0 吗？ | 典型使用理由 |
|---|---|:---:|---|
| OLS / `LinearRegression` | 不正则化 | 否 | 设计稳定，使用未惩罚估计 |
| [Ridge](ridge.md) | 更倾向于共享信号 | 否 | 稳定相关特征，但不做变量删除 |
| **Lasso** | 可能只选同组中的一个 | 会 | 稀疏预测 / 自动特征选择 |
| [Elastic Net](elastic-net.md) | 比 Lasso 更照顾相关变量组 | 会 | 既要稀疏，又有较强共线性 |
| Adaptive Lasso | 使用数据驱动的 L1 权重 | 会 | 在更强的稀疏模型假设下减轻统一 penalty 的偏差 |

如果你在纠结“Ridge 还是 Lasso”，先问自己：**精确删除变量真的有价值吗？** 如果没有，Ridge 往往是更稳健的低方差选择。

## CPU 与 GPU 示例

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

显式 `device="cuda"` 和 `device="torch"` 会在支持时使用相应 GPU backend；如果显式请求的设备不可用，应报错，而不是静默改变执行路径。

## 进阶：求解器支持

| `solver` 值 | CPU | CuPy / Torch | 含义 |
|---|:---:|:---:|---|
| `fista`（默认） | 支持 | 支持 | L1 目标的稳定近端梯度路径 |
| `auto` | FISTA | FISTA | squared-error + L1 当前自动分发结果 |
| `fista_bb` | 支持 | 支持 | 使用 Barzilai-Borwein 步长调整的 FISTA |
| `admm` | 支持 | 支持 | 拆分求解替代路径；仅均匀样本权重 |
| `coordinate_descent` | 支持 | 不支持 | 单次 squared-error 拟合的 CPU-only 兼容路径 |

非光滑 L1 目标会拒绝 `newton`、`lbfgs`、`irls` 和 `exact`。`cpu_solver` 由 Lasso 的 CV/path 辅助接口使用，不会覆盖单次 `Lasso.fit` 的 `solver`。通用算法机制见[求解器指南](../guides/solver-algorithms.md)。

## 进阶：Lasso 之后的推断

数据驱动选择之后的统计推断，比一个预先指定的 OLS 模型困难得多。statgpu 暴露了多个实用路径，但它们**不代表相同的统计保证**。

| `inference_method` | 主要用途 | 重要限制 |
|---|---|---|
| `cpu_ols_inference` | 轻量 CPU post-selection 诊断 | OLS 风格启发式区间；不是严格 selective-inference 区间 |
| `gpu_ols_inference` | 同类诊断，同时减少 GPU→CPU 大块传输 | 同样存在选择后有效性限制 |
| `debiased` | de-biased / de-sparsified 系数推断 | 当前 `_conf_int` 是单个系数的边际区间；高维去偏方法本身的假设仍然必须满足 |
| `bootstrap` | residual-bootstrap 替代方案 | 开销更高，并且仍条件于具体重采样与模型假设 |

`compute_inference=True` 时，公开报告可包含 `_bse`、根据推断方法使用 t 或 z 口径的统计量、`_pvalues` 和 `_conf_int`。

对于 `inference_method="debiased"`，还可以开启 simultaneous interval：

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

marginal_ci = model._conf_int
simultaneous_ci = model._conf_int_simultaneous
```

边际区间和 simultaneous family-wise 区间回答的是不同问题，不能互相替代描述。

## 常见误区

- **不要把“被选中”理解成“已经证明有因果作用”，甚至也不能直接理解成“总体效应必然非零”。** Lasso 选择依赖样本和 tuning。
- **不要忽略特征尺度。** L1 penalty 不具备尺度不变性。
- **不要期待高度重复的变量之间有稳定选择。** 纯 Lasso 可能任意偏好一个相关变量；Elastic Net 往往更适合。
- **不要通过最大化训练 $R^2$ 选择 `alpha`。** 应使用 held-out validation 或 cross-validation。
- **不要在变量选择以后直接套普通 OLS p 值，再假装模型是预先指定的。** 推断方法必须匹配你的统计问题。
- **不要把数值收敛等同于统计正确。** KKT residual 很小只说明声明的优化问题求解得足够准确。

## API 与验证

导入：

```python
from statgpu.linear_model import Lasso
```

进阶构造参数包括 `max_iter`、`tol`、`cpu_solver`、`gpu_memory_cleanup`、`inference_method` 和 simultaneous-inference 相关设置。它们被有意放到本页后半段。

维护中的验证覆盖求解器收敛、CPU/GPU 一致性、KKT stopping、de-biased inference、bootstrap/推断路径以及需要时的真实 GPU 行为。相关入口包括 `dev/tests/test_lasso_debiased_inference.py`、`dev/benchmarks/benchmark_lasso_inference_gpu_vs_cpu.py` 与 `dev/comparisons/compare_lasso_kkt_stopping.py`。

## 参考文献

- Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *Journal of the Royal Statistical Society: Series B*, 58(1), 267–288.
- Bühlmann, P., & van de Geer, S. (2011). *Statistics for High-Dimensional Data*. Springer.
- Zhang, C.-H., & Zhang, S. S. (2014). Confidence intervals for low-dimensional parameters in high-dimensional linear models. *Journal of the Royal Statistical Society: Series B*, 76(1), 217–242.
- Javanmard, A., & Montanari, A. (2014). Confidence intervals and hypothesis testing for high-dimensional regression. *Journal of Machine Learning Research*, 15, 2869–2909.
