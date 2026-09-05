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

假设四个真正有用的特征组成两对高度相关变量，而且每一对里的两个变量都确实携带信号。

纯 Lasso 可能把同一对变量分得很不均匀：

```text
               x1     x2     x3     x4
Lasso         1.62   0.71  -0.68  -1.24
Elastic Net   1.19   1.14  -0.95  -0.98
```

两种模型都可能有不错的预测，但 Elastic Net 更符合“同一对变量其实是共享潜在信号的近似测量”这种情形。

## 直觉

Elastic Net 要模型同时满足两种正则化偏好：

1. **L1 部分：** 弱系数可以被直接推到 0；
2. **L2 部分：** 大而不稳定的系数会平滑收缩，使相关变量更像一个组来分担信号。

两个参数分别控制两件事：

- `alpha` 控制**总共正则化多少**；
- `l1_ratio` 控制**正则化更像 Lasso 还是更像 Ridge**。

可以把它想成一条连续轴：

```text
l1_ratio = 0.0        0.5             1.0
              Ridge ←──── Elastic Net ────→ Lasso
```

在 statgpu 的目标函数尺度下，`l1_ratio=0` 使用与 Ridge 相同的 L2 penalty 尺度，而 `l1_ratio=1` 对应 Lasso penalty。

## 什么时候使用？

Elastic Net 是很好的选择，当：

- 你想做特征选择，但很多特征彼此相关；
- 特征天然成组，组内变量携带相近信息；
- 纯 Lasso 会在几个几乎重复的变量之间频繁改变选择；
- 候选特征很多，希望得到一个稀疏但比纯 Lasso 更稳定的模型；
- 愿意通过验证同时调节 regularization strength 和 mixture。

以下情况更适合别的方法：

- **不需要精确 0**，主要目标只是稳定预测——Ridge 更简单；
- 强烈希望从每组中只保留极少数代表变量，而且相关性不严重——Lasso 可能已经够用；
- 因变量不适合 Gaussian 线性回归——使用相应的 penalized GLM 或其他模型族；
- 特征选择本身没有科学意义，例如变量只是任意编码或高度混杂。

## 模型与目标函数

带截距 $b$ 时，Elastic Net 最小化

$$
\frac{1}{2n}\sum_{i=1}^{n}
\left(y_i-b-x_i^\top\beta\right)^2
+\alpha\lambda\lVert\beta\rVert_1
+\frac{\alpha}{2}(1-\lambda)\lVert\beta\rVert_2^2,
$$

其中 $\lambda$ 就是 `l1_ratio`。

这里：

- $\alpha\ge 0$ 控制总体 penalty 强度；
- $0\le\lambda\le 1$ 混合 L1 与 L2；
- 截距不参与 penalty。

提高 `alpha` 会让整体收缩更强；提高 `l1_ratio` 会增加产生精确 0 的倾向；降低 `l1_ratio` 会让模型更接近 Ridge。

## 最小可运行示例

下面构造两对几乎重复、但都真正有用的变量，然后比较纯 Lasso 与 Elastic Net。

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

lasso = Lasso(
    alpha=0.08,
    device="cpu",
    compute_inference=False,
).fit(X, y)

elastic = ElasticNet(
    alpha=0.08,
    l1_ratio=0.5,
    device="cpu",
    compute_inference=False,
).fit(X, y)

print("Lasso:      ", np.round(lasso.coef_, 2))
print("Elastic Net:", np.round(elastic.coef_, 2))
```

固定随机种子后，Lasso 对第一对相关变量的分配大约是 `1.62` 与 `0.71`，第二对大约是 `-0.68` 与 `-1.24`。Elastic Net 则更均匀，约为 `1.19`、`1.14`、`-0.95`、`-0.98`，同时两个噪声变量应保持为 0 或接近 0。

重点并不是“相关变量的系数必须相等”，而是：L2 部分能减少纯 Lasso 在高度相关变量之间常见的 winner-takes-most 行为。

## 如何理解结果？

- `coef_[j] == 0` 表示在当前 `alpha` 与 `l1_ratio` 下，该特征被组合 penalty 从线性预测器中移除。
- 非零系数仍然已经收缩，不能直接当成未惩罚 OLS 系数。
- `intercept_` 独立拟合，不参与 penalty。
- `predict(X_new)` 返回连续响应预测。
- `score(X, y)` 返回 $R^2$。
- active set 同时依赖 `alpha` 和 `l1_ratio`；任意一个改变都可能改变最终保留的变量。

两个相关变量同时保持非零，是很典型的 Elastic Net 行为，但这并不证明它们各自存在独立因果效应。

## 关键参数应该怎么选？

| 参数 | 默认值 | 应该怎么理解 |
|---|---:|---|
| `alpha` | `1.0` | 总体正则化强度。越大，收缩越强，也可能删除更多变量。应通过验证选择。 |
| `l1_ratio` | `0.5` | L1/L2 混合比例。接近 1 更像 Lasso，接近 0 更像 Ridge。最好和 `alpha` 一起调。 |
| `fit_intercept` | `True` | 一般保持开启，除非理论上截距固定，或设计矩阵已经含截距。 |
| `device` | `"auto"` | 小问题 CPU 最简单；优化规模足够大时 GPU 才更有优势。 |
| `solver` | `"fista"` | 这个非光滑目标的稳定默认值。通常因为数值/性能原因才修改。 |
| `stopping` | `"coef_delta"` | 如果更关心是否满足最优性条件，可用 `"kkt"`。 |
| `compute_inference` | `False` | 普通预测/选择建议保持关闭；只有确实需要受支持的拟合后推断时再开启。 |

预测任务中，通常应优先使用 `ElasticNetCV`，而不是手工猜 `alpha` 与 `l1_ratio`。

### 先标准化特征

L1 与 L2 都直接作用于系数大小，因此不同变量单位会改变实际 penalty 强度。

除非原始尺度就是你有意设计的一部分，否则 regularized regression 通常应先标准化连续特征。

## 与 Ridge / Lasso 比较

| 性质 | Ridge | Lasso | **Elastic Net** |
|---|:---:|:---:|:---:|
| 平滑收缩系数 | 是 | 是 | 是 |
| 产生精确 0 | 通常不会 | 会 | 会 |
| 面对相关变量时的稳定性 | 强 | 可能不稳定 | 比纯 Lasso 更强 |
| 主要 tuning 参数 | `alpha` | `alpha` | `alpha` + `l1_ratio` |
| 最简单的记忆方式 | 稳定 | 选择 | 选择 + 稳定 |

一个实用经验是：

- 关心预测、相关变量很多，但不在乎删变量：先用 **Ridge**；
- 稀疏性最重要，相关性不严重：先用 **Lasso**；
- 既想稀疏，又知道相关变量组很重要：先用 **Elastic Net**。

## CPU 与 GPU 示例

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

公开 wrapper 在可用时支持 NumPy CPU、CuPy CUDA 与 Torch CUDA。实际速度取决于样本量、特征维数、dtype、数据驻留位置和传输成本，应针对真实 workload 做 benchmark。

单次拟合还可以通过 `fit(initial_coef=...)` 提供 warm start。

## 进阶：求解器与优化细节

只要 `l1_ratio > 0`，Elastic Net 就包含非光滑 L1 部分，因此近端方法是正常数值路径。

| `solver` 值 | CPU | CuPy / Torch | 说明 |
|---|:---:|:---:|---|
| `fista`（默认） | 支持 | 支持 | 推荐近端路径 |
| `auto` | FISTA | FISTA | squared-error + Elastic Net 当前自动分发 |
| `fista_bb` | 支持 | 支持 | 自适应谱步长 |
| `admm` | 支持 | 支持 | 拆分求解替代路径；仅均匀样本权重 |
| `coordinate_descent` | 支持 | 不支持 | CPU-only 兼容路径 |

非光滑 Elastic Net 估计器接口会拒绝 `newton`、`lbfgs`、`irls` 与 `exact`。单模型拟合时，`cpu_solver` 不会覆盖 `solver`。完整数值机制见[求解器指南](../guides/solver-algorithms.md)。

系数的一阶 KKT 条件为

$$
\frac{1}{n}X^\top(X\hat\beta-y)
+\alpha(1-\lambda)\hat\beta
+\alpha\lambda\,\partial\lVert\hat\beta\rVert_1
=0.
$$

`stopping="kkt"` 检查的就是这类最优性条件；它不会定义另一种统计近似模型。

## 进阶：推断

`ElasticNet` 默认只做估计。设置 `compute_inference=True` 后，statgpu 会在不改变 penalized coefficients 的前提下运行拟合后推断。

| `inference_method` | 主要用途 | 重要限制 |
|---|---|---|
| `debiased`（默认推断方法） | 使用共享 penalized-linear engine 做 bias-corrected coefficient inference | de-biasing 本身的假设必须满足；推断条件于已经选定的正则参数 |
| `cpu_ols` | 轻量 post-selection OLS 风格路径 | 选择之后只是启发式；不构成一般 selective-inference 保证 |
| `bootstrap` | 重采样替代方案 | 计算开销更高，并依赖具体 bootstrap 假设 |

推断成功后，根据所选方法可获得 `summary()`、标准误、z-style 统计量、p 值和置信区间等报告。

对于 `ElasticNetCV`，`compute_inference=True` 只作用于 `alpha` 与 `l1_ratio` 选择完成后的全数据最终重拟合；各 fold 模型仍然只负责估计和评分。

## 常见误区

- **不要只调 `alpha`，却把 `l1_ratio` 当成无关参数。** mixture 会直接改变你拟合的是哪一类模型。
- **不要把相关且非零的变量理解为已经识别了各自因果效应。** Elastic Net 改善的是预测/选择稳定性，不会自动识别因果结构。
- **不要忘记标准化。** L1 和 L2 都依赖系数尺度。
- **不要用训练 $R^2$ 选择超参数。** 应使用 held-out 或 cross-validation 表现。
- **弱信号下不要期待 active set 对微小数据扰动完全不变。** Elastic Net 改善相关变量稳定性，但不能消除 sampling uncertainty。
- **不要在数据驱动选择后直接套普通未惩罚推断而不说明 selection。** 应使用受支持的拟合后方法，并遵守其假设。

## API 与验证

导入：

```python
from statgpu.linear_model import ElasticNet
```

公开 wrapper 还提供 `max_iter`、`tol`、`cpu_solver`、`lipschitz_L`、`gpu_memory_cleanup`、`cov_type`、`hac_maxlags` 等进阶控制。它没有独立的 `backend`、`warm_start` 或 `random_state` 构造参数；后端由 `device` 控制，单次 warm start 使用 `fit(initial_coef=...)`。

维护中的验证会检查声明的 Elastic Net 目标函数、solver/KKT 行为、CPU 与支持的 GPU 路径、拟合后推断，以及 `ElasticNetCV` final-refit inference contract。适用时，真实 CUDA 验证仍属于 exact release acceptance。

## 参考文献

- Zou, H., & Hastie, T. (2005). Regularization and variable selection via the Elastic Net. *Journal of the Royal Statistical Society: Series B*, 67(2), 301–320.
- Beck, A., & Teboulle, M. (2009). A fast iterative shrinkage-thresholding algorithm for linear inverse problems. *SIAM Journal on Imaging Sciences*, 2(1), 183–202.
