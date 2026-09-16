# 分位数回归

> 语言：中文  
> 最后更新：2026-09-16  
> 页面定位：模型文档  
> 切换：[英文版](../../en/models/quantile.md)

## 概述

`QuantileLoss` 实现分位数回归的 **check loss（又称 pinball loss）**。这两个名称指的是同一个非对称绝对损失，而不是两种不同的损失函数。`PenalizedQuantileRegression` 在此基础上提供带惩罚估计；平滑的 L2/无惩罚问题使用普通 Quantile IRLS，SCAD/MCP 则使用专门的 Proximal IRLS-CD 路径。

| 组件 | 路径 |
|------|------|
| 损失函数 | `statgpu.losses.QuantileLoss` |
| 独立模型 | `statgpu.linear_model.QuantileRegression` |
| 带惩罚模型 | `statgpu.linear_model.penalized.PenalizedQuantileRegression` |
| 专用求解器 | `statgpu.solvers._proximal_irls_quantile.proximal_irls_quantile_solver` |
| R 中的对应方法 | `quantreg::rq()` |

## 目标函数

在分位数 $\tau\in(0,1)$ 处，check / pinball 损失定义为

$$
\ell(\eta,y)=\rho_\tau(y-\eta),
\qquad
\rho_\tau(u)=u\left(\tau-\mathbf 1\{u<0\}\right).
$$

等价地，

$$
\rho_\tau(u)=
\begin{cases}
\tau u, & u\ge 0,\\
(\tau-1)u, & u<0.
\end{cases}
$$

它对正残差和负残差使用不同的线性斜率，因此最优解对应条件 $\tau$ 分位数。其折线形状是 “pinball” 名称的来源；“check loss” 是分位数回归文献中的传统名称。

当 $\tau=0.5$ 时，

$$
\rho_{0.5}(u)=\frac12|u|,
$$

因此中位数回归与最小绝对偏差只差一个不影响最优解的常数比例。

逐样本次梯度为

$$
\frac{\partial\ell}{\partial\eta}
=-\tau+\mathbf 1\{y-\eta<0\},
$$

在残差为 0 的折点处使用次梯度解释。梯度是阶梯函数，因此 `has_hessian=False`、`smooth_gradient=False`。

## 参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `quantile` | `0.5` | 目标分位数，取值范围 `(0,1)`；`0.5` 为中位数回归 |

## 求解器兼容性

下面的“支持”首先描述无权重时的算法能力。传入 `sample_weight` 后，还必须满足对应损失函数与求解器的带权支持约定；不能从无权重支持直接推出任意非均匀权重也受支持。

| 求解器 | 支持 | 说明 |
|--------|:---:|------|
| Proximal IRLS-CD | ✅ | 专用 IRLS 上界 + LLA，主要用于 SCAD/MCP；当前维护路径支持相应解析权重 |
| IRLS | ✅ | **L2/无惩罚 Quantile 下 `solver="auto"` 的默认路径**；`QuantileLoss.irls()` 有显式 `sample_weight` 支持 |
| FISTA | ✅（稀疏路径） | L1/ElasticNet 等近端路径继续维护。L2/无惩罚 Quantile 不维护显式 FISTA 请求；应使用 `auto` 或 `irls` |
| FISTA-BB | ✅ | 可在受支持的稀疏路径上显式选择；平滑 Quantile 的 `auto` 不会选择它 |
| L-BFGS | ✅（底层无权重/均匀权重边界） | 公开 `PenalizedQuantileRegression` 因 Quantile 没有 Hessian-compatible smooth contract 而拒绝 L-BFGS；通用 `LossBase` 的真正非均匀 direct weighted L-BFGS 也会 fail closed |
| ADMM | ❌ | 共享 ADMM 的 w-update 使用 accelerated gradient descent，因此要求光滑损失梯度。Quantile 的梯度是阶梯函数；模型层 `solver="admm"` 与公开 `admm_solver(QuantileLoss, ...)` 都会在数值迭代前 fail closed |
| Newton | ❌ | 分位数损失没有 Hessian |
| Proximal Newton | ❌ | 分位数损失没有 Hessian |

对平滑 Quantile 目标，`auto`、IRLS 和 FISTA 的语义现在是明确分开的：`PenalizedQuantileRegression(..., solver="auto", penalty="l2")` 会解析到 IRLS；显式 `solver="irls"` 直接请求同一维护算法。显式 smooth `solver="fista"` 会明确失败，而不会在内部静默替换成 IRLS。稀疏 Quantile 惩罚继续保留 FISTA-family 路径。

## 惩罚兼容性

| 惩罚 | `solver="auto"` 的主要路径 | 说明 |
|---------|----------------------------|-------|
| L2 / 无惩罚 | IRLS | `none` 会先规范化为 `L2(alpha=0)`；显式 `solver="irls"` 选择同一维护路径 |
| L1 / ElasticNet | FISTA | 近端/次梯度路径 |
| SCAD / MCP | Proximal IRLS-CD | Quantile 专用 IRLS 上界 + LLA |
| adaptive_l1 | FISTA | 先准备 adaptive weights，再进入 Quantile FISTA |
| group_lasso / adaptive group | 分组 FISTA | 面向分组的近端路径 |
| group_scad / group_mcp | 分组 FISTA-LLA | Group LLA surrogate + group-aware FISTA 内层 |

## `sample_weight` 语义

对已经声明支持非均匀解析权重的分位数路径，数据拟合项使用加权 check/pinball 目标。以逐样本损失 $\rho_\tau(r_i)$ 为例，归一化形式为

$$
L_w(\beta)
=\frac{\sum_i w_i\rho_\tau(y_i-x_i^\top\beta)}{\sum_i w_i}.
$$

但 `sample_weight` **不是所有求解器自动具备的统一能力**。当前尤其需要区分：

- Quantile IRLS / Proximal IRLS-CD 等维护中的带权路径具有显式带权实现；
- 受支持的 FISTA 路径使用损失层的归一化带权目标；
- 通用 `LossBase` 的共享函数值和梯度可以计算归一化带权目标；
- 直接调用 `lbfgs_solver` 时，真正非均匀的分位数权重仍会被明确拒绝；
- ADMM 在任何权重设置下都不是维护中的 Quantile 路径，因为共享 ADMM 的 w-update 要求光滑损失梯度。

需要比较其他损失函数和求解器的带权范围时，见 [求解器 × 惩罚项兼容性矩阵](../guides/solver-penalty-matrix.md) 和 [求解器算法](../guides/solver-algorithms.md)。

## 示例

### 独立模型（含统计推断）

```python
from statgpu.linear_model import QuantileRegression

model = QuantileRegression(
    quantile=0.5,
    compute_inference=True,
    inference_method="kernel",
    kernel="epa",
    bandwidth="hsheather",
)
model.fit(X, y)
print(model.coef_)
print(model._bse)
print(model._pvalues)
print(model._conf_int)
```

### 带惩罚分位数回归

```python
from statgpu.linear_model.penalized import PenalizedQuantileRegression

# 这个平滑 L2 Quantile 问题中，solver="auto" 解析到 IRLS。
model = PenalizedQuantileRegression(
    quantile=0.5,
    penalty="l2",
    alpha=0.1,
    solver="auto",
)
model.fit(X, y)

# SCAD/MCP 使用专用 Proximal IRLS-CD 延续路径。
scad_model = PenalizedQuantileRegression(
    quantile=0.5,
    penalty="scad",
    alpha=0.1,
)
scad_model.fit(X, y)
```

### 显式 Quantile IRLS

```python
irls_model = PenalizedQuantileRegression(
    quantile=0.5,
    penalty="l2",
    alpha=0.01,
    solver="irls",
)
irls_model.fit(X, y)
```

显式 IRLS 只应在维护中的 L2/无惩罚边界使用。ElasticNet 等非光滑惩罚应使用 FISTA 而不是 IRLS；直接调用底层 `QuantileLoss.irls()` 也不属于维护中的 ElasticNet 拟合路径。

### GPU（Torch CUDA）

```python
import torch

X_t = torch.tensor(X, dtype=torch.float64).cuda()
y_t = torch.tensor(y, dtype=torch.float64).cuda()

model = PenalizedQuantileRegression(
    quantile=0.5,
    penalty="scad",
    alpha=0.1,
)
model.fit(X_t, y_t)
```

### 加权分位数回归

```python
sample_weight = np.ones(n)
sample_weight[:50] = 5.0

model = PenalizedQuantileRegression(
    quantile=0.5,
    penalty="l2",
    alpha=0.01,
)
model.fit(X, y, sample_weight=sample_weight)
```

这里展示的是模型层已经维护的带权路径，不能据此推断显式选择任意底层求解器时都支持同样的非均匀权重。

## 算法详解

### Proximal IRLS-CD（SCAD/MCP）

详细更新公式见 [求解器算法](../guides/solver-algorithms.md#1-proximal-irls-cd)。其核心是把 check loss 的 IRLS 二次上界与 SCAD/MCP 的局部线性近似结合起来。

### IRLS（L2/无惩罚）

IRLS 是 L2/无惩罚 Quantile 目标的维护中 `auto` 路径，也可以显式请求。令

$$
r_i=y_i-x_i^\top\beta.
$$

分位数 IRLS 权重为

$$
w_i^{\mathrm{IRLS}}
=\frac{\tau+(1-2\tau)\mathbf1\{r_i<0\}}
{\max(|r_i|,\varepsilon)}.
$$

若同时传入解析权重 $s_i$，维护中的实现先把它归一化为

$$
\widetilde s_i=\frac{n s_i}{\sum_j s_j},
$$

再使用

$$
w_i=\widetilde s_i w_i^{\mathrm{IRLS}}.
$$

记 $W=\operatorname{diag}(w)$，无惩罚时更新满足

$$
(X^\top W X+\varepsilon I)\beta_{\mathrm{new}}
=X^\top W y.
$$

L2 路径在左侧加入相应 Ridge 对角项；截距坐标不参与惩罚。完整实现细节见 [IRLS 算法参考](../guides/solver-algorithms.md#6-irls迭代重加权最小二乘)。

## 输出

| 属性 | 类型 | 说明 |
|------|------|------|
| `coef_` | `(p,)` float | 估计系数 |
| `intercept_` | float | 截距 |
| `n_iter_` | int | 迭代次数 |
| `quantile` | float | 目标分位数 |

## 说明

- `score()` 使用 check/pinball loss，并返回其相反数以符合 sklearn“越大越好”的约定。
- `sample_weight` 支持是 **loss × solver × estimator** 路径能力，而不是所有 solver 自动拥有的属性。
- 显式 solver 请求保持权威；不受支持的 smooth Quantile FISTA 与 Quantile ADMM 请求都会在数值迭代前明确失败，而不是被静默替换或运行不受支持的算法。
- 不支持的显式带权求解器组合应在数值迭代前失败，而不是静默替换成其他 solver。
- 维护中的 GPU 路径（`cuda`/`torch`）不能静默回退到 CPU。

## 参考文献

- Koenker, R. & Bassett, G. (1978). Regression Quantiles. *Econometrica*, 46(1), 33-50.
- Koenker, R. (2005). *Quantile Regression*. Cambridge University Press.
- Wu, Y. & Liu, Y. (2009). Variable Selection in Quantile Regression. *Statistica Sinica*, 19, 801-817.
- Hunter, D. R. & Li, R. (2005). Variable Selection using MM Algorithms. *Annals of Statistics*, 33(4), 1617-1642.
