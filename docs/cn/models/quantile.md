# 分位数回归

> 语言：中文  
> 最后更新：2026-09-13  
> 页面定位：模型文档  
> 切换：[英文版](../../en/models/quantile.md)

## 概述

`QuantileLoss` 实现分位数回归的 **check loss（又称 pinball loss）**。这两个名称指的是同一个非对称绝对损失，而不是两种不同的损失函数。`PenalizedQuantileRegression` 在此基础上提供带惩罚估计，并包含针对 SCAD/MCP 的 Proximal IRLS-CD 路径。

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
| FISTA | ✅ | 非光滑/近端路径；权重支持以当前维护的 FISTA 路径为准 |
| FISTA-BB | ✅ | 支持的稀疏路径可用；带权能力由当前损失函数和求解器路径共同决定 |
| IRLS | ✅ | L2/无惩罚；`QuantileLoss.irls()` 有显式 `sample_weight` 路径 |
| L-BFGS | ✅（无权重/均匀权重） | 直接使用真正非均匀权重时，当前通用 `LossBase` 路径会明确拒绝；见 Issue #153 |
| ADMM | ✅（无权重/均匀权重） | 共享 `admm_solver` 当前拒绝真正非均匀的 `sample_weight` |
| Newton | ❌ | 分位数损失没有 Hessian |
| Proximal Newton | ❌ | 分位数损失没有 Hessian |

## 惩罚兼容性

| 惩罚 | `solver="auto"` 的主要路径 | 说明 |
|---------|---------------|-------|
| L2 / 无惩罚 | IRLS | 分位数专用 IRLS |
| L1 / ElasticNet | FISTA | 近端/次梯度路径 |
| SCAD / MCP | Proximal IRLS-CD | IRLS 上界 + LLA |
| 自适应 L1 | FISTA-LLA | 加权 L1 近端 |
| 分组惩罚 | FISTA-LLA / 分组路径 | 使用对应分组近端算子 |

## `sample_weight` 语义

对已经声明支持非均匀解析权重的分位数路径，数据拟合项使用加权 check/pinball 目标。以逐样本损失 $\rho_\tau(r_i)$ 为例，归一化形式为

$$
L_w(\beta)
=\frac{\sum_i w_i\rho_\tau(y_i-x_i^\top\beta)}{\sum_i w_i}.
$$

但 `sample_weight` **不是所有求解器自动具备的统一能力**。当前尤其需要区分：

- Quantile IRLS / Proximal IRLS-CD 等维护中的带权路径具有显式带权实现；
- 通用 `LossBase` 的共享函数值和梯度已经可以计算归一化带权目标；
- 直接调用 `lbfgs_solver` 时，真正非均匀的分位数权重仍会被明确拒绝；
- 共享 `admm_solver` 目前只接受未传权重或均匀权重。

`LossBase` 的统一带权支持约定由 GitHub Issue #153 跟踪。

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

model = PenalizedQuantileRegression(
    quantile=0.5,
    penalty="scad",
    alpha=0.1,
)
model.fit(X, y)
```

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

令

$$
r_i=y_i-x_i^\top\beta,
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
| `intercept_` | float | 估计截距 |
| `n_iter_` | int | 迭代次数 |
| `quantile` | float | 目标分位数 |

## 注意事项

- `score()` 使用 check/pinball 损失，并按照 sklearn “越大越好”的约定返回其负值。
- `sample_weight` 是否受支持，取决于损失函数、求解器和具体模型路径三者的组合，而不是“所有求解器自动支持”。
- 显式请求不受支持的带权求解器组合时，应在数值迭代前报错，而不是更换求解器。
- GPU 设备（`cuda`/`torch`）在维护中的支持路径上不应自动回退到 CPU。

## 参考文献

- Koenker, R. & Bassett, G. (1978). Regression Quantiles. *Econometrica*, 46(1), 33-50.
- Koenker, R. (2005). *Quantile Regression*. Cambridge University Press.
- Wu, Y. & Liu, Y. (2009). Variable Selection in Quantile Regression. *Statistica Sinica*, 19, 801-817.
- Hunter, D. R. & Li, R. (2005). Variable Selection using MM Algorithms. *Annals of Statistics*, 33(4), 1617-1642.