# 分位数回归

> 语言：中文  
> 最后更新：2026-09-18  
> 页面定位：模型文档  
> 切换：[英文版](../../en/models/quantile.md)

## 概述

`QuantileLoss` 实现分位数回归的 **check loss（又称 pinball loss）**。这两个名称指的是同一个非对称绝对损失，而不是两种不同的损失函数。`PenalizedQuantileRegression` 在此基础上提供带惩罚估计：L2/无惩罚目标的自动路径使用普通 Quantile IRLS，凸目标也可以显式选择普通 FISTA，标量 SCAD/MCP 使用 Proximal IRLS-CD，而 Group SCAD/MCP 的自动路径使用分组 Proximal IRLS-LLA。

| 组件 | 路径 |
|---|---|
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

下面的“支持”首先描述无权重时的算法能力。传入 `sample_weight` 后，还必须满足对应损失函数与求解器的带权约定；不能从无权重支持直接推出任意非均匀权重也受支持。

| 求解器 | 支持 | 说明 |
|---|:---:|---|
| Proximal IRLS-CD | ✅ | 专用 IRLS 上界 + LLA，用于标量 SCAD/MCP；支持解析权重 |
| 分组 Proximal IRLS-LLA | ✅ | Group SCAD/MCP 的自动路径；Quantile IRLS/MM 与凸 Adaptive Group Lasso 加权最小二乘子问题组合 |
| IRLS | ✅ | **L2/无惩罚 Quantile 下 `solver="auto"` 的默认路径**；`QuantileLoss.irls()` 明确支持 `sample_weight` |
| FISTA | ✅ | 用于 L1/ElasticNet 等近端路径；L2/无惩罚下显式 `solver="fista"` 也会真正执行普通 FISTA，而 `auto` 仍优先 IRLS |
| FISTA-BB | ❌ | BB 步长通过光滑梯度差估计局部曲率；Quantile 的次梯度是阶梯函数，因此估计器/CV 的显式请求与公开底层 `fista_bb_solver(QuantileLoss, ...)` 都会直接报错 |
| L-BFGS | ✅（仅底层无权重/均匀权重兼容边界） | 公开 `PenalizedQuantileRegression` / `PenalizedGLM_CV` 的显式 L-BFGS 请求不受支持；底层 `lbfgs_solver(QuantileLoss, ...)` 保留历史上的无权重/均匀权重兼容面，真正的非均匀权重仍会被拒绝 |
| ADMM | ❌（不能直接用于 Quantile） | 公开 `admm_solver(QuantileLoss, ...)` 不受支持，因为其通用 w 子问题要求光滑损失。分组 Proximal IRLS-LLA 只会在 Quantile IRLS 已经构造出光滑加权最小二乘代理问题之后，内部使用变量分裂法求解该凸子问题 |
| Newton | ❌ | 分位数损失没有 Hessian |
| Proximal Newton | ❌ | 分位数损失没有 Hessian |

对于 L2/无惩罚 Quantile 目标，`PenalizedQuantileRegression(..., solver="auto")` 会解析到 IRLS，显式 `solver="irls"` 也直接请求同一算法。显式普通 `solver="fista"` 同样受支持：它会真正进入通用 FISTA 求解过程，而不会静默替换成 IRLS。

IRLS 仍是自动选择，因为 pinball loss 本身非光滑；这里的 Quantile FISTA 是一阶近端/次梯度路径，并不声称满足教科书中光滑梯度 FISTA 的经典收敛假设。FISTA-BB 仍不支持，因为它的 BB 曲率更新明确依赖有意义的光滑梯度差分。

## 惩罚项兼容性

| 惩罚项 | `solver="auto"` 的主要路径 | 说明 |
|---|---|---|
| L2 / 无惩罚 | IRLS | `none` 会先规范化为 `L2(alpha=0)`；显式 `irls` 选择同一路径，显式普通 `fista` 也可按需选择 |
| L1 / ElasticNet | FISTA | 近端/次梯度路径 |
| SCAD / MCP | Proximal IRLS-CD | Quantile 专用 IRLS 上界 + LLA |
| adaptive_l1 | FISTA | 先准备自适应权重，再进入 Quantile FISTA |
| group_lasso / adaptive group | 分组 FISTA | 面向分组的近端路径 |
| group_scad / group_mcp | 分组 Proximal IRLS-LLA | 分组 LLA + Quantile IRLS/MM；每个凸 Adaptive Group Lasso 加权最小二乘代理问题都在所选后端求解 |

上表描述的是自动路径。若对 Group SCAD/MCP 显式指定 `solver="fista"`，该请求仍然保持为显式近端 FISTA，不会被静默改写成自动的分组 Proximal IRLS-LLA。公开底层 `fista_lla_path(...)` 也继续表示真正的 FISTA-LLA，而不是该自动估计器路径的别名。

## `sample_weight` 语义

对于已经声明支持非均匀解析权重的分位数路径，数据拟合项使用加权 check/pinball 目标。以逐样本损失 $\rho_\tau(r_i)$ 为例，归一化形式为

$$
L_w(\beta)
=\frac{\sum_i w_i\rho_\tau(y_i-x_i^\top\beta)}{\sum_i w_i}.
$$

但 `sample_weight` **不是所有求解器自动具备的统一能力**。当前尤其需要区分：

- Quantile IRLS / Proximal IRLS-CD 具有明确的带权实现；
- 分组 Proximal IRLS-LLA 使用同一组归一化解析权重构造 Quantile IRLS/MM 上界，然后求解相应的凸分组代理问题；
- 普通 FISTA（包括显式选择的 L2/无惩罚 FISTA）使用损失函数层的归一化带权目标；
- Adaptive L1 需要从初始化拟合学习 adaptive penalty weights 时，会使用同一组解析训练权重；若用户已经显式给定固定 adaptive weights，则直接使用，不再额外运行一次无效初始化；
- 通用 `LossBase` 的共享函数值和梯度可以计算归一化带权目标；
- FISTA-BB 与公开的直接 ADMM 不支持 Quantile，因为这些通用算法依赖 check loss 不具备的光滑梯度结构；
- 底层 Quantile L-BFGS 保留无权重/均匀权重兼容面，但真正非均匀权重会报错；模型/CV 层显式 L-BFGS 仍不支持。

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

# 这个 L2 惩罚的 Quantile 问题中，solver="auto" 解析到 IRLS。
model = PenalizedQuantileRegression(
    quantile=0.5,
    penalty="l2",
    alpha=0.1,
    solver="auto",
)
model.fit(X, y)

# 标量 SCAD/MCP 使用专用 Proximal IRLS-CD 延续路径。
scad_model = PenalizedQuantileRegression(
    quantile=0.5,
    penalty="scad",
    alpha=0.1,
)
scad_model.fit(X, y)
```

### 显式 Quantile IRLS 或 FISTA

```python
irls_model = PenalizedQuantileRegression(
    quantile=0.5,
    penalty="l2",
    alpha=0.01,
    solver="irls",
)
irls_model.fit(X, y)

fista_model = PenalizedQuantileRegression(
    quantile=0.5,
    penalty="l2",
    alpha=0.01,
    solver="fista",
)
fista_model.fit(X, y)
```

对于 L2/无惩罚，`auto` 与显式 IRLS 使用同一 IRLS 路径。显式普通 FISTA 是独立的算法控制选项，并且会真正执行 FISTA，而不是 IRLS 的别名或备用路径。ElasticNet 等非光滑惩罚本来就使用普通 FISTA。

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

这里展示的是支持解析权重的模型路径，不能据此推断显式选择任意底层求解器时都支持同样的非均匀权重。

## 算法详解

### Proximal IRLS-CD（SCAD/MCP）

详细更新公式见 [求解器算法](../guides/solver-algorithms.md#1-proximal-irls-cd)。其核心是把 check loss 的 IRLS 二次上界与 SCAD/MCP 的局部线性近似结合起来。

### 分组 Proximal IRLS-LLA（Group SCAD/MCP）

自动 Group SCAD/MCP 路径的外层 LLA 会把非凸分组惩罚转化成加权 Adaptive Group Lasso 凸代理问题。记 $D_g^{(k)}$ 为当前分组惩罚对 $\|\beta_g\|_2$ 的导数，Quantile 专用内层首先根据当前残差构造

$$
w_i^{(t)}
=
\widetilde s_i
\frac{\tau+(1-2\tau)\mathbf 1\{r_i^{(t)}<0\}}
{\max(|r_i^{(t)}|,\varepsilon)},
\qquad
\widetilde s_i=\frac{n s_i}{\sum_j s_j},
$$

无解析权重时取 $s_i=1$。随后求解凸的加权最小二乘代理问题

$$
\min_\beta
\frac{1}{2n}
\sum_i w_i^{(t)}
\left(y_i-x_i^\top\beta\right)^2
+
\sum_g D_g^{(k)}\|\beta_g\|_2.
$$

该凸子问题使用后端原生的变量分裂求解：二次更新对应加权最小二乘线性系统，近端更新则使用精确的 Adaptive Group Lasso 分组收缩。截距包含在二次模型中，但不参与惩罚。如果所有 $D_g^{(k)}$ 都为 0，则当前 LLA 目标恰好退化为无惩罚 Quantile 回归，此时直接用普通带权 Quantile IRLS 闭合。

这条自动路径与显式 FISTA 控制相互独立：`solver="fista"` 以及底层 `fista_lla_path(...)` 都继续表示基于 FISTA 的算法。

### IRLS（L2/无惩罚）

IRLS 是 L2/无惩罚 Quantile 目标的 `auto` 路径，也可以显式请求。令

$$
r_i=y_i-x_i^\top\beta.
$$

分位数 IRLS 权重为

$$
w_i^{\mathrm{IRLS}}
=\frac{\tau+(1-2\tau)\mathbf1\{r_i<0\}}
{\max(|r_i|,\varepsilon)}.
$$

若同时传入解析权重 $s_i$，实现先把它归一化为

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

L2 路径在左侧加入相应的 Ridge 对角项；截距坐标不参与惩罚。完整实现细节见 [IRLS 算法参考](../guides/solver-algorithms.md#6-irls迭代重加权最小二乘)。

### 普通 FISTA（显式 L2/无惩罚与稀疏凸路径）

Quantile/check loss 本身非光滑，因此这里不应解释为满足经典光滑梯度 FISTA 的全部理论假设。statgpu 使用注册的 Quantile 次梯度、已有的一阶步长策略以及所请求惩罚项的近端算子。该路径用于显式算法控制以及凸稀疏 Quantile 路径；L2/无惩罚的自动策略仍然是 IRLS。

## 输出

| 属性 | 类型 | 说明 |
|---|---|---|
| `coef_` | `(p,)` float | 估计系数 |
| `intercept_` | float | 截距 |
| `n_iter_` | int | 迭代次数 |
| `quantile` | float | 目标分位数 |

## 说明

- `score()` 使用 check/pinball loss，并返回其相反数，以符合 sklearn“越大越好”的约定。NumPy、CuPy 与 Torch 的响应变量/权重容器在受支持路径上均可直接传入；评分只在最终返回 Python 标量时将所需数据转为 CPU 快照。
- `sample_weight` 支持是**损失函数 × 求解器 × 估计器**路径能力，而不是所有求解器自动拥有的属性。
- 对自动标量 SCAD/MCP 和自动 Group SCAD/MCP 交叉验证，如果某一折在目标 α 上未建立收敛，则该折不会计分；strict selection 只接受所有折都有有限得分的 α。选中 α 后的全数据最终重拟合沿用直接估计器的收敛报告语义：目标 IRLS/LLA 预算耗尽会发出 `ConvergenceWarning` 并返回最终迭代结果。
- Quantile 非凸 continuation 路径会拒绝停止控制的隐式类型转换。`max_iter` 必须是正整数，`tol` 必须是有限正实数；直接标量 SCAD/MCP 与自动 Group SCAD/MCP 还要求布尔型 `lla=True`、整数 `max_lla_iters` 和有限正数 `lla_tol`。当前自动 Quantile continuation 含 3 个 alpha step，因此 `max_lla_iters` 至少为 3，才能保证每一步至少执行一次 LLA 更新。中间 continuation step 使用缩减后的 IRLS 预算，但不会超过公开的 `max_iter`；目标 step 最多使用完整预算。显式 Group SCAD/MCP `solver="fista"` 不进入 LLA continuation，因此 `lla`、`max_lla_iters` 与 `lla_tol` 不控制这条显式算法。
- 公开底层 Quantile solver 调用——包括普通 `fista_solver`、保留兼容边界的直接 `lbfgs_solver`、`QuantileLoss.irls()`、`proximal_irls_quantile_solver()` 与 Quantile `fista_lla_path()`——都会对监督输入形状 fail closed：`X` 必须为二维、`y` 必须为一维，且二者行数一致。IRLS/continuation 专用边界还会拒绝非法的截距、停止、路径和权重控制，不依赖 broadcasting 或隐式类型转换；直接 continuation 调用的 `alpha_path` 还必须是一维非空、元素均为有限正数、并从起点到目标值保持非递增的序列。
- 显式普通 L2/无惩罚 Quantile FISTA 受支持并保持权威；Group SCAD/MCP 的显式 FISTA 同样不会被改写成自动的分组 Proximal IRLS-LLA。
- FISTA-BB、公开直接 ADMM、Newton、Proximal Newton 与 L-BFGS-B 都不支持 Quantile，并会在进入数值迭代前报错。普通 estimator/CV L-BFGS 同样不支持 Quantile；只有底层 `lbfgs_solver` 的未加权/均匀权重历史兼容边界继续保留。 历史 `quantile_cd_solver` 名称仅作为 import 兼容的 fail-closed 符号保留：旧实现会忽略 `sample_weight`，且无法可靠表示不受惩罚的截距，因此维护中的标量 SCAD/MCP 拟合改用 Proximal IRLS-CD。
- 模型/CV 层的 Quantile L-BFGS 不受支持；底层公开 L-BFGS 只保留历史的无权重/均匀权重兼容边界。
- 受支持的 GPU 路径不会静默回退到 CPU。

## 相关文档

- [损失函数](losses.md) — `QuantileLoss` 的底层定义
- [求解器 × 惩罚项兼容性矩阵](../guides/solver-penalty-matrix.md) — 组合支持范围
- [求解器算法](../guides/solver-algorithms.md) — IRLS、FISTA 与 Proximal IRLS-CD 的数值细节
- [交叉验证](../guides/cross-validation.md) — CV 选择与最终重拟合