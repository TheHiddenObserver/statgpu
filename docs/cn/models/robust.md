# 稳健回归

> 语言：中文  
> 最后更新：2026-07-01  
> 页面定位：模型文档  
> 切换：[English](../../en/models/robust.md)

## 概述

通过 M-估计实现稳健回归，支持自动尺度估计。`PenalizedRobustRegression` 将 Huber、Bisquare 和 Fair 损失与 penalty-aware 求解器分发组合起来；SCAD/MCP 使用模型专属 FISTA-LLA continuation 路径。

| 组件 | 路径 |
|------|------|
| Huber 损失 | `statgpu.losses.HuberLoss` |
| Bisquare 损失 | `statgpu.losses.BisquareLoss` |
| Fair 损失 | `statgpu.losses.FairLoss` |
| 惩罚模型 | `statgpu.linear_model.penalized.PenalizedRobustRegression` |
| R 等价 | `MASS::rlm()` |

## 损失函数

### Huber 损失

$$
\ell(\eta, y) = \begin{cases}
\frac{1}{2}(y - \eta)^2 & |y - \eta| \le \delta \\
\delta|y - \eta| - \frac{1}{2}\delta^2 & \text{否则}
\end{cases}
$$

- `smooth_gradient=True`、`has_hessian=True`
- δ→∞ 退化为 OLS；δ→0 退化为 LAD
- 默认 ε=1.345 在高斯分布下 95% 效率

### Bisquare (Tukey biweight) 损失

$$
\ell(\eta, y) = \rho_c(y - \eta),\quad
\rho_c(u) = \begin{cases}
\frac{c^2}{6}\bigl[1 - (1 - (u/c)^2)^3\bigr] & |u| \le c \\
c^2/6 & |u| > c
\end{cases}
$$

- `smooth_gradient=True`、`has_hessian=True`
- |u|>c 时完全忽略残差（梯度=0）
- 比 Huber 更高的 breakdown point
- 默认 ε=4.685 在高斯分布下 95% 效率

### Fair 损失

$$
\ell(\eta, y) = c^2\left[\frac{|y-\eta|}{c} - \log(1 + \frac{|y-\eta|}{c})\right]
$$

- `smooth_gradient=True`、`has_hessian=True`
- 比 Huber 更温和，小残差更接近 OLS

## 参数

### HuberLoss

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `delta` | `1.0` | 固定阈值 |
| `epsilon` | `1.345` | 稳健性调节（auto-scale 模式） |
| `method` | `"MAD"` | 尺度估计方法 |

### BisquareLoss

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `epsilon` | `4.685` | 稳健性调节 |
| `method` | `"MAD"` | 尺度估计方法 |

## 尺度估计

`epsilon` 模式（auto-scale）下，尺度 σ 在拟合前估计：

- **MAD**: σ̂ = median(|r_i|) / 0.6745
- **Huber Proposal 2**: 迭代重估计

然后 δ = ε · σ̂（Huber）或 c = ε · σ̂（Bisquare）。

## 求解器支持

| `solver` 值 | Huber | Bisquare | Fair | 限制 |
|---|:---:|:---:|:---:|---|
| `auto` | 支持 | 支持 | 支持 | L2/none 用 Newton；稀疏惩罚用 FISTA |
| `fista` / `fista_bb` | 支持 | 支持 | 支持 | 通用近端路径 |
| `newton` / `lbfgs` | L2/none | L2/none | L2/none | 仅光滑惩罚 |
| `admm` | 兼容惩罚 | 兼容惩罚 | 兼容惩罚 | 只支持均匀样本权重 |
| `irls` | 不支持 | 暂不作为文档化估计器路径 | 暂不作为文档化估计器路径 | Huber 会拒绝 IRLS contract；Bisquare/Fair 待估计器调用签名对齐后再开放 |
| `exact` | 不支持 | 不支持 | 不支持 | 仅适用于 Ridge |

Proximal Newton 不是 `PenalizedRobustRegression(solver=...)` 的取值；它是求解器库中的低层 facade。当前稳健回归的 SCAD/MCP 估计器路径是 FISTA-LLA。

## 惩罚兼容性

| 惩罚 | `auto` 路径 | 说明 |
|---|---|---|
| l2 / none | Newton | 当前自动光滑路径 |
| l1 / elasticnet | FISTA | 近端稀疏路径 |
| SCAD / MCP | FISTA-LLA | Continuation 与局部线性近似 |
| adaptive_l1 | 加权 L1 FISTA | 数据驱动 proximal 权重 |
| group penalties | Group proximal 路径 | 具体分发取决于 penalty |

## 示例

### CPU

```python
from statgpu.linear_model.penalized import PenalizedRobustRegression

# Huber + SCAD
model = PenalizedRobustRegression(loss='huber', penalty='scad', alpha=0.1)
model.fit(X, y)

# Bisquare + MCP
model = PenalizedRobustRegression(loss='bisquare', penalty='mcp', alpha=0.1)
model.fit(X, y)

# Fair + L2
model = PenalizedRobustRegression(loss='fair', penalty='l2', alpha=0.01)
model.fit(X, y)
```

### GPU (torch-CUDA)

```python
import torch
X_t = torch.tensor(X, dtype=torch.float64).cuda()
y_t = torch.tensor(y, dtype=torch.float64).cuda()

model = PenalizedRobustRegression(loss='huber', penalty='scad', alpha=0.1)
model.fit(X_t, y_t)
```

### 直接调用求解器 API

```python
from statgpu.losses import HuberLoss, BisquareLoss
from statgpu.penalties import SCADPenalty
from statgpu.solvers import fista_solver

loss = HuberLoss(epsilon=1.345)
coef, n_iter = fista_solver(loss, SCADPenalty(alpha=0.1), X, y)
```

## 算法详解

### FISTA-LLA (SCAD/MCP)

1. 从数据驱动的起始惩罚构造递减 continuation path，直至用户指定的 `alpha`。
2. 在当前系数处线性化非凸惩罚。
3. 使用 FISTA 求解得到的加权 L1 子问题。
4. 为下一次 LLA/continuation 步骤热启动，并根据系数与 LLA 容差停止。

对于 L2/none，`auto` 使用稳健损失梯度与 Hessian 的 Newton 路径。低层 loss class 中的其他实验方法不会自动成为估计器级 `solver=` 取值。

## 输出

| 属性 | 类型 | 说明 |
|------|------|------|
| `coef_` | (p,) float | 估计系数 |
| `intercept_` | float | 估计截距 |
| `n_iter_` | int | 迭代次数 |
| `loss` | str | 损失名称 |

## 外部验证

- **Huber**: 与 R `MASS::rlm(psi=psi.huber)` 对齐，系数一致。
- **Bisquare**: 与 R `MASS::rlm(psi=psi.bisquare)` 对齐；SCAD/MCP active set 与 FISTA-LLA 一致。
- **Fair**: 与 R `MASS::rlm(psi=psi.fair)` 对齐。

## 注意事项

- `BisquareLoss` + SCAD/MCP：在 LAST continuation step（target α）warm-start（v0.2.1 修复）。
- 尺度估计使用 CPU numpy；GPU 数据自动转换。
- 所有损失接受 `sample_weight`。
- 三种损失的 `has_hessian=True` 使 Newton/L-BFGS 可用于光滑 L2/无惩罚目标；SCAD/MCP 仍分发到 FISTA-LLA。

## 参考文献

- Huber, P. J. (1964). Robust Estimation of a Location Parameter. *Annals of Mathematical Statistics*, 35(1), 73-101.
- Beaton, A. E. & Tukey, J. W. (1974). The Fitting of Power Series. *Technometrics*, 16(2), 147-185.
- Holland, P. W. & Welsch, R. E. (1977). Robust Regression using IRLS. *Communications in Statistics*, A6(9), 813-827.
