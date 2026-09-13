# 稳健回归

> 语言：中文  
> 最后更新：2026-09-13  
> 页面定位：模型文档  
> 切换：[English](../../en/models/robust.md)

## 概述

statgpu 通过 M-估计提供稳健回归，并支持稳健尺度估计。`PenalizedRobustRegression` 组合 Huber、Bisquare 和 Fair 损失与多类惩罚项；光滑目标默认使用 Newton，稀疏与非凸惩罚进入 FISTA / LLA 路径。

| 组件 | 路径 |
|------|------|
| Huber 损失 | `statgpu.losses.HuberLoss` |
| Bisquare 损失 | `statgpu.losses.BisquareLoss` |
| Fair 损失 | `statgpu.losses.FairLoss` |
| 带惩罚模型 | `statgpu.linear_model.penalized.PenalizedRobustRegression` |
| R 中的对应方法 | `MASS::rlm()` |

## 损失函数

### Huber 损失

$$
\ell(\eta, y) = \begin{cases}
\frac{1}{2}(y - \eta)^2 & |y - \eta| \le \delta \\
\delta|y - \eta| - \frac{1}{2}\delta^2 & \text{否则}
\end{cases}
$$

- `smooth_gradient=True`、`has_hessian=True`
- $\delta\to\infty$ 时趋近 OLS；阈值减小时对大残差的影响进一步受限
- 默认 `epsilon=1.35`；在自动尺度模式下有效阈值为 `epsilon × scale`

### Bisquare（Tukey biweight）损失

$$
\ell(\eta, y) = \rho_c(y - \eta),\quad
\rho_c(u) = \begin{cases}
\frac{c^2}{6}\bigl[1 - (1 - (u/c)^2)^3\bigr] & |u| \le c \\
c^2/6 & |u| > c
\end{cases}
$$

- `smooth_gradient=True`、`has_hessian=True`
- $|u|>c$ 时损失保持常数、梯度为 0
- 默认 `epsilon=4.685`，常用于获得高斯分布下约 95% 的效率

### Fair 损失

$$
\ell(\eta, y) = c^2\left[\frac{|y-\eta|}{c} - \log\left(1 + \frac{|y-\eta|}{c}\right)\right]
$$

- `smooth_gradient=True`、`has_hessian=True`
- 对残差的降权比硬截断型损失更平缓

## 参数

### `HuberLoss`

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `delta` | `None` | 可选固定阈值；一旦给定，会使用固定阈值模式 |
| `epsilon` | `1.35` | 与估计尺度相乘得到有效 Huber 阈值 |
| `method` | `"MAD"` | `"MAD"`、`"huber_prop2"` 或 `"joint"` |

`method="joint"` 联合优化系数与 `log_sigma`，与固定尺度的系数优化问题不同。

### `BisquareLoss`

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `delta` | `None` | 可选固定阈值 |
| `epsilon` | `4.685` | 与估计尺度相乘得到有效阈值 |
| `method` | `"MAD"` | `"MAD"` 或 `"huber_prop2"` |

### `FairLoss`

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `c` | `1.4` | Fair 损失的调节常数 |

## 尺度估计

`RobustLossBase` 提供 MAD 与 Huber Proposal 2 尺度计算。当前 `PenalizedRobustRegression` 拟合路径会在进入数值求解器前调用 `precompute_scale(...)`，因此普通 `MAD` / `huber_prop2` 系数优化在求解阶段使用已经确定的有效阈值。

- **MAD**：$\hat\sigma=\operatorname{median}(|r_i|)/0.6745$
- **Huber Proposal 2**：通过固定点迭代估计尺度
- Huber 使用 $\delta=\epsilon\hat\sigma$
- Bisquare 使用 $c=\epsilon\hat\sigma$

显式给定 `delta` 时直接使用固定阈值。`method="joint"` 则是单独的系数—尺度联合优化问题。

## 求解器兼容性

下表描述当前公开模型/底层求解路径；`sample_weight` 仍需同时满足具体损失函数和求解器的带权支持约定。

| 求解器 | Huber | Bisquare | Fair | 说明 |
|--------|:---:|:---:|:---:|------|
| Proximal Newton | ✅（光滑目标） | ✅（光滑目标） | ✅（光滑目标） | 当前通用实现对 L2/无惩罚执行 Newton；非光滑请求转到 FISTA |
| FISTA | ✅ | ✅ | ✅ | 稀疏/近端路径 |
| FISTA-BB | ✅（受支持组合） | ✅（受支持组合） | ✅（受支持组合） | 自适应步长 |
| FISTA-LLA | ✅ | ✅ | ✅ | SCAD/MCP 等非凸惩罚的 LLA 路径 |
| IRLS | ❌（当前未开放） | ✅（L2/无惩罚） | ✅（L2/无惩罚） | Huber IRLS 的恢复与验证由 Issue #156 跟踪 |
| Newton | ✅ | ✅ | ✅ | `solver="auto"` 下光滑 L2/无惩罚的主要路径 |
| L-BFGS | ✅（光滑、无权重/均匀权重） | ✅（光滑、无权重/均匀权重） | ✅（光滑、无权重/均匀权重） | 通用非 GLM `LossBase` 当前未声明直接非均匀带权 L-BFGS |
| ADMM | ✅（受支持形式） | ✅（受支持形式） | ✅（受支持形式） | 共享入口当前只接受未传或均匀 `sample_weight` |

### `solver="auto"` 的主要分发

| 惩罚 | 主要路径 | 说明 |
|---------|---------------|-------|
| L2 / 无惩罚 | Newton | 稳健损失提供梯度与 Hessian |
| L1 / ElasticNet | FISTA | 近端稀疏路径 |
| SCAD / MCP | FISTA + LLA | 非凸惩罚通过局部线性近似形成加权凸子问题 |
| 自适应 L1 | FISTA / LLA 路径 | 使用自适应加权近端形式 |
| 分组惩罚 | Group FISTA / Group FISTA-LLA | 使用对应的分组近端算子 |

## 示例

```python
from statgpu.linear_model.penalized import PenalizedRobustRegression

# Huber + SCAD
model = PenalizedRobustRegression(loss="huber", penalty="scad", alpha=0.1)
model.fit(X, y)

# Bisquare + MCP
model = PenalizedRobustRegression(loss="bisquare", penalty="mcp", alpha=0.1)
model.fit(X, y)

# Fair + L2：solver="auto" 使用光滑 Newton 路径
model = PenalizedRobustRegression(loss="fair", penalty="l2", alpha=0.01)
model.fit(X, y)
```

### GPU（Torch CUDA）

```python
import torch

X_t = torch.tensor(X, dtype=torch.float64).cuda()
y_t = torch.tensor(y, dtype=torch.float64).cuda()

model = PenalizedRobustRegression(loss="huber", penalty="scad", alpha=0.1)
model.fit(X_t, y_t)
```

### 底层求解器接口

```python
from statgpu.losses import HuberLoss
from statgpu.penalties import SCADPenalty
from statgpu.solvers import fista_solver

loss = HuberLoss(epsilon=1.35)
coef, n_iter = fista_solver(loss, SCADPenalty(alpha=0.1), X, y)
```

## 算法说明

### 光滑 Huber / Bisquare / Fair

对于 L2/无惩罚目标，当前自动分发使用 Newton。损失函数提供实际梯度和 Hessian，求解器使用线性系统与 Armijo 回溯完成更新。

### SCAD / MCP

非凸惩罚通过 LLA 转换成局部加权凸问题，再由 FISTA 系列内层求解。当前通用 Proximal Newton 不把欧氏近端近似当作 Hessian 度量近端子问题，因此非光滑请求不会静默走旧的 Proximal-Newton 快捷路径。

### Huber IRLS 的当前状态

`HuberLoss.irls()` 当前是明确拒绝的占位入口，且 `_supports_irls=False`，因此 `PenalizedRobustRegression(..., solver="irls")` 不会进入 Huber IRLS。固定阈值 Huber 的标准 IRLS 权重

$$
w_i=\frac{\psi_\delta(r_i)}{r_i}
=\min\left(1,\frac{\delta}{|r_i|}\right)
$$

与 Huber 一阶条件具有直接关系；是否恢复为维护中的公开求解路径正在 Issue #156 中重新验证。本 PR 只记录当前实现状态，不改变数值源码。

## 输出

| 属性 | 类型 | 说明 |
|------|------|------|
| `coef_` | `(p,)` float | 估计系数 |
| `intercept_` | float | 估计截距 |
| `n_iter_` | int | 迭代次数 |
| `loss` | str | 损失名称 |

## 外部验证

- **Huber**：历史验证与 R `MASS::rlm(psi=psi.huber)` 对齐；恢复 IRLS 后需要在相同尺度约定下重新验证该显式求解路径。
- **Bisquare**：与 R `MASS::rlm(psi=psi.bisquare)` 对齐；非凸惩罚路径使用当前 LLA/FISTA 实现。
- **Fair**：与 R `MASS::rlm(psi=psi.fair)` 对齐。

## 注意事项

- 尺度计算目前会使用 NumPy 主机数组；完成尺度预计算后，维护中的数值优化路径继续使用所选 NumPy/CuPy/Torch 后端。
- `sample_weight` 是否可用取决于损失函数、求解器和具体模型路径，而不是“所有 robust solver 自动支持”。
- 三种损失都提供 Hessian 数值原语；这支持光滑 Newton 路径，但不等价于“任意非光滑惩罚都支持 Proximal Newton”。
- Huber IRLS 当前未作为公开求解路径开放；Issue #156 跟踪其数学与实现验证。

## 参考文献

- Huber, P. J. (1964). Robust Estimation of a Location Parameter. *Annals of Mathematical Statistics*, 35(1), 73-101.
- Beaton, A. E. & Tukey, J. W. (1974). The Fitting of Power Series. *Technometrics*, 16(2), 147-185.
- Holland, P. W. & Welsch, R. E. (1977). Robust Regression using IRLS. *Communications in Statistics*, A6(9), 813-827.