# 损失函数 (LossBase)

> 语言：中文  
> 最后更新：2026-07-01  
> 页面定位：模型文档  
> 切换：[English](../../en/models/losses.md)

## 概述

`LossBase` 是 statgpu 中所有损失函数的通用基类。它为优化求解器和惩罚函数提供统一接口。

> 求解器算法详见：[求解器算法](../guides/solver-algorithms.md)
>
> 各损失详细文档参见：
> - [分位数回归](quantile.md) — pinball 损失、PenalizedQuantileRegression、Proximal IRLS-CD
> - [稳健回归](robust.md) — Huber、Bisquare、Fair 损失、PenalizedRobustRegression
> - [CoxPH](coxph.md) — Cox 部分似然、Efron ties

五种新损失类型扩展了 `LossBase`（在已有 7 种 GLM 家族之外）：

| 损失 | 类 | R 等价 | 用途 |
|------|------|--------|------|
| 分位数 | `QuantileLoss` | `quantreg::rq()` | 条件分位数、中位数回归 |
| Huber | `HuberLoss` | `MASS::rlm()` | 稳健回归（M-估计器） |
| Bisquare | `BisquareLoss` | `MASS::rlm(psi="bisquare")` | 重降 M-估计器 |
| Fair | `FairLoss` | `MASS::rlm(psi="fair")` | Fair M-估计器 |
| Cox PH | `CoxPartialLikelihoodLoss` | `survival::coxph()` | 生存分析 |

所有损失自动继承 10 种惩罚类型和 8 种求解器。
惩罚封装器：`PenalizedQuantileRegression`、`PenalizedRobustRegression`、`PenalizedCoxRegression`。

## 路径

```
statgpu.losses.LossBase
statgpu.losses.QuantileLoss
statgpu.losses.HuberLoss
statgpu.losses.BisquareLoss
statgpu.losses.FairLoss
statgpu.losses.CoxPartialLikelihoodLoss
```

## 架构

```
LossBase (statgpu/losses/_base.py)
├── GLMLoss (statgpu/glm_core/_base.py) — 添加 _mu_from_eta、IRLS 提示
│   ├── SquaredErrorLoss、LogisticLoss、PoissonLoss 等
├── QuantileLoss — pinball 损失，非光滑
├── HuberLoss — 稳健，光滑
├── BisquareLoss — 重降，光滑
├── FairLoss — Fair 损失，光滑
└── CoxPartialLikelihoodLoss — 生存分析，有 Hessian
```

## 目标函数

所有损失最小化：
$$
\min_{\beta} \frac{1}{n} \sum_{i=1}^n \ell(X_i \beta, y_i) + \text{penalty}(\beta)
$$

### Quantile 损失 (Pinball)

$$
\ell(\eta, y) = \rho_\tau(y - \eta), \quad \rho_\tau(u) = u \cdot (\tau - \mathbf{1}\{u < 0\})
$$

当 $\tau = 0.5$ 时即为绝对损失（中位数回归）。

### Huber 损失

$$
\ell(\eta, y) = \begin{cases}
\frac{1}{2}(y - \eta)^2 & \text{若 } |y - \eta| \le \delta \\
\delta(|y - \eta| - \frac{1}{2}\delta) & \text{否则}
\end{cases}
$$

### Bisquare 损失 (Tukey biweight)

$$ \ell(\eta, y) = \rho_c(y - \eta) $$，其中
$$ \rho_c(u) = \begin{cases} \frac{c^2}{6}\left[1 - \left(1 - (\frac{u}{c})^2\right)^3\right] & |u| \le c \\ \frac{c^2}{6} & |u| > c \end{cases} $$

### Cox 部分似然（负对数）

$$ \ell(\beta) = -\frac{1}{n} \log L(\beta) $$

其中 $L(\beta)$ 为 Breslow 或 Efron 部分似然。

## 求解器兼容性

| 求解器 | Quantile | Huber | Bisquare | Fair | Cox PH |
|--------|----------|-------|----------|------|--------|
| FISTA | ✅ | ✅ | ✅ | ✅ | ✅ |
| FISTA-BB | ✅ | ✅ | ✅ | ✅ | ✅ |
| FISTA-LLA | ✅ (SCAD/MCP) | ✅ | ✅ | ✅ | ✅ |
| Proximal IRLS-CD | ✅ (SCAD/MCP) | ❌ | ❌ | ❌ | ❌ |
| Proximal Newton | ❌ (无 Hessian) | ✅ (5-10 iter) | ✅ (5-10 iter) | ✅ | ✅ (5-10 iter) |
| Newton | ❌ (无 Hessian) | ✅ | ✅ | ✅ | ✅ |
| L-BFGS | ✅ | ✅ | ✅ | ✅ | ✅ |
| ADMM | ✅ | ✅ | ✅ | ✅ | ✅ |
| IRLS | ✅ (仅 L2) | ❌ | ❌ | ❌ | ❌ |

## 参数

### QuantileLoss

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `quantile` | `0.5` | 目标分位数，取值范围 (0, 1) |

### HuberLoss

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `delta` | `1.0` | 阈值：\|u\| ≤ delta 时二次，否则线性 |
| `epsilon` | `1.345` | 稳健性调节（95% 高斯效率） |
| `method` | `"MAD"` | 尺度估计方法：`"MAD"` 或 `"huber_prop2"` |

### BisquareLoss

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `epsilon` | `4.685` | 稳健性调节（95% 高斯效率） |
| `method` | `"MAD"` | 尺度估计方法 |

### FairLoss

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `c` | `1.4` | 调节常数 |

### CoxPartialLikelihoodLoss

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `ties` | `"breslow"` | ties 处理方法：`"breslow"` 或 `"efron"` |

## 示例

### CPU

```python
import numpy as np
from statgpu.losses import QuantileLoss, HuberLoss
from statgpu.solvers import lbfgs_solver

n, p = 200, 10
X = np.random.randn(n, p)
y = X @ np.array([1.0, 0, -0.5, 0, 0.3, 0, 0, 0, 0, 0]) + np.random.randn(n) * 0.5

# Quantile 回归（中位数）
loss = QuantileLoss(quantile=0.5)
coef, n_iter = lbfgs_solver(loss, None, X, y)

# 稳健回归
loss = HuberLoss(epsilon=1.345)
coef, n_iter = lbfgs_solver(loss, None, X, y)
```

### GPU (torch-CUDA)

```python
import torch
X_t = torch.tensor(X, dtype=torch.float64).cuda()
y_t = torch.tensor(y, dtype=torch.float64).cuda()

from statgpu.losses import HuberLoss
from statgpu.penalties import SCADPenalty
from statgpu.solvers import fista_solver

loss = HuberLoss(epsilon=1.345)
coef, n_iter = fista_solver(loss, SCADPenalty(alpha=0.1), X_t, y_t)
```

### Penalized Quantile + SCAD（CPU/GPU）

```python
from statgpu.linear_model.penalized import PenalizedQuantileRegression

# CPU
model = PenalizedQuantileRegression(quantile=0.5, penalty='scad', alpha=0.1)
model.fit(X, y)

# GPU
model = PenalizedQuantileRegression(quantile=0.5, penalty='scad', alpha=0.1)
model.fit(X_t, y_t)
```

## 外部验证

- **QuantileLoss**: 与 R `quantreg::rq()`（Frisch-Newton IRLS）和 sklearn `QuantileRegressor`（HiGHS LP 求解器）对齐。系数精度 1e-6。
- **HuberLoss**: 与 R `MASS::rlm()` Huber psi 函数对齐。
- **BisquareLoss**: 与 R `MASS::rlm(psi="bisquare")` 对齐。支持 SCAD/MCP 通过 proximal Newton（5-10 次迭代收敛）。
- **CoxPartialLikelihoodLoss**: Efron tied-event 梯度/Hessian 与 `statsmodels PHReg(ties='efron')` 对齐。CI 包含 reference parity 测试。

## 注意事项

- `CoxPartialLikelihoodLoss` 支持 CuPy CUDA / PyTorch-CUDA kernel（Breslow 和 Efron）。显式 GPU 输入在 GPU 路径失败时 `raise RuntimeError`；CPU 输入使用 numpy 实现。
- `QuantileLoss` 的 `smooth_gradient=False` 且 `has_hessian=False`；对 SCAD/MCP 使用 FISTA 或 proximal IRLS-CD。
- `HuberLoss` 和 `BisquareLoss` 的 `has_hessian=True`；proximal Newton 对 SCAD/MCP 5-10 次迭代收敛。
- 所有损失接受 `sample_weight`（`CoxPartialLikelihoodLoss` 除外，会 `raise NotImplementedError`）。
- 详见 [Loss × Penalty × Solver 框架](../guides/loss-penalty-solver-framework.md)。

## 参考文献

- Koenker, R. & Bassett, G. (1978). Regression Quantiles. *Econometrica*, 46(1), 33-50.
- Huber, P. J. (1964). Robust Estimation of a Location Parameter. *Annals of Mathematical Statistics*, 35(1), 73-101.
- Beaton, A. E. & Tukey, J. W. (1974). The Fitting of Power Series. *Technometrics*, 16(2), 147-185. (Bisquare)
- Cox, D. R. (1972). Regression Models and Life-Tables. *Journal of the Royal Statistical Society*, B34, 187-220.
- Wu, Y. & Liu, Y. (2009). Variable Selection in Quantile Regression. *Statistica Sinica*, 19, 801-817.
- Fan, J. & Li, R. (2001). Variable Selection via Nonconcave Penalized Likelihood. *JASA*, 96, 1348-1360. (SCAD)
