# 损失函数 (LossBase)

> 语言：中文
>
> 最后更新：2026-09-12
>
> 页面定位：模型文档
>
> 切换：[English](../../en/models/losses.md)

## 概述

`LossBase` 是 statgpu 中所有损失函数的通用基类。它为优化求解器和惩罚函数提供统一接口。

> 求解器算法详见：[求解器算法](../guides/solver-algorithms.md)
>
> 各损失详细文档参见：
> - [分位数回归](quantile.md) — pinball 损失、PenalizedQuantileRegression、Proximal IRLS-CD
> - [稳健回归](robust.md) — Huber、Bisquare、Fair 损失、PenalizedRobustRegression
> - [CoxPH](coxph.md) — Breslow/Efron/Exact、start-stop、分层、推断与 CV

五类 non-GLM loss 扩展了 `LossBase`：

| 损失 | 类 | R 等价 | 用途 |
|------|------|--------|------|
| 分位数 | `QuantileLoss` | `quantreg::rq()` | 条件分位数、中位数回归 |
| Huber | `HuberLoss` | `MASS::rlm()` | 稳健回归（M-估计器） |
| Bisquare | `BisquareLoss` | `MASS::rlm(psi="bisquare")` | 重降 M-估计器 |
| Fair | `FairLoss` | `MASS::rlm(psi="fair")` | Fair M-估计器 |
| Cox PH | `CoxPartialLikelihoodLoss` | `survival::coxph()` | 生存分析 |

`LossBase` 提供统一接口，但可用组合仍由具体 loss 和 public estimator 的 contract 决定。某个 solver 可以调用某个 loss，并不意味着这个 loss 对该 solver 的所有可选控制——特别是 `sample_weight`——都具有相同统计含义。

惩罚封装器包括 `PenalizedQuantileRegression`、`PenalizedRobustRegression` 和 `PenalizedCoxPHModel`；其中 Cox 封装器当前验证 L1、L2、Elastic Net、SCAD、MCP 五类惩罚。

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

无权重 loss 最小化平均损失与声明 penalty：

$$
\min_{\beta} \frac{1}{n} \sum_{i=1}^n \ell(X_i \beta, y_i) + \text{penalty}(\beta).
$$

对于**明确支持 analytic objective weights** 的 loss/solver 组合，维护中的 convention 是

$$
\frac{\sum_i w_i\ell_i}{\sum_i w_i}.
$$

但 weight support 属于具体的 **loss × solver × estimator contract**；不能因为 `LossBase` 某个方法带有 `sample_weight` 参数，就推断所有 solver 都支持任意 non-uniform weights。

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

`CoxPartialLikelihoodLoss` 接收 `[time, event]` 二列响应，$L(\beta)$ 为 Breslow 或 Efron 部分似然。需要 Exact ties、$(\text{start},\text{stop}]$、`strata` 或 `subject_id` 时，应使用 [`CoxPH`/`CoxPHCV`](coxph.md) 的计数过程实现。

## 求解器兼容性

| 求解器 | Quantile | Huber | Bisquare | Fair | Cox PH |
|--------|----------|-------|----------|------|--------|
| FISTA | ✅ | ✅ | ✅ | ✅ | ✅ |
| FISTA-BB | ✅ | ✅ | ✅ | ✅ | ✅ |
| FISTA-LLA | ✅ (SCAD/MCP) | ✅ | ✅ | ✅ | ✅ (SCAD/MCP) |
| Proximal IRLS-CD | ✅ (SCAD/MCP) | ❌ | ❌ | ❌ | ❌ |
| Proximal Newton | ❌ (无 Hessian) | ✅ (5-10 iter) | ✅ (5-10 iter) | ✅ | ❌ |
| Newton | ❌ (无 Hessian) | ✅ | ✅ | ✅ | ✅ |
| L-BFGS | ✅ | ✅ | ✅ | ✅ | ✅ |
| ADMM | ✅ | ✅ | ✅ | ✅ | ✅ |
| IRLS | ✅ (仅 L2) | ❌ | ❌ | ❌ | ❌ |

上表描述的是普通**无权重** solver compatibility，并不表示所有这些 loss 都能通过每个列出的 solver 接受 genuine non-uniform weights。

### Non-uniform weights 与 direct L-BFGS

`lbfgs_solver` 对 genuine non-uniform weighted execution 采用 loss-level opt-in。维护中的 `GLMLoss` 子类会 opt-in，因为它们的 fused value/gradient contract 明确定义了归一化 analytic-weight objective。

本页的 generic non-GLM losses——Quantile、Huber/Bisquare/Fair、Cox——当前对 direct L-BFGS 的 genuine non-uniform `sample_weight` 保持 fail closed，除非之后为该 loss 单独定义并验证相应统计 contract。uniform weights 仍保持历史 unweighted L-BFGS 路径。

这个边界很重要：共享 solver 为 GLM 增加 weighted capability，不应静默改变 robust、quantile 或 survival objective 的统计含义。

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

此处的 loss 对象不接受 `ties="exact"`。Exact 是 `statgpu.survival.CoxPH` 和 `CoxPHCV` 的 estimator 级能力。

## 示例

### CPU

```python
import numpy as np
from statgpu.losses import QuantileLoss, HuberLoss
from statgpu.solvers import lbfgs_solver

n, p = 200, 10
X = np.random.randn(n, p)
y = X @ np.array([1.0, 0, -0.5, 0, 0.3, 0, 0, 0, 0, 0]) + np.random.randn(n) * 0.5

# 以下是无权重 direct-solver 示例。
loss = QuantileLoss(quantile=0.5)
coef, n_iter = lbfgs_solver(loss, None, X, y)

loss = HuberLoss(epsilon=1.345)
coef, n_iter = lbfgs_solver(loss, None, X, y)
```

不要从这些 unweighted 示例推断 Quantile/Huber 已支持 genuine non-uniform weighted L-BFGS。加权 robust/quantile procedure 应以对应 model 文档声明为准。

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

model = PenalizedQuantileRegression(quantile=0.5, penalty='scad', alpha=0.1)
model.fit(X, y)

model_gpu = PenalizedQuantileRegression(quantile=0.5, penalty='scad', alpha=0.1)
model_gpu.fit(X_t, y_t)
```

## 外部验证

loss-specific numerical validation 使用对应维护中的 model/solver contract。跨 NumPy/CuPy/Torch 的数值一致性与“某种 weighting interpretation 在统计上是否有效”是两个不同问题；一个尚未声明 weighted contract 的 loss，不能只靠三后端数值一致就被视为支持该 weighting method。

## 注意事项

- `CoxPartialLikelihoodLoss` 的 Breslow/Efron 路径在 NumPy、CuPy CUDA 和 Torch CUDA 后端执行；其 weight support 仍由 Cox-specific contract 单独定义。
- `QuantileLoss` 的 `smooth_gradient=False` 且 `has_hessian=False`；对 SCAD/MCP 使用 FISTA 或 proximal IRLS-CD。
- robust losses 有各自 estimator-level weight semantics；这并不自动扩展成 direct non-uniform weighted L-BFGS。
- 详见 [Loss × Penalty × Solver 框架](../guides/loss-penalty-solver-framework.md)。

## 参考文献

- Koenker, R. & Bassett, G. (1978). Regression Quantiles. *Econometrica*, 46(1), 33-50.
- Huber, P. J. (1964). Robust Estimation of a Location Parameter. *Annals of Mathematical Statistics*, 35(1), 73-101.
- Beaton, A. E. & Tukey, J. W. (1974). The Fitting of Power Series. *Technometrics*, 16(2), 147-185. (Bisquare)
- Cox, D. R. (1972). Regression Models and Life-Tables. *Journal of the Royal Statistical Society*, B34, 187-220.
- Wu, Y. & Liu, Y. (2009). Variable Selection in Quantile Regression. *Statistica Sinica*, 19, 801-817.
- Fan, J. & Li, R. (2001). Variable Selection via Nonconcave Penalized Likelihood. *JASA*, 96, 1348-1360. (SCAD)
