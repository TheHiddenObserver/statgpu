# 损失函数（LossBase）

> 语言：中文  
> 最后更新：2026-09-12  
> 页面定位：底层损失函数参考  
> 切换：[English](../../en/models/losses.md)

## 概览

`LossBase` 是 statgpu 中损失函数、惩罚项和优化求解器之间的底层统一接口。对大多数用户，应该优先从具体模型类开始；只有在需要确认求解器兼容性或直接调用底层求解器时，才需要参考本页。

相关模型文档：

- [分位数回归](quantile.md) — check（又称 pinball）损失与分位数回归模型
- [稳健回归](robust.md) — Huber、Bisquare、Fair 损失与稳健回归模型
- [CoxPH](coxph.md) — Cox 部分似然、并列事件处理、计数过程数据与推断
- [GeneralizedLinearModel](generalized-linear-model.md) — GLM 目标函数与解析权重语义

五类非 GLM 损失使用这套共享接口：

| 损失 | 类 | R 中的对应方法 | 常见用途 |
|------|------|--------|----------|
| 分位数 | `QuantileLoss` | `quantreg::rq()` | 条件分位数、中位数回归 |
| Huber | `HuberLoss` | `MASS::rlm()` | 稳健 M-估计 |
| Bisquare | `BisquareLoss` | `MASS::rlm(psi="bisquare")` | 重降型 M-估计 |
| Fair | `FairLoss` | `MASS::rlm(psi="fair")` | Fair 稳健损失 |
| Cox PH | `CoxPartialLikelihoodLoss` | `survival::coxph()` | 生存分析 |

**统一接口不等于统一能力。** 某个底层函数带有 `sample_weight` 参数，并不意味着该损失函数在所有求解器、所有统计模型下都支持任意非均匀权重。

本页只列出 loss 层的接口。`QuantileRegression`、`PenalizedQuantileRegression`、`PenalizedRobustRegression`、`CoxPH`、`PenalizedCoxPHModel` 等都是 estimator，应在各自模型文档中说明，而不是作为 `statgpu.losses` 的公开入口列在这里。

Panel 模型也不属于 `LossBase` 架构。当前 panel estimator 继承独立的 `BasePanelModel`，共享的是 panel 数据准备、变换后 OLS、协方差/推断和生命周期逻辑，因此不应加入本页的 loss 层次结构。

## 公开入口

```text
statgpu.losses.LossBase
statgpu.losses.QuantileLoss
statgpu.losses.HuberLoss
statgpu.losses.BisquareLoss
statgpu.losses.FairLoss
statgpu.losses.CoxPartialLikelihoodLoss
```

## 架构

```text
LossBase (statgpu/losses/_base.py)
├── GLMLoss (statgpu/glm_core/_base.py) — GLM 专用函数值/梯度/Hessian 接口
│   ├── SquaredErrorLoss、LogisticLoss、PoissonLoss 等
├── QuantileLoss — check / pinball 损失，非光滑
├── HuberLoss — 稳健、光滑
├── BisquareLoss — 重降型稳健损失
├── FairLoss — Fair 稳健损失
└── CoxPartialLikelihoodLoss — 生存分析损失，提供 Hessian
```

## 目标函数与权重语义

无权重的损失 + 惩罚问题可写为

$$
\min_{\beta}\frac{1}{n}\sum_{i=1}^n\ell_i(\beta)+P(\beta).
$$

`LossBase` 的共享 `value()`、`gradient()` 和 `fused_value_and_gradient()` 已经接受 `sample_weight`。在这些共享一阶原语上，非均匀解析权重使用归一化加权平均

$$
\frac{\sum_i w_i\ell_i(\beta)}{\sum_i w_i}.
$$

但是，这并不自动定义完整的带权求解器能力。Hessian、Fisher 信息、Lipschitz 常数、IRLS、Newton、L-BFGS、ADMM 等还需要具体损失函数和求解器分别声明相应语义。当前这部分统一化工作由 GitHub Issue #153 跟踪。

因此应区分两件事：

1. `LossBase` 已经提供部分共享带权数值原语；
2. 某个 **loss × solver × estimator** 路径是否支持非均匀权重，仍必须单独确认。

### Quantile 损失（check / pinball loss）

Quantile 回归中的 **check loss** 与 **pinball loss** 指的是同一个函数，不是两种不同损失。中文文档保留 `check` 这个统计学名称，避免把它误解成日常语义中的“检查”。

$$
\ell(\eta,y)=\rho_\tau(y-\eta),
\qquad
\rho_\tau(u)=u\left(\tau-\mathbf 1\{u<0\}\right).
$$

也可以写成更直观的分段形式：

$$
\rho_\tau(u)=
\begin{cases}
\tau u, & u\ge 0,\\
(\tau-1)u, & u<0.
\end{cases}
$$

它对正、负残差施加不同的线性斜率，因此可以定位任意条件分位数；其折线形状也解释了 “pinball” 这一名称。当 $\tau=0.5$ 时，

$$
\rho_{0.5}(u)=\frac12|u|,
$$

与绝对损失只差一个常数比例，对应中位数回归。

### Huber 损失

$$
\ell(\eta,y)=
\begin{cases}
\frac12(y-\eta)^2, & |y-\eta|\le\delta,\\
\delta\left(|y-\eta|-\frac12\delta\right), & \text{否则}.
\end{cases}
$$

### Bisquare 损失（Tukey biweight）

令 $u=y-\eta$。Bisquare 损失为

$$
\ell(\eta,y)=\rho_c(u),
$$

其中

$$
\rho_c(u)=
\begin{cases}
\frac{c^2}{6}\left[1-\left(1-(u/c)^2\right)^3\right], & |u|\le c,\\
\frac{c^2}{6}, & |u|>c.
\end{cases}
$$

### Cox 部分似然（负对数尺度）

$$
\ell(\beta)=-\frac1n\log L(\beta).
$$

低层 `CoxPartialLikelihoodLoss` 面向标准右删失数据，接受 `{"time": ..., "event": ...}` 或 `(n,2)` 的 `[time, event]` 响应，并使用 Breslow 或 Efron 部分似然。高层 `statgpu.survival.CoxPH` 另外支持 Exact 并列事件处理、延迟进入/start-stop 数据和 `strata`。

Cox partial likelihood 对线性预测子中的整体常数平移不变。若

$$
\eta_i=x_i^\top\beta+c,
$$

则风险集中的分子和分母都会乘上同一个 $e^c$，因此 $c$ 完全抵消；等价地，该常数可以吸收到未知的 baseline hazard 中。所以 Cox 截距在 partial-likelihood 模型里不可识别，`PenalizedCoxPHModel` 固定 `fit_intercept=False` 是模型定义的一部分，而不是待实现功能。

## 求解器兼容性

下表描述的是当前维护的**无权重**底层求解器兼容性，不能直接当成带权支持矩阵。

| 求解器 | Quantile | Huber | Bisquare | Fair | Cox PH |
|--------|----------|-------|----------|------|--------|
| FISTA | ✅ | ✅ | ✅ | ✅ | ✅ |
| FISTA-BB | ✅ | ✅ | ✅ | ✅ | ✅ |
| FISTA-LLA | ✅（SCAD/MCP） | ✅ | ✅ | ✅ | ✅（SCAD/MCP） |
| Proximal IRLS-CD | ✅（SCAD/MCP） | ❌ | ❌ | ❌ | ❌ |
| Proximal Newton | ❌（无 Hessian） | ✅ | ✅ | ✅ | ❌ |
| Newton | ❌（无 Hessian） | ✅ | ✅ | ✅ | ✅ |
| L-BFGS | ✅ | ✅ | ✅ | ✅ | ✅ |
| ADMM | ✅ | ✅ | ✅ | ✅ | ✅ |
| IRLS | ✅（仅 L2） | ❌ | ❌ | ❌ | ❌ |

### 非均匀权重与直接调用 L-BFGS

直接调用 `lbfgs_solver` 时，非均匀 `sample_weight` 需要由对应损失函数明确支持：

| 路径 | 非均匀 `sample_weight` |
|---|---|
| 当前维护的 `GLMLoss` | ✅ 支持 |
| Quantile / Huber / Bisquare / Fair | ❌ 不能由“无权重 L-BFGS 可用”推出 |
| Cox 部分似然 | ❌ 当前明确不支持 `sample_weight` |

`GLMLoss` 能够支持这一能力，是因为其融合的函数值/梯度接口明确定义了归一化解析权重目标。通用非 GLM 损失继续拒绝真正非均匀的直接 L-BFGS 权重，除非该损失以后单独定义并验证相应统计语义。

均匀权重继续保持历史无权重 L-BFGS 行为。

## 参数

### `QuantileLoss`

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `quantile` | `0.5` | 目标分位数，取值 `(0,1)` |

### `HuberLoss`

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `delta` | `None` | 可选固定阈值；一旦提供，会忽略 `epsilon` 与 `method` |
| `epsilon` | `1.35` | 与估计尺度配合使用的稳健性调节常数 |
| `method` | `"MAD"` | 尺度处理方式：`"MAD"`、`"huber_prop2"` 或 `"joint"` |

### `BisquareLoss`

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `epsilon` | `4.685` | 稳健性调节常数（常用于获得高斯分布下约 95% 的效率） |
| `method` | `"MAD"` | 尺度估计方法 |

### `FairLoss`

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `c` | `1.4` | Fair 损失的调节常数 |

### `CoxPartialLikelihoodLoss`

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `ties` | `"breslow"` | `"breslow"` 或 `"efron"`；Exact 并列事件处理请使用 `CoxPH` |

## 示例

### CPU 直接调用求解器

```python
from statgpu.losses import QuantileLoss, HuberLoss
from statgpu.solvers import lbfgs_solver

# 下面两个例子有意使用无权重的底层直接调用。
quantile_loss = QuantileLoss(quantile=0.5)
coef_q, n_iter_q = lbfgs_solver(quantile_loss, None, X, y)

huber_loss = HuberLoss(epsilon=1.345)
coef_h, n_iter_h = lbfgs_solver(huber_loss, None, X, y)
```

这些例子只说明无权重 L-BFGS 可以直接调用，不能据此推断 Quantile/Huber 已支持非均匀带权 L-BFGS。需要加权稳健回归或分位数回归时，请以对应模型文档为准。

### GPU（Torch CUDA）

```python
import torch
from statgpu.losses import HuberLoss
from statgpu.penalties import SCADPenalty
from statgpu.solvers import fista_solver

X_t = torch.tensor(X, dtype=torch.float64).cuda()
y_t = torch.tensor(y, dtype=torch.float64).cuda()

loss = HuberLoss(epsilon=1.345)
coef, n_iter = fista_solver(loss, SCADPenalty(alpha=0.1), X_t, y_t)
```

### Cox 部分似然

```python
import numpy as np
from statgpu.losses import CoxPartialLikelihoodLoss

y_surv = np.column_stack([time, event])
loss = CoxPartialLikelihoodLoss(ties="efron")
coef = np.zeros(X.shape[1])
value = loss.value(X, y_surv, coef)
gradient = loss.gradient(X, y_surv, coef)
hessian = loss.hessian(X, y_surv, coef)
```

迭代中的数值数组会保留在选定的 NumPy、CuPy 或 Torch 后端。Cox 预处理会把排序后的 `time` 与 `event` 一次性复制到主机，用于构造确定性的失效组元数据；随后索引缓存到选定设备，而设计矩阵、线性预测子、目标函数、梯度和 Hessian 在迭代中不会被搬回 CPU。

具体 Cox estimator（包括无惩罚 `CoxPH`、`CoxPHCV` 和 `PenalizedCoxPHModel`）的 API、数据范围与推断能力见 [CoxPH 模型文档](coxph.md)。

## 验证与注意事项

跨 NumPy/CuPy/Torch 的数值一致性和“某种权重解释是否已经声明为受支持”是两个不同问题。三后端一致本身不能证明一个尚未声明带权统计语义的损失函数支持该加权方式。

- `QuantileLoss` 非光滑且没有 Hessian；模型层的 SCAD/MCP 路径使用 FISTA 或 Proximal IRLS-CD。
- 稳健损失有各自的模型层权重语义；这不会自动扩展成直接调用 L-BFGS 时的非均匀权重支持。
- `CoxPartialLikelihoodLoss` 当前明确拒绝 `sample_weight`；如果以后定义 Cox case/frequency/sampling weight，需要单独冻结统计语义并验证。
- Panel estimator 使用独立的 `BasePanelModel` 架构，不是 `LossBase` 子类，也不应从本页推断其目标函数或权重语义。
- 更完整的兼容性见 [Loss × Penalty × Solver 框架](../guides/loss-penalty-solver-framework.md)。

## 参考文献

- Koenker, R. & Bassett, G. (1978). Regression Quantiles. *Econometrica*, 46(1), 33-50.
- Huber, P. J. (1964). Robust Estimation of a Location Parameter. *Annals of Mathematical Statistics*, 35(1), 73-101.
- Beaton, A. E. & Tukey, J. W. (1974). The Fitting of Power Series. *Technometrics*, 16(2), 147-185.
- Cox, D. R. (1972). Regression Models and Life-Tables. *Journal of the Royal Statistical Society*, B34, 187-220.
- Wu, Y. & Liu, Y. (2009). Variable Selection in Quantile Regression. *Statistica Sinica*, 19, 801-817.
- Fan, J. & Li, R. (2001). Variable Selection via Nonconcave Penalized Likelihood. *JASA*, 96, 1348-1360.