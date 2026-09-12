# 损失函数（LossBase）

> 语言：中文  
> 最后更新：2026-09-12  
> 页面定位：底层损失函数参考  
> 切换：[English](../../en/models/losses.md)

## 概览

`LossBase` 是 statgpu 中损失函数、惩罚项与优化求解器之间的底层统一接口。对大多数用户而言，应该优先从具体模型类开始；只有在需要确认底层求解器支持范围或直接调用求解器时，才需要查阅本页。

相关模型文档：

- [分位数回归](quantile.md) — pinball loss 与 `PenalizedQuantileRegression`
- [稳健回归](robust.md) — Huber、Bisquare、Fair loss 与 `PenalizedRobustRegression`
- [CoxPH](coxph.md) — Cox 部分似然、ties、计数过程数据与推断
- [GeneralizedLinearModel](generalized-linear-model.md) — GLM 目标函数与解析权重定义

五类非 GLM 损失函数使用这套共享接口：

| 损失 | 类 | R 中的对应方法 | 常见用途 |
|------|------|----------------|----------|
| 分位数 | `QuantileLoss` | `quantreg::rq()` | 条件分位数、中位数回归 |
| Huber | `HuberLoss` | `MASS::rlm()` | 稳健 M-估计 |
| Bisquare | `BisquareLoss` | `MASS::rlm(psi="bisquare")` | 重降型 M-估计 |
| Fair | `FairLoss` | `MASS::rlm(psi="fair")` | Fair 稳健损失 |
| Cox PH | `CoxPartialLikelihoodLoss` | `survival::coxph()` | 生存分析 |

**共享接口不等于共享全部能力。** 某个求解器能够调用某个损失函数，并不意味着这个损失函数在所有惩罚项、所有求解器控制项以及 `sample_weight` 上都采用相同的统计定义。

带惩罚的封装类包括 `PenalizedQuantileRegression`、`PenalizedRobustRegression` 和 `PenalizedCoxPHModel`。其中 Cox 封装当前支持 L1、L2、ElasticNet、SCAD、MCP；它只提供估计，且不拟合截距。

## 公开入口

```text
statgpu.losses.LossBase
statgpu.losses.QuantileLoss
statgpu.losses.HuberLoss
statgpu.losses.BisquareLoss
statgpu.losses.FairLoss
statgpu.losses.CoxPartialLikelihoodLoss
statgpu.linear_model.PenalizedCoxPHModel
```

## 架构

```text
LossBase (statgpu/losses/_base.py)
├── GLMLoss (statgpu/glm_core/_base.py) — GLM 专用 value/gradient/Hessian 接口
│   ├── SquaredErrorLoss、LogisticLoss、PoissonLoss 等
├── QuantileLoss — pinball loss，非光滑
├── HuberLoss — 稳健、光滑
├── BisquareLoss — 重降型稳健损失
├── FairLoss — Fair 稳健损失
└── CoxPartialLikelihoodLoss — 生存分析损失，提供 Hessian
```

## 目标函数与权重

无权重的损失函数与惩罚项问题可以写成

$$
\min_{\beta} \frac{1}{n}\sum_{i=1}^n \ell_i(\beta) + P(\beta).
$$

如果某个具体的损失函数、求解器和模型组合**明确支持解析权重**，statgpu 使用归一化加权数据拟合项：

$$
\frac{\sum_i w_i\ell_i(\beta)}{\sum_i w_i}.
$$

关键点是：`sample_weight` 的支持范围由完整的模型路径决定。某个底层方法带有 `sample_weight` 参数，并不能推出该损失函数在所有求解器下都支持任意非均匀权重。

### Quantile loss（pinball）

$$
\ell(\eta, y) = \rho_\tau(y - \eta), \quad
\rho_\tau(u) = u\left(\tau - \mathbf{1}\{u < 0\}\right).
$$

当 $\tau = 0.5$ 时，就是绝对损失（中位数回归）。

### Huber loss

$$
\ell(\eta, y) = \begin{cases}
\frac{1}{2}(y - \eta)^2 & \text{若 } |y - \eta| \le \delta, \\
\delta\left(|y - \eta| - \frac{1}{2}\delta\right) & \text{否则}.
\end{cases}
$$

### Cox 部分似然（负对数尺度）

$$
\ell(\beta) = -\frac{1}{n}\log L(\beta).
$$

`CoxPartialLikelihoodLoss` 接受 `[time, event]` 两列目标变量，并使用 Breslow 或 Efron 部分似然。高层 `statgpu.survival.CoxPH` 另外支持 Exact ties、延迟进入/start-stop 数据和分层。

## 求解器兼容性

下表描述的是当前维护的**无权重**底层求解器兼容性，不能把它直接理解为加权支持矩阵。

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

### 非均匀权重与直接 L-BFGS

直接调用 `lbfgs_solver` 时，非均匀权重需要由具体损失函数明确支持：

| 直接 L-BFGS 路径 | 非均匀 `sample_weight` |
|---|---|
| 当前维护的 `GLMLoss` | ✅ 支持 |
| Quantile / Huber / Bisquare / Fair | ❌ 不能由“无权重 L-BFGS 可用”推导出来 |
| Cox 部分似然 | ❌ 遵循独立的 Cox 权重限制 |

`GLMLoss` 能支持这一路径，是因为它的 value/gradient 接口明确定义了统一的归一化加权目标。其他非 GLM 损失函数只有在单独定义并验证相应统计含义后，才能获得同样的能力。

均匀权重继续兼容历史无权重 L-BFGS 路径。一个损失函数支持无权重 L-BFGS，并不代表它自动支持真正非均匀的 `sample_weight`。

## 参数

### `QuantileLoss`

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `quantile` | `0.5` | 目标分位数，取值 `(0, 1)` |

### `HuberLoss`

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `delta` | `None` | 可选固定阈值；提供后忽略 `epsilon` 与 `method` |
| `epsilon` | `1.35` | 与估计尺度配合使用的稳健性调节常数 |
| `method` | `"MAD"` | 尺度处理方式：`"MAD"`、`"huber_prop2"` 或 `"joint"` |

### `CoxPartialLikelihoodLoss`

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `ties` | `"breslow"` | `"breslow"` 或 `"efron"`；Exact ties 请使用 `CoxPH` |

## 示例

### CPU 直接调用求解器

```python
from statgpu.losses import QuantileLoss, HuberLoss
from statgpu.solvers import lbfgs_solver

# 下面两个例子有意使用无权重的底层求解器。
quantile_loss = QuantileLoss(quantile=0.5)
coef_q, n_iter_q = lbfgs_solver(quantile_loss, None, X, y)

huber_loss = HuberLoss(epsilon=1.345)
coef_h, n_iter_h = lbfgs_solver(huber_loss, None, X, y)
```

这些例子只说明无权重的直接 L-BFGS 可以工作，不能据此推断 Quantile/Huber 已支持非均匀加权 L-BFGS。需要加权稳健回归或分位数回归时，请以对应模型文档为准。

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

### Penalized Quantile + SCAD（CPU/GPU）

```python
from statgpu.linear_model.penalized import PenalizedQuantileRegression

model = PenalizedQuantileRegression(quantile=0.5, penalty="scad", alpha=0.1)
model.fit(X, y)

model_gpu = PenalizedQuantileRegression(quantile=0.5, penalty="scad", alpha=0.1)
model_gpu.fit(X_t, y_t)
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

迭代中的数值数组会保留在选定的 NumPy、CuPy 或 Torch 计算后端。Cox 预处理会把排序后的 `time` 与 `event` 一次性复制到 CPU，用于构造确定性的失败组元数据；随后索引会缓存到选定设备，设计矩阵、线性预测子、目标函数、梯度与 Hessian 在求解器迭代中不会搬回 CPU。

### 正则化生存模型

```python
from statgpu.linear_model import PenalizedCoxPHModel

model = PenalizedCoxPHModel(
    penalty="scad",
    alpha=0.1,
    ties="efron",
    device="cuda",
    fit_intercept=False,
    compute_inference=False,
)
model.fit(X, y_surv)
```

支持的惩罚项为 `l1`、`l2`、`elasticnet`、`scad`、`mcp`。SCAD/MCP 使用 FISTA-LLA 延续路径。`compute_inference=True` 会抛出 `NotImplementedError`；需要无惩罚推断时使用 `statgpu.survival.CoxPH`。

## 验证与注意事项

不同损失函数的数值验证以各自模型和求解器的支持约定为准。NumPy/CuPy/Torch 三个后端给出一致结果，并不能单独证明一个尚未声明权重定义的损失函数支持某种加权方法。

- `QuantileLoss` 非光滑且没有 Hessian；SCAD/MCP 模型路径使用 FISTA 或 Proximal IRLS-CD。
- 稳健损失函数有各自的模型级权重定义；这不会自动扩展成直接调用 L-BFGS 时的非均匀权重支持。
- `CoxPartialLikelihoodLoss` 保留 Cox 专用的 `sample_weight` 限制；数据结构与推断支持以 Cox 模型文档为准。
- 更完整的兼容性见 [Loss × Penalty × Solver 框架](../guides/loss-penalty-solver-framework.md)。

## 参考文献

- Koenker, R. & Bassett, G. (1978). Regression Quantiles. *Econometrica*, 46(1), 33-50.
- Huber, P. J. (1964). Robust Estimation of a Location Parameter. *Annals of Mathematical Statistics*, 35(1), 73-101.
- Beaton, A. E. & Tukey, J. W. (1974). The Fitting of Power Series. *Technometrics*, 16(2), 147-185.
- Cox, D. R. (1972). Regression Models and Life-Tables. *Journal of the Royal Statistical Society*, B34, 187-220.
- Wu, Y. & Liu, Y. (2009). Variable Selection in Quantile Regression. *Statistica Sinica*, 19, 801-817.
- Fan, J. & Li, R. (2001). Variable Selection via Nonconcave Penalized Likelihood. *JASA*, 96, 1348-1360.
