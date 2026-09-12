# 损失函数（LossBase）

> 语言：中文  
> 最后更新：2026-09-12  
> 页面定位：底层 loss 参考  
> 切换：[English](../../en/models/losses.md)

## 概览

`LossBase` 是 statgpu 中 loss、penalty 与优化 solver 之间的底层统一接口。对大多数用户来说，应该优先从具体模型类开始，而不是直接实例化 loss；当你需要确认 solver compatibility 或直接调用底层 solver 时，再参考本页。

相关模型文档：

- [分位数回归](quantile.md) — pinball loss 与 `PenalizedQuantileRegression`
- [稳健回归](robust.md) — Huber、Bisquare、Fair loss 与 `PenalizedRobustRegression`
- [CoxPH](coxph.md) — Cox partial likelihood、ties、counting-process 数据与推断
- [GeneralizedLinearModel](generalized-linear-model.md) — GLM objective 与 analytic-weight 语义

五类非 GLM loss 使用这套共享接口：

| 损失 | 类 | R 等价 | 常见用途 |
|------|------|--------|----------|
| 分位数 | `QuantileLoss` | `quantreg::rq()` | 条件分位数、中位数回归 |
| Huber | `HuberLoss` | `MASS::rlm()` | 稳健 M-估计 |
| Bisquare | `BisquareLoss` | `MASS::rlm(psi="bisquare")` | redescending M-估计 |
| Fair | `FairLoss` | `MASS::rlm(psi="fair")` | Fair 稳健损失 |
| Cox PH | `CoxPartialLikelihoodLoss` | `survival::coxph()` | 生存分析 |

**统一接口不等于统一能力。** 某个 solver 能调用某个 loss，并不意味着这个 loss 对所有 penalty、所有 solver 控制项、尤其是 `sample_weight`，都具有相同的统计含义。

惩罚封装器包括 `PenalizedQuantileRegression`、`PenalizedRobustRegression` 和 `PenalizedCoxPHModel`。Cox wrapper 当前支持 L1、L2、ElasticNet、SCAD、MCP；它只提供 estimation，且不拟合截距。

## 公开入口

```
statgpu.losses.LossBase
statgpu.losses.QuantileLoss
statgpu.losses.HuberLoss
statgpu.losses.BisquareLoss
statgpu.losses.FairLoss
statgpu.losses.CoxPartialLikelihoodLoss
statgpu.linear_model.PenalizedCoxPHModel
```

## 架构

```
LossBase (statgpu/losses/_base.py)
├── GLMLoss (statgpu/glm_core/_base.py) — GLM 专用 value/gradient/Hessian contract
│   ├── SquaredErrorLoss、LogisticLoss、PoissonLoss 等
├── QuantileLoss — pinball loss，非光滑
├── HuberLoss — 稳健、光滑
├── BisquareLoss — redescending robust loss
├── FairLoss — Fair robust loss
└── CoxPartialLikelihoodLoss — 生存分析 loss，提供 Hessian
```

## 目标函数与权重语义

无权重的 loss + penalty 问题可以写成

$$
\min_{\beta} \frac{1}{n}\sum_{i=1}^n \ell_i(\beta) + P(\beta).
$$

如果某个具体的 loss × solver × estimator 路径**明确支持 analytic objective weights**，statgpu 使用归一化的 weighted data-fit term：

$$
\frac{\sum_i w_i\ell_i(\beta)}{\sum_i w_i}.
$$

关键点是：**weight support 属于完整的 loss × solver × estimator 路径**。某个底层方法带有 `sample_weight` 参数，并不能推出该 loss 在所有 solver 下都支持任意非均匀权重。

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

### Cox partial likelihood（负对数尺度）

$$
\ell(\beta) = -\frac{1}{n}\log L(\beta).
$$

`CoxPartialLikelihoodLoss` 接受 `[time, event]` 两列 response，并使用 Breslow 或 Efron partial likelihood。高层 `statgpu.survival.CoxPH` 另外支持 Exact ties、delayed-entry/start-stop 数据和 strata。

## 求解器兼容性

下表描述的是维护中的**无权重**低层 solver compatibility，不能把它当成 weighted-support matrix。

| 求解器 | Quantile | Huber | Bisquare | Fair | Cox PH |
|--------|----------|-------|----------|------|--------|
| FISTA | ✅ | ✅ | ✅ | ✅ | ✅ |
| FISTA-BB | ✅ | ✅ | ✅ | ✅ | ✅ |
| FISTA-LLA | ✅ (SCAD/MCP) | ✅ | ✅ | ✅ | ✅ (SCAD/MCP) |
| Proximal IRLS-CD | ✅ (SCAD/MCP) | ❌ | ❌ | ❌ | ❌ |
| Proximal Newton | ❌（无 Hessian） | ✅ | ✅ | ✅ | ❌ |
| Newton | ❌（无 Hessian） | ✅ | ✅ | ✅ | ✅ |
| L-BFGS | ✅ | ✅ | ✅ | ✅ | ✅ |
| ADMM | ✅ | ✅ | ✅ | ✅ | ✅ |
| IRLS | ✅（仅 L2） | ❌ | ❌ | ❌ | ❌ |

### 非均匀权重与直接 L-BFGS

对非均匀权重，direct L-BFGS 采用保守的 opt-in 规则：

| direct `lbfgs_solver` 路径 | 真正非均匀的 `sample_weight` |
|---|---|
| 维护中的 `GLMLoss` | ✅ 由 GLM weighted-objective contract 明确支持 |
| Quantile / Huber / Bisquare / Fair | ❌ 不能由“无权重 L-BFGS 可用”推出 |
| Cox partial likelihood | ❌ 遵循独立的 Cox weight 边界 |

`GLMLoss` 能够 opt in，是因为其 fused value/gradient contract 明确定义了一个归一化 analytic-weight objective。generic non-GLM loss 则继续拒绝真正非均匀的 direct L-BFGS weights，除非该 loss 以后单独定义并验证相应统计 contract。

uniform weights 继续保持历史 unweighted L-BFGS 行为。这个区分可以避免“给 GLM solver 增加 weighted capability”意外改变 robust、quantile 或 survival objective 的统计含义。

## 参数

### `QuantileLoss`

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `quantile` | `0.5` | 目标分位数，取值 `(0, 1)` |

### `HuberLoss`

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `delta` | `None` | 可选固定 threshold；一旦提供，会忽略 `epsilon` 与 `method` |
| `epsilon` | `1.35` | 与估计 scale 配合使用的稳健性 tuning constant |
| `method` | `"MAD"` | scale 处理方式：`"MAD"`、`"huber_prop2"` 或 `"joint"` |

### `CoxPartialLikelihoodLoss`

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `ties` | `"breslow"` | `"breslow"` 或 `"efron"`；Exact ties 请使用 `CoxPH` |

## 示例

### CPU 直接调用 solver

```python
from statgpu.losses import QuantileLoss, HuberLoss
from statgpu.solvers import lbfgs_solver

# 下面两个例子有意使用无权重的低层 direct solver。
quantile_loss = QuantileLoss(quantile=0.5)
coef_q, n_iter_q = lbfgs_solver(quantile_loss, None, X, y)

huber_loss = HuberLoss(epsilon=1.345)
coef_h, n_iter_h = lbfgs_solver(huber_loss, None, X, y)
```

这些例子只说明无权重 direct L-BFGS 可以工作，不能据此推断 Quantile/Huber 已支持非均匀 weighted L-BFGS。需要加权 robust/quantile procedure 时，请以对应模型文档为准。

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

### Cox partial likelihood

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

迭代中的数值数组会留在选定的 NumPy、CuPy 或 Torch backend。Cox preprocessing 会把排序后的 `time` 与 `event` 一次性复制到 host，用于构造确定性的 failure-group metadata；随后索引会缓存到选定 device，而 design matrix、predictor、objective、gradient、Hessian 在 solver 迭代中不会被搬回 CPU。

### 正则化 survival

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

支持的 penalty 为 `l1`、`l2`、`elasticnet`、`scad`、`mcp`。SCAD/MCP 使用 FISTA-LLA continuation。`compute_inference=True` 会抛出 `NotImplementedError`；需要 unpenalized inference 时使用 `statgpu.survival.CoxPH`。

## 验证与注意事项

loss-specific numerical validation 以对应的 model/solver contract 为准。跨 NumPy/CuPy/Torch 的数值一致性和“某种 weighting interpretation 是否具有声明过的统计含义”是两个不同问题；三后端一致本身不能证明一个尚未声明 weighted contract 的 loss 支持该加权方式。

- `QuantileLoss` 非光滑且没有 Hessian；model-level SCAD/MCP 路径使用 FISTA 或 proximal IRLS-CD。
- robust loss 有各自的 estimator-level weight semantics；这不会自动扩展成 direct non-uniform weighted L-BFGS。
- `CoxPartialLikelihoodLoss` 保留 Cox-specific sample-weight 限制；数据结构与 inference 支持以 Cox 模型文档为准。
- 更完整的 compatibility 见 [Loss × Penalty × Solver 框架](../guides/loss-penalty-solver-framework.md)。

## 参考文献

- Koenker, R. & Bassett, G. (1978). Regression Quantiles. *Econometrica*, 46(1), 33-50.
- Huber, P. J. (1964). Robust Estimation of a Location Parameter. *Annals of Mathematical Statistics*, 35(1), 73-101.
- Beaton, A. E. & Tukey, J. W. (1974). The Fitting of Power Series. *Technometrics*, 16(2), 147-185.
- Cox, D. R. (1972). Regression Models and Life-Tables. *Journal of the Royal Statistical Society*, B34, 187-220.
- Wu, Y. & Liu, Y. (2009). Variable Selection in Quantile Regression. *Statistica Sinica*, 19, 801-817.
- Fan, J. & Li, R. (2001). Variable Selection via Nonconcave Penalized Likelihood. *JASA*, 96, 1348-1360.
