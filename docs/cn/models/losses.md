# 损失函数（LossBase）

> 语言：中文  
> 最后更新：2026-09-17  
> 页面定位：底层损失函数参考  
> 切换：[英文版](../../en/models/losses.md)

## 概览

`LossBase` 是 statgpu 中用于描述优化问题数据拟合项的底层损失函数接口。对大多数用户，应优先从具体模型类开始；本页主要用于查阅损失函数本身的数学定义、数值原语、参数以及直接的损失函数 API。

本页刻意只讨论**损失函数层**。模型层怎样选择求解器、不同惩罚项如何分发、交叉验证、统计推断以及完整的 `sample_weight` 支持范围，应放在对应模型页和求解器文档中说明。

相关文档：

- [分位数回归](quantile.md) — 分位数回归模型、求解器选择、权重与推断
- [稳健回归](robust.md) — Huber、Bisquare、Fair 模型
- [CoxPH](coxph.md) — Cox 模型、并列事件、计数过程数据与推断
- [广义线性模型](generalized-linear-model.md) — GLM 目标函数与解析权重语义
- [求解器 × 惩罚项兼容性矩阵](../guides/solver-penalty-matrix.md) — 实际支持的求解组合
- [求解器算法](../guides/solver-algorithms.md) — 数值算法与适用前提

## 公开入口

```text
statgpu.losses.LossBase
statgpu.losses.QuantileLoss
statgpu.losses.HuberLoss
statgpu.losses.BisquareLoss
statgpu.losses.FairLoss
statgpu.losses.CoxPartialLikelihoodLoss
```

## 损失函数层次结构

```text
LossBase (statgpu/losses/_base.py)
├── GLMLoss (statgpu/glm_core/_base.py)
│   ├── SquaredErrorLoss、LogisticLoss、PoissonLoss 等
├── QuantileLoss
├── HuberLoss
├── BisquareLoss
├── FairLoss
└── CoxPartialLikelihoodLoss
```

统一层次结构表示这些对象共享一套数值接口，并不意味着它们属于同一个统计模型族。

## 共享目标函数与权重语义

对逐观测损失 $\ell_i(\beta)$，无权重的数据拟合项为

$$
L(\beta)=\frac{1}{n}\sum_{i=1}^n\ell_i(\beta).
$$

当某个 `LossBase` 实现通过共享一阶原语接受解析 `sample_weight` 时，使用归一化加权目标

$$
L_w(\beta)
=\frac{\sum_i w_i\ell_i(\beta)}{\sum_iw_i}.
$$

`value()`、`gradient()` 与 `fused_value_and_gradient()` 暴露损失函数层的这些数值原语。部分损失函数还提供 Hessian 或其他专用原语；某些损失函数也可能拒绝特定的带权操作。只有当某条求解路径所需的全部数值量都与同一个目标函数一致时，完整的求解器/估计器路径才是受支持的。具体组合请查阅 [求解器 × 惩罚项兼容性矩阵](../guides/solver-penalty-matrix.md)。

## 已实现的非 GLM 损失

| 损失 | 类 | 响应类型 | 光滑梯度 | Hessian | 常见用途 |
|---|---|---|:---:|:---:|---|
| 分位数 | `QuantileLoss` | 连续 | ❌ | ❌ | 条件分位数、中位数回归 |
| Huber | `HuberLoss` | 连续 | ✅ | ✅ | 稳健 M-估计 |
| Bisquare | `BisquareLoss` | 连续 | ✅ | ✅ | 重降型稳健 M-估计 |
| Fair | `FairLoss` | 连续 | ✅ | ✅ | 光滑稳健 M-估计 |
| Cox PH | `CoxPartialLikelihoodLoss` | 生存 | ✅ | ✅ | Cox 部分似然优化 |

这些性质描述的是损失函数对象本身。它们是求解器选择的输入信息，但不能替代完整的求解器支持矩阵。

### 分位数损失（check / pinball）

令残差 $u=y-\eta$，分位数 $\tau\in(0,1)$，则

$$
\rho_\tau(u)
=u\left(\tau-\mathbf 1\{u<0\}\right).
$$

当 $\tau=0.5$ 时，$\rho_{0.5}(u)=\tfrac12|u|$。该损失是分段线性的，因此 `QuantileLoss` 的次梯度为阶梯函数，并且没有 Hessian。分位数回归中的求解器选择、惩罚项、权重、CV 与推断请见 [分位数回归](quantile.md)。

### Huber 损失

令残差 $u=y-\eta$，则

$$
\rho_\delta(u)=
\begin{cases}
\frac12u^2, & |u|\le\delta,\\
\delta\left(|u|-\frac12\delta\right), & |u|>\delta.
\end{cases}
$$

### Bisquare 损失（Tukey biweight）

令残差 $u=y-\eta$，则

$$
\rho_c(u)=
\begin{cases}
\frac{c^2}{6}\left[1-\left(1-(u/c)^2\right)^3\right], & |u|\le c,\\
\frac{c^2}{6}, & |u|>c.
\end{cases}
$$

### Fair 损失

令残差 $u=y-\eta$，则

$$
\rho_c(u)
=c^2\left(\frac{|u|}{c}-\log\left(1+\frac{|u|}{c}\right)\right).
$$

它的得分函数随残差平滑变化，并逐渐降低大残差的影响。

### Cox 部分似然

`CoxPartialLikelihoodLoss` 表示标准右删失 Cox 数据的负对数部分似然。底层接口接受 `{"time": ..., "event": ...}` 或 `(n, 2)` 的 `[time, event]` 响应，并支持 Breslow 或 Efron 并列事件处理。

Cox 部分似然对所有线性预测子同时加上一个常数保持不变，因此无法从该损失中识别截距。更高层的 Cox 数据结构、Exact ties、延迟进入、`strata` 和推断请见 [CoxPH](coxph.md)。

## 参数

### `QuantileLoss`

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `quantile` | `0.5` | 目标分位数，取值 `(0, 1)` |

### `HuberLoss`

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `delta` | `None` | 可选固定阈值；一旦提供，会忽略 `epsilon` 与 `method` |
| `epsilon` | `1.35` | 与估计尺度配合使用的稳健性调节常数 |
| `method` | `"MAD"` | 尺度处理方式：`"MAD"`、`"huber_prop2"` 或 `"joint"` |

### `BisquareLoss`

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `epsilon` | `4.685` | 稳健性调节常数 |
| `method` | `"MAD"` | 尺度估计方法 |

### `FairLoss`

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `delta` | `None` | 可选固定阈值；一旦提供，会忽略 `epsilon` 与 `method` |
| `epsilon` | `1.35` | 与估计尺度配合使用的稳健性调节常数 |
| `method` | `"MAD"` | 尺度处理方式：`"MAD"` 或 `"huber_prop2"` |

### `CoxPartialLikelihoodLoss`

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `ties` | `"breslow"` | `"breslow"` 或 `"efron"`；Exact ties 请使用 `CoxPH` |

## 直接调用损失函数层 API

### 函数值与梯度

```python
import numpy as np
from statgpu.losses import HuberLoss

loss = HuberLoss(delta=1.5)
coef = np.zeros(X.shape[1])

value = loss.value(X, y, coef)
gradient = loss.gradient(X, y, coef)
```

### 带权一阶计算

```python
sample_weight = np.ones(X.shape[0])
sample_weight[:50] = 5.0

value = loss.value(X, y, coef, sample_weight=sample_weight)
gradient = loss.gradient(X, y, coef, sample_weight=sample_weight)
```

这里展示的只是损失函数层的一阶带权原语；完整的估计器/求解器路径是否支持相同权重，需要以对应模型与求解器文档为准。

### Cox 部分似然

```python
from statgpu.losses import CoxPartialLikelihoodLoss

loss = CoxPartialLikelihoodLoss(ties="efron")
y_surv = np.column_stack([time, event])
coef = np.zeros(X.shape[1])

value = loss.value(X, y_surv, coef)
gradient = loss.gradient(X, y_surv, coef)
hessian = loss.hessian(X, y_surv, coef)
```

## 后端行为

当具体损失函数支持相应操作时，损失计算沿用选定的 NumPy、CuPy 或 Torch 后端。不同后端的数值结果一致，只说明数值计算执行一致；它本身不能推出模型层的求解器、权重、CV 或推断能力。

Cox 预处理会把排序后的 `time` 与 `event` 一次性复制到主机端，用于构造确定性的失效组元数据；随后索引会缓存到选定设备，而设计矩阵、线性预测子、目标函数、梯度和 Hessian 在迭代计算中保持在数值后端。

## 下一步文档

- 模型页：查看估计器 API、默认行为、权重、CV 与推断。
- [求解器 × 惩罚项兼容性矩阵](../guides/solver-penalty-matrix.md)：查看实际支持的组合。
- [求解器算法](../guides/solver-algorithms.md)：查看更新公式和算法前提。
- [损失函数 × 惩罚项 × 求解器框架](../guides/loss-penalty-solver-framework.md)：查看完整计算架构。

## 参考文献

- Koenker, R. & Bassett, G. (1978). Regression Quantiles. *Econometrica*, 46(1), 33-50.
- Huber, P. J. (1964). Robust Estimation of a Location Parameter. *Annals of Mathematical Statistics*, 35(1), 73-101.
- Beaton, A. E. & Tukey, J. W. (1974). The Fitting of Power Series. *Technometrics*, 16(2), 147-185.
- Cox, D. R. (1972). Regression Models and Life-Tables. *Journal of the Royal Statistical Society*, B34, 187-220.
