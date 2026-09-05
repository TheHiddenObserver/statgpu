# Loss × Penalty × Solver 框架

> 语言：中文
>
> 最后更新：2026-09-04
>
> 切换：[English](../../en/guides/loss-penalty-solver-framework.md)

## 概述

statgpu 支持 **损失函数 × 惩罚类型 × 求解器 × 后端** 的组合空间。本文档记录完整框架架构、调度逻辑和覆盖矩阵。

## 架构

```
fit(X, y, sample_weight)
  ├── _resolve_loss()   → LossBase 子类
  ├── _resolve_penalty() → Penalty 子类
  ├── _select_solver()   → solver 名称（auto 或显式）
  ├── _pre_fit()         → 后端转换、截距增强
  └── _fit_loss_backend() → 路由到具体 solver 路径
       ├── fista / fista_bb              → 通用 proximal 路径
       ├── newton / irls / lbfgs        → 光滑路径
       ├── quantile refinement          → 分位数 IRLS 或 proximal IRLS + LLA
       ├── robust refinement            → Newton、FISTA 或 FISTA-LLA
       ├── Cox refinement               → Cox 专用 Newton/FISTA/FISTA-LLA
       └── admm                         → 受支持的分裂路径
```

## 1. 损失函数

### LossBase

抽象基类位于 `statgpu/losses/_base.py`。子类实现 `per_sample_value()` 和 `per_sample_gradient()`。基类自动派生 `value()`、`gradient()`、`fused_value_and_gradient()`。

```python
class LossBase:
    name: str               # "quantile", "huber" 等
    y_type: str             # "continuous" / "survival"
    smooth_gradient: bool   # 光滑性能力
    has_hessian: bool       # Hessian contract，不等于公开 solver 保证
    _supports_irls: bool    # 低层 IRLS contract
```

### 全部损失函数

| 损失 | 类 | `has_hessian` | `smooth_gradient` | `_supports_irls` | R 等价 |
|------|-------|:---:|:---:|:---:|--------------|
| 平方误差 | `GLMLoss` (squared_error) | ✅ | ✅ | ✅ | `lm()` |
| Logistic | `GLMLoss` (logistic) | ✅ | ✅ | ✅ | `glm(…, binomial)` |
| Poisson | `GLMLoss` (poisson) | ✅ | ✅ | ✅ | `glm(…, poisson)` |
| Gamma | `GLMLoss` (gamma) | ✅ | ✅ | ✅ | `glm(…, Gamma)` |
| 逆高斯 | `GLMLoss` (inverse_gaussian) | ✅ | ✅ | ✅ | `glm(…, inverse.gaussian)` |
| 负二项 | `GLMLoss` (negative_binomial) | ✅ | ✅ | ✅ | `glm.nb()` |
| Tweedie | `GLMLoss` (tweedie) | ✅ | ✅ | ✅ | `glm(…, tweedie)` |
| Quantile | `QuantileLoss` | ❌ | ❌ | ✅ | `quantreg::rq()` |
| Huber | `HuberLoss` | ✅ | ✅ | ❌ | `MASS::rlm()` |
| Bisquare | `BisquareLoss` | ✅ | ✅ | ✅ | `MASS::rlm(psi="bisquare")` |
| Fair | `FairLoss` | ✅ | ✅ | ✅ | `MASS::rlm(psi="fair")` |
| Cox PH | `CoxPartialLikelihoodLoss` | ✅ | ✅ | ❌ | `survival::coxph()` |

这些标志描述的是 loss 对象能力，并不会自动把某个名称变成估计器合法的
`solver=` 取值。尤其是 Bisquare 和 Fair 虽然在 loss 层声明了 IRLS contract，
目前并未作为高层估计器 IRLS 路径写入支持范围；请以对应模型页为准。

### 逐样本公式

**Quantile (Pinball)**:
$$\ell(u) = u \cdot (\tau - \mathbf{1}_{u<0}), \quad u = y - \eta$$

**Huber** (delta-k = 1.345):
$$\ell(u) = \begin{cases} \frac{1}{2}u^2 & |u| \leq k \\ k|u| - \frac{1}{2}k^2 & |u| > k \end{cases}$$

**Bisquare (Tukey biweight)** (c = 4.685):
$$\ell(u) = \begin{cases} \frac{c^2}{6}[1 - (1-(u/c)^2)^3] & |u| \leq c \\ c^2/6 & |u| > c \end{cases}$$

**Cox 部分似然**（`CoxPartialLikelihoodLoss` 的 Breslow / Efron ties）：
$$L(\beta) = \prod_{i:\delta_i=1} \frac{\exp(X_i\beta)}{\sum_{j:T_j \geq T_i} \exp(X_j\beta)}$$

该 loss 对象接收 `[time, event]` 二列响应并服务于惩罚 Cox estimator。完整
`CoxPH`/`CoxPHCV` 还支持 Exact ties 与计数过程风险集

$$
R_s(t)=\{j:\operatorname{strata}_j=s,\;\operatorname{start}_j<t\leq
\operatorname{stop}_j\},
$$

以及 `subject_id` 语义；这些不是通用 loss 对象当前的输入轴。

## 2. 惩罚函数

### 全部惩罚

| 惩罚 | `is_convex` | `is_smooth` | Proximal 算子 | LLA 支持 | P(β) |
|---------|:---:|:---:|:---:|:---:|------|
| None / Null | ✅ | ✅ | identity | ❌ | 0 |
| L2 (Ridge) | ✅ | ✅ | — | ❌ | α·‖β‖²₂ |
| L1 (Lasso) | ✅ | ❌ | 软阈值 | ❌ | α·‖β‖₁ |
| ElasticNet | ✅ | ❌ | 软阈值 | ❌ | α(r‖β‖₁+(1-r)‖β‖²₂) |
| SCAD | ❌ | ❌ | 三段式 | ✅ | 分段函数 |
| MCP | ❌ | ❌ | 三段式 | ✅ | 分段函数 |
| Adaptive L1 | ✅ | ❌ | 加权软阈值 | ✅ | α/|β̂|^ν · |β| |
| Group Lasso | ✅ | ❌ | 块软阈值 | ❌ | · |
| Group MCP | ❌ | ❌ | 块 proximal | ✅ | · |
| Group SCAD | ❌ | ❌ | 块 proximal | ✅ | · |

### SCAD 公式
$$P(|\beta|) = \begin{cases} \alpha|\beta| & |\beta| \leq \alpha \\ \frac{-(|\beta|^2 - 2a\alpha|\beta| + \alpha^2)}{2(a-1)} & \alpha < |\beta| \leq a\alpha \\ \frac{(a+1)\alpha^2}{2} & |\beta| > a\alpha \end{cases}$$

### LLA（局部线性近似）
非凸惩罚（SCAD、MCP）通过 LLA 求解：
1. 在当前迭代点计算权重 `w_j = P'(|β_j|)`
2. 求解加权 L1 问题：`min L(β) + Σ w_j·|β_j|`
3. 重复直到收敛（通常 2-5 次迭代）

## 3. 求解器

::: warning 估计器边界
求解器库包含的可调用函数多于高层估计器公开的 `solver=` 取值。选择算法时应先看
模型页，本页只解释内部调度关系。
:::

### 公开控制项与内部路径

| 层级 | 名称 | 含义 |
|---|---|---|
| 共享惩罚估计器选择器 | `auto`、`fista`、`fista_bb`、`irls`、`newton`、`lbfgs`、`admm`、`exact` | 候选值；每个估计器只接受其中一部分 |
| CPU 平方误差兼容路径 | `coordinate_descent` | 仅 L1/Elastic Net；不是分位数 CD |
| 内部或低层函数 | `fista_lla_path`、`proximal_irls_quantile_solver`、`quantile_cd_solver`、`proximal_newton_solver`、`lbfgs_b_solver` | 实现 API，不是通用估计器关键字 |
| 固定算法估计器 | 无选择器 | Logistic 回归、普通分位数回归、普通 Cox PH 和有序模型自行选择算法 |

模型专属路由会先于部分通用 solver 分支执行。对 SCAD、MCP 和 Adaptive Lasso，
一个通过共享校验的通用名称未必会切换到不同算法。除非模型页明确说明路径不同，
不要用这些名称做算法性能对比。

### 当前自动路径

| 模型或目标 | 自动或固定路径 |
|---|---|
| 普通 GLM | IRLS |
| Ridge | wrapper 默认 `exact`；共享 `auto` 在 NumPy 上用 exact，在 CuPy/Torch 上用 Newton |
| Lasso / Elastic Net | FISTA |
| Adaptive Lasso | 初始估计后执行加权 L1 FISTA |
| SCAD / MCP | 模型专属 FISTA-LLA continuation |
| 惩罚分位数，L2/none | 分位数 IRLS refinement |
| 惩罚分位数，SCAD/MCP | proximal IRLS + LLA |
| 惩罚稳健回归，L2/none | Newton |
| 惩罚稳健回归，稀疏惩罚 | FISTA；SCAD/MCP 使用 FISTA-LLA |
| 普通 Cox PH / 有序模型 | 固定 Newton 变体，不提供公开选择器 |

平方误差/L2 的 `exact` 求解器与 `CoxPH(ties="exact")` 无关。精确的接受与拒绝
组合见[求解器 × 惩罚兼容矩阵](solver-penalty-matrix.md)。

### 专用内部路径

- 分位数 SCAD/MCP 使用二次 IRLS majorization 和 LLA 阈值步骤。估计器暴露的是
  高层 penalty-aware 路径，而不是低层函数名。
- SCAD/MCP FISTA-LLA 先走 continuation path，再更新局部线性权重，并求解加权 L1
  内层问题。
- Proximal Newton 仍是低层求解器库 facade。当前稳健回归 SCAD/MCP 的估计器路径是
  FISTA-LLA，而不是 Proximal Newton。

## 4. 后端范围

共享 FISTA、FISTA-BB、加权 L1/FISTA-LLA、分位数 IRLS、光滑
Newton/L-BFGS 和 Cox 专属路径，在对应估计器开放时具有 NumPy、CuPy 与 Torch
实现。后端覆盖不会扩大模型的公开选择器；例如 `LogisticRegression` 在所有后端
都仍然固定使用 IRLS。

## 5. 模型专属参考

| 模型族 | 以此求解器章节为准 |
|---|---|
| GLM wrappers | [广义线性模型](../models/generalized-linear-model.md#求解器支持)、[逻辑回归](../models/logistic-regression.md#求解器支持)、[泊松回归](../models/poisson-regression.md#求解器支持) |
| 凸正则化 | [Ridge](../models/ridge.md#求解器支持)、[Lasso](../models/lasso.md#求解器支持)、[Elastic Net](../models/elastic-net.md#求解器支持) |
| 加权或非凸正则化 | [Adaptive Lasso](../models/adaptive-lasso.md#求解器支持)、[SCAD](../models/scad.md#求解器支持)、[MCP](../models/mcp.md#求解器支持) |
| 专门回归 | [分位数回归](../models/quantile.md#求解器支持)、[稳健回归](../models/robust.md#求解器支持)、[Cox PH](../models/coxph.md#求解器支持) |
| 其他模型专属算法 | [有序模型](../models/ordered.md#求解器支持)、[核方法](../models/kernel-methods.md#求解器支持)、[PCA](../unsupervised/pca.md#求解器支持)、[NMF](../unsupervised/nmf.md#求解器支持) |

`PenalizedCoxPHModel` 仅提供惩罚估计且不拟合截距。`compute_inference=True`
会抛出 `NotImplementedError`；需要推断、baseline hazard 或生存曲线时使用普通
`CoxPH`。

## 6. 快速参考

```python
# Quantile 回归 + SCAD
from statgpu.linear_model.penalized import PenalizedQuantileRegression
model = PenalizedQuantileRegression(quantile=0.5, penalty='scad', alpha=0.1)
model.fit(X, y)

# 稳健回归 + MCP
from statgpu.linear_model.penalized import PenalizedRobustRegression
model = PenalizedRobustRegression(loss='huber', penalty='mcp', alpha=0.1)
model.fit(X, y)

# Cox PH + SCAD 惩罚（FISTA-LLA；响应为 [time, event]）
import numpy as np
from statgpu.linear_model import PenalizedCoxPHModel

y_surv = np.column_stack([time, event])
model = PenalizedCoxPHModel(
    penalty='scad', alpha=0.1,
    fit_intercept=False, compute_inference=False,
)
model.fit(X, y_surv)

# 全部惩罚 + 损失通过 PenalizedGeneralizedLinearModel
from statgpu.linear_model.penalized import PenalizedGeneralizedLinearModel
model = PenalizedGeneralizedLinearModel(loss='gamma', penalty='scad', alpha=0.1)
model.fit(X, y)
```

## 参考文献

- Fan & Li (2001): Variable selection via nonconcave penalized likelihood (SCAD)
- Zhang (2010): Nearly unbiased variable selection under minimax concave penalty (MCP)
- Wu & Liu (2009): Variable selection in quantile regression
- Hunter & Li (2005): MM algorithms for nonconvex penalized estimation
- Barzilai & Borwein (1988): Two-point step size gradient methods (BB)
- O'Donoghue & Candes (2015): Adaptive restart for accelerated gradient schemes
