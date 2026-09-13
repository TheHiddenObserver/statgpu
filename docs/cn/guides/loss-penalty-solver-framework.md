# 损失函数 × 惩罚项 × 求解器框架

> 语言：中文
>
> 最后更新：2026-09-13
>
> 切换：[英文版](../../en/guides/loss-penalty-solver-framework.md)

## 概述

statgpu 将公共接口与数值计算接口分层组织：**模型类面向用户并负责组织拟合；损失函数与惩罚项共同定义优化问题；求解器读取该问题并执行数值优化；计算后端则贯穿这些步骤。**

“损失函数 × 惩罚类型 × 求解器 × 计算后端”构成模型拟合内部可组合的计算结构，并与模型类的继承层次彼此独立。本文档记录当前运行调用关系、调度逻辑和覆盖矩阵。

## 架构

### 面向用户的真实运行调用链

```text
用户
  │
  │  model = Estimator(...)
  │  model.fit(X, y, sample_weight=...)
  ▼
模型类 / 公共 API
  │
  ├── 公式与 X、y 的解析和验证
  ├── 计算后端 / 设备选择
  ├── _resolve_loss()      → LossBase 子类实例
  ├── _resolve_penalty()   → Penalty 子类实例
  ├── _select_solver()     → 求解器名称（auto 或显式）
  ├── sample_weight / 截距 / 初值处理
  │
  ▼
优化问题
  │
  │      F(β) = L(β) + P(β)
  │       ▲           ▲
  │       │           │
  │    损失函数      惩罚项
  │
  ▼
求解器
  │
  ├── exact / IRLS / Newton / L-BFGS
  ├── FISTA / FISTA-BB / FISTA-LLA
  ├── Proximal IRLS / Proximal Newton
  └── ADMM / 专用路径
  │
  │  返回系数、截距、迭代次数与收敛状态
  ▼
拟合后的模型状态
  │
  ├── coef_ / intercept_
  ├── 推断结果 / 拟合状态元数据
  ├── 计算后端 / 求解器来源记录
  └── predict() / summary()
```

在当前带惩罚模型的主拟合路径中，`_PenalizedFitMixin.fit()` 承担上述组织工作：先构造 `self._loss` 与 `self._penalty`，选择计算后端和求解器，再进入 `_fit_loss_backend()`、`_dispatch_irls()` 或 SCAD/MCP 等专用路径。`LossBase` 在这一层负责描述优化目标，并由模型对象构造后交给求解器使用。

### 各层职责

| 组件 | 主要职责 | 是否直接面向普通用户 |
|---|---|:---:|
| 模型类 | 公共 API、公式/数据验证、计算后端和求解器选择、状态管理、推断、预测 | ✅ |
| 损失函数 | 定义数据拟合项 `L(β)` 以及函数值、梯度、Hessian 等基础数值操作 | 通常否 |
| 惩罚项 | 定义 `P(β)` 以及梯度、近端算子、LLA 等正则化操作 | 通常否 |
| 求解器 | 根据损失函数和惩罚项声明的能力执行具体数值算法 | 否 |
| 计算后端 | NumPy/CuPy/Torch 数组、设备与数值操作；贯穿上述各层 | 通过模型类选择 |

损失函数与惩罚项**并列**组成目标函数：

$$
F(\beta)=L(\beta)+P(\beta).
$$

求解器随后调用诸如

$$
L(\beta),\quad \nabla L(\beta),\quad \nabla^2L(\beta),\quad
P(\beta),\quad \nabla P(\beta),\quad
\operatorname{prox}_{\gamma P}(v)
$$

这些基础操作来实现 Newton、L-BFGS、FISTA、ADMM 等算法。

### 计算后端是横切执行维度

NumPy、CuPy、Torch 贯穿模型准备、目标函数计算、惩罚项运算和求解器迭代。模型对象先确定实际使用的计算后端和设备；随后 `X`、`y`、`sample_weight`、损失函数导数、惩罚项的近端运算以及求解器迭代，都在接口约定允许的范围内尽量保留在同一计算后端。只有明确允许的元数据或最终小型结果会回到主机端。

```text
                  NumPy / CuPy / Torch
                ┌────────────────────────┐
模型类    ──────┤ 后端与设备选择          │
损失函数  ──────┤ 函数值 / 梯度 / Hessian │
惩罚项    ──────┤ 函数值 / 梯度 / 近端    │
求解器    ──────┤ 数值迭代                │
                └────────────────────────┘
```

本页聚焦于“模型构造损失函数与惩罚项，再交给通用求解器”的拟合路径。面板模型采用面板数据变换、OLS/GLS/分期回归和面板专用推断组织计算，其实现架构见 [面板模型架构](../panel/architecture.md)。

## 1. 损失函数

### LossBase

抽象基类位于 `statgpu/losses/_base.py`。子类实现 `per_sample_value()` 和 `per_sample_gradient()`，基类自动派生 `value()`、`gradient()` 与 `fused_value_and_gradient()`。

`LossBase` 是**优化问题的定义接口**。模型通常通过 `_resolve_loss()` 或相应工厂构造损失对象，再把它与惩罚对象一起交给求解器。

```python
class LossBase:
    name: str               # "quantile", "huber" 等
    y_type: str             # "continuous" / "survival"
    smooth_gradient: bool   # True → Newton 可用
    has_hessian: bool       # True → Proximal Newton 可用
    _supports_irls: bool    # True → 提供 irls() 方法
```

### 全部损失函数

| 损失 | 类 | `has_hessian` | `smooth_gradient` | `_supports_irls` | R 中的对应方法 |
|------|-------|:---:|:---:|:---:|--------------|
| 平方误差 | `GLMLoss` (`squared_error`) | ✅ | ✅ | ✅ | `lm()` |
| 逻辑回归 | `GLMLoss` (`logistic`) | ✅ | ✅ | ✅ | `glm(…, binomial)` |
| Poisson | `GLMLoss` (`poisson`) | ✅ | ✅ | ✅ | `glm(…, poisson)` |
| Gamma | `GLMLoss` (`gamma`) | ✅ | ✅ | ✅ | `glm(…, Gamma)` |
| 逆高斯 | `GLMLoss` (`inverse_gaussian`) | ✅ | ✅ | ✅ | `glm(…, inverse.gaussian)` |
| 负二项 | `GLMLoss` (`negative_binomial`) | ✅ | ✅ | ✅ | `glm.nb()` |
| Tweedie | `GLMLoss` (`tweedie`) | ✅ | ✅ | ✅ | `glm(…, tweedie)` |
| 分位数 | `QuantileLoss` | ❌ | ❌ | ✅ | `quantreg::rq()` |
| Huber | `HuberLoss` | ✅ | ✅ | ✅ | `MASS::rlm()` |
| Bisquare | `BisquareLoss` | ✅ | ✅ | ✅ | `MASS::rlm(psi="bisquare")` |
| Fair | `FairLoss` | ✅ | ✅ | ✅ | `MASS::rlm(psi="fair")` |
| Cox PH | `CoxPartialLikelihoodLoss` | ✅ | ✅ | ❌ | `survival::coxph()` |

### 逐样本公式

**分位数损失（check，又称 pinball）**：
$$\ell(u) = u \cdot (\tau - \mathbf{1}_{u<0}), \quad u = y - \eta$$

**Huber**（$k=1.345$）：
$$\ell(u) = \begin{cases} \frac{1}{2}u^2 & |u| \leq k \\ k|u| - \frac{1}{2}k^2 & |u| > k \end{cases}$$

**Bisquare（Tukey biweight，$c=4.685$）**：
$$\ell(u) = \begin{cases} \frac{c^2}{6}[1 - (1-(u/c)^2)^3] & |u| \leq c \\ c^2/6 & |u| > c \end{cases}$$

**Cox 部分似然**（`CoxPartialLikelihoodLoss` 中的 Breslow / Efron 并列事件处理）：
$$L(\beta) = \prod_{i:\delta_i=1} \frac{\exp(X_i\beta)}{\sum_{j:T_j \geq T_i} \exp(X_j\beta)}$$

该损失对象接收 `[time, event]` 两列响应，并服务于带惩罚 Cox 模型。完整的 `CoxPH`/`CoxPHCV` 还支持 Exact 并列事件处理和计数过程风险集

$$
R_s(t)=\{j:\operatorname{strata}_j=s,\;\operatorname{start}_j<t\leq
\operatorname{stop}_j\},
$$

以及 `subject_id` 语义；这些是高层 Cox 模型接口提供的附加数据结构。

## 2. 惩罚函数

### 全部惩罚

| 惩罚 | `is_convex` | `is_smooth` | 近端算子 | LLA 支持 | $P(\beta)$ |
|---------|:---:|:---:|:---:|:---:|------|
| None / Null | ✅ | ✅ | 恒等映射 | ❌ | 0 |
| L2（Ridge） | ✅ | ✅ | — | ❌ | α·‖β‖²₂ |
| L1（Lasso） | ✅ | ❌ | 软阈值 | ❌ | α·‖β‖₁ |
| ElasticNet | ✅ | ❌ | 软阈值 | ❌ | α(r‖β‖₁+(1-r)‖β‖²₂) |
| SCAD | ❌ | ❌ | 三段式 | ✅ | 分段函数 |
| MCP | ❌ | ❌ | 三段式 | ✅ | 分段函数 |
| 自适应 L1 | ✅ | ❌ | 加权软阈值 | ✅ | α/|β̂|^ν · |β| |
| 分组 Lasso | ✅ | ❌ | 分组软阈值 | ❌ | · |
| 分组 MCP | ❌ | ❌ | 分组近端 | ✅ | · |
| 分组 SCAD | ❌ | ❌ | 分组近端 | ✅ | · |

### SCAD 公式
$$P(|\beta|) = \begin{cases} \alpha|\beta| & |\beta| \leq \alpha \\ \frac{-(|\beta|^2 - 2a\alpha|\beta| + \alpha^2)}{2(a-1)} & \alpha < |\beta| \leq a\alpha \\ \frac{(a+1)\alpha^2}{2} & |\beta| > a\alpha \end{cases}$$

### LLA（局部线性近似）
非凸惩罚（SCAD、MCP）通过 LLA 求解：
1. 在当前迭代点计算权重 `w_j = P'(|β_j|)`；
2. 求解加权 L1 问题：`min L(β) + Σ w_j·|β_j|`；
3. 重复直到收敛（通常 2–5 次迭代）。

## 3. 求解器

### 自动调度表

`solver="auto"` 按以下优先级调度：

| 优先级 | 求解器 | 条件 |
|----------|--------|------|
| 1 | `exact` | `squared_error` + L2 + NumPy |
| 2 | `newton` | `squared_error` + L2 + GPU |
| 3 | `fista`（LLA） | 所有非凸惩罚（SCAD/MCP/自适应） |
| 4 | `fista` | 分位数损失（无 Hessian） |
| 5 | `fista` / `fista_bb` | 平方误差/GLM + 稀疏惩罚 |
| 6 | `lbfgs` / `newton` | 交叉验证 + L2 + 特定损失函数 |
| 7 | `newton` / `irls` | 光滑惩罚 + 光滑损失 |

### 全部求解器

`sample_weight` 的支持取决于求解器、损失函数统计语义以及函数值、梯度、曲率等数值能力。下表列出当前主要路径；完整的支持约定由 #153 跟踪。

| 求解器 | 损失约束 | 惩罚约束 | `sample_weight` | `warm_start` |
|--------|:-----------------|:---------------------|:------------|:----------:|
| `exact` | 仅平方误差 | 仅 L2 | ✅ | ❌ |
| `irls` | 支持 IRLS 的损失 | L2 / 无惩罚 | 对应损失的 IRLS 路径支持时可用 | ❌ |
| `newton` | 有 Hessian 的损失 | L2 / 无惩罚 | 由损失函数能力决定；普通 GLM ✅ | ❌ |
| `lbfgs` | 光滑损失 | L2 / 无惩罚 | 受能力声明约束；普通 GLM ✅ | ❌ |
| `lbfgs_b` | 光滑盒约束问题 | L2 / 无惩罚 | 尚未声明通用的非均匀权重契约 | ❌ |
| `fista` | 支持梯度/近端路径的损失 | 全部 | 由具体损失路径决定 | ✅ |
| `fista_bb` | 支持梯度/近端路径的损失 | 全部（非凸分组惩罚除外） | 由具体损失路径决定 | ✅ |
| `fista_lla` | 支持当前 LLA 路径的损失 | SCAD/MCP/自适应 | 由具体损失路径决定 | ✅ |
| `proximal_irls_cd` | 仅分位数损失 | SCAD/MCP | ✅ | ✅ |
| `proximal_newton` | 指定的 Hessian 损失 | SCAD/MCP/自适应（经 LLA） | 由具体损失路径决定 | ✅ |
| `admm` | 当前维护的 ADMM 损失 | 全部 | 仅未传权重或均匀权重；真正非均匀权重会明确报错 | ✅ |

### 专用求解器

**Proximal IRLS-CD**（分位数 + SCAD/MCP）：
1. 计算 IRLS 权重：`w_i = τ_i / max(|r_i|, ε)`；
2. 构造二次上界：`Q(β) = ½ Σ w_i(y_i - X_iβ)²`；
3. 执行并行对角上界更新并结合 LLA 阈值；
4. GPU 上的收敛比较留在设备端，只同步最终布尔结果。

**Proximal Newton**（Huber/Bisquare + SCAD/MCP）：
1. 计算 Hessian `H = ∇²ℓ(β)` 和梯度 `g = ∇ℓ(β)`；
2. 计算 Newton 方向：`d = -H⁻¹·g`；
3. 执行 Armijo 线搜索和近端更新；
4. 通常 5–10 次迭代收敛。

**FISTA-LLA**（通用非凸路径；也是 Cox + SCAD/MCP 的当前路径）：
1. 延续路径：从 `λ_max` 逐步到目标 `α`（3–5 步）；
2. LLA 外层循环（每步 2–5 次迭代）；
3. 根据损失函数选择 FISTA 或 Proximal Newton 内层；Cox 明确使用后端原生 FISTA，因为当前通用复合近端 Newton 的线搜索尚不适用于风险集目标。

## 4. 后端覆盖

| 求解器 / 路径 | NumPy | CuPy | Torch |
|:---------------|:---:|:---:|:---:|
| Proximal IRLS-CD | ✅ | ✅ | ✅ |
| Proximal Newton | ✅ | ✅ | ✅ |
| FISTA（加权） | ✅ | ✅ | ✅ |
| FISTA-BB（加权） | ✅ | ✅ | ✅ |
| FISTA-LLA（加权） | ✅ | ✅ | ✅ |
| 分位数 IRLS（光滑惩罚） | ✅ | ✅ | ✅ |
| CoxPH Breslow/Efron 损失 | ✅ | ✅（后端原生） | ✅（后端原生） |
| CoxPH Exact / `start-stop` / `strata` / `subject` | ✅ | ✅（共享计数过程实现） | ✅（共享计数过程实现） |
| DBSCAN | ✅ | GPU 距离计算 + 主机同步的连通分量 | ✅（设备端） |
| UMAP | ✅ | 识别后端 + 必要的主机传输 | 识别后端 + 必要的主机传输 |

## 5. 面向用户的带惩罚模型

这些类是用户通常直接构造并调用 `.fit()` 的公共模型层；它们内部解析损失函数、惩罚项、求解器与计算后端。

| 类 | 损失 | 惩罚 | 求解器 |
|-------|------|-----------|---------|
| `PenalizedGeneralizedLinearModel` | 任意 | 全部 10 种 | 全部 10 种 |
| `PenalizedLinearRegression` | `squared_error` | l1/l2/elasticnet/scad/mcp/adaptive_l1 | exact/fista |
| `PenalizedLogisticRegression` | `logistic` | l1/l2/elasticnet/scad/mcp/adaptive_l1 | irls/fista |
| `PenalizedPoissonRegression` | `poisson` | l1/l2/elasticnet/scad/mcp/adaptive_l1 | irls/fista |
| `PenalizedQuantileRegression` | `quantile` | scad/mcp/l2 | proximal_irls_cd/fista/irls |
| `PenalizedRobustRegression` | huber/bisquare | scad/mcp/l2 | proximal_newton/irls |
| `PenalizedCoxPHModel` | `cox_ph` | l1/l2/elasticnet/scad/mcp | FISTA；SCAD/MCP 使用 FISTA-LLA |

`PenalizedCoxPHModel` 提供带惩罚 Cox 系数估计；需要协方差、显著性检验、基线风险或生存曲线时，使用 `statgpu.survival.CoxPH`。

## 6. 快速参考

```python
# 分位数回归 + SCAD
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

# 通过 PenalizedGeneralizedLinearModel 组合不同损失与惩罚
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
- O'Donoghue & Candes (2015): Adaptive restart for accelerated gradient schemes (BB)