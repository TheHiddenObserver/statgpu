# Loss × Penalty × Solver 框架

> 语言：中文
>
> 最后更新：2026-07-12
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
       ├── fista / fista_bb / fista_lla → FISTA 家族
       ├── newton / irls                  → 光滑路径
       ├── proximal_irls_cd              → quantile + SCAD/MCP
       ├── proximal_newton               → Huber/Bisquare + SCAD/MCP
       ├── Cox + SCAD/MCP                → Cox 专用 FISTA-LLA
       └── lbfgs / admm                 → 拟牛顿 / 增广拉格朗日
```

## 1. 损失函数

### LossBase

抽象基类位于 `statgpu/losses/_base.py`。子类实现 `per_sample_value()` 和 `per_sample_gradient()`。基类自动派生 `value()`、`gradient()`、`fused_value_and_gradient()`。

```python
class LossBase:
    name: str               # "quantile", "huber" 等
    y_type: str             # "continuous" / "survival"
    smooth_gradient: bool   # True → Newton 可用
    has_hessian: bool       # True → proximal Newton 可用
    _supports_irls: bool    # True → 有 irls() 方法
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
| Huber | `HuberLoss` | ✅ | ✅ | ✅ | `MASS::rlm()` |
| Bisquare | `BisquareLoss` | ✅ | ✅ | ✅ | `MASS::rlm(psi="bisquare")` |
| Fair | `FairLoss` | ✅ | ✅ | ✅ | `MASS::rlm(psi="fair")` |
| Cox PH | `CoxPartialLikelihoodLoss` | ✅ | ✅ | ❌ | `survival::coxph()` |

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

### 自动调度表

`solver="auto"` 按以下优先级调度：

| 优先级 | 求解器 | 条件 |
|----------|--------|------|
| 1 | `exact` | squared_error + l2 + numpy |
| 2 | `newton` | squared_error + l2 + GPU |
| 3 | `fista` (LLA) | 所有非凸惩罚 (SCAD/MCP/adaptive) |
| 4 | `fista` | quantile（无 Hessian） |
| 5 | `fista` / `fista_bb` | squared_error/GLM + 稀疏惩罚 |
| 6 | `lbfgs` / `newton` | CV + L2 + loss 特定 |
| 7 | `newton` / `irls` | 光滑惩罚 + 光滑损失 |

### 全部求解器

| 求解器 | 损失约束 | 惩罚约束 | sample_weight | warm_start |
|--------|:-----------------|:---------------------|:------------:|:----------:|
| `exact` | 仅 squared_error | 仅 l2 | ✅ | ❌ |
| `irls` | 任意支持 IRLS 的损失 | l2 / none | ✅ | ❌ |
| `newton` | 任意有 Hessian 的损失 | l2 / none | ❌ | ❌ |
| `lbfgs` | 任意 | l2 / none | ❌ | ❌ |
| `fista` | 任意 | 全部 | ✅ | ✅ |
| `fista_bb` | 任意 | 全部（非凸 group 除外） | ✅ | ✅ |
| `fista_lla` | 任意 | SCAD/MCP/adaptive | ✅ | ✅ |
| `proximal_irls_cd` | 仅 quantile | SCAD/MCP | ✅ | ✅ |
| `proximal_newton` | 有 Hessian 的损失 | SCAD/MCP/adaptive (LLA) | ✅ | ✅ |
| `admm` | 任意 | 全部 | ❌ | ✅ |

### 专用求解器

**Proximal IRLS-CD** (quantile + SCAD/MCP):
1. 计算 IRLS 权重：`w_i = τ_i / max(|r_i|, ε)`
2. 二次上界：`Q(β) = ½ Σ w_i(y_i - X_iβ)²`
3. 并行对角化 + LLA 阈值
4. GPU：收敛检查在 device 上比较，仅同步 bool

**Proximal Newton** (Huber/Bisquare + SCAD/MCP):
1. 计算 Hessian `H = ∇²ℓ(β)` 和梯度 `g = ∇ℓ(β)`
2. Newton 方向：`d = -H⁻¹·g`
3. Armijo 线搜索 + proximal 步
4. 通常 5-10 次迭代收敛

**FISTA-LLA**（通用非凸路径；也是 Cox + SCAD/MCP 的当前路径）：
1. Continuation path：λ_max → 目标 α（3-5 步）
2. LLA 外层循环（每步 2-5 次迭代）
3. 根据 loss 选择 FISTA 或 Proximal Newton 内层；Cox 明确使用后端原生 FISTA，
   因为通用 composite proximal-Newton 的线搜索尚不适用于风险集目标

## 4. 后端覆盖

| 求解器 / 路径 | NumPy | CuPy | Torch |
|:---------------|:---:|:---:|:---:|
| Proximal IRLS-CD | ✅ | ✅ | ✅ |
| Proximal Newton | ✅ | ✅ | ✅ |
| FISTA (加权) | ✅ | ✅ | ✅ |
| FISTA-BB (加权) | ✅ | ✅ | ✅ |
| FISTA-LLA (加权) | ✅ | ✅ | ✅ |
| Quantile IRLS (光滑惩罚) | ✅ | ✅ | ✅ |
| CoxPH Breslow/Efron loss | ✅ | ✅（后端原生） | ✅（后端原生） |
| CoxPH Exact / start-stop / strata / subject | ✅ | ✅（共享计数过程实现） | ✅（共享计数过程实现） |
| DBSCAN | ✅ | GPU 距离 + host-sync 连通分量 | ✅ on-device |
| UMAP | ✅ | backend-aware + host transfer | backend-aware + host transfer |

## 5. 惩罚模型类

| 类 | 损失 | 惩罚 | 求解器 |
|-------|------|-----------|---------|
| `PenalizedGeneralizedLinearModel` | 任意 | 全部 10 种 | 全部 10 种 |
| `PenalizedLinearRegression` | squared_error | l1/l2/elasticnet/scad/mcp/adaptive_l1 | exact/fista |
| `PenalizedLogisticRegression` | logistic | l1/l2/elasticnet/scad/mcp/adaptive_l1 | irls/fista |
| `PenalizedPoissonRegression` | poisson | l1/l2/elasticnet/scad/mcp/adaptive_l1 | irls/fista |
| `PenalizedQuantileRegression` | quantile | scad/mcp/l2 | proximal_irls_cd/fista/irls |
| `PenalizedRobustRegression` | huber/bisquare | scad/mcp/l2 | proximal_newton/irls |
| `PenalizedCoxPHModel` | cox_ph | l1/l2/elasticnet/scad/mcp | FISTA；SCAD/MCP 为 FISTA-LLA |

`PenalizedCoxPHModel` 仅提供惩罚估计，不拟合截距，也不提供协方差、显著性检验、
baseline 或生存曲线。`fit_intercept=True` 会报错；`compute_inference=True` 会抛出
`NotImplementedError`。需要这些结果时使用 `statgpu.survival.CoxPH`。

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
