# 损失函数 × 惩罚项 × 求解器框架

> 语言：中文
>
> 最后更新：2026-09-16
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
    smooth_gradient: bool   # 逐样本梯度是否为光滑梯度
    has_hessian: bool       # 是否提供 Hessian 数值原语
    _supports_irls: bool    # 是否声明可进入维护中的 IRLS 调度路径
```

这些字段描述的是损失函数提供的**数值原语或调度能力**，并不单独决定完整的 solver × penalty 支持关系。

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
| Huber | `HuberLoss` | ✅ | ✅ | ❌ | `MASS::rlm()` |
| Bisquare | `BisquareLoss` | ✅ | ✅ | ✅ | `MASS::rlm(psi="bisquare")` |
| Fair | `FairLoss` | ✅ | ✅ | ✅ | 需要自定义 psi/外部 reference；MASS 没有内建 Fair psi |
| Cox PH | `CoxPartialLikelihoodLoss` | ✅ | ✅ | ❌ | `survival::coxph()` |

Huber 当前的 `_supports_irls=False` 表示公共调度不会进入 Huber IRLS；因此显式请求该路径时按当前兼容性约定拒绝，而不是静默切换到其他求解器。Fair loss 的外部比较也需要显式匹配 Fair psi 实现，不能使用并不存在的 `MASS::rlm(psi="fair")` 内建选项。

### 逐样本公式

**分位数损失（check，又称 pinball）**：
$$\ell(u) = u \cdot (\tau - \mathbf{1}_{u<0}), \quad u = y - \eta$$

**Huber**（有效阈值为 $\delta$）：
$$\ell(u) = \begin{cases} \frac{1}{2}u^2 & |u| \leq \delta \\ \delta|u| - \frac{1}{2}\delta^2 & |u| > \delta \end{cases}$$

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
| L2（Ridge） | ✅ | ✅ | — | ❌ | $\frac{\alpha}{2}\|\beta\|_2^2$ |
| L1（Lasso） | ✅ | ❌ | 软阈值 | ❌ | α·‖β‖₁ |
| ElasticNet | ✅ | ❌ | 软阈值 | ❌ | $\alpha\left(r\|\beta\|_1+\frac{1-r}{2}\|\beta\|_2^2\right)$ |
| SCAD | ❌ | ❌ | 三段式 | ✅ | 分段函数 |
| MCP | ❌ | ❌ | 三段式 | ✅ | 分段函数 |
| 自适应 L1 | ✅ | ❌ | 加权软阈值 | ✅ | α/|β̂|^ν · |β| |
| 分组 Lasso | ✅ | ❌ | 分组软阈值 | ❌ | · |
| 分组 MCP | ❌ | ❌ | 分组近端 | ✅ | · |
| 分组 SCAD | ❌ | ❌ | 分组近端 | ✅ | · |

`AdaptiveL1Penalty` 在数据驱动的坐标权重确定后是凸惩罚。表中的 “LLA 支持” 不代表 standalone adaptive L1 是非凸 LLA 路径；当前 auto dispatch 在初始化权重后把它作为凸稀疏惩罚处理。

### SCAD 公式
$$P(|\beta|) = \begin{cases} \alpha|\beta| & |\beta| \leq \alpha \\ \frac{-(|\beta|^2 - 2a\alpha|\beta| + \alpha^2)}{2(a-1)} & \alpha < |\beta| \leq a\alpha \\ \frac{(a+1)\alpha^2}{2} & |\beta| > a\alpha \end{cases}$$

### LLA（局部线性近似）
非凸惩罚（SCAD、MCP）通过 LLA 求解：
1. 在当前迭代点计算权重 `w_j = P'(|β_j|)`；
2. 求解加权 L1 问题：`min L(β) + Σ w_j·|β_j|`；
3. 重复直到收敛（通常 2–5 次迭代）。

## 3. 求解器

### 自动调度表

`solver="auto"` 的主要分发可概括如下。公开的 `none` / `null` 会在求解器选择前规范化为 `L2(alpha=0)`，因此无惩罚的光滑行与 L2 使用同一个分支。

| 优先级 | 求解器 | 条件 |
|----------|--------|------|
| 1 | `exact` | `squared_error` + L2/none + NumPy |
| 2 | `newton` | `squared_error` + L2/none + GPU |
| 3 | 专用延续路径 | Quantile SCAD/MCP → Proximal IRLS-CD；其他 SCAD/MCP 与分组非凸惩罚使用对应维护中的 LLA wrapper |
| 4 | `irls` | Quantile + L2/none |
| 5 | `fista` / `fista_bb` | Quantile 凸稀疏 → 普通 FISTA；其他凸稀疏惩罚（包括初始化后的 adaptive L1）按 loss/backend/CV 选择 FISTA 或 FISTA-BB |
| 6 | `lbfgs` / `newton` | 交叉验证 + L2 + 特定损失函数 |
| 7 | `newton` | GLM/稳健/Cox 等具有维护中 Hessian 的光滑 L2/无惩罚路径 |

对 Quantile，普通 FISTA 只用于受支持的凸稀疏模型路径；L2/无惩罚使用 IRLS，SCAD/MCP 使用 Proximal IRLS-CD。显式 smooth Quantile `fista`，以及模型/CV 层任意 Quantile `fista_bb`、`lbfgs`、`admm` 请求，都会在数值 dispatch 前明确失败。底层 solver API 中，FISTA-BB 与 ADMM 同样对 Quantile fail closed，而直接无权重/均匀权重 Quantile L-BFGS 继续作为既有兼容面保留。这里的 `exact` 是平方误差/L2 的闭式求解器，与 `CoxPH(ties="exact")` 无关。需要 family/backend-specific 的精确分派时，请查看 [求解器 × 惩罚项兼容性矩阵](solver-penalty-matrix.md)。

### 全部求解器

`sample_weight` 的支持取决于求解器、损失函数统计语义以及函数值、梯度、曲率等数值能力。下表列出当前维护的主要路径；未在表中声明的组合不应从其他求解器能力类推。

| 求解器 | 损失约束 | 惩罚约束 | `sample_weight` | `warm_start` |
|--------|:-----------------|:---------------------|:------------|:----------:|
| `exact` | 仅平方误差 | 仅 L2 | ✅ | ❌ |
| `irls` | 声明 IRLS 调度能力的损失 | L2 / 无惩罚 | 对应损失的 IRLS 路径支持时可用 | ❌ |
| `newton` | 有 Hessian 的损失 | L2 / 无惩罚 | 由损失函数能力决定；普通 GLM ✅ | ❌ |
| `lbfgs` | 光滑损失；另保留未传/均匀权重的底层 Quantile 兼容面 | L2 / 无惩罚 | 受能力声明约束；普通 GLM ✅；Quantile 非均匀权重 ❌ | ❌ |
| `lbfgs_b` | 光滑盒约束问题 | L2 / 无惩罚 | 尚未声明通用的非均匀权重约定 | ❌ |
| `fista` | 支持梯度/近端路径的损失 | 受支持的近端惩罚 | 由具体损失路径决定 | ✅ |
| `fista_bb` | 具有有效 smooth-gradient difference 的损失；不含 Quantile | 受支持的稀疏惩罚 | 由具体损失路径决定 | ✅ |
| `fista_lla` | 支持当前 LLA 路径的损失 | SCAD/MCP 与分组非凸 LLA 路径 | 由具体损失路径决定 | ✅ |
| `proximal_irls_cd` | 仅分位数损失 | SCAD/MCP | ✅ | ✅ |
| `proximal_newton` | 有 Hessian 的光滑损失 | L2 / 无惩罚 | 由具体损失路径决定 | ✅ |
| `admm` | 具有维护中光滑 w-update 的 ADMM 损失；不含 Quantile | 受支持的近端形式 | 仅未传权重或均匀权重；真正非均匀权重会明确报错 | ✅ |

`lbfgs` 这一行描述的是通用底层 solver 能力，不是模型层 dispatch。`PenalizedQuantileRegression` 与 `PenalizedGLM_CV` 仍会在数值拟合前拒绝显式 `solver="lbfgs"`。

### 专用求解器

**Proximal IRLS-CD**（分位数 + SCAD/MCP）：
1. 计算 IRLS 权重：`w_i = τ_i / max(|r_i|, ε)`；
2. 构造二次上界：`Q(β) = ½ Σ w_i(y_i - X_iβ)²`；
3. 执行并行对角上界更新并结合 LLA 阈值；
4. GPU 上的收敛比较留在设备端，只同步最终布尔结果。

**Proximal Newton**（当前维护的光滑路径）使用完整目标

$$
F(\beta)=L(\beta)+P(\beta).
$$

对支持解析权重的逐样本损失，可写成

$$
L(\beta)=\frac{1}{s}\sum_{i=1}^n w_i\,\ell_i(x_i^\top\beta),
\qquad
s=\sum_i w_i,
$$

无权重时取 $w_i=1$、$s=n$。若记

$$
\psi_i=\frac{\partial\ell_i}{\partial\eta_i},
\qquad
h_i=\frac{\partial^2\ell_i}{\partial\eta_i^2},
\qquad
\eta_i=x_i^\top\beta,
$$

则在相应损失路径定义这些逐样本曲率时，

$$
\nabla L(\beta)
=\frac{X^\top(w\odot\psi)}{s},
\qquad
\nabla^2L(\beta)
=\frac{X^\top\operatorname{diag}(w\odot h)X}{s}.
$$

结构化损失直接使用自身实现的 Hessian 接口。对于当前维护的 L2 惩罚，

$$
P(\beta)=\frac{\alpha}{2}\|\beta\|_2^2,
\qquad
\nabla P(\beta)=\alpha\beta,
\qquad
\nabla^2P(\beta)=\alpha I.
$$

因此第 $k$ 次迭代构造完整目标的梯度与 Hessian

$$
g_k=\nabla L(\beta_k)+\alpha\beta_k,
\qquad
H_k=\nabla^2L(\beta_k)+\alpha I,
$$

无惩罚时令 $\alpha=0$。实现先对 Hessian 对称化，再加入固定稳定项：

$$
\bar H_k=\frac12(H_k+H_k^\top),
\qquad
\widetilde H_k=\bar H_k+10^{-10}I.
$$

若

$$
\|g_k\|_2\le\texttt{tol},
$$

则停止；否则解

$$
\widetilde H_k d_k=g_k.
$$

代码采用“减去方向”的记号，因此候选点为

$$
\beta_k(t)=\beta_k-t d_k.
$$

若线性系统被识别为奇异或病态，当前实现不调用最小二乘回退，而直接令

$$
d_k=g_k.
$$

同时要求下降量

$$
q_k=g_k^\top d_k>0.
$$

若 $q_k$ 非有限或不大于 0，则同样使用最速下降：

$$
d_k=g_k,
\qquad
q_k=\|g_k\|_2^2.
$$

Armijo 回溯从 $t_0=1$ 开始，依次尝试

$$
t_m=2^{-m},
\qquad m=0,1,\ldots,24,
$$

接受第一个满足

$$
F(\beta_k-t_m d_k)
\le
F(\beta_k)-10^{-4}t_m q_k
$$

的候选点，并更新

$$
\beta_{k+1}=\beta_k-t_m d_k.
$$

若 25 个候选步长全部失败，则恢复 $\beta_{k+1}=\beta_k$，发出线搜索警告并停止。默认值为 `max_iter=50`、`tol=1e-6`；未提供 `init_coef` 时取 $\beta_0=0$。

对于真正的非光滑复合目标，Proximal Newton 应求解 Hessian 度量下的近端子问题

$$
\Delta_k
=\arg\min_{\Delta}
\left\{
\nabla L(\beta_k)^\top\Delta
+\frac12\Delta^\top\nabla^2L(\beta_k)\Delta
+P(\beta_k+\Delta)
\right\}.
$$

当前求解器没有实现该 Hessian-metric proximal 子问题，因此非光滑惩罚会在进入 Newton 迭代前转交 FISTA。也就是说，当前 L2/无惩罚的 `proximal_newton` 数值上是稳定化的 damped Newton + Armijo，不会额外套一次欧氏近端算子，从而避免重复计算 L2 曲率。

**FISTA-LLA**（通用非凸路径）：
1. continuation：λ_max → 目标 α（3–5 步）；
2. LLA 外循环（每步 2–5 次）；
3. 通用复合路径使用加权凸 FISTA 内层。只有当损失函数显式提供正确的 Hessian-metric proximal 子问题时，才应启用 Proximal-Newton 内层。Cox 的 SCAD/MCP 当前继续使用 FISTA-LLA。

## 4. 后端覆盖

| 求解器 / 路径 | NumPy | CuPy | Torch |
|:---------------|:---:|:---:|:---:|
| Proximal IRLS-CD | ✅ | ✅ | ✅ |
| Proximal Newton（光滑路径） | ✅ | ✅ | ✅ |
| FISTA（带权） | ✅ | ✅ | ✅ |
| FISTA-BB（带权） | ✅ | ✅ | ✅ |
| FISTA-LLA（带权） | ✅ | ✅ | ✅ |
| Quantile IRLS（平滑 L2/无惩罚 auto 或显式请求） | ✅ | ✅ | ✅ |
| Cox partial likelihood（Breslow/Efron） | ✅ 原生 | ✅ 原生 | ✅ 原生 |
| CoxPH counting process / strata / Exact | ✅ 原生 | ✅ 原生 | ✅ 原生 |
| DBSCAN | ✅ | GPU 距离 + host-sync CC | ✅ on-device |
| UMAP | 是 | 支持，存在显式 SciPy host graph 边界 | 支持，存在显式 SciPy host graph 边界 |

## 5. 面向用户的带惩罚模型

这些是普通用户直接构造并调用 `.fit()` 的公开模型类；内部会解析 Loss、Penalty、Solver 与 Backend。

| 类 | 损失 | 惩罚 | 主要求解路径 |
|----|------|------|--------------|
| `PenalizedGeneralizedLinearModel` | 任意已注册损失 | 已注册惩罚 | 根据完整 loss × penalty × backend 组合进行 auto dispatch，或使用显式 solver |
| `PenalizedLinearRegression` | squared_error | l1/l2/elasticnet/scad/mcp/adaptive_l1 | L2/none：CPU exact / GPU Newton；凸稀疏：FISTA；SCAD/MCP：FISTA-LLA |
| `PenalizedLogisticRegression` | logistic | l1/l2/elasticnet/scad/mcp/adaptive_l1 | L2/none：Newton；direct 凸稀疏：FISTA-BB；SCAD/MCP：FISTA-LLA |
| `PenalizedPoissonRegression` | poisson | l1/l2/elasticnet/scad/mcp/adaptive_l1 | L2/none：Newton；direct 凸稀疏：FISTA-BB；SCAD/MCP：FISTA-LLA |
| `PenalizedQuantileRegression` | quantile | scad/mcp/l2 及其他受支持惩罚 | L2/none 的 auto 或显式请求：IRLS；凸稀疏：普通 FISTA；SCAD/MCP：Proximal IRLS-CD |
| `PenalizedRobustRegression` | huber/bisquare/fair | l1/l2/elasticnet/scad/mcp 等 | L2/none：Newton；凸稀疏：FISTA；SCAD/MCP：FISTA-LLA；Bisquare/Fair 另有显式 IRLS |
| `PenalizedCoxPHModel` | cox_ph | l1/l2/elasticnet/scad/mcp | L2/none：Newton；direct L1/ElasticNet：FISTA-BB；SCAD/MCP：FISTA-LLA |

`PenalizedCoxPHModel` 只负责带惩罚 Cox 系数估计；需要协方差、显著性检验、基线风险或生存曲线时应使用 `statgpu.survival.CoxPH`。

## 6. 快速示例

```python
# Quantile + SCAD
from statgpu.linear_model.penalized import PenalizedQuantileRegression
model = PenalizedQuantileRegression(quantile=0.5, penalty='scad', alpha=0.1)
model.fit(X, y)

# Robust + MCP
from statgpu.linear_model.penalized import PenalizedRobustRegression
model = PenalizedRobustRegression(loss='huber', penalty='mcp', alpha=0.1)
model.fit(X, y)

# Cox PH + SCAD
import numpy as np
from statgpu.linear_model.penalized import PenalizedCoxPHModel

y_surv = np.column_stack([time, event])
model = PenalizedCoxPHModel(
    penalty='scad', alpha=0.1,
    fit_intercept=False, compute_inference=False,
)
model.fit(X, y_surv)

# 通过通用 PenalizedGeneralizedLinearModel 组合损失与惩罚
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
