# 损失函数 × 惩罚项 × 求解器框架

> 语言：中文  
> 最后更新：2026-09-17  
> 页面定位：计算架构指南  
> 切换：[英文版](../../en/guides/loss-penalty-solver-framework.md)

## 概述

statgpu 将面向用户的统计模型接口与内部数值组件分层组织：

- **模型（Model）**负责用户可见的统计契约和完整拟合生命周期；
- **损失函数（Loss）**定义数据拟合项以及它能够提供的数值原语；
- **惩罚项（Penalty）**定义正则化项及其数值原语；
- **求解器（Solver）**消费这些原语并执行优化；
- **计算后端（Backend）**贯穿数值层，在 NumPy、CuPy 或 Torch 上提供执行环境；
- **CV / meta-estimator**重复构造并评价兼容的拟合，再执行最终重拟合。

本页只解释这些组件**如何组合以及职责边界在哪里**。逐个损失函数的数学公式、具体模型的 solver 特例、求解器更新方程和完整兼容性矩阵不在本页重复。

需要具体信息时请查阅：

- [损失函数](../models/losses.md) — 损失函数定义与 loss-level 数值性质
- [求解器 × 惩罚项兼容性矩阵](solver-penalty-matrix.md) — 实际 loss × penalty × solver 路径
- [求解器算法](solver-algorithms.md) — 更新公式、收敛行为与算法前提
- [广义线性模型](../models/generalized-linear-model.md)、[分位数回归](../models/quantile.md)、[稳健回归](../models/robust.md)、[CoxPH](../models/coxph.md) 等模型页 — 模型专属统计行为

## 1. 运行架构

一次典型拟合可以概括为：

```text
用户
  │
  │  estimator = Model(...)
  │  estimator.fit(X, y, sample_weight=...)
  ▼
模型 / 公共 API
  │
  ├── 解析并验证数据 / formula 输入
  ├── 解析 backend 与 device
  ├── 构造 Loss
  ├── 构造 Penalty
  ├── 验证或解析 Solver
  ├── 准备截距 / 初值 / fit-local 状态
  │
  ▼
优化问题
  │
  │               F(β) = L(β) + P(β)
  │                      ▲       ▲
  │                      │       │
  │                    Loss   Penalty
  ▼
Solver
  │
  ├── 调用算法需要的数值原语
  ├── 在选定后端上迭代
  ├── 执行停止 / 收敛判据
  │
  ▼
拟合后的模型状态
  │
  ├── 系数 / 截距
  ├── 收敛信息与 solver provenance
  ├── estimator 暴露的推断或 CV 状态
  └── 预测 / 评分接口
```

模型类是整个流程的编排边界。某个 loss、penalty 或底层 solver 可以独立使用，但“这个底层组件存在”本身并不能推出完整的公开 estimator 路径已经受支持。

## 2. 各组件的职责

| 组件 | 负责 | 单独不能决定 |
|---|---|---|
| **Model / estimator** | 公共参数、数据验证、formula、loss/penalty 构造、solver 选择、拟合状态、预测、estimator-level 推断 | 底层算法的更新方程 |
| **Loss** | 数据拟合项 `L(β)`、函数值/梯度，以及可选的曲率或专用原语 | 哪些 penalty 或公开 estimator 一定可用 |
| **Penalty** | 正则化项 `P(β)`、梯度/prox/LLA 或分组元数据 | loss 是否提供某 solver 所需的数值量 |
| **Solver** | 数值更新规则、线搜索/步长策略、停止与收敛行为 | loss、权重、估计目标或 CV 的统计含义 |
| **Backend** | 数组类型、device、线性代数和 backend-native 数值操作 | 某条统计路径是否受支持 |
| **CV / meta-estimator** | fold-local 重建、评分、选择和最终重拟合 | 绕过一条原本不支持的 direct-fit 契约 |

因此阅读 API 时不能只看单个函数签名。例如一个 solver 带有 `sample_weight` 参数，并不意味着任意 loss 和任意 estimator 都自动获得同样的带权能力。

## 3. 契约如何组合

通用优化目标为

$$
F(\beta)=L(\beta)+P(\beta).
$$

不同求解器需要不同的数值原语。具体算法可能使用以下量中的一部分：

$$
L(\beta),\qquad
\nabla L(\beta),\qquad
\nabla^2L(\beta),\qquad
P(\beta),\qquad
\nabla P(\beta),\qquad
\operatorname{prox}_{\gamma P}(v),
$$

也可能需要某个 loss 或 penalty 专有的 IRLS 上界、局部线性 surrogate、分组结构等操作。

一条公开路径只有在所有这些组件描述的是**同一个目标函数和参数化**时才是完整一致的。单独每个函数都“可以调用”并不足够。特别需要保证：

- objective 与 gradient 使用相同的缩放和权重约定；
- 依赖曲率的算法使用与目标函数一致的曲率；
- proximal 方法使用与其 prox 或 surrogate 步骤相匹配的 penalty 表示；
- 截距处理在 loss、penalty、初始化和 solver 更新之间保持一致；
- backend/device 的选择不能通过静默换路径改变原本的数值问题。

当前具体组合请查阅 [求解器 × 惩罚项兼容性矩阵](solver-penalty-matrix.md)。

## 4. Capability matching 与 solver 分发

模型层大体按以下顺序解析一条拟合路径：

1. 规范化公共别名并验证模型参数；
2. 构造具体 Loss 与 Penalty 对象；
3. 解析数值 backend/device；
4. 若用户显式指定 solver，验证完整的 loss × penalty × estimator 路径是否支持它；
5. 若请求 `solver="auto"`，根据模型的兼容性策略选择自动路径；
6. 执行该 solver，不静默改写用户的显式 solver 请求；
7. 保存 estimator 对外暴露的实际 solver/backend 信息。

因此 `solver="auto"` 是一种**分发策略**，不是一种具体数值算法。其结果可以随 loss、penalty、backend 和 estimator 类型变化。直接拟合与 CV 也可以因为计算负载不同而有意采用不同的自动路径。

内部 resolved label 还可能表示 continuation 或 composite path；这些标签不一定都是合法的公共 `solver=` 参数值。兼容性矩阵负责区分用户可显式请求的 solver 与内部解析后的路径。

准确的当前分发规则见 [求解器 × 惩罚项兼容性矩阵](solver-penalty-matrix.md)；各算法如何计算见 [求解器算法](solver-algorithms.md)。

## 5. `sample_weight` 与目标函数一致性

`sample_weight` 不是 solver 的通用属性。它的统计含义和可用范围属于完整的 **loss × solver × estimator** 契约。

对于采用归一化解析权重目标的路径，数据拟合项形如

$$
L_w(\beta)
=
\frac{\sum_i w_i\,\ell_i(\beta)}{\sum_i w_i}.
$$

一条完整的带权路径必须把同一种约定应用到算法所使用的全部数值量上。根据 solver 的不同，这可能包括 objective、gradient、Hessian 或其他曲率、线搜索候选点、majorization 权重、停止量以及 validation score。

因此：

- 共享方法签名中出现 `sample_weight`，并不能推出完整 estimator 已经支持带权拟合；
- 提供权重不会自动把用户显式指定的 solver 改成另一个 solver；
- 不支持的带权组合应明确报错，而不是在用户不可见的情况下改变统计或数值问题；
- 只有明确采用上述归一化目标的路径，才可以据此推出共同缩放权重不改变目标最优点。

模型专属的权重语义由各模型页说明；loss-level 的一阶权重语义见 [损失函数](../models/losses.md)。

## 6. Backend 与 device 边界

Backend 是贯穿各层的执行维度，而不是独立的统计模型层：

```text
                  NumPy / CuPy / Torch
                ┌──────────────────────┐
Model      ──────┤ device 选择          │
Loss       ──────┤ objective 原语       │
Penalty    ──────┤ prox / gradient      │
Solver     ──────┤ 数值迭代             │
                └──────────────────────┘
```

在受支持路径上，数值数组和迭代计算保持在解析后的 backend/device 上；只有明确记录的报告元数据或小型同步边界才需要跨到 host。显式 GPU 请求不表示允许静默改为 CPU。

Backend 能力仍然是 route-specific 的：某个 solver 同时实现 NumPy、CuPy、Torch，并不能证明所有 loss、estimator 或 inference procedure 都支持这三个后端。模型专属的预处理和设备边界应写在对应模型文档中。

## 7. CV 与 meta-estimator 边界

CV / meta-estimator 在普通拟合之外增加一层编排。它负责：

- 构造 fold-local estimator 状态；
- 重建可变的 loss/penalty 状态，避免不同候选之间泄漏；
- 使用当前 fold 的训练数据和权重；
- 计算声明的 validation score；
- 选择调参结果；
- 按预定路径执行最终全数据重拟合。

每个 candidate 内部仍必须遵守公开的 loss/penalty/solver 契约。CV 不会使一个原本不支持的 solver 组合突然变成合法组合，也不应静默重新解释显式 solver 请求。

为了计算效率，CV 的 `solver="auto"` 可以有意与 direct-fit 的 `solver="auto"` 不同。这些差异属于兼容性参考，统一记录在 [CV 矩阵](solver-penalty-matrix.md#4-cv-solverauto-penalizedglm_cv) 中。

strict / approximate CV、loss-specific scoring、continuation path 构造和模型专属 final-refit 行为等统计细节，应放在对应模型或 CV 文档中，而不是在 framework 页重复。

## 8. 公共 estimator API 与底层 API

普通用户应优先使用模型类作为入口。文档明确公开的底层 loss、penalty、solver API 也可以直接使用，但其契约可能比 estimator 路径更窄，或者本来就是不同层次的接口。

不要根据以下任意一个单独事实推断 estimator 已支持某项能力：

- 某个 loss 有 `gradient()` 或 `hessian()`；
- 某个 penalty 有 `prox()`；
- 某个 solver 函数签名里存在某个参数；
- 某个底层调用在一个 backend 或一个无权重例子中可以运行。

反过来，一个 estimator 也可以把多个底层组件组合成一条解析后的路径。公开支持范围由完整 estimator 契约与兼容性矩阵定义，而不是由单个底层组件决定。

## 9. 文档职责分工

为了保持文档层次稳定，各类问题使用以下 canonical 文档：

| 问题 | 主要文档 |
|---|---|
| 这个 loss 的数学定义是什么？提供哪些数值原语？ | [损失函数](../models/losses.md) |
| 某个 loss × penalty 组合实际使用哪个 solver？ | [求解器 × 惩罚项兼容性矩阵](solver-penalty-matrix.md) |
| Newton / FISTA / ADMM 等具体如何更新？ | [求解器算法](solver-algorithms.md) |
| Loss、Penalty、Solver、Backend、CV 如何组合？ | **本页** |
| 某个具体模型支持什么、为什么？ | 对应模型页 |
| 某种 inference procedure / estimand 是否可用？ | [推断模式](inference-modes.md)及模型专属推断文档 |

这种分工是有意的：某个模型增加或修改一个特例时，不应该把 framework 页再次变成第二份兼容性矩阵或第二份模型手册。

## 相关文档

- [损失函数](../models/losses.md)
- [求解器 × 惩罚项兼容性矩阵](solver-penalty-matrix.md)
- [求解器算法](solver-algorithms.md)
- [广义线性模型](../models/generalized-linear-model.md)
- [分位数回归](../models/quantile.md)
- [稳健回归](../models/robust.md)
- [CoxPH](../models/coxph.md)
- [推断模式](inference-modes.md)
