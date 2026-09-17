# 求解器 × 惩罚项兼容性矩阵

> 语言：中文  
> 最后更新：2026-09-17  
> 页面定位：兼容性参考  
> 切换：[英文版](../../en/guides/solver-penalty-matrix.md)

## 概览

本页是模型层的紧凑参考，用来查询**给定 loss × penalty 组合时，statgpu 会自动选择或允许哪些求解器**。它的主要作用是查表，而不是重复各模型的算法专题说明。

本页主要回答三个问题：

1. 直接拟合时 `solver="auto"` 会选择什么？
2. 显式指定 solver 时，需要满足哪些数值前提？
3. `PenalizedGLM_CV` 在 `solver="auto"` 下会选择什么？

模型专属行为放在对应模型页；更新公式和算法前提放在 [求解器算法](solver-algorithms.md)。

通用约定：

- `none` / `null` 在求解器选择前规范化为 `L2(alpha=0)`；
- 显式 solver 请求会在数值拟合前验证，不会因为存在权重而被静默替换；
- 只有后端确实改变实际执行路径时，矩阵才显示后端差异；
- FISTA-LLA、Proximal IRLS-CD 等内部 resolved label 可以出现在矩阵中，即使它们不是公开的 `solver=` 参数值。

`AdaptiveGroupLassoPenalty` 可以作为公开惩罚对象使用，但调用者必须显式提供组权重，因此没有字符串注册别名。

## 1. 直接拟合的 `solver="auto"`

| 损失 | l2 / none | l1 | elasticnet | scad | mcp | adaptive_l1 | group_lasso | group_scad | group_mcp |
|------|:---------:|:--:|:----------:|:----:|:---:|:-----------:|:-----------:|:----------:|:---------:|
| **squared_error** | CPU `exact` / GPU Newton | FISTA | FISTA | FISTA-LLA | FISTA-LLA | FISTA | 分组 FISTA | 分组 FISTA-LLA | 分组 FISTA-LLA |
| **logistic** | Newton | FISTA-BB | FISTA-BB | FISTA-LLA | FISTA-LLA | FISTA-BB | 分组 FISTA | 分组 FISTA-LLA | 分组 FISTA-LLA |
| **poisson** | Newton | FISTA-BB | FISTA-BB | FISTA-LLA | FISTA-LLA | FISTA-BB | 分组 FISTA | 分组 FISTA-LLA | 分组 FISTA-LLA |
| **gamma** | Newton | FISTA | FISTA | FISTA-LLA | FISTA-LLA | FISTA | 分组 FISTA | 分组 FISTA-LLA | 分组 FISTA-LLA |
| **inverse_gaussian** | Newton | FISTA | FISTA | FISTA-LLA | FISTA-LLA | FISTA | 分组 FISTA | 分组 FISTA-LLA | 分组 FISTA-LLA |
| **negative_binomial** | Newton | FISTA-BB | FISTA-BB | FISTA-LLA | FISTA-LLA | FISTA-BB | 分组 FISTA | 分组 FISTA-LLA | 分组 FISTA-LLA |
| **tweedie** | Newton | CPU FISTA-BB / GPU FISTA | CPU FISTA-BB / GPU FISTA | FISTA-LLA | FISTA-LLA | CPU FISTA-BB / GPU FISTA | 分组 FISTA | 分组 FISTA-LLA | 分组 FISTA-LLA |
| **quantile** | IRLS | FISTA | FISTA | Proximal IRLS-CD | Proximal IRLS-CD | FISTA | 分组 FISTA | 分组 FISTA-LLA | 分组 FISTA-LLA |

### 如何阅读这张表

- 单元格表示**实际的自动分发路径**，不代表全部可用的显式 solver 选择。
- FISTA-LLA 表示标量 SCAD/MCP 使用的非凸延续路径；Group FISTA-LLA 是对应的分组路径。
- Proximal IRLS-CD 是专用 resolved route，而不是公开显式 solver 关键字。
- Group Lasso 和 Adaptive Group Lasso 使用 group-aware FISTA；Group SCAD/MCP 使用 group-aware LLA 路径。
- 某个单元格为什么采用该算法，属于对应模型页的解释范围，而不是矩阵页的正文重点。

分布族/链接函数的定义域限制、权重语义或特殊初始化规则见 [广义线性模型](../models/generalized-linear-model.md)。Quantile 的非光滑性质和具体求解器选择见 [分位数回归](../models/quantile.md)。

## 2. 显式求解器约束

下表总结的是**模型层显式 solver 请求**的主要数值前提。底层 solver 函数可能具有更窄或不同的接口约定，应以相应 API/算法文档为准。

| 求解器 | 主要数值前提 | 常见惩罚范围 | 公开请求说明 |
|--------|--------------|--------------|--------------|
| `exact` | 二次型平方误差目标 | L2 / none | 仅平方误差路径 |
| `irls` | loss 提供模型层 IRLS 路径 | L2 / none | 依损失函数/分布族而定 |
| `newton` | 光滑目标且提供 Hessian | L2 / none | Newton + 线搜索 |
| `lbfgs` | 光滑目标且梯度与目标一致 | L2 / none | 不显式形成完整 Hessian |
| `fista` | 一阶 loss 原语与惩罚近端步骤兼容 | 凸近端路径及部分显式路径 | 具体支持范围依路径而定 |
| `fista_bb` | 可用于 BB 步长更新的 smooth-gradient difference | 受支持稀疏近端路径 | loss 缺少有效光滑梯度差时不可用 |
| `admm` | 支持相应分裂形式且 w-subproblem 光滑 | 受支持近端形式 | 具体支持范围依路径而定 |
| `irls_cd` | 专用标量 IRLS/坐标下降结构 | 专用路径 | 不是通用 fallback |
| `proximal_irls_cd` | 专用 proximal IRLS majorization | 专用非凸路径 | 内部 resolved label；不是公开显式 `solver=` 关键字 |
| `proximal_newton` | loss 与 penalty 具备相容的 Newton/近端结构 | 依路径而定 | 行为取决于具体 loss 与 penalty |

不支持的显式 estimator 组合会在数值拟合前报错。本节只描述共享数值前提；具体模型的排除项与替代路径应查对应模型页。

## 3. 求解器能力概览

| 求解器 | 核心要求 | 常见用途 | `sample_weight` | `warm_start` |
|--------|----------|----------|-----------------|:------------:|
| `exact` | 二次型闭式解 / 特征分解 | 平方误差 + L2 | 在声明支持的路径上可用 | ❌ |
| `irls` | loss-specific reweighted least squares 更新 | 受支持的 L2/无惩罚路径 | 依 loss/estimator 而定 | ❌ |
| `newton` | 梯度 + Hessian | 光滑 L2/无惩罚目标 | 依 loss/estimator 而定 | ❌ |
| `lbfgs` | 一致的光滑梯度 | 光滑 L2/无惩罚目标 | 依 loss/estimator 而定 | ❌ |
| `fista` | 一阶 loss 原语 + 近端步骤 | 凸近端目标与 LLA 内层 | 依路径而定 | ✅ |
| `fista_bb` | smooth-gradient difference + 近端步骤 | 使用 BB 自适应步长的稀疏目标 | 依路径而定 | ✅ |
| `admm` | 相容的变量分裂与光滑 w-update | 近端形式 | 依路径而定 | ✅ |
| `irls_cd` | 专用 IRLS + 坐标下降 | 专用标量路径 | 依路径而定 | ✅ |

`sample_weight` 的支持范围是 **loss × solver × estimator** 的联合约定，不能仅根据 solver 函数签名判断。权重语义和不支持组合请查对应模型页以及 [损失函数 × 惩罚项 × 求解器框架](loss-penalty-solver-framework.md)。

## 4. CV 的 `solver="auto"`（`PenalizedGLM_CV`）

交叉验证可能有意使用与直接拟合不同的数值路径，因为候选拟合、后端执行和最终全数据重拟合具有不同的计算权衡。

| 损失 | l2 | l1 / elasticnet | scad / mcp | adaptive_l1 | group_lasso / adaptive group | group_scad / group_mcp |
|------|:--:|:---------------:|:----------:|:-----------:|:----------------------------:|:-----------------------:|
| **squared_error** | eig-batch / CPU exact-style；GPU 最终拟合在适用时用 Newton | FISTA | FISTA-LLA | FISTA | 分组 FISTA | 分组 FISTA-LLA |
| **logistic** | Newton | FISTA | FISTA-LLA | FISTA | 分组 FISTA | 分组 FISTA-LLA |
| **poisson** | Newton | CPU FISTA；GPU L1 可按规模选择 FISTA-BB，GPU ElasticNet 使用 FISTA-BB | FISTA-LLA | FISTA | 分组 FISTA | 分组 FISTA-LLA |
| **gamma** | L-BFGS | FISTA | FISTA-LLA | FISTA | 分组 FISTA | 分组 FISTA-LLA |
| **inverse_gaussian** | L-BFGS | FISTA | FISTA-LLA | FISTA | 分组 FISTA | 分组 FISTA-LLA |
| **negative_binomial** | L-BFGS | FISTA-BB；GPU ElasticNet 的特定规模区间使用 FISTA | FISTA-LLA | FISTA-BB | 分组 FISTA | 分组 FISTA-LLA |
| **tweedie** | Newton | CPU FISTA-BB / GPU FISTA | FISTA-LLA | CPU FISTA-BB / GPU FISTA | 分组 FISTA | 分组 FISTA-LLA |
| **quantile** | IRLS | FISTA | Proximal IRLS-CD | FISTA | 分组 FISTA | 分组 FISTA-LLA |

### CV 说明

- 表格记录实际的自动路径；只有确实影响分发的后端/规模差异才直接写在对应单元格中。
- 如果某个 loss × penalty × solver 组合支持显式 solver，请求在 CV 中仍保持有效，不会仅因为 folds 或权重存在而被静默替换。
- 候选拟合与最终全数据重拟合会保留解析后的 loss、penalty、groups 与 solver 约定。
- group 输入会在候选拟合前验证；详细规则见 [损失函数 × 惩罚项 × 求解器框架](loss-penalty-solver-framework.md)。
- strict/two-stage CV 语义和模型专属的验证细节放在对应模型页，不在本页重复。

## 5. 惩罚项参考

| 惩罚项 | 公式 | 近端 / surrogate 形式 | 主要参数 |
|---------|------|----------------------|----------|
| `l2` | ½α‖β‖² | ridge scaling | `alpha` |
| `l1` | α‖β‖₁ | soft threshold | `alpha` |
| `elasticnet` | α[λ‖β‖₁ + ½(1-λ)‖β‖²] | soft threshold + L2 scaling | `alpha`, `l1_ratio` |
| `scad` | SCAD(β; α, a) | SCAD thresholding / LLA | `alpha`, `a` |
| `mcp` | MCP(β; α, γ) | MCP thresholding / LLA | `alpha`, `gamma` |
| `adaptive_l1` | αΣ_j w_j|β_j| | weighted soft threshold | `alpha`、weights |
| `group_lasso` | αΣ_g √p_g‖β_g‖₂ | block soft threshold | `alpha`、`groups` |
| `AdaptiveGroupLassoPenalty` | αΣ_g w_g√p_g‖β_g‖₂ | weighted block soft threshold | `alpha`、`groups`、`weights`；仅对象形式 |
| `group_scad` | Σ_g SCAD(‖β_g‖₂; α√p_g, a) | group LLA surrogate | `alpha`、`groups`、`a` |
| `group_mcp` | Σ_g MCP(‖β_g‖₂; α√p_g, γ) | group LLA surrogate | `alpha`、`groups`、`gamma` |

Group SCAD/MCP 的凸 LLA surrogate 通过 adaptive group-lasso 问题表示。group 元数据必须与最终设计矩阵宽度一致；精确验证规则见 [损失函数 × 惩罚项 × 求解器框架](loss-penalty-solver-framework.md)。

## 6. 相关参考

本页刻意不重复模型专属推导、优化 safeguard、推断契约或验证流程。

- [求解器算法](solver-algorithms.md) — 更新公式、收敛/停止行为与算法前提
- [损失函数 × 惩罚项 × 求解器框架](loss-penalty-solver-framework.md) — 计算架构与分发概念
- [损失函数](../models/losses.md) — loss layer 的数学定义与数值原语
- [广义线性模型](../models/generalized-linear-model.md) — GLM 分布族/链接函数、权重、CV 与推断
- [分位数回归](../models/quantile.md) — Quantile 专属求解器、惩罚、权重与推断行为
- [稳健回归](../models/robust.md) — 稳健损失模型行为
- [惩罚 GLM 推断](penalized-glm-inference.md) 与 [推断模式](inference-modes.md) — 推断支持与统计解释
