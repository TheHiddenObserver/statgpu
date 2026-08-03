# Solver × Penalty 兼容性矩阵

> 语言：中文  
> 最后更新：2026-08-03  
> 页面定位：参考指南  
> 切换：[English](../../en/guides/solver-penalty-matrix.md)

## 概述

`PenalizedGeneralizedLinearModel` 支持 **7 个损失族 × 9 种惩罚 × 9 个求解器**。本页说明 `solver='auto'` 的分发、显式求解器限制，以及 group penalty 的目标函数与验证契约。

## 1. 自动分发表

| Loss | l2 / none | l1 | elasticnet | scad | mcp | adaptive_l1 | group_lasso | group_scad | group_mcp |
|------|:---------:|:--:|:----------:|:----:|:---:|:-----------:|:-----------:|:----------:|:---------:|
| **squared_error** | exact | fista | fista | irls_cd → fista_lla | irls_cd → fista_lla | fista | fista | group fista_lla | group fista_lla |
| **logistic** | irls | fista | fista | fista_lla | fista_lla | fista | fista | group fista_lla | group fista_lla |
| **poisson** | irls | fista | fista | fista_lla | fista_lla | fista | fista | group fista_lla | group fista_lla |
| **gamma** | newton | fista | fista | fista_lla | fista_lla | fista | fista | group fista_lla | group fista_lla |
| **inverse_gaussian** | newton | fista | fista | fista_lla | fista_lla | fista | fista | group fista_lla | group fista_lla |
| **negative_binomial** | irls | fista | fista | fista_lla | fista_lla | fista | fista | group fista_lla | group fista_lla |
| **tweedie** | irls | fista | fista | fista_lla | fista_lla | fista | fista | group fista_lla | group fista_lla |

**分发说明**：
- `fista_lla` 是内部 continuation 路径；直接调用公开 `fista_lla_path()` 时也执行相同的 surrogate contract。
- 标量 squared-error SCAD/MCP 可使用坐标下降 continuation；Group SCAD/MCP 始终使用 weighted Group Lasso surrogate 与 group-aware FISTA 内层。
- 所有 Group Lasso 模型都使用实际 loss gradient 与精确的欧氏 Group Lasso proximal，包括 squared error、robust/GLM loss、`sample_weight`、CV fold 和最终 refit。
- 旧的 Gaussian block 更新不再进入公开路由。对一般相关 group Gram block，先解 Gram 系统再做欧氏 block threshold 并不是原 Group Lasso 子问题的精确解；它只有在 group block 正交归一时成立。

## 2. 显式求解器约束

| 求解器 | 接受 | 拒绝 | 说明 |
|--------|------|------|------|
| `exact` | 仅 l2 + squared_error | 其他所有 | 特征分解闭式解 |
| `irls` | 光滑 l2 路径 | 非光滑惩罚 | IRLS |
| `newton` | 光滑目标 | l1、非凸和 group_* | Newton + 线搜索 |
| `lbfgs` | 光滑目标 | l1、非凸和 group_* | L-BFGS |
| `fista` | 支持 proximal 的惩罚 | — | Nesterov FISTA |
| `fista_bb` | 支持的稀疏组合 | 不支持的组合明确失败 | BB 自适应步长 |
| `admm` | 支持的 proximal 组合 | 不支持的组合明确失败 | ADMM |
| `irls_cd` | 标量 scad/mcp/adaptive_l1 | group_* | IRLS + 坐标下降 |
| `proximal_newton` | 支持的标量非凸 Hessian 路径 | group_* | Newton + Armijo + proximal |

不支持的组合在数值拟合前抛出 `ValueError`。

## 3. CV 支持

| Loss | l2 | l1 / elasticnet | scad / mcp | adaptive_l1 | group_lasso | group_scad / group_mcp |
|------|:--:|:---------------:|:----------:|:-----------:|:-----------:|:-----------------------:|
| **squared_error** | eig-batch | 稀疏 FISTA | LLA + FISTA/CD | 通用 fit | Group FISTA | Group FISTA-LLA |
| **logistic** | 通用 fit | 稀疏 FISTA | LLA + FISTA | 通用 fit | Group FISTA | Group FISTA-LLA |
| **其他 GLM/robust** | 通用 fit | 稀疏/FISTA | LLA + FISTA | 通用 fit | Group FISTA | Group FISTA-LLA |

Group validation 在 alpha grid、fold construction 与 candidate fitting 前执行。Groups 按最终设计矩阵宽度解释，包括 formula 展开后的 dummy/transform 列。无显式 adaptive weights 时，遗漏特征只会一次性补为 singleton groups；越界索引和不完整 adaptive weighted groups 会事务性失败。CV score、selected alpha 和最终 refit 共用同一 groups、loss、sample weights 与 solver contract。

## 4. 惩罚定义

| 惩罚 | 公式 | Proximal |
|------|------|----------|
| `l2` | ½α‖β‖² | ridge scale |
| `l1` | α‖β‖₁ | soft threshold |
| `elasticnet` | α[λ‖β‖₁ + ½(1-λ)‖β‖²] | soft threshold + L2 scale |
| `group_lasso` | αΣ_g √p_g‖β_g‖₂ | block soft threshold |
| `group_scad` | Σ_g SCAD(‖β_g‖₂; α√p_g, a) | SCAD block threshold |
| `group_mcp` | Σ_g MCP(‖β_g‖₂; α√p_g, γ) | MCP block threshold |

对 Group SCAD/MCP，记关于 `‖β_g‖₂` 的导数为 `D_g`。精确凸 surrogate 是 `Σ_g D_g‖β_g‖₂`，内部表示为 `AdaptiveGroupLassoPenalty(alpha=1, weights_g=D_g/√p_g)`，不会再次乘 target alpha 或 group size。Group LLA 固定采用 FISTA 内层，因为通用 proximal-Newton 路径可能拒绝全部 Armijo steps 而不暴露失败状态。

Group 输入采用严格契约：索引/ID 必须是非负整数值 numeric；显式 groups 不得为空或重复；flat IDs 必须从 0 连续；公开数值方法要求 coefficient vector 与 group feature width 完全一致。只有内部 fused group-LLA surrogate 通过私有 capability 允许一个未惩罚的 trailing intercept。

## 5. 推断支持

| 惩罚 | 状态 |
|------|------|
| `l2` | 标准路径可用 |
| `l1` | 支持的 debiased 路径可用 |
| `scad` / `mcp` | 依 estimator/method 契约 |
| `group_*` | Group-debiased 尚未实现；不支持请求在拟合前明确失败 |

## 6. 选择求解器

```
                    ┌─ squared_error + l2? ─── 是 ──→ exact
                    │
                    ├─ 光滑惩罚? ───────────── 是 ──→ irls / newton / lbfgs
                    │
solver='auto' ──────├─ 标量非凸? ───────────── 是 ──→ scalar LLA
                    │
                    ├─ group_lasso? ────────── 是 ──→ exact Group FISTA
                    │
                    └─ group SCAD/MCP? ─────── 是 ──→ Group FISTA-LLA
```
