# Solver × Penalty 兼容性矩阵

> 语言：中文  
> 最后更新：2026-08-03  
> 页面定位：参考指南  
> 切换：[English](../../en/guides/solver-penalty-matrix.md)

## 概述

`PenalizedGeneralizedLinearModel` 支持 **7 个损失族 × 9 个注册惩罚名称 × 9 个求解器**。此外，公开的 `AdaptiveGroupLassoPenalty` 可作为 penalty object 使用；由于调用方必须显式提供 group weights，它有意不提供字符串 registry alias。

支持的 loss × penalty 组合在 `solver='auto'` 下自动分发；显式求解器请求会在数值计算前验证。

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
- `AdaptiveGroupLassoPenalty` 沿 `group_lasso` 列分发，但使用调用方给定的 per-group weights。
- `fista_lla` 是内部 continuation 路径；直接调用公开 `fista_lla_path()` 时也执行相同 surrogate contract。
- 标量 squared-error SCAD/MCP 可使用坐标下降 continuation；Group SCAD/MCP 始终使用 weighted Group Lasso surrogate 与 group-aware FISTA 内层。
- Group Lasso 与 Adaptive Group Lasso 都使用实际 loss gradient 和精确欧氏 group proximal，包括 robust/GLM loss、`sample_weight`、CV fold 与最终 selected-alpha refit。
- 旧 Gaussian block 更新不再进入公开路由；其 inverse-Gram 后欧氏阈值只对正交归一 group block 精确。

## 2. 显式求解器约束

| 求解器 | 接受 | 拒绝 | 说明 |
|--------|------|------|------|
| `exact` | 仅 l2 + squared_error | 其他所有 | 特征分解闭式解 |
| `irls` | 光滑 l2 路径 | 非光滑惩罚 | IRLS |
| `newton` | 光滑目标 | l1、非凸及全部 group penalty | Newton + 线搜索 |
| `lbfgs` | 光滑目标 | l1、非凸及全部 group penalty | L-BFGS |
| `fista` | 支持 proximal 的惩罚 | — | Nesterov FISTA |
| `fista_bb` | 支持的稀疏组合 | 不支持的组合明确失败 | BB 自适应步长 |
| `admm` | 支持的 proximal 组合 | 不支持的组合明确失败 | ADMM |
| `irls_cd` | 标量 scad/mcp/adaptive_l1 | 全部 group penalty | IRLS + 坐标下降 |
| `proximal_newton` | 支持的标量非凸 Hessian 路径 | 全部 group penalty | Newton + Armijo + proximal |

不支持的组合在数值拟合前抛出 `ValueError`。

## 3. CV 支持

| Loss | l2 | l1 / elasticnet | scad / mcp | adaptive_l1 | group_lasso / adaptive group | group_scad / group_mcp |
|------|:--:|:---------------:|:----------:|:-----------:|:----------------------------:|:-----------------------:|
| **squared_error** | eig-batch | 稀疏 FISTA | LLA + FISTA/CD | 通用 fit | Group FISTA | Group FISTA-LLA |
| **logistic** | 通用 fit | 稀疏 FISTA | LLA + FISTA | 通用 fit | Group FISTA | Group FISTA-LLA |
| **其他 GLM/robust** | 通用 fit | 稀疏/FISTA | LLA + FISTA | 通用 fit | Group FISTA | Group FISTA-LLA |

Group validation 在 alpha grid、fold construction 与 candidate fitting 前执行。Groups 按最终设计矩阵宽度解释，包括 formula 展开列。无显式 adaptive weights 时，遗漏特征补为 singleton groups；越界索引和不完整 adaptive weighted groups 会事务性失败。

CV 使用 fit-local penalty state，不修改调用方的 penalty object 或 `penalty_kwargs` 字典。Penalty object 会在每个 candidate alpha 下重建；最终 estimator 公开一个无私有 marker 的 penalty 快照，其 alpha 与 groups 和实际 resolved objective 一致。顶层 CV estimator 保留原 constructor parameter。

Coefficient 与 intercept warm start 作为同一个一次性状态进入拟合，并在成功或失败后共同清除。

## 4. 惩罚定义

| 惩罚 | 公式 | Proximal | 参数 |
|------|------|----------|------|
| `l2` | ½α‖β‖² | ridge scale | `alpha` |
| `l1` | α‖β‖₁ | soft threshold | `alpha` |
| `elasticnet` | α[λ‖β‖₁ + ½(1-λ)‖β‖²] | soft threshold + L2 scale | `alpha`, `l1_ratio` |
| `group_lasso` | αΣ_g √p_g‖β_g‖₂ | block soft threshold | `alpha`, `groups` |
| `AdaptiveGroupLassoPenalty` | αΣ_g w_g√p_g‖β_g‖₂ | weighted block soft threshold | `alpha`, `groups`, `weights`；仅 object |
| `group_scad` | Σ_g SCAD(‖β_g‖₂; α√p_g, a) | SCAD block threshold | `alpha`, `groups`, `a` |
| `group_mcp` | Σ_g MCP(‖β_g‖₂; α√p_g, γ) | MCP block threshold | `alpha`, `groups`, `gamma` |

对 Group SCAD/MCP，记关于 `‖β_g‖₂` 的导数为 `D_g`。精确凸 surrogate 是 `Σ_g D_g‖β_g‖₂`，内部表示为 `AdaptiveGroupLassoPenalty(alpha=1, weights_g=D_g/√p_g)`，不会再次乘 target alpha 或 group size。Group LLA 固定采用 FISTA 内层，因为通用 proximal-Newton 路径可能拒绝全部 Armijo steps 而不暴露失败状态。

Group 输入采用严格契约：alpha 与其他超参数必须是有限 numeric scalar，不能是 boolean 或可强制转换的字符串；索引/ID 必须是可由 signed `int64` 表示的非负整数值 numeric；显式 groups 不得为空或重复；flat IDs 必须从 0 连续；公开数值方法要求 coefficient vector 与 group feature width 完全一致。只有内部 fused group-LLA surrogate 通过私有 capability 允许一个未惩罚 trailing intercept。

## 5. 推断支持

| 惩罚 | 状态 |
|------|------|
| `l2` | 标准路径可用 |
| `l1` | 支持的 debiased 路径可用 |
| `scad` / `mcp` | 依 estimator/method 契约 |
| Group Lasso / Adaptive Group Lasso / Group SCAD / Group MCP | Group-preserving covariance/bootstrap 尚未实现；所有 inference 请求在拟合前明确失败 |

## 6. 选择求解器

```
                    ┌─ squared_error + l2? ─── 是 ──→ exact
                    │
                    ├─ 光滑惩罚? ───────────── 是 ──→ irls / newton / lbfgs
                    │
solver='auto' ──────├─ 标量非凸? ───────────── 是 ──→ scalar LLA
                    │
                    ├─ 凸 group penalty? ───── 是 ──→ exact Group FISTA
                    │
                    └─ group SCAD/MCP? ─────── 是 ──→ Group FISTA-LLA
```
