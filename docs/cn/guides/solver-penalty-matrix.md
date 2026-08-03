# Solver × Penalty 兼容性矩阵

> 语言：中文  
> 最后更新：2026-08-03  
> 页面定位：参考指南  
> 切换：[English](../../en/guides/solver-penalty-matrix.md)

## 概述

`PenalizedGeneralizedLinearModel` 支持 **7 个损失族 × 9 种惩罚 × 9 个求解器** 的组合空间。本页记录哪些组合受支持、`solver='auto'` 如何分发、以及显式指定求解器时的行为。

**核心规则**：所有 loss × penalty 组合在 `solver='auto'` 下均可工作。限制仅在显式指定求解器时生效。

## 1. 自动分发表

当 `solver='auto'`（默认）时，模型为每个 loss × penalty 对选择最佳求解器：

| Loss | l2 / none | l1 | elasticnet | scad | mcp | adaptive_l1 | group_lasso | group_scad | group_mcp |
|------|:---------:|:--:|:----------:|:----:|:---:|:-----------:|:-----------:|:----------:|:---------:|
| **squared_error** | exact | fista | fista | irls_cd → fista_lla | irls_cd → fista_lla | fista | fista (CD) | fista_lla | fista_lla |
| **logistic** | irls | fista | fista | fista_lla | fista_lla | fista | fista | fista_lla | fista_lla |
| **poisson** | irls | fista | fista | fista_lla | fista_lla | fista | fista | fista_lla | fista_lla |
| **gamma** | newton | fista | fista | fista_lla | fista_lla | fista | fista | fista_lla | fista_lla |
| **inverse_gaussian** | newton | fista | fista | fista_lla | fista_lla | fista | fista | fista_lla | fista_lla |
| **negative_binomial** | irls | fista | fista | fista_lla | fista_lla | fista | fista | fista_lla | fista_lla |
| **tweedie** | irls | fista | fista | fista_lla | fista_lla | fista | fista | fista_lla | fista_lla |

**分发说明**：
- `fista_lla` 不是用户可填写的 `solver=` 关键字；它在非凸惩罚（SCAD、MCP、group_scad、group_mcp）时由内部调用。直接调用公开的 `fista_lla_path()` 时，也会自动建立与惩罚定义一致的凸 surrogate。
- `irls_cd` 优先用于 squared_error + 标量 SCAD/MCP。GLM + SCAD/MCP 根据损失结构使用 FISTA 或 proximal-Newton 内层步骤。
- Group SCAD/MCP 的内层 surrogate 是自适应 **Group Lasso**，不是逐坐标 adaptive L1。
- GPU 路径可能在合适时采用 `fista_bb`。

## 2. 显式求解器约束

| 求解器 | 接受 | 拒绝 | 说明 |
|--------|------|------|------|
| `exact` | 仅 l2，仅 squared_error | 其他所有 | 特征分解闭式解 |
| `irls` | 仅 l2（任意 loss） | 所有非光滑 | 迭代重加权最小二乘 |
| `newton` | l2 / none | l1, elasticnet, scad, mcp, adaptive_l1, group_* | 牛顿法 + 线搜索 |
| `lbfgs` | l2 / none | l1, elasticnet, scad, mcp, adaptive_l1, group_* | L-BFGS + 线搜索 |
| `fista` | 所有惩罚 | — | FISTA + Nesterov 动量 |
| `fista_bb` | 所有惩罚 | — | FISTA + Barzilai-Borwein 步长 |
| `admm` | 所有惩罚 | — | ADMM + proximal z 更新 |
| `irls_cd` | scad, mcp, adaptive_l1 | l1, elasticnet, group_* | IRLS 外层 + 坐标下降内层 |
| `proximal_irls_cd` | scad, mcp（仅 quantile） | group_* 及其他 loss | IRLS 上界 + LLA |
| `proximal_newton` | scad, mcp, adaptive_l1（有 Hessian 的 loss） | 其他所有 | Newton 方向 + Armijo + proximal |

不支持的组合会在开始数值拟合前明确抛出 `ValueError`。

## 3. 求解器能力

| 求解器 | sample_weight | warm_start | 推断 | 最佳用途 |
|--------|:------------:|:----------:|:---:|----------|
| `exact` | ✅ | ❌ | ✅ (OLS) | squared_error + l2 |
| `irls` | ✅ | ❌ | ❌ | GLM + l2 |
| `newton` | ❌ | ❌ | ❌ | GLM + l2 |
| `lbfgs` | ❌ | ❌ | ❌ | 大规模光滑问题 |
| `fista` | ✅ | ✅ | ❌ | 光滑 + 非光滑惩罚 |
| `fista_bb` | ✅ | ✅ | ❌ | 自适应步长稀疏问题 |
| `admm` | ✅ | ✅ | ❌ | 增广拉格朗日路径 |
| `irls_cd` | ✅ | ✅ | ❌ | squared_error + 标量 SCAD/MCP |

## 4. CV 支持 (`PenalizedGLM_CV`)

| Loss | l2 | l1 / elasticnet | scad / mcp | adaptive_l1 | group_* |
|------|:--:|:---------------:|:----------:|:-----------:|:-------:|
| **squared_error** | 特征批处理 | 稀疏 FISTA | LLA + FISTA/CD | 通用 fit | 通用 fit |
| **logistic** | 通用 fit | logistic 稀疏路径 | LLA + FISTA | 通用 fit | 通用 fit |
| **poisson** | 通用 fit | 折批处理 GPU | LLA + FISTA | 通用 fit | 通用 fit |
| **gamma** | 通用 fit | 折批处理 GPU | LLA + FISTA | 通用 fit | 通用 fit |
| **inverse_gaussian** | 通用 fit | 折批处理 GPU | LLA + FISTA | 通用 fit | 通用 fit |
| **negative_binomial** | 通用 fit | 折批处理 GPU | LLA + FISTA | 通用 fit | 通用 fit |
| **tweedie** | 通用 fit | 折批处理 GPU | LLA + FISTA | 通用 fit | 通用 fit |

**路径说明**：
- 标量 SCAD/MCP 的 LLA 产生 weighted L1 surrogate。
- Group SCAD/MCP 的 LLA 产生 weighted Group Lasso surrogate。
- CV fold score、selected alpha 与最终全数据 refit 使用同一 surrogate contract，并支持 `sample_weight`。

## 5. 惩罚参考

| 惩罚 | 公式 | Proximal | 参数 |
|------|------|----------|------|
| `l2` | ½α‖β‖² | β/(1+α·step) | `alpha` |
| `l1` | α‖β‖₁ | soft_threshold | `alpha` |
| `elasticnet` | α[λ‖β‖₁ + ½(1-λ)‖β‖²] | soft_threshold + L2 缩放 | `alpha`, `l1_ratio` |
| `scad` | SCAD(β; α, a) | SCAD 阈值 | `alpha`, `a` |
| `mcp` | MCP(β; α, γ) | MCP 阈值 | `alpha`, `gamma` |
| `adaptive_l1` | αΣ_j w_j|β_j| | 加权 soft_threshold | `alpha`, `_weights` |
| `group_lasso` | αΣ_g √p_g‖β_g‖₂ | 块 soft_threshold | `alpha`, `groups` |
| `group_scad` | Σ_g SCAD(‖β_g‖₂; α√p_g, a) | SCAD 块阈值 | `alpha`, `groups`, `a` |
| `group_mcp` | Σ_g MCP(‖β_g‖₂; α√p_g, γ) | MCP 块阈值 | `alpha`, `groups`, `gamma` |

**非凸惩罚说明**：
- 标量 SCAD/MCP 在每个 continuation step 线性化为 weighted L1。
- 对 Group SCAD/MCP，记惩罚关于 `‖β_g‖₂` 的导数为 `D_g`，正确的凸 surrogate 为 `Σ_g D_g‖β_g‖₂`。内部使用 `AdaptiveGroupLassoPenalty(alpha=1, weights_g=D_g/√p_g)` 精确表示，因此不会再次乘 target alpha，也不会额外乘 group size。
- 默认 continuation 当前对通常的光滑/Hessian 路径使用 5 步，对非光滑路径使用 3 步；CV 提供的 alpha path 决定其自身步数。
- Group SCAD 要求 `a > 2`，Group MCP 要求 `gamma > 1`；非法值明确失败。

## 6. 推断支持

| 惩罚 | 推断方法 | 状态 |
|------|---------|------|
| `l2` | 标准 OLS/GLS 推断 | ✅ 可用 |
| `l1` | Debiased Lasso | ✅ 可用 |
| `elasticnet` | Debiased Lasso 适配 | 待实现 |
| `scad` / `mcp` | Debiased 非凸 | 待实现 |
| `adaptive_l1` | Debiased adaptive Lasso | 待实现 |
| `group_*` | Group debiased | 待实现；不支持的请求会明确失败 |

## 7. 选择求解器

```
                    ┌─ squared_error + l2? ─── 是 ──→ exact
                    │
                    ├─ 仅光滑惩罚? ────────── 是 ──→ irls / newton / lbfgs
                    │
solver='auto' ──────├─ 非凸惩罚? ───────────── 是 ──→ fista_lla
                    │
                    ├─ l1 / elasticnet? ────── 是 ──→ fista / fista_bb
                    │
                    └─ group penalty? ───────── 是 ──→ group-aware proximal / block path
```

- 标量 squared_error + SCAD/MCP 可使用 `irls_cd`。
- Group SCAD/MCP 使用 group-aware LLA，不走逐坐标 `irls_cd`。
