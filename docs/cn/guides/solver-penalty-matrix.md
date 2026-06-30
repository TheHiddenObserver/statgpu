# Solver × Penalty 兼容性矩阵

> 语言：中文  
> 最后更新：2026-06-12  
> 页面定位：参考指南  
> 切换：[English](../en/guides/solver-penalty-matrix.md)

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
- `fista_lla` 不是用户可指定的求解器关键字。它在非凸惩罚（SCAD、MCP、group_scad、group_mcp）时被内部调用。外层 `solver` 关键字仅控制内层循环（fista、fista_bb 或 irls_cd）。
- `irls_cd` 优先用于 squared_error + SCAD/MCP（Gauss-Seidel CD 对 OLS 更快）。GLM + SCAD/MCP 使用 `fista_lla` 配合 FISTA 内层循环。
- GPU 路径可能用 `fista_bb` 替代 `fista`，当 Barzilai-Borwein 步长更有利时。

## 2. 显式求解器约束

当显式设置 `solver=` 时，以下约束生效：

| 求解器 | 接受 | 拒绝 | 说明 |
|--------|------|------|------|
| `exact` | 仅 l2，仅 squared_error | 其他所有 | 特征分解闭式解 |
| `irls` | 仅 l2（任意 loss） | 所有非光滑 | 迭代重加权最小二乘 |
| `newton` | l2 / none（任意 loss） | l1, elasticnet, scad, mcp, adaptive_l1, group_* | 牛顿法 + 线搜索 |
| `lbfgs` | l2 / none（任意 loss） | l1, elasticnet, scad, mcp, adaptive_l1, group_* | L-BFGS + 线搜索 |
| `fista` | 所有惩罚（任意 loss） | — | FISTA + Nesterov 动量 |
| `fista_bb` | 所有惩罚（任意 loss） | — | FISTA + Barzilai-Borwein 步长 |
| `admm` | 所有惩罚（任意 loss） | — | ADMM + proximal z 更新 |
| `irls_cd` | scad, mcp, adaptive_l1 | l1, elasticnet, group_* | IRLS 外层 + 坐标下降内层 |
| `proximal_irls_cd` | scad, mcp（仅 quantile） | l1, elasticnet, group_* 及其他 loss | IRLS 上界 + LLA + 并行对角化 |
| `proximal_newton` | scad, mcp, adaptive_l1（有 Hessian 的 loss） | 其他所有 | Newton 方向 + Armijo + proximal 算子 |

**尝试不支持的组合会抛出 `ValueError`**，提示哪些 solver–penalty 对有效。

## 3. 求解器能力

| 求解器 | sample_weight | warm_start | 推断 | 最佳用途 |
|--------|:------------:|:----------:|:---:|----------|
| `exact` | ✅ | ❌ | ✅ (OLS) | squared_error + l2（小 p） |
| `irls** | ✅ | ❌ | ❌ | GLM + l2（标准 link） |
| `newton** | ❌ | ❌ | ❌ | GLM + l2（非标准 link） |
| `lbfgs** | ❌ | ❌ | ❌ | GLM + l2（大 p） |
| `fista** | ✅ | ✅ | ❌ | 光滑 + 非光滑惩罚 |
| `fista_bb** | ✅ | ✅ | ❌ | GLM + 非光滑（自适应步长） |
| `admm** | ✅ | ✅ | ❌ | 任意惩罚（增广拉格朗日） |
| `irls_cd** | ✅ | ✅ | ❌ | squared_error + SCAD/MCP（快速 CD） |
| `proximal_irls_cd` | ✅ | ✅ | ❌ | quantile + SCAD/MCP（IRLS 上界） |
| `proximal_newton` | ✅ | ✅ | ❌ | Huber/Bisquare/Cox + SCAD/MCP（5-10 iters） |

## 4. CV 支持 (`PenalizedGLM_CV`)

CV 估计器在可用时使用专用快速路径，其余回退到逐折 `fit()`：

| Loss | l2 | l1 / elasticnet | scad / mcp | adaptive_l1 | group_* |
|------|:--:|:---------------:|:----------:|:-----------:|:-------:|
| **squared_error** | 特征批处理 O(p³) | 稀疏 FISTA 路径 | LLA + FISTA/CD | 通用 fit | 通用 fit |
| **logistic** | 通用 fit | logistic 稀疏路径 | LLA + FISTA | 通用 fit | 通用 fit |
| **poisson** | 通用 fit | 折批处理 GPU | LLA + FISTA | 通用 fit | 通用 fit |
| **gamma** | 通用 fit | 折批处理 GPU | LLA + FISTA | 通用 fit | 通用 fit |
| **inverse_gaussian** | 通用 fit | 折批处理 GPU | LLA + FISTA | 通用 fit | 通用 fit |
| **negative_binomial** | 通用 fit | 折批处理 GPU | LLA + FISTA | 通用 fit | 通用 fit |
| **tweedie** | 通用 fit | 折批处理 GPU | LLA + FISTA | 通用 fit | 通用 fit |

**快速路径说明**：
- **特征批处理**：预计算 X'X 特征分解一次，批量求解所有 alpha/fold。O(p³) 初始化 + O(p·n_alphas·n_folds) 求解。
- **稀疏 FISTA 路径**：squared_error + l1/elasticnet 的专用 FISTA 循环。
- **logistic 稀疏路径**：logistic + l1/elasticnet 的专用 FISTA 循环。
- **折批处理 GPU**：所有 fold × alpha 在一次 GPU kernel launch 中求解。用于 GLM + l1/elasticnet GPU 路径。
- **LLA + FISTA**：非凸惩罚的局部线性近似（LLA）延续路径。从 λ_max 追踪到目标 α。
- **通用 fit**：回退到逐折 `PenalizedGeneralizedLinearModel.fit()`。所有组合可用但较慢。

## 5. 惩罚参考

| 惩罚 | 公式 | Proximal | 参数 |
|------|------|----------|------|
| `l2` | ½α‖β‖² | β/(1+α·step) | `alpha` |
| `l1` | α‖β‖₁ | soft_threshold(β, α·step) | `alpha` |
| `elasticnet` | α[λ‖β‖₁ + ½(1-λ)‖β‖²] | soft_threshold / (1+α(1-λ)step) | `alpha`, `l1_ratio` |
| `scad` | SCAD(β; α, a) | SCAD 阈值 | `alpha`, `a`（默认 3.7） |
| `mcp` | MCP(β; α, γ) | MCP 阈值 | `alpha`, `gamma`（默认 3.0） |
| `adaptive_l1` | α·w·‖β‖₁ | 加权 soft_threshold | `alpha`, `_weights` |
| `group_lasso` | αΣ_g‖β_g‖₂ | 块 soft_threshold | `alpha`, `groups` |
| `group_scad` | SCAD 组 | SCAD 块阈值 | `alpha`, `groups`, `a` |
| `group_mcp` | MCP 组 | MCP 块阈值 | `alpha`, `groups`, `gamma` |

**非凸惩罚说明**：
- SCAD 和 MCP 通过 **LLA（局部线性近似）** 求解：每个延续步将非凸惩罚在当前估计处线性化，产生加权 L1 问题，FISTA/CD 可解。
- 延续路径从 `λ_max`（所有系数为零）追踪到目标 `α`，使用 20-100 步。避免陷入不良局部最小值。
- SCAD 的 `a=2.0` 和 MCP 的 `gamma=1.0` 数值奇异。代码将这些值 clamp 到安全范围（`a ≥ 2+1e-6`，`gamma ≥ 1+1e-6`）。

## 6. 推断支持

| 惩罚 | 推断方法 | 状态 |
|------|---------|------|
| `l2` | 标准 OLS/GLS 推断 | ✅ 可用 |
| `l1` | Debiased Lasso（nodewise 回归） | ✅ 可用（`compute_inference=True`） |
| `elasticnet` | Debiased Lasso（适配版） | 待实现 |
| `scad` / `mcp` | Debiased 非凸 | 待实现 |
| `adaptive_l1` | Debiased adaptive Lasso | 待实现 |
| `group_*` | Group debiased | 待实现 |

## 7. 选择求解器

```
                    ┌─ squared_error + l2? ─── 是 ──→ exact（闭式解）
                    │
                    ├─ 仅光滑惩罚? ────────── 是 ──→ irls / newton / lbfgs
                    │
solver='auto' ──────├─ 非凸 (SCAD/MCP)? ───── 是 ──→ fista_lla（自动）
                    │
                    ├─ l1 / elasticnet? ────── 是 ──→ fista / fista_bb
                    │
                    └─ 组惩罚? ─────────────── 是 ──→ fista + 块 CD
```

**手动选择求解器指南**：
- 使用 `solver='fista_bb'` 处理 GLM + 非光滑惩罚，当你需要自适应步长时（通常比固定步长 FISTA 更快）。
- 使用 `solver='admm'` 当你需要特定的增广拉格朗日公式，或当 proximal 算子计算廉价时。
- 使用 `solver='irls_cd'` 处理 squared_error + SCAD/MCP，当你需要 Gauss-Seidel CD 时（对小 p 收敛快于 Jacobi 块 CD）。
