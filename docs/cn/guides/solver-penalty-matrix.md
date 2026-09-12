# Solver × Penalty 兼容性矩阵

> 语言：中文  
> 最后更新：2026-09-12  
> 页面定位：参考指南  
> 切换：[English](../../en/guides/solver-penalty-matrix.md)

## 概览

本页用于回答一个最实际的问题：在 `PenalizedGeneralizedLinearModel` 或 `PenalizedGLM_CV` 中，某个 loss / penalty 应该由哪个 solver 处理？

首先要区分 **direct fit** 和 **CV**：

- 普通 direct `solver="auto"` 使用第 1 节的分发表；
- `PenalizedGLM_CV` 对 smooth-L2 有一套相关但有意不同的策略，见第 4 节；
- 如果用户显式指定 solver，statgpu 会在数值计算前验证该请求，不会因为加入 `sample_weight` 而静默换 solver。

`AdaptiveGroupLassoPenalty` 可以作为公开 penalty object 使用，但调用方必须显式提供 group weights，因此它有意不提供字符串 registry alias。

## 1. Direct-fit `solver="auto"`

| Loss | l2 / none | l1 | elasticnet | scad | mcp | adaptive_l1 | group_lasso | group_scad | group_mcp |
|------|:---------:|:--:|:----------:|:----:|:---:|:-----------:|:-----------:|:----------:|:---------:|
| **squared_error** | l2：CPU exact / GPU Newton；none：FISTA | FISTA | FISTA | IRLS-CD → FISTA-LLA | IRLS-CD → FISTA-LLA | FISTA | FISTA | Group FISTA-LLA | Group FISTA-LLA |
| **logistic** | Newton | FISTA | FISTA | FISTA-LLA | FISTA-LLA | FISTA | FISTA | Group FISTA-LLA | Group FISTA-LLA |
| **poisson** | Newton | FISTA | FISTA | FISTA-LLA | FISTA-LLA | FISTA | FISTA | Group FISTA-LLA | Group FISTA-LLA |
| **gamma** | Newton | FISTA | FISTA | FISTA-LLA | FISTA-LLA | FISTA | FISTA | Group FISTA-LLA | Group FISTA-LLA |
| **inverse_gaussian** | Newton | FISTA | FISTA | FISTA-LLA | FISTA-LLA | FISTA | FISTA | Group FISTA-LLA | Group FISTA-LLA |
| **negative_binomial** | Newton | FISTA | FISTA | FISTA-LLA | FISTA-LLA | FISTA | FISTA | Group FISTA-LLA | Group FISTA-LLA |
| **tweedie** | Newton | FISTA | FISTA | FISTA-LLA | FISTA-LLA | FISTA | FISTA | Group FISTA-LLA | Group FISTA-LLA |

### 如何阅读这张表

- `fista_lla` 是内部 continuation path，不是公开的 `solver=` keyword；直接调用 `fista_lla_path()` 时使用同一 surrogate。
- 标量 squared-error SCAD/MCP 可以使用 coordinate-descent continuation；Group SCAD/MCP 使用 weighted Group-Lasso surrogate + group-aware FISTA inner solve。
- Group Lasso 与 Adaptive Group Lasso 使用声明的 loss gradient 与欧氏 group proximal operator，包括维护中的 `sample_weight` 和 CV route。
- `sample_weight` 不会重写显式 solver request。受支持的 weighted Newton/L-BFGS route 在整个优化中使用同一个归一化 weighted objective；不支持的 loss/solver/weight 组合直接报错。

### inverse-power Gamma 的边界

对普通 `GammaRegression(link="inverse_power")`，真正非均匀的 weighted explicit Newton/L-BFGS 要求 `fit_intercept=True`，这样 statgpu 才能构造严格为正、满足该 family/link 定义域的初始 predictor。

只有 **非均匀权重 + no-intercept + 显式 Newton/L-BFGS** 这一组合会被拒绝。未传权重、uniform 权重或等效 uniform 权重继续保持历史 no-intercept 行为。

## 2. 显式求解器约束

| 求解器 | 接受 | 拒绝 / 限制 | 说明 |
|--------|------|-------------|------|
| `exact` | 仅 L2 + squared error | 其他所有 | 闭式/特征分解路径 |
| `irls` | 支持 IRLS 的 loss 上的 L2 | 非光滑 penalty | loss/family-specific IRLS |
| `newton` | 有 Hessian 的 smooth loss + L2/none | L1、ElasticNet、非凸及 group penalty | Newton + Armijo line search |
| `lbfgs` | smooth loss + L2/none | L1、ElasticNet、非凸及 group penalty | limited-memory BFGS + line search |
| `fista` | 支持 proximal 的 penalty | 不支持的模型组合 | Nesterov proximal gradient |
| `fista_bb` | 受支持的 sparse penalty | 不支持的组合 | FISTA + BB step adaptation |
| `admm` | 受支持的 proximal formulation | 不支持的组合 | variable splitting + proximal update |
| `irls_cd` | 标量 SCAD/MCP/adaptive-L1 route | L1/ElasticNet 与 group penalty | IRLS outer + coordinate descent inner |
| `proximal_irls_cd` | quantile + 标量 SCAD/MCP | non-quantile loss 与 group penalty | quantile majorization + LLA |
| `proximal_newton` | L2/none 使用 Newton；非光滑 direct call 显式改走 FISTA | 不支持的 penalty structure | 不静默使用 Euclidean-prox 近似 |

不支持的显式组合会在数值拟合前失败。

## 3. 求解器能力

| 求解器 | `sample_weight` | `warm_start` | 推断 | 最适合 |
|--------|:---------------:|:------------:|:----:|--------|
| `exact` | ✅（对应维护路径） | ❌ | ✅（OLS path） | squared error + L2 |
| `irls` | 依 estimator/loss | ❌ | 依 estimator | 维护中的 IRLS GLM |
| `newton` | 维护中的 GLM 支持 analytic weights | ❌ | 依 estimator | 有 Hessian 的 smooth objective |
| `lbfgs` | 维护中的 GLM 支持 analytic weights；其他 loss 依具体 route | ❌ | 依 estimator | 不希望形成完整 Hessian 的 smooth objective |
| `fista` | 维护中的 weighted route ✅ | ✅ | 依 estimator | convex sparse/group 与 LLA inner solve |
| `fista_bb` | 维护中的 weighted route ✅ | ✅ | 依 estimator | 受支持 sparse objective 的自适应 step |
| `admm` | 依组合 | ✅ | 依 estimator | 受支持 proximal formulation |
| `irls_cd` | 维护中的 route ✅ | ✅ | 依 estimator | 标量 non-convex continuation route |

对 Newton/L-BFGS，`sample_weight` support 是 **loss × estimator contract**，不能仅由 solver 函数签名推出。维护中的 GLM loss 对 data-fit term 使用

`sum(w_i * loss_i) / sum(w_i)`。

generic robust、quantile、Cox direct L-BFGS 则继续遵循各自的 weight 边界。

Group warm start 会把 coefficient 与 intercept state 一起带入一次 fit，并在成功或失败后清除。

## 4. CV 支持（`PenalizedGLM_CV`）

`PenalizedGLM_CV` 保留 public `solver="auto"` request，但 candidate fit 与 selected full-data final refit 的 smooth-L2 policy 按 family 区分。

| Loss | l2 | l1 / elasticnet | scad / mcp | adaptive_l1 | group_lasso / adaptive group | group_scad / group_mcp |
|------|:--:|:---------------:|:----------:|:-----------:|:----------------------------:|:-----------------------:|
| **squared_error** | eig-batch / exact-style path | sparse FISTA | LLA + FISTA/CD | general fit | Group FISTA | Group FISTA-LLA |
| **logistic** | Newton | sparse FISTA | LLA + FISTA | general fit | Group FISTA | Group FISTA-LLA |
| **poisson** | Newton | sparse/FISTA path | LLA + FISTA | general fit | Group FISTA | Group FISTA-LLA |
| **gamma** | L-BFGS | sparse/FISTA path | LLA + FISTA | general fit | Group FISTA | Group FISTA-LLA |
| **inverse_gaussian** | L-BFGS | sparse/FISTA path | LLA + FISTA | general fit | Group FISTA | Group FISTA-LLA |
| **negative_binomial** | L-BFGS | sparse/FISTA path | LLA + FISTA | general fit | Group FISTA | Group FISTA-LLA |
| **tweedie** | Newton | sparse/FISTA path | LLA + FISTA | general fit | Group FISTA | Group FISTA-LLA |

weights 不会静默替换 solver。上面三个 L-BFGS smooth-L2 family 的 weighted candidate/final-refit 支持遵循维护中的 GLM weighted-objective contract。

Group validation 在 alpha grid、fold construction 和 candidate fitting 前完成。Groups 按最终 design width 解释，包括 formula 展开的列。没有显式 adaptive weights 时，遗漏 feature 会补成 singleton group；越界 index 和不完整 adaptive weighted group 会在 candidate fitting 前失败。

CV 使用 fit-local penalty state，不修改调用方的 penalty object 或 `penalty_kwargs`。对于 penalty object，每个 candidate 都按当前 alpha 重建；selected final estimator 暴露的 penalty snapshot 与实际 resolved objective 的 alpha/groups 一致。顶层 CV estimator 保留原 constructor parameter。

## 5. Penalty 参考

| Penalty | 公式 | Proximal 形式 | 主要参数 |
|---------|------|---------------|----------|
| `l2` | ½α‖β‖² | ridge scaling | `alpha` |
| `l1` | α‖β‖₁ | soft threshold | `alpha` |
| `elasticnet` | α[λ‖β‖₁ + ½(1-λ)‖β‖²] | soft threshold + L2 scaling | `alpha`, `l1_ratio` |
| `scad` | SCAD(β; α, a) | SCAD thresholding / LLA route | `alpha`, `a` |
| `mcp` | MCP(β; α, γ) | MCP thresholding / LLA route | `alpha`, `gamma` |
| `adaptive_l1` | αΣ_j w_j|β_j| | weighted soft threshold | `alpha`, weights |
| `group_lasso` | αΣ_g √p_g‖β_g‖₂ | block soft threshold | `alpha`, `groups` |
| `AdaptiveGroupLassoPenalty` | αΣ_g w_g√p_g‖β_g‖₂ | weighted block soft threshold | `alpha`, `groups`, `weights`；仅 object |
| `group_scad` | Σ_g SCAD(‖β_g‖₂; α√p_g, a) | group LLA surrogate | `alpha`, `groups`, `a` |
| `group_mcp` | Σ_g MCP(‖β_g‖₂; α√p_g, γ) | group LLA surrogate | `alpha`, `groups`, `gamma` |

对 Group SCAD/MCP，记关于 `‖β_g‖₂` 的导数为 `D_g`。精确凸 surrogate 是 `Σ_g D_g‖β_g‖₂`，内部表示为 `AdaptiveGroupLassoPenalty(alpha=1, weights_g=D_g/√p_g)`；target alpha 与 group size 不会重复相乘。Group LLA 使用 FISTA inner solve，而不是 generic proximal-Newton branch。

Group 输入采用严格规则：hyperparameter 必须是有限 numeric scalar；group index/ID 必须是 signed `int64` 可表示的非负整数值；显式 group 不得为空或重复；flat ID 必须从 0 连续；公开 penalty 数值方法要求 coefficient dimension 与 grouped feature width 完全一致。

## 6. 推断支持

| Penalty | 推断方法 | 状态 |
|---------|----------|------|
| `l2` | estimator 暴露的 standard / M-estimation | ✅ 维护中的 route 可用 |
| `l1` | Debiased Lasso | ✅ 受支持 route |
| `elasticnet` | 依 method | 见 estimator contract |
| `scad` / `mcp` | 已实现的 oracle/bootstrap | 见 estimator contract |
| `adaptive_l1` | 依 method | 见 estimator contract |
| Group Lasso / Adaptive Group Lasso / Group SCAD / Group MCP | group-preserving covariance/bootstrap | 尚未实现；inference request 在拟合前失败 |

## 7. 选择求解器

对大多数用户，建议先使用 `solver="auto"`，只有在确实希望固定某个算法时才显式覆盖。

```
direct solver="auto"
├── squared_error + L2?                 → CPU exact / GPU Newton
├── squared_error + none?               → FISTA
├── smooth non-Gaussian GLM + L2/none?  → Newton
├── scalar non-convex penalty?          → scalar LLA path
├── convex group penalty?               → Group FISTA
└── group SCAD/MCP?                     → Group FISTA-LLA
```

`PenalizedGLM_CV` 请使用上面的独立 CV 表。Gamma、Inverse-Gaussian、Negative-Binomial 的 L2 row 在 CV/final refit 中有意使用 L-BFGS，虽然 direct-fit `auto` 使用 Newton。
