# Solver × Penalty 兼容性矩阵

> 语言：中文  
> 最后更新：2026-09-04
> 页面定位：参考指南  
> 切换：[English](../../en/guides/solver-penalty-matrix.md)

## 概述

共享惩罚引擎支持多种 loss、penalty 与求解路径，但不存在适用于所有估计器的通用取值列表。Wrapper 默认值与模型专属路由可能进一步收窄共享引擎。

::: warning 以模型页为准
应在对应模型页选择 `solver=`。本页解释共享兼容性，不会把低层函数变成估计器 keyword。请从[模型求解器速查](../models/#求解器速查)开始。
:::

## 1. 自动分发规则

`auto` 会同时考虑 loss、penalty、解析后的后端、CV 模式，有时还考虑问题规模。当前主要规则为：

| 模型目标 | 自动分发或细化路径 |
|---|---|
| squared error + L2 | CPU 用 exact；CuPy/Torch 用 Newton |
| squared error + L1 / Elastic Net | FISTA |
| 普通 `GeneralizedLinearModel` | IRLS |
| 光滑惩罚 GLM / robust / Cox | Newton；部分 CV 情况使用 L-BFGS |
| 稀疏 GLM | 根据 family、后端、CV 模式与规模选择 FISTA 或 FISTA-BB |
| Adaptive Lasso | 加权 L1 FISTA |
| 标量 SCAD / MCP | FISTA 分发再细化为模型专属 FISTA-LLA；quantile 使用 proximal IRLS + LLA |
| group penalties | 模型专属 group proximal/LLA 路径 |
| quantile + L2/none | FISTA 分发后在内部细化为 quantile IRLS |

Wrapper 默认值可能比共享 `auto` 更具体：`Ridge` 默认 `exact`，Lasso 与 Elastic Net 默认 `fista`，`LogisticRegression` 没有选择器并固定使用 IRLS，有序模型则固定使用 trust-region Newton。不要从本表反推 keyword，应沿模型目录链接进入具体页面。

## 2. 显式求解器约束

| 求解器 | 接受 | 拒绝 | 说明 |
|--------|------|------|------|
| `exact` | 仅 l2 + squared_error | 其他所有 | 特征分解闭式解 |
| `irls` | l2/none 且 loss 声明 IRLS contract | 非光滑惩罚；共享惩罚接口中的 squared error 与 Huber | 模型专属 IRLS |
| `newton` | l2 / none 且 loss 提供 Hessian | 非光滑惩罚与 quantile loss | Newton + 线搜索 |
| `lbfgs` | l2 / none 且 loss 光滑 | 非光滑惩罚与 quantile loss | L-BFGS |
| `fista` | 支持 proximal 的惩罚 | — | Nesterov FISTA |
| `fista_bb` | 支持的稀疏组合 | 不支持的组合明确失败 | BB 自适应步长 |
| `admm` | 受支持的 loss/penalty 组合 | 非均匀样本权重 | ADMM |
| `coordinate_descent` | CPU squared-error L1/Elastic Net 兼容路径 | GPU 与非 squared loss | 估计器兼容路径，不是 quantile CD |

`irls_cd`、`proximal_irls_quantile_solver`、`fista_lla_path`、`proximal_newton_solver` 与 `lbfgs_b_solver` 属于内部或低层直接 API，不是通用估计器 `solver=` 值。部分模型专属 penalty 会把通用分发标签继续细化为固定内部路径；具体边界以模型页为准。

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
