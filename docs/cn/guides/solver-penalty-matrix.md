# Solver × Penalty 兼容性矩阵

> 语言：中文  
> 最后更新：2026-09-12  
> 页面定位：参考指南  
> 切换：[English](../../en/guides/solver-penalty-matrix.md)

## 概览

本页回答一个实际问题：在 `PenalizedGeneralizedLinearModel` 或 `PenalizedGLM_CV` 中，某个损失函数与惩罚项组合应该由哪个求解器处理？

首先要区分**直接拟合**和**交叉验证（CV）**：

- 普通直接拟合的 `solver="auto"` 使用第 1 节的分发表；
- `PenalizedGLM_CV` 对光滑 L2 模型有一套相关但有意不同的规则，见第 4 节；
- 若显式指定 `solver`，`sample_weight` 不会改变这一请求；不支持的组合会在数值计算前报错。

`AdaptiveGroupLassoPenalty` 可以作为公开惩罚对象使用，但调用方必须显式提供组权重，因此它不提供字符串别名。

## 1. 直接拟合的 `solver="auto"`

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

- `fista_lla` 是内部延续路径，不是公开的 `solver=` 参数值；直接调用 `fista_lla_path()` 时使用同一近似问题。
- 标量 squared-error SCAD/MCP 可以使用坐标下降延续路径；Group SCAD/MCP 使用加权 Group Lasso 近似问题与 group-aware FISTA 内层。
- Group Lasso 与 Adaptive Group Lasso 使用相应损失函数的梯度和欧氏 group proximal operator，包括受支持的 `sample_weight` 与交叉验证路径。
- `sample_weight` 不会改变显式指定的 `solver`。受支持的加权 Newton/L-BFGS 在整个优化中使用同一个归一化加权目标；不支持的损失函数、求解器和权重组合会直接报错。

### inverse-power Gamma 的边界

对普通 `GammaRegression(link="inverse_power")`，真正的非均匀权重配合显式 Newton/L-BFGS 时要求 `fit_intercept=True`，这样才能构造严格为正、满足该分布族和链接函数定义域的初始线性预测子。

只有 **非均匀权重 + 无截距 + 显式 Newton/L-BFGS** 这一组合会被拒绝。未传权重、均匀权重或等效均匀权重继续保持历史无截距行为。

## 2. 显式求解器约束

| 求解器 | 接受 | 拒绝 / 限制 | 说明 |
|--------|------|-------------|------|
| `exact` | 仅 L2 + squared error | 其他所有 | 闭式/特征分解路径 |
| `irls` | 支持 IRLS 的损失函数上的 L2 | 非光滑惩罚 | 分布族/损失函数专用 IRLS |
| `newton` | 有 Hessian 的光滑损失 + L2/none | L1、ElasticNet、非凸及 group penalty | Newton + Armijo 线搜索 |
| `lbfgs` | 光滑损失 + L2/none | L1、ElasticNet、非凸及 group penalty | limited-memory BFGS + 线搜索 |
| `fista` | 支持近端算子的惩罚 | 不支持的模型组合 | Nesterov 近端梯度 |
| `fista_bb` | 受支持的稀疏惩罚 | 不支持的组合 | FISTA + BB 自适应步长 |
| `admm` | 受支持的近端形式 | 不支持的组合 | 变量分裂 + 近端更新 |
| `irls_cd` | 标量 SCAD/MCP/adaptive-L1 | L1/ElasticNet 与 group penalty | IRLS 外层 + 坐标下降内层 |
| `proximal_irls_cd` | quantile + 标量 SCAD/MCP | 非 quantile loss 与 group penalty | 分位数上界近似 + LLA |
| `proximal_newton` | L2/none 使用 Newton；非光滑直接调用改用 FISTA | 不支持的惩罚结构 | 当前不采用 Euclidean-prox 近似 |

不支持的显式组合会在数值拟合前报错。

## 3. 求解器能力

| 求解器 | `sample_weight` | `warm_start` | 推断 | 最适合 |
|--------|:---------------:|:------------:|:----:|--------|
| `exact` | ✅（对应支持路径） | ❌ | ✅（OLS 路径） | squared error + L2 |
| `irls` | 依模型/损失函数而定 | ❌ | 依模型而定 | 支持 IRLS 的 GLM |
| `newton` | 当前 GLM 支持解析权重 | ❌ | 依模型而定 | 有 Hessian 的光滑目标 |
| `lbfgs` | 当前 GLM 支持解析权重；其他损失函数依具体路径 | ❌ | 依模型而定 | 不希望形成完整 Hessian 的光滑目标 |
| `fista` | 受支持的加权路径 ✅ | ✅ | 依模型而定 | 凸稀疏/group 目标与 LLA 内层 |
| `fista_bb` | 受支持的加权路径 ✅ | ✅ | 依模型而定 | 稀疏目标的自适应步长 |
| `admm` | 依组合而定 | ✅ | 依模型而定 | 受支持的近端形式 |
| `irls_cd` | 受支持的路径 ✅ | ✅ | 依模型而定 | 标量非凸延续路径 |

对 Newton/L-BFGS，`sample_weight` 的支持范围由损失函数和模型共同决定，不能仅根据底层求解器函数签名判断。当前 GLM 的数据拟合项使用

`sum(w_i * loss_i) / sum(w_i)`。

通用稳健、分位数和 Cox 的直接 L-BFGS 则遵循各自的权重限制。

Group warm start 会把系数和截距状态一起带入一次拟合，并在成功或失败后清除。

## 4. CV 支持（`PenalizedGLM_CV`）

`PenalizedGLM_CV` 保留公开的 `solver="auto"` 请求，但光滑 L2 候选模型与最终全数据重拟合会按分布族选择求解器。

| Loss | l2 | l1 / elasticnet | scad / mcp | adaptive_l1 | group_lasso / adaptive group | group_scad / group_mcp |
|------|:--:|:---------------:|:----------:|:-----------:|:----------------------------:|:-----------------------:|
| **squared_error** | eig-batch / exact-style | sparse FISTA | LLA + FISTA/CD | general fit | Group FISTA | Group FISTA-LLA |
| **logistic** | Newton | sparse FISTA | LLA + FISTA | general fit | Group FISTA | Group FISTA-LLA |
| **poisson** | Newton | sparse/FISTA | LLA + FISTA | general fit | Group FISTA | Group FISTA-LLA |
| **gamma** | L-BFGS | sparse/FISTA | LLA + FISTA | general fit | Group FISTA | Group FISTA-LLA |
| **inverse_gaussian** | L-BFGS | sparse/FISTA | LLA + FISTA | general fit | Group FISTA | Group FISTA-LLA |
| **negative_binomial** | L-BFGS | sparse/FISTA | LLA + FISTA | general fit | Group FISTA | Group FISTA-LLA |
| **tweedie** | Newton | sparse/FISTA | LLA + FISTA | general fit | Group FISTA | Group FISTA-LLA |

权重不会改变上表选择的求解器。Gamma、Inverse-Gaussian、Negative-Binomial 这三个 L-BFGS 光滑 L2 分布族的加权候选拟合和最终重拟合使用 GLM 的统一加权目标函数。

Group validation 会在 alpha 网格、fold 构造和候选拟合前完成。Groups 按最终设计矩阵宽度解释，包括 formula 展开的列。没有显式 adaptive weights 时，遗漏特征会补成单独一组；越界索引和不完整的 adaptive weighted group 会在候选拟合前报错。

CV 使用本次拟合局部的 penalty 状态，不会修改调用方传入的 penalty object 或 `penalty_kwargs`。对于 penalty object，每个候选模型都会按当前 alpha 重建；最终选中模型中暴露的 penalty snapshot 与实际目标函数的 alpha/groups 一致。顶层 CV 对象保留原始构造参数。

## 5. Penalty 参考

| Penalty | 公式 | Proximal 形式 | 主要参数 |
|---------|------|---------------|----------|
| `l2` | ½α‖β‖² | ridge scaling | `alpha` |
| `l1` | α‖β‖₁ | soft threshold | `alpha` |
| `elasticnet` | α[λ‖β‖₁ + ½(1-λ)‖β‖²] | soft threshold + L2 scaling | `alpha`, `l1_ratio` |
| `scad` | SCAD(β; α, a) | SCAD thresholding / LLA | `alpha`, `a` |
| `mcp` | MCP(β; α, γ) | MCP thresholding / LLA | `alpha`, `gamma` |
| `adaptive_l1` | αΣ_j w_j|β_j| | weighted soft threshold | `alpha`, weights |
| `group_lasso` | αΣ_g √p_g‖β_g‖₂ | block soft threshold | `alpha`, `groups` |
| `AdaptiveGroupLassoPenalty` | αΣ_g w_g√p_g‖β_g‖₂ | weighted block soft threshold | `alpha`, `groups`, `weights`；仅对象形式 |
| `group_scad` | Σ_g SCAD(‖β_g‖₂; α√p_g, a) | group LLA 近似 | `alpha`, `groups`, `a` |
| `group_mcp` | Σ_g MCP(‖β_g‖₂; α√p_g, γ) | group LLA 近似 | `alpha`, `groups`, `gamma` |

对 Group SCAD/MCP，记关于 `‖β_g‖₂` 的导数为 `D_g`。精确凸近似为 `Σ_g D_g‖β_g‖₂`，内部表示成 `AdaptiveGroupLassoPenalty(alpha=1, weights_g=D_g/√p_g)`；目标 `alpha` 与组大小不会重复相乘。Group LLA 使用 FISTA 内层，而不是通用 Proximal Newton 分支。

Group 输入采用严格规则：超参数必须是有限数值标量；group 索引/ID 必须是 signed `int64` 可表示的非负整数值；显式 group 不得为空或重复；平坦 group ID 必须从 0 连续；公开 penalty 数值方法要求系数维度与分组后的特征宽度完全一致。

## 6. 推断支持

| Penalty | 推断方法 | 状态 |
|---------|----------|------|
| `l2` | 模型暴露的标准推断 / M-estimation | ✅ 受支持路径可用 |
| `l1` | Debiased Lasso | ✅ 受支持路径可用 |
| `elasticnet` | 依具体方法 | 见模型说明 |
| `scad` / `mcp` | 已实现的 oracle/bootstrap | 见模型说明 |
| `adaptive_l1` | 依具体方法 | 见模型说明 |
| Group Lasso / Adaptive Group Lasso / Group SCAD / Group MCP | 保持 group 结构的 covariance/bootstrap | 尚未实现；推断请求会在拟合前报错 |

## 7. 选择求解器

对大多数用户，建议先使用 `solver="auto"`，只有确实希望固定算法时才显式覆盖。

```text
direct solver="auto"
├── squared_error + L2?                 → CPU exact / GPU Newton
├── squared_error + none?               → FISTA
├── smooth non-Gaussian GLM + L2/none?  → Newton
├── scalar non-convex penalty?          → scalar LLA
├── convex group penalty?               → Group FISTA
└── group SCAD/MCP?                     → Group FISTA-LLA
```

`PenalizedGLM_CV` 请使用上面的独立 CV 表。Gamma、Inverse-Gaussian、Negative-Binomial 的 L2 在 CV 和最终重拟合中有意使用 L-BFGS，而直接拟合的 `auto` 使用 Newton。
