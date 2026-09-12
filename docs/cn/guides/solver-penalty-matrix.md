# 求解器 × 惩罚项兼容性矩阵

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

| 损失 | l2 / none | l1 | elasticnet | scad | mcp | adaptive_l1 | group_lasso | group_scad | group_mcp |
|------|:---------:|:--:|:----------:|:----:|:---:|:-----------:|:-----------:|:----------:|:---------:|
| **squared_error** | l2：CPU `exact` / GPU Newton；none：FISTA | FISTA | FISTA | IRLS-CD → FISTA-LLA | IRLS-CD → FISTA-LLA | FISTA | FISTA | Group FISTA-LLA | Group FISTA-LLA |
| **logistic** | Newton | FISTA | FISTA | FISTA-LLA | FISTA-LLA | FISTA | FISTA | Group FISTA-LLA | Group FISTA-LLA |
| **poisson** | Newton | FISTA | FISTA | FISTA-LLA | FISTA-LLA | FISTA | FISTA | Group FISTA-LLA | Group FISTA-LLA |
| **gamma** | Newton | FISTA | FISTA | FISTA-LLA | FISTA-LLA | FISTA | FISTA | Group FISTA-LLA | Group FISTA-LLA |
| **inverse_gaussian** | Newton | FISTA | FISTA | FISTA-LLA | FISTA-LLA | FISTA | FISTA | Group FISTA-LLA | Group FISTA-LLA |
| **negative_binomial** | Newton | FISTA | FISTA | FISTA-LLA | FISTA-LLA | FISTA | FISTA | Group FISTA-LLA | Group FISTA-LLA |
| **tweedie** | Newton | FISTA | FISTA | FISTA-LLA | FISTA-LLA | FISTA | FISTA | Group FISTA-LLA | Group FISTA-LLA |

### 如何阅读这张表

- `fista_lla` 是内部延续路径，不是公开的 `solver=` 参数值；直接调用 `fista_lla_path()` 时使用同一近似问题。
- 标量 `squared_error` + SCAD/MCP 可以使用坐标下降延续路径；Group SCAD/MCP 使用加权 Group Lasso 近似问题和面向分组的 FISTA 内层。
- Group Lasso 与 Adaptive Group Lasso 使用相应损失函数的梯度和欧氏组近端算子，包括受支持的 `sample_weight` 与交叉验证路径。
- `sample_weight` 不会改变显式指定的 `solver`。受支持的加权 Newton/L-BFGS 在整个优化中使用同一个归一化加权目标；不支持的损失函数、求解器和权重组合会直接报错。

### inverse-power Gamma 的当前限制

对普通 `GammaRegression(link="inverse_power")`，真正的非均匀权重配合显式 Newton/L-BFGS 时目前要求 `fit_intercept=True`，这样才能构造严格为正、满足该分布族和链接函数定义域的初始线性预测子。

只有 **非均匀权重 + 无截距 + 显式 Newton/L-BFGS** 这一组合当前会被拒绝。未传权重、均匀权重或等效均匀权重继续保持历史无截距行为。

这是当前实现缺少可行无截距初值构造所导致的限制，而不是模型本身的理论限制；后续支持由 [GitHub issue #152](https://github.com/TheHiddenObserver/statgpu/issues/152) 跟踪。

## 2. 显式求解器约束

| 求解器 | 接受 | 拒绝 / 限制 | 说明 |
|--------|------|-------------|------|
| `exact` | 仅 L2 + 平方误差 | 其他所有 | 闭式/特征分解路径 |
| `irls` | 支持 IRLS 的损失函数上的 L2 | 非光滑惩罚 | 分布族/损失函数专用 IRLS |
| `newton` | 有 Hessian 的光滑损失 + L2/none | L1、ElasticNet、非凸及组惩罚 | Newton + Armijo 线搜索 |
| `lbfgs` | 光滑损失 + L2/none | L1、ElasticNet、非凸及组惩罚 | 有限内存 BFGS + 线搜索 |
| `fista` | 支持近端算子的惩罚 | 不支持的模型组合 | Nesterov 近端梯度 |
| `fista_bb` | 受支持的稀疏惩罚 | 不支持的组合 | FISTA + BB 自适应步长 |
| `admm` | 受支持的近端形式 | 不支持的组合 | 变量分裂 + 近端更新 |
| `irls_cd` | 标量 SCAD/MCP/adaptive-L1 | L1/ElasticNet 与组惩罚 | IRLS 外层 + 坐标下降内层 |
| `proximal_irls_cd` | 分位数损失 + 标量 SCAD/MCP | 非分位数损失与组惩罚 | 分位数上界近似 + LLA |
| `proximal_newton` | L2/none 使用 Newton；非光滑直接调用改用 FISTA | 不支持的惩罚结构 | 当前不采用欧氏近端近似 |

不支持的显式组合会在数值拟合前报错。

## 3. 求解器能力

| 求解器 | `sample_weight` | `warm_start` | 推断 | 最适合 |
|--------|:---------------:|:------------:|:----:|--------|
| `exact` | ✅（对应支持路径） | ❌ | ✅（OLS 路径） | 平方误差 + L2 |
| `irls` | 依模型/损失函数而定 | ❌ | 依模型而定 | 支持 IRLS 的 GLM |
| `newton` | 当前 GLM 支持解析权重 | ❌ | 依模型而定 | 有 Hessian 的光滑目标 |
| `lbfgs` | 当前 GLM 支持解析权重；其他损失函数依具体路径 | ❌ | 依模型而定 | 不希望形成完整 Hessian 的光滑目标 |
| `fista` | 受支持的加权路径 ✅ | ✅ | 依模型而定 | 凸稀疏/分组目标与 LLA 内层 |
| `fista_bb` | 受支持的加权路径 ✅ | ✅ | 依模型而定 | 稀疏目标的自适应步长 |
| `admm` | 依组合而定 | ✅ | 依模型而定 | 受支持的近端形式 |
| `irls_cd` | 受支持的路径 ✅ | ✅ | 依模型而定 | 标量非凸延续路径 |

对 Newton/L-BFGS，`sample_weight` 的支持范围由损失函数和模型共同决定，不能仅根据底层求解器函数签名判断。当前 GLM 的数据拟合项使用

`sum(w_i * loss_i) / sum(w_i)`。

通用的稳健回归、分位数回归和 Cox 路径在直接调用 L-BFGS 时分别遵循各自的权重限制。

分组模型的 `warm_start` 会把系数和截距状态一起带入一次拟合，并在成功或失败后清除。

## 4. 交叉验证支持（`PenalizedGLM_CV`）

`PenalizedGLM_CV` 保留公开的 `solver="auto"` 请求，但光滑 L2 候选模型与最终全数据重拟合会按分布族选择求解器。

| 损失 | l2 | l1 / elasticnet | scad / mcp | adaptive_l1 | group_lasso / adaptive group | group_scad / group_mcp |
|------|:--:|:---------------:|:----------:|:-----------:|:----------------------------:|:-----------------------:|
| **squared_error** | `eig-batch` / `exact` 风格路径 | 稀疏 FISTA | LLA + FISTA/CD | 通用拟合 | Group FISTA | Group FISTA-LLA |
| **logistic** | Newton | 稀疏 FISTA | LLA + FISTA | 通用拟合 | Group FISTA | Group FISTA-LLA |
| **poisson** | Newton | 稀疏/FISTA | LLA + FISTA | 通用拟合 | Group FISTA | Group FISTA-LLA |
| **gamma** | L-BFGS | 稀疏/FISTA | LLA + FISTA | 通用拟合 | Group FISTA | Group FISTA-LLA |
| **inverse_gaussian** | L-BFGS | 稀疏/FISTA | LLA + FISTA | 通用拟合 | Group FISTA | Group FISTA-LLA |
| **negative_binomial** | L-BFGS | 稀疏/FISTA | LLA + FISTA | 通用拟合 | Group FISTA | Group FISTA-LLA |
| **tweedie** | Newton | 稀疏/FISTA | LLA + FISTA | 通用拟合 | Group FISTA | Group FISTA-LLA |

权重不会改变上表选择的求解器。Gamma、Inverse-Gaussian、Negative-Binomial 这三个 L-BFGS 光滑 L2 分布族的加权候选拟合和最终重拟合使用 GLM 的统一加权目标函数。

分组验证会在 `alpha` 网格、交叉验证折构造和候选拟合前完成。分组按照最终设计矩阵宽度解释，包括 `formula` 展开的列。没有显式自适应权重时，遗漏特征会补成单独一组；越界索引和不完整的自适应加权分组会在候选拟合前报错。

交叉验证使用本次拟合局部的惩罚状态，不会修改调用方传入的惩罚对象或 `penalty_kwargs`。对于惩罚对象，每个候选模型都会按当前 `alpha` 重建；最终选中模型中暴露的惩罚对象快照与实际目标函数的 `alpha`/`groups` 一致。顶层 CV 对象保留原始构造参数。

## 5. 惩罚项参考

| 惩罚项 | 公式 | 近端形式 | 主要参数 |
|---------|------|---------------|----------|
| `l2` | ½α‖β‖² | L2 缩放 | `alpha` |
| `l1` | α‖β‖₁ | 软阈值 | `alpha` |
| `elasticnet` | α[λ‖β‖₁ + ½(1-λ)‖β‖²] | 软阈值 + L2 缩放 | `alpha`, `l1_ratio` |
| `scad` | SCAD(β; α, a) | SCAD 阈值 / LLA | `alpha`, `a` |
| `mcp` | MCP(β; α, γ) | MCP 阈值 / LLA | `alpha`, `gamma` |
| `adaptive_l1` | αΣ_j w_j|β_j| | 加权软阈值 | `alpha`, weights |
| `group_lasso` | αΣ_g √p_g‖β_g‖₂ | 分组软阈值 | `alpha`, `groups` |
| `AdaptiveGroupLassoPenalty` | αΣ_g w_g√p_g‖β_g‖₂ | 加权分组软阈值 | `alpha`, `groups`, `weights`；仅对象形式 |
| `group_scad` | Σ_g SCAD(‖β_g‖₂; α√p_g, a) | 分组 LLA 近似 | `alpha`, `groups`, `a` |
| `group_mcp` | Σ_g MCP(‖β_g‖₂; α√p_g, γ) | 分组 LLA 近似 | `alpha`, `groups`, `gamma` |

对 Group SCAD/MCP，记关于 `‖β_g‖₂` 的导数为 `D_g`。精确凸近似为 `Σ_g D_g‖β_g‖₂`，内部表示成 `AdaptiveGroupLassoPenalty(alpha=1, weights_g=D_g/√p_g)`；目标 `alpha` 与组大小不会重复相乘。Group LLA 使用 FISTA 内层，而不是通用 Proximal Newton 分支。

分组输入采用严格规则：超参数必须是有限数值标量；组索引/ID 必须是有符号 `int64` 可表示的非负整数值；显式分组不得为空或重复；平坦组 ID 必须从 0 连续；公开惩罚项数值方法要求系数维度与分组后的特征宽度完全一致。

## 6. 推断支持

| 惩罚项 | 推断方法 | 状态 |
|---------|----------|------|
| `l2` | 模型提供的标准推断 / M-估计 | ✅ 受支持路径可用 |
| `l1` | 去偏 Lasso | ✅ 受支持路径可用 |
| `elasticnet` | 依具体方法 | 见模型说明 |
| `scad` / `mcp` | 已实现的 oracle 型推断 / 自助法 | 见模型说明 |
| `adaptive_l1` | 依具体方法 | 见模型说明 |
| Group Lasso / Adaptive Group Lasso / Group SCAD / Group MCP | 保持分组结构的协方差估计 / 自助法 | 尚未实现；推断请求会在拟合前报错 |

## 7. 选择求解器

对大多数用户，建议先使用 `solver="auto"`，只有确实希望固定算法时才显式覆盖。

```text
直接拟合 solver="auto"
├── squared_error + L2?                 → CPU exact / GPU Newton
├── squared_error + none?               → FISTA
├── 光滑非高斯 GLM + L2/none?          → Newton
├── 标量非凸惩罚?                       → 标量 LLA
├── 凸组惩罚?                           → Group FISTA
└── group SCAD/MCP?                     → Group FISTA-LLA
```

`PenalizedGLM_CV` 请使用上面的独立交叉验证表。Gamma、Inverse-Gaussian、Negative-Binomial 的 L2 在交叉验证和最终重拟合中有意使用 L-BFGS，而直接拟合的 `auto` 使用 Newton。