# 求解器 × 惩罚项兼容性矩阵

> 语言：中文  
> 最后更新：2026-09-15  
> 页面定位：参考指南  
> 切换：[英文版](../../en/guides/solver-penalty-matrix.md)

## 概览

本页回答一个实际问题：在 `PenalizedGeneralizedLinearModel` 或 `PenalizedGLM_CV` 中，某个损失函数与惩罚项组合应该由哪个求解器处理？

首先要区分**直接拟合**和**交叉验证（CV）**：

- 普通直接拟合的 `solver="auto"` 使用第 1 节的调度表；
- `PenalizedGLM_CV` 使用一套相关但有意不同的规则，见第 4 节；
- 若显式指定 `solver`，`sample_weight` 不会改变这一请求；不支持的组合会在数值计算前报错。

`none` / `null` 在求解器选择之前会先规范化为 `L2(alpha=0)`。因此无惩罚的光滑路径与 L2 使用同一个 `auto` 分支，不会因为公开参数写成 `none` 就自动变成 FISTA。

`AdaptiveGroupLassoPenalty` 可以作为公开惩罚对象使用，但调用方必须显式提供组权重，因此它不提供字符串别名。

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

### 如何阅读这张表

- 表中写的是**实际维护的执行路径**，而不仅是 `_select_solver()` 返回的内部字符串。
- `fista_lla` 是内部延续路径，不是公开的 `solver=` 参数值。`squared_error + SCAD/MCP` 会在 `fit()` 中直接进入融合的 `fista_lla_path()`。Quantile + SCAD/MCP 不同，它使用分位数文档中说明的专用 Proximal IRLS 路径。
- direct logistic、Poisson 与负二项的凸稀疏行会落到默认 FISTA-BB 规则；Gamma 和逆高斯的稀疏行被显式固定为 FISTA；Tweedie 的稀疏行在 CuPy/Torch 上走 FISTA、在 CPU 上落到 FISTA-BB。
- 分组 Lasso 与自适应分组 Lasso 使用 group-aware FISTA；Group SCAD/MCP 使用加权 Group-Lasso LLA surrogate 与 group-aware FISTA 内层。
- `sample_weight` 不会改变显式指定的 `solver`。受支持的加权 Newton/L-BFGS 在整个优化中使用同一个归一化加权目标；不支持的损失函数、求解器和权重组合会直接报错。

### `inverse_power` Gamma 的光滑定义域约定

对 inverse-power Gamma，

\[
\eta_i=x_i^\top\beta>0,
\qquad
\ell_i(\eta_i)=y_i\eta_i-\log\eta_i.
\]

当前维护的显式 `newton` / `lbfgs` 路径会在实际执行后端上构造位于内部的初值，并保证每次**训练目标函数**的有效观测线性预测子都处在未触发 clipping 的数值区间内，使 value、gradient 与 Hessian 对应同一个光滑目标。Armijo 的定义域步长上限只在 Newton/L-BFGS 已经完成奇异/非下降回退、确定最终实际搜索方向之后计算。

因此 `fit_intercept=False` 不再被类别式拒绝：只要 statgpu 能够为实际设计矩阵认证一个内部初值，并且优化过程没有在数值定义域边界处停滞，就可以使用显式 Newton/L-BFGS。未传权重、均匀权重、等效均匀权重和真正非均匀解析权重使用相同的定义域/初始化原则；在真正加权目标中，解析权重严格为 0 的行不约束训练定义域。

若有限设计矩阵无法通过维护的数值过程认证内部初值，或优化在梯度尚未收敛时被定义域边界卡住，该路径会显式失败，而不是发布依赖 clipping 的近似拟合。公开预测与留出验证仍保留原有 clipping 语义，因此这里的训练定义域保证不应理解为对所有未来新样本的线性预测子作额外限制。

## 2. 显式求解器约束

| 求解器 | 接受 | 拒绝 / 限制 | 说明 |
|--------|------|-------------|------|
| `exact` | 仅 L2 + 平方误差 | 其他所有 | 闭式/特征分解路径 |
| `irls` | 声明维护中 IRLS 支持的损失函数上的 L2/无惩罚 | 非光滑惩罚 | 损失函数/分布族专用 IRLS；Quantile 的 `auto` 虽然使用 FISTA，但显式 Quantile IRLS 仍支持 L2/无惩罚 |
| `newton` | 有 Hessian 的光滑损失 + L2/none | L1、ElasticNet、非凸及分组惩罚 | Newton + Armijo 线搜索 |
| `lbfgs` | 光滑损失 + L2/none | L1、ElasticNet、非凸及分组惩罚 | 有限内存 BFGS + 线搜索 |
| `fista` | 支持近端算子的惩罚 | 不支持的模型组合 | Nesterov 近端梯度 |
| `fista_bb` | 受支持的稀疏惩罚 | 不支持的组合 | FISTA + BB 自适应步长 |
| `admm` | 受支持的近端形式 | 不支持的组合 | 变量分裂 + 近端更新 |
| `irls_cd` | 专用标量路径 | 不支持的组合 | 不是当前 `squared_error + SCAD/MCP` 的公开 auto 路径 |
| `proximal_irls_cd` | 分位数损失 + 标量 SCAD/MCP | 非分位数损失与分组惩罚 | 分位数上界近似 + LLA |
| `proximal_newton` | L2/none 使用 Newton；非光滑直接调用改用 FISTA | 不支持的惩罚结构 | 当前不采用欧氏近端近似 |

不支持的显式组合会在数值拟合前报错。

## 3. 求解器能力

| 求解器 | `sample_weight` | `warm_start` | 推断 | 最适合 |
|--------|:---------------:|:------------:|:----:|--------|
| `exact` | ✅（对应支持路径） | ❌ | ✅（OLS 路径） | 平方误差 + L2 |
| `irls` | 依模型/损失函数而定 | ❌ | 依模型而定 | 维护中的 IRLS 路径 |
| `newton` | 当前 GLM 支持解析权重 | ❌ | 依模型而定 | 有 Hessian 的光滑目标 |
| `lbfgs` | 当前 GLM 支持解析权重；其他损失函数依具体路径 | ❌ | 依模型而定 | 不希望形成完整 Hessian 的光滑目标 |
| `fista` | 受支持的加权路径 ✅ | ✅ | 依模型而定 | 凸稀疏/分组目标与 LLA 内层 |
| `fista_bb` | 受支持的加权路径 ✅ | ✅ | 依模型而定 | 稀疏目标的自适应步长 |
| `admm` | 共享 `admm_solver` 仅支持未传/均匀权重 | ✅ | 依模型而定 | 受支持的近端形式 |
| `irls_cd` | 依具体路径而定 | ✅ | 依模型而定 | 专用标量坐标下降路径 |

对 Newton/L-BFGS，`sample_weight` 的支持范围由损失函数和模型共同决定，不能仅根据底层求解器函数签名判断。当前 GLM 的数据拟合项使用

`sum(w_i * loss_i) / sum(w_i)`。

通用的稳健回归、分位数回归和 Cox 路径在直接调用 L-BFGS 时分别遵循各自的权重限制。共享 `admm_solver` 则在入口处要求权重为未传或均匀；真正非均匀解析权重会在数值迭代前报错。

分组模型的 `warm_start` 会把系数和截距状态一起带入一次拟合，并在成功或失败后清除。

## 4. 交叉验证支持（`PenalizedGLM_CV`）

`PenalizedGLM_CV` 保留公开的 `solver="auto"` 请求，但候选拟合与最终全数据重拟合会按损失函数、后端，有时还按问题规模选择实际求解器。

| 损失 | l2 | l1 / elasticnet | scad / mcp | adaptive_l1 | group_lasso / adaptive group | group_scad / group_mcp |
|------|:--:|:---------------:|:----------:|:-----------:|:----------------------------:|:-----------------------:|
| **squared_error** | CPU `eig-batch` / `exact` 风格路径；适用的 GPU 最终拟合使用 Newton | FISTA | FISTA-LLA | FISTA | 分组 FISTA | 分组 FISTA-LLA |
| **logistic** | Newton | FISTA | FISTA-LLA | FISTA | 分组 FISTA | 分组 FISTA-LLA |
| **poisson** | Newton | CPU FISTA；GPU L1 可使用按规模门控的 FISTA-BB，GPU ElasticNet 使用 FISTA-BB | FISTA-LLA | FISTA | 分组 FISTA | 分组 FISTA-LLA |
| **gamma** | L-BFGS | FISTA | FISTA-LLA | FISTA | 分组 FISTA | 分组 FISTA-LLA |
| **inverse_gaussian** | L-BFGS | FISTA | FISTA-LLA | FISTA | 分组 FISTA | 分组 FISTA-LLA |
| **negative_binomial** | L-BFGS | 通常 FISTA-BB；维护中的 GPU ElasticNet 中等规模区间使用 FISTA | FISTA-LLA | FISTA-BB | 分组 FISTA | 分组 FISTA-LLA |
| **tweedie** | Newton | CPU FISTA-BB / GPU FISTA | FISTA-LLA | CPU FISTA-BB / GPU FISTA | 分组 FISTA | 分组 FISTA-LLA |

Poisson GPU L1 的 FISTA-BB 是按规模门控的：维护中的快路径用于大约两百万个 design elements 以下，较大的问题使用 FISTA。Negative-Binomial GPU ElasticNet 在维护中的中等规模区间（约 200k–1M 个 design elements）使用 FISTA，区间之外使用 FISTA-BB。这些阈值属于内部 dispatch policy，不是通用性能保证。

权重不会改变上表选择的求解器。Gamma、逆高斯、负二项这三个使用 L-BFGS 的光滑 L2 分布族，其加权候选拟合和最终重拟合都使用 GLM 的统一加权目标函数。对 `loss="gamma"` 且 `loss_kwargs={"link": "inverse_power"}` 的光滑 L2 CV，候选拟合、验证损失、`alpha` 选择和最终重拟合会一直使用实际的 inverse-power Gamma 损失，而不会进入只适用于 log-link Gamma 的快速验证公式。

分组验证会在 `alpha` 网格、交叉验证折构造和候选拟合前完成。分组按照最终设计矩阵宽度解释，包括 `formula` 展开的列。没有显式自适应权重时，遗漏特征会补成单独一组；越界索引和不完整的自适应加权分组会在候选拟合前报错。

交叉验证使用本次拟合局部的惩罚状态，不会修改调用方传入的惩罚对象或 `penalty_kwargs`。对于惩罚对象，每个候选模型都会按当前 `alpha` 重建；最终选中模型中暴露的惩罚对象快照与实际目标函数的 `alpha`/`groups` 一致。顶层交叉验证对象保留原始构造参数。

## 5. 惩罚项参考

| 惩罚项 | 公式 | 近端形式 | 主要参数 |
|---------|------|---------|----------|
| `l2` | ½α‖β‖² | L2 缩放 | `alpha` |
| `l1` | α‖β‖₁ | 软阈值 | `alpha` |
| `elasticnet` | α[λ‖β‖₁ + ½(1-λ)‖β‖²] | 软阈值 + L2 缩放 | `alpha`, `l1_ratio` |
| `scad` | SCAD(β; α, a) | SCAD 阈值 / LLA | `alpha`, `a` |
| `mcp` | MCP(β; α, γ) | MCP 阈值 / LLA | `alpha`, `gamma` |
| `adaptive_l1` | αΣ_j w_j|β_j| | 加权软阈值 | `alpha`, `weights` |
| `group_lasso` | αΣ_g √p_g‖β_g‖₂ | 分组软阈值 | `alpha`, `groups` |
| `AdaptiveGroupLassoPenalty` | αΣ_g w_g√p_g‖β_g‖₂ | 加权分组软阈值 | `alpha`, `groups`, `weights`；仅对象形式 |
| `group_scad` | Σ_g SCAD(‖β_g‖₂; α√p_g, a) | 分组 LLA surrogate | `alpha`, `groups`, `a` |
| `group_mcp` | Σ_g MCP(‖β_g‖₂; α√p_g, γ) | 分组 LLA surrogate | `alpha`, `groups`, `gamma` |

对 Group SCAD/MCP，记 `D_g` 为关于 `‖β_g‖₂` 的导数。精确凸 surrogate 为 `Σ_g D_g‖β_g‖₂`，内部表示为 `AdaptiveGroupLassoPenalty(alpha=1, weights_g=D_g/√p_g)`。目标 alpha 与组大小不会被重复乘入。Group LLA 使用 FISTA，而不是通用 Proximal Newton 分支。

分组输入采用严格契约：超参数必须是有限数值标量；group index/ID 必须是可表示为 signed `int64` 的非负整数值；显式 group 不能为空且不能重复；扁平 group ID 必须从 0 连续；公开数值惩罚方法的维度必须与被分组特征维度完全一致。

## 6. 推断支持

| 惩罚项 | 推断方法 | 状态 |
|--------|----------|------|
| `l2` | estimator 暴露的 standard / M-estimation | ✅ 维护路径可用 |
| `l1` | Debiased Lasso | ✅ 支持路径可用 |
| `elasticnet` | 依方法而定 | 见 estimator 契约 |
| `scad` / `mcp` | 已实现处使用 oracle/bootstrap | 见 estimator 契约 |
| `adaptive_l1` | 依方法而定 | 见 estimator 契约 |
| Group Lasso / Adaptive Group Lasso / Group SCAD / Group MCP | 保留分组结构的 covariance/bootstrap | 尚未实现；推断请求会在拟合前失败 |

## 7. 如何选择求解器

大多数情况下建议从 `solver="auto"` 开始，只有在确实需要指定某个算法时才显式覆盖。

```text
direct solver="auto"
├── squared_error + L2/none?             → CPU exact / GPU Newton
├── smooth non-Gaussian GLM + L2/none?  → Newton
├── squared_error convex sparse?         → FISTA
├── gamma / inverse-Gaussian sparse?     → FISTA
├── logistic / poisson / NB sparse?      → FISTA-BB
├── tweedie sparse?                      → CPU FISTA-BB / GPU FISTA
├── scalar SCAD/MCP?                     → FISTA-LLA
├── convex group penalty?                → 分组 FISTA
└── group SCAD/MCP?                      → 分组 FISTA-LLA
```

`PenalizedGLM_CV` 请使用上面的独立交叉验证表。Gamma、逆高斯、负二项的 L2 在交叉验证和最终重拟合中有意使用 L-BFGS，而直接拟合的 `auto` 使用 Newton。
