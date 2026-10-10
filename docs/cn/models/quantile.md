# 分位数回归

> 语言：中文  
> 最后更新：2026-10-09<br>
> 页面定位：模型文档  
> 切换：[英文版](../../en/models/quantile.md)

## 什么时候使用分位数回归？

分位数回归直接建模给定特征时响应分布的某个位置：`quantile=0.5` 对应条件中位数，`0.9` 对应条件第 90 百分位。它适合研究分布尾部或不对称响应；若问题关心条件均值，应考虑普通线性回归等均值模型。

`QuantileRegression` 提供无惩罚拟合与受限条件下的推断，`PenalizedQuantileRegression` 增加正则化。两者使用同一个非对称绝对损失，即 check loss，也称 pinball loss。中位数回归对大响应残差的惩罚是线性的，不是平方的；这不意味着对异常特征值也自动稳健。

<a id="cpu-example"></a>

## 完整 CPU 示例

按顺序运行以下小节，先拟合条件中位数，再在未参与拟合的数据上评价。

### 1. 导入

<!-- learner-example: quantile-basic -->
```python
import numpy as np
from statgpu.linear_model import QuantileRegression
```

### 2. 准备连续响应

`X` 为 `(320, 2)` 矩阵，每行一条观测、每列一个数值特征。`y` 为长度 320 的连续响应。这里噪声对称且条件中位数为零，因此真实条件中位数为 `1 + X @ [1.5, -0.7]`。前 240 行训练，后 80 行留出。实际数据应先处理非有限值，并保持预测列顺序。

```python
rng = np.random.default_rng(23)
X = rng.normal(size=(320, 2))
y = 1.0 + X @ np.array([1.5, -0.7]) + rng.normal(scale=0.6, size=320)
```

### 3. 拟合条件中位数

`quantile=0.5` 选择中位数。此时不启用推断，先查看系数与预测。

```python
model = QuantileRegression(
    quantile=0.5, device="cpu", max_iter=3000, tol=1e-6,
).fit(X[:240], y[:240])
print("Slopes:", np.round(model.coef_, 3))
```

斜率约为 `[1.476, -0.636]`。其他特征固定时，第一列增加一单位，拟合的条件中位数约增加 1.476 个响应单位。该解释针对所选分位数，不自动是因果效应或均值效应。

### 4. 预测与评价

```python
prediction = model.predict(X[240:])
pinball_loss = -model.score(X[240:], y[240:])
print("Predicted medians:", np.round(prediction[:3], 3))
print("Held-out pinball loss:", round(float(pinball_loss), 3))
```

预测形状为 `(80,)`，前三项约为 `[1.429, 1.455, 1.632]`。留出 pinball 损失约为 `0.255`；固定数据和目标分位数时越小越好。`score` 返回其负值，不是 R²。一个分位数预测不是单次未来观测的置信区间。
<!-- example-end: quantile-basic -->

## 参数选择与常见问题

- 根据问题选择 `quantile`，不要通过挑选训练损失最小的分位数来改变研究目标。极端分位数需要更多数据。
- 带惩罚拟合中，特征尺度会影响收缩；在训练折内学习缩放与 `alpha`，保留最终测试集。
- 分别拟合多个分位数时可能发生分位数交叉；这一 API 不自动施加不交叉约束。
- 检查数值警告与迭代稳定性。`n_iter_` 不是精度保证，增加预算也不能修复不可识别的设计。

## 目标函数

在分位数 $\tau\in(0,1)$ 处，check / pinball 损失定义为

$$
\ell(\eta,y)=\rho_\tau(y-\eta),
\qquad
\rho_\tau(u)=u\left(\tau-\mathbf 1\{u<0\}\right).
$$

等价地，

$$
\rho_\tau(u)=
\begin{cases}
\tau u, & u\ge 0,\\
(\tau-1)u, & u<0.
\end{cases}
$$

它对正残差和负残差使用不同的线性斜率，因此最优解对应条件 $\tau$ 分位数。其折线形状是 “pinball” 名称的来源；“check loss” 是分位数回归文献中的传统名称。

当 $\tau=0.5$ 时，

$$
\rho_{0.5}(u)=\frac12|u|,
$$

因此中位数回归与最小绝对偏差只差一个不影响最优解的常数比例。

逐样本次梯度为

$$
\frac{\partial\ell}{\partial\eta}
=-\tau+\mathbf 1\{y-\eta<0\},
$$

在残差为 0 的折点处使用次梯度解释。梯度是阶梯函数，因此 `has_hessian=False`、`smooth_gradient=False`。

## 常用参数

下表只列出目标分位数。完整独立模型参数、推断控制与方法见[QuantileRegression 公开实现](../../../statgpu/linear_model/wrappers/_quantile.py)或 `help(QuantileRegression)`；带惩罚类的专用参数见[公开实现](../../../statgpu/linear_model/penalized/_penalized_quantile.py)，共享控制见[通用带惩罚 API](../reference/linear-model-api.md#penalizedgeneralizedlinearmodel)。

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `quantile` | `0.5` | 目标分位数，取值范围 `(0,1)`；`0.5` 为中位数回归 |

## 进阶用法

### 独立模型的系数推断

独立 `QuantileRegression` 的 `kernel` 与 `bootstrap` 推断只支持未传权重或均匀 `sample_weight`；真正非均匀的分析权重在该类中仅支持估计，若同时设置 `compute_inference=True` 会明确报错。`bootstrap` 采用 **i.i.d. 残差 bootstrap**：先按目标分位数的经验分位点对拟合残差做中心化，使重抽样误差分布的经验 τ 分位数为 0，再把中心化残差视为可交换样本进行重抽样，并用后端原生的批量 Quantile IRLS/MM 完成重拟合。它不是 wild/multiplier bootstrap，也不宣称对一般异方差具有稳健覆盖；异方差 Quantile 回归需要不同的 bootstrap 构造。`bootstrap` 推断还要求 `n_bootstrap >= 2`。`kernel` 推断则要求所选带宽规则使 `q ± h` 保持在 `(0, 1)` 内，并得到有限且严格为正的零点残差密度估计；条件不满足时会直接报错，而不是发布非有限标准误。

先完成 [CPU 示例](#cpu-example)，再复用其中的导入与训练数据，下面选择核密度方法估计系数不确定性。区间对应系数，不是响应的预测区间。

<!-- example-requires: quantile-basic -->
<!-- learner-example: quantile-inference -->
```python
inference_model = QuantileRegression(
    quantile=0.5, device="cpu", max_iter=3000, tol=1e-6,
    compute_inference=True, inference_method="kernel",
    kernel="epa", bandwidth="hsheather",
).fit(X[:240], y[:240])
print("Standard errors:", inference_model._bse)
print("Intervals:", inference_model._conf_int)
```
<!-- example-end: quantile-inference -->

本例中区间数组为 `(3, 2)`：截距在前，随后是两个斜率；`_bse` 与 `_pvalues` 也采用同一顺序。

### 加入 L2 惩罚

先完成 [CPU 示例](#cpu-example)，再复用其中的 `X`、`y`。`alpha` 控制收缩强度；示例数值只是演示，应根据训练数据内的验证结果选择。`solver="auto"` 对 L2/无惩罚选择 IRLS。

<!-- example-requires: quantile-basic -->
<!-- learner-example: quantile-penalized -->
```python
from statgpu.linear_model.penalized import PenalizedQuantileRegression

penalized_model = PenalizedQuantileRegression(
    quantile=0.5, penalty="l2", alpha=0.1, device="cpu",
).fit(X[:240], y[:240])
penalized_prediction = penalized_model.predict(X[240:])
```
<!-- example-end: quantile-penalized -->

### 可选：SCAD 惩罚

先完成 [CPU 示例](#cpu-example)与“加入 L2 惩罚”小节，再复用其中的导入和训练数据。标量 SCAD/MCP 通过专用 Proximal IRLS-CD 延续路径拟合非凸惩罚；需要单独评估调参和数值稳定性。

<!-- example-requires: quantile-penalized -->
<!-- learner-example: quantile-scad -->
```python
scad_model = PenalizedQuantileRegression(
    quantile=0.5, penalty="scad", alpha=0.1, device="cpu",
).fit(X[:240], y[:240])
```
<!-- example-end: quantile-scad -->

### 显式选择 IRLS 或 FISTA

先完成 [CPU 示例](#cpu-example)与“加入 L2 惩罚”小节，再复用导入和训练数据，固定目标和惩罚，只更换算法。`auto` 与显式 IRLS 使用同一路径；显式普通 FISTA 会真正执行 FISTA，不会重定向到 IRLS。非光滑惩罚如 ElasticNet 本来就使用普通 FISTA。

<!-- example-requires: quantile-penalized -->
<!-- learner-example: quantile-explicit-solvers -->
```python
irls_model = PenalizedQuantileRegression(
    quantile=0.5, penalty="l2", alpha=0.01, solver="irls", device="cpu",
).fit(X[:240], y[:240])
```

用相同数据与目标请求 FISTA：

```python
fista_model = PenalizedQuantileRegression(
    quantile=0.5, penalty="l2", alpha=0.01, solver="fista", device="cpu",
).fit(X[:240], y[:240])
```
<!-- example-end: quantile-explicit-solvers -->

### 加权分位数回归

先完成 [CPU 示例](#cpu-example)与“加入 L2 惩罚”小节，再复用其中的导入和训练数据。下面的权重为长度 240 的有限非负向量，使前 50 条训练观测在拟合目标中获得更大权重；权重总和必须为正。

<!-- example-requires: quantile-penalized -->
<!-- learner-example: quantile-weighted -->
```python
sample_weight = np.ones(240)
sample_weight[:50] = 5.0
weighted_model = PenalizedQuantileRegression(
    quantile=0.5, penalty="l2", alpha=0.01, device="cpu",
).fit(X[:240], y[:240], sample_weight=sample_weight)
```
<!-- example-end: quantile-weighted -->

此例只展示支持分析权重的估计路径，不代表任意底层求解器或推断方法都支持这些权重。

### GPU（Torch CUDA）

先完成 [CPU 示例](#cpu-example)与“加入 L2 惩罚”小节，再复用 `PenalizedQuantileRegression` 导入和 `X`、`y`。这里显式请求 Torch CUDA；需要已安装 Torch 和可用 CUDA 设备，不可用时会报错。

```python
gpu_model = PenalizedQuantileRegression(
    quantile=0.5, penalty="scad", alpha=0.1, device="torch",
).fit(X[:240], y[:240])
```

## 求解器兼容性

下面的“支持”首先描述无权重时的算法能力。传入 `sample_weight` 后，还必须满足对应损失函数与求解器的带权约定；不能从无权重支持直接推出任意非均匀权重也受支持。

| 求解器 | 支持 | 说明 |
|---|:---:|---|
| Proximal IRLS-CD | ✅ | 专用 IRLS 上界 + LLA，用于标量 SCAD/MCP；支持分析权重 |
| 分组 Proximal IRLS-LLA | ✅ | Group SCAD/MCP 的自动路径；Quantile IRLS/MM 与凸 Adaptive Group Lasso 加权最小二乘子问题组合 |
| IRLS | ✅ | **L2/无惩罚 Quantile 下 `solver="auto"` 的默认路径**；`QuantileLoss.irls()` 明确支持 `sample_weight` |
| FISTA | ✅ | 用于 L1/ElasticNet 等近端路径；L2/无惩罚下显式 `solver="fista"` 也会真正执行普通 FISTA，而 `auto` 仍优先 IRLS |
| FISTA-BB | ❌ | BB 步长通过光滑梯度差估计局部曲率；Quantile 的次梯度是阶梯函数，因此估计器/CV 的显式请求与公开底层 `fista_bb_solver(QuantileLoss, ...)` 都会直接报错 |
| L-BFGS | ✅（仅底层无权重/均匀权重兼容边界） | 公开 `PenalizedQuantileRegression` / `PenalizedGLM_CV` 的显式 L-BFGS 请求不受支持；底层 `lbfgs_solver(QuantileLoss, ...)` 保留历史上的无权重/均匀权重兼容面，真正的非均匀权重仍会被拒绝 |
| ADMM | ❌（不能直接用于 Quantile） | 公开 `admm_solver(QuantileLoss, ...)` 不受支持，因为其通用 w 子问题要求光滑损失。分组 Proximal IRLS-LLA 只会在 Quantile IRLS 已经构造出光滑加权最小二乘代理问题之后，内部使用变量分裂法求解该凸子问题 |
| Newton | ❌ | 分位数损失没有 Hessian |
| Proximal Newton | ❌ | 分位数损失没有 Hessian |

对于 L2/无惩罚 Quantile 目标，`PenalizedQuantileRegression(..., solver="auto")` 会解析到 IRLS，显式 `solver="irls"` 也直接请求同一算法。显式普通 `solver="fista"` 同样受支持：它会真正进入通用 FISTA 求解过程，而不会静默替换成 IRLS。

IRLS 仍是自动选择，因为 pinball loss 本身非光滑；这里的 Quantile FISTA 是一阶近端/次梯度路径，并不声称满足教科书中光滑梯度 FISTA 的经典收敛假设。FISTA-BB 仍不支持，因为它的 BB 曲率更新明确依赖有意义的光滑梯度差分。

## 惩罚项兼容性

| 惩罚项 | `solver="auto"` 的主要路径 | 说明 |
|---|---|---|
| L2 / 无惩罚 | IRLS | `none` 会先规范化为 `L2(alpha=0)`；显式 `irls` 选择同一路径，显式普通 `fista` 也可按需选择 |
| L1 / ElasticNet | FISTA | 近端/次梯度路径 |
| SCAD / MCP | Proximal IRLS-CD | Quantile 专用 IRLS 上界 + LLA |
| adaptive_l1 | FISTA | 先准备自适应权重，再进入 Quantile FISTA |
| group_lasso / adaptive group | 分组 FISTA | 面向分组的近端路径 |
| group_scad / group_mcp | 分组 Proximal IRLS-LLA | 分组 LLA + Quantile IRLS/MM；每个凸 Adaptive Group Lasso 加权最小二乘代理问题都在所选后端求解 |

上表描述的是自动路径。若对 Group SCAD/MCP 显式指定 `solver="fista"`，该请求仍然保持为显式近端 FISTA，不会被静默改写成自动的分组 Proximal IRLS-LLA。公开底层 `fista_lla_path(...)` 的标量 Quantile SCAD/MCP 现在执行同一套专用 Proximal IRLS-LLA 求解；分组惩罚以及带热启动或返回路径的底层调用继续使用融合式 FISTA-LLA 引擎。该委托路径的 `n_iter` 记录 IRLS 迭代次数，并给出自身的预算耗尽告警。

## `sample_weight` 语义

对于已经声明支持非均匀分析权重的分位数路径，数据拟合项使用加权 check/pinball 目标。以逐样本损失 $\rho_\tau(r_i)$ 为例，归一化形式为

$$
L_w(\beta)
=\frac{\sum_i w_i\rho_\tau(y_i-x_i^\top\beta)}{\sum_i w_i}.
$$

但 `sample_weight` **不是所有求解器自动具备的统一能力**。当前尤其需要区分：

- Quantile IRLS / Proximal IRLS-CD 具有明确的带权实现；
- 分组 Proximal IRLS-LLA 使用同一组归一化分析权重构造 Quantile IRLS/MM 上界，然后求解相应的凸分组代理问题；
- 普通 FISTA（包括显式选择的 L2/无惩罚 FISTA）使用损失函数层的归一化带权目标；
- Adaptive L1 需要通过初始化拟合学习自适应惩罚权重时，会使用同一组解析训练权重；若用户已经显式给定固定的自适应权重，则直接使用，不再额外运行一次无效初始化；
- 通用 `LossBase` 的共享函数值和梯度可以计算归一化带权目标；
- FISTA-BB 与公开的直接 ADMM 不支持 Quantile，因为这些通用算法依赖 check loss 不具备的光滑梯度结构；
- 底层 Quantile L-BFGS 保留无权重/均匀权重兼容面，但真正非均匀权重会报错；模型/CV 层显式 L-BFGS 仍不支持。

需要比较其他损失函数和求解器的带权范围时，见 [求解器 × 惩罚项兼容性矩阵](../guides/solver-penalty-matrix.md) 和 [求解器算法](../guides/solver-algorithms.md)。

## 算法详解

### Proximal IRLS-CD（SCAD/MCP）

详细更新公式见 [求解器算法](../guides/solver-algorithms.md#1-proximal-irls-cd)。其核心是把 check loss 的 IRLS 二次上界与 SCAD/MCP 的局部线性近似结合起来。

### 分组 Proximal IRLS-LLA（Group SCAD/MCP）

自动 Group SCAD/MCP 路径的外层 LLA 会把非凸分组惩罚转化成加权 Adaptive Group Lasso 凸代理问题。记 $D_g^{(k)}$ 为当前分组惩罚对 $\|\beta_g\|_2$ 的导数，Quantile 专用内层首先根据当前残差构造

$$
w_i^{(t)}
=
\widetilde s_i
\frac{\tau+(1-2\tau)\mathbf 1\{r_i^{(t)}<0\}}
{\max(|r_i^{(t)}|,\varepsilon)},
\qquad
\widetilde s_i=\frac{n s_i}{\sum_j s_j},
$$

无分析权重时取 $s_i=1$。随后求解凸的加权最小二乘代理问题

$$
\min_\beta
\frac{1}{2n}
\sum_i w_i^{(t)}
\left(y_i-x_i^\top\beta\right)^2
+
\sum_g D_g^{(k)}\|\beta_g\|_2.
$$

该凸子问题使用后端原生的变量分裂求解：二次更新对应加权最小二乘线性系统，近端更新则使用精确的 Adaptive Group Lasso 分组收缩。截距包含在二次模型中，但不参与惩罚。如果所有 $D_g^{(k)}$ 都为 0，则当前 LLA 目标恰好退化为无惩罚 Quantile 回归，此时直接使用普通带权 Quantile IRLS 求解。

这条自动路径与显式 FISTA 控制相互独立：`solver="fista"` 仍然表示显式 FISTA 请求，而底层 `fista_lla_path(...)` 的标量 Quantile SCAD/MCP 执行同一套专用 Proximal IRLS-LLA 求解（分组惩罚以及带热启动或返回路径的调用继续使用融合式 FISTA-LLA 引擎）。

### IRLS（L2/无惩罚）

IRLS 是 L2/无惩罚 Quantile 目标的 `auto` 路径，也可以显式请求。令

$$
r_i=y_i-x_i^\top\beta.
$$

分位数 IRLS 权重为

$$
w_i^{\mathrm{IRLS}}
=\frac{\tau+(1-2\tau)\mathbf1\{r_i<0\}}
{\max(|r_i|,\varepsilon)}.
$$

若同时传入分析权重 $s_i$，实现先把它归一化为

$$
\widetilde s_i=\frac{n s_i}{\sum_j s_j},
$$

再使用

$$
w_i=\widetilde s_i w_i^{\mathrm{IRLS}}.
$$

记 $W=\operatorname{diag}(w)$，无惩罚时更新满足

$$
(X^\top W X+\varepsilon I)\beta_{\mathrm{new}}
=X^\top W y.
$$

L2 路径在左侧加入相应的 Ridge 对角项；截距坐标不参与惩罚。完整实现细节见 [IRLS 算法参考](../guides/solver-algorithms.md#6-irls迭代重加权最小二乘)。

### 普通 FISTA（显式 L2/无惩罚与稀疏凸路径）

Quantile/check loss 本身非光滑，因此这里不应解释为满足经典光滑梯度 FISTA 的全部理论假设。statgpu 使用注册的 Quantile 次梯度、已有的一阶步长策略以及所请求惩罚项的近端算子。该路径用于显式算法控制以及凸稀疏 Quantile 路径；L2/无惩罚的自动策略仍然是 IRLS。

## 输出

| 属性 | 类型 | 说明 |
|---|---|---|
| `coef_` | `(p,)` float | 估计系数 |
| `intercept_` | float | 截距 |
| `n_iter_` | int | 迭代次数 |
| `quantile` | float | 目标分位数 |

## 说明

- `QuantileRegression.score()` 与类型化封装 `PenalizedQuantileRegression.score()` 使用 check/pinball 损失，并返回其相反数，以符合 sklearn“越大越好”的约定。NumPy、CuPy 与 Torch 的响应变量/权重容器在受支持路径上均可直接传入；评分会将响应、权重和预测转为 NumPy，并返回 Python 标量。通用 `PenalizedGeneralizedLinearModel(loss="quantile")` 则保留共享的标量响应 `score()` 契约，因此返回响应尺度上的 R²；`PenalizedGLM_CV.score()` 委托给选中 α 后的最终重拟合模型。CV 的 `best_score_` 是验证集损失的相反数，与拟合后的 `score()` 本来就是不同指标。
- `sample_weight` 支持是**损失函数 × 求解器 × 估计器**路径能力，而不是所有求解器自动拥有的属性。
- 严格 Quantile 交叉验证对所有惩罚家族都要求每一折的验证得分为有限值。只有当对应求解器路径明确把某一折标记为目标 α 的收敛失败时，该折才不会计分；只有所有折都有有限得分的 α 才有候选资格。如果任一折拟合失败或携带这种明确的目标 α 收敛失败信号，则该 α 整列都视为无效，不能只对剩余有限折求均值。通用求解器单独发出 `ConvergenceWarning` 并不会自动抹去一个已经得到的有限候选结果。两阶段策略的第一阶段筛选仍有意采用较宽松的准则；所有折均需得到有限得分的要求在后续的严格精炼与严格选择阶段执行。选中 α 后的全数据最终重拟合沿用直接估计器的收敛报告语义，并可按该语义发出 `ConvergenceWarning`，而不是作为 CV 候选失败处理。
- Quantile 非凸延续路径会拒绝停止控制的隐式类型转换。`max_iter` 必须是正整数，`tol` 必须是有限正实数；直接标量 SCAD/MCP 与自动 Group SCAD/MCP 还要求布尔型 `lla=True`、整数 `max_lla_iters` 和有限正数 `lla_tol`。当前自动 Quantile 延续路径包含 3 个 α 步骤，因此 `max_lla_iters` 至少为 3，才能保证每一步至少执行一次 LLA 更新。中间延续步骤使用缩减后的 IRLS 预算，但不会超过公开的 `max_iter`；目标步骤最多使用完整预算。显式 Group SCAD/MCP `solver="fista"` 不进入 LLA 延续路径，因此 `lla`、`max_lla_iters` 与 `lla_tol` 不控制这条显式算法。
- 公开底层 Quantile 求解器调用——包括普通 `fista_solver`、保留兼容边界的直接 `lbfgs_solver`、`QuantileLoss.irls()`、`proximal_irls_quantile_solver()` 与 Quantile `fista_lla_path()`——都会在数值计算前拒绝非法的监督输入形状：`X` 必须为二维、`y` 必须为一维，且二者行数一致，并且二者都必须只包含有限实数。IRLS/延续路径专用边界还会拒绝非法的截距、停止、路径和权重控制，不依赖广播或隐式类型转换；直接延续路径调用的 `alpha_path` 还必须是一维非空、元素均为有限正数、并从起点到目标值保持非递增的序列。
- 显式请求的普通 L2/无惩罚 Quantile FISTA 受支持并按请求执行；Group SCAD/MCP 的显式 FISTA 同样不会被改写成自动的分组 Proximal IRLS-LLA。
- FISTA-BB、公开直接 ADMM、Newton、Proximal Newton 与 L-BFGS-B 都不支持 Quantile，并会在进入数值迭代前报错。普通估计器/CV 层的 L-BFGS 同样不支持 Quantile；只有底层 `lbfgs_solver` 的未加权/均匀权重历史兼容边界继续保留。`quantile_cd_solver` 仅作为导入兼容符号保留，调用时会抛出 `NotImplementedError`。底层标量 SCAD/MCP 拟合请使用 `proximal_irls_quantile_solver`。
- 受支持的 GPU 路径不会静默回退到 CPU。

## 相关文档

- [损失函数](losses.md) — `QuantileLoss` 的底层定义
- [求解器 × 惩罚项兼容性矩阵](../guides/solver-penalty-matrix.md) — 组合支持范围
- [求解器算法](../guides/solver-algorithms.md) — IRLS、FISTA 与 Proximal IRLS-CD 的数值细节
- [交叉验证](../guides/cross-validation.md) — CV 选择与最终重拟合
底层损失对象 `QuantileLoss` 从 `statgpu.losses` 导入。与 R `quantreg::rq()` 比较无惩罚结果时，应对齐分位数、设计矩阵、截距、权重和优化精度；比较推断还需要相容的密度估计或重抽样假设。

## 参考文献

- Koenker, R. & Bassett, G. (1978). Regression Quantiles. *Econometrica*, 46(1), 33–50.
- Koenker, R. (2005). *Quantile Regression*. Cambridge University Press.
- Feng, X., He, X. & Hu, J. (2011). Wild bootstrap for quantile regression. *Biometrika*, 98(4), 995–999.
- Wu, Y. & Liu, Y. (2009). Variable Selection in Quantile Regression. *Statistica Sinica*, 19, 801–817.
- Hunter, D. R. & Li, R. (2005). Variable Selection using MM Algorithms. *Annals of Statistics*, 33(4), 1617–1642.
