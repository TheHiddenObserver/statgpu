# 推断模式

> 语言：中文  
> 最后更新：2026-09-17  
> 页面定位：选择并解释 coefficient inference method  
> 切换：[English](../../en/guides/inference-modes.md)

## 本页解决什么问题

statgpu 提供多种推断方法，是因为经典低维回归、固定惩罚 GLM，以及通过 L1 正则化选择出的稀疏模型对应的是不同统计问题。

本页主要回答两个问题：

1. **当前 fitted model 应该使用哪一种 inference method？**
2. **报告出来的区间到底对应原拟合参数、偏差修正参数，还是另一个 refit？**

具体 backend kernel、内部 result storage 与 validation evidence 不属于本用户指南。

## 方法总览

| Fitted model / 场景 | Inference method | 解释 |
|---|---|---|
| Gaussian linear/Ridge | classical 或 robust covariance | 对 fitted linear-model coefficient 做推断 |
| smooth non-Gaussian L2/no-penalty penalized GLM | `m_estimation` | fixed-penalty estimating-equation inference |
| Gaussian Lasso/ElasticNet | `debiased` | de-biased / de-sparsified coefficient inference |
| Gaussian Lasso/ElasticNet | `post_selection_ols` | 在已选择 active set 上做 OLS/WLS 诊断性 refit |
| 受支持的 Gaussian penalized model | `bootstrap` | tuning 固定时的 residual-bootstrap distribution |
| 受支持的 SCAD/MCP | 显式请求 `oracle` | 以已选择 active set 为条件的推断 |
| 不支持的 loss/penalty/method 组合 | — | 报错，而不是静默替换成另一种 inferential target |

完整支持矩阵见 [Penalized GLM 推断](penalized-glm-inference.md)与对应模型页。

## Gaussian 线性模型推断

普通 Gaussian linear model 与共享的 squared-error L2/Ridge 路径通过 `cov_type` 选择协方差与参考分布。

常用选项包括：

- `nonrobust`：经典协方差，Student-t 参考分布；
- `hc0`、`hc1`、`hc2`、`hc3`：异方差稳健协方差，正态参考分布；
- `hac`：Bartlett-kernel HAC 协方差，正态参考分布。

数值推断在 fitted model 所支持的 backend 上完成。数值阶段完成后，小型 reporting array 可以转换为 NumPy；这种 reporting conversion 不表示显式 CUDA/Torch 拟合被静默重新放到 CPU 上计算。

模型专属 covariance 选择见对应模型页。

## Penalized GLM 的 fixed-penalty M-estimation

对受支持的 smooth non-Gaussian L2/no-penalty model，

```python
inference_method="auto"
```

会解析为 `m_estimation`。

正的 L2 penalty 对应**给定惩罚强度下的 penalized estimating equation**；无惩罚拟合对应普通 unpenalized parameter。当前 non-Gaussian fixed-penalty covariance 支持 `nonrobust`、`hc0`、`hc1`；在不支持的行上请求 HC2/HC3/HAC 会直接报错。

### Analytic weights

在所选 smooth solver 支持 analytic `sample_weight` 时，拟合与对应 M-estimation 使用同一个归一化加权目标：

$$
L_w(\beta)
=
\frac{\sum_i w_i\,\ell_i(\beta)}{\sum_i w_i}.
$$

因此所有正 analytic weight 同乘一个常数不会改变统计 target。若某个 loss 并没有定义对应的 weighted fitting，例如不支持的 weighted Cox route，statgpu 会拒绝请求而不是丢弃权重。

协方差公式和精确 loss/penalty 支持范围见 [Penalized GLM 推断](penalized-glm-inference.md)。

## 稀疏 Gaussian 推断

对于 Gaussian `Lasso`、`ElasticNet` 以及等价的 squared-error L1/ElasticNet penalized model，三个 public mode 对应不同目标。

### `debiased`

De-biased/de-sparsified inference 通过估计 design precision 的近似逆并进行 one-step correction，修正 penalized estimator 的一阶正则化偏差。

当目标是高维 marginal coefficient inference，并且相应假设对应用场景合理时，可使用这一模式。node-wise precision 的 tuning 与主模型 penalty 分开控制；见 [Node-wise Lasso inference tuning migration](nodewise-alpha-migration.md)。

支持 analytic weights 时，debiased inference 与 sparse Gaussian fit 使用相同的 weighted-centered statistical problem，因此全局正权重缩放不会改变其统计 target。

如果开启 simultaneous inference，statgpu 使用 max-|Z| calibration，而不是把普通 marginal interval 当成 simultaneous interval。相关控制见 Lasso/ElasticNet 模型文档。

### `post_selection_ols`

`post_selection_ols` 先取得 penalized model 选出的 active set，然后在**完全相同的 active set 上重新拟合无惩罚 OLS 或 WLS**。

因此必须区分：

- prediction 仍使用原来的 penalized `coef_` / `intercept_`；
- 该 inference mode 的 coefficient table 属于 active-set OLS/WLS refit；
- 在同一份数据上先选变量再做普通 OLS/WLS 区间，**并不自动成为一般 selective-inference confidence interval**。

因此应把它理解为 post-selection diagnostic 或 conditional refit，而不是自动修正 variable-selection uncertainty 的方法。

```python
from statgpu.linear_model import Lasso

model = Lasso(
    alpha=0.1,
    solver="fista",
    compute_inference=True,
    inference_method="post_selection_ols",
)
model.fit(X, y)

# prediction 参数仍属于 penalized fit
prediction_coef = model.coef_
```

历史上的 `cpu_ols` 与 `gpu_ols` 只是 `post_selection_ols` 的弃用兼容别名，并不负责选择执行 device。

### `bootstrap`

Gaussian residual-bootstrap 会保持 fitted design 与 tuning configuration 不变。每个 bootstrap draw 依次：

1. 计算 fitted value 与 residual；
2. 有放回地重采样 residual；
3. 构造 bootstrap response；
4. 使用相同 tuning configuration 重新拟合同一个 penalized model；
5. 汇总 bootstrap coefficient distribution。

它不是适用于所有 GLM family 的通用 bootstrap。当前支持的 residual-bootstrap 路径要求 `sample_weight=None` 且 `cov_type="nonrobust"`；weighted residual bootstrap、robust/HAC block bootstrap、non-Gaussian bootstrap 与 Cox bootstrap 不由这一模式提供。

因此该结果描述的是 fixed-design、fixed-tuning residual-bootstrap procedure 下的不确定性，并不会自动调整 tuning 或 variable-selection uncertainty。

## SCAD/MCP active-set 推断

在支持 `inference_method="oracle"` 的模型上，statgpu 以 non-convex penalized fit 选择出的 active set 为条件进行推断。`auto` 不会静默选择这种解释，因为“以选定 support 为条件”本身就是一个实质性的 inferential assumption。

请求 `oracle` 前请查看 [Penalized GLM 推断](penalized-glm-inference.md)中的当前 model/backend 限制。

## Cross-validation 后推断

CV selection 与 coefficient inference 是两个阶段：

```text
在 folds 上拟合候选模型
    -> 选择 tuning parameter
    -> 在全部观测上 refit 所选模型
    -> 只对 final refit 执行一次 inference
```

因此，除非某种方法明确说明做了额外修正，报告的不确定性都应解释为**以 CV 已经选定的 tuning configuration 为条件**，不会自动调整 tuning-selection uncertainty。

选择/refit 语义见 [交叉验证](cross-validation.md)。

## Inference method 不负责选择硬件

`inference_method` 选择统计程序，`device` 决定在该程序支持范围内的执行硬件。

- `device="cpu"` 请求 NumPy CPU；
- `device="cuda"` 要求 CuPy CUDA route；
- `device="torch"` 要求 Torch CUDA route；
- `device="auto"` 可以在可用 backend 之间自动选择。

不支持的显式 backend/method 组合会报错，而不会把 CPU 结果伪装成用户请求的 accelerator 结果。某些 inference method 的 backend 范围比 parent estimator 更窄；如果 device placement 很重要，应查看 method-specific 文档。

## 如何理解 fitted parameter 与 inference parameter

普通 unpenalized model 中，两者通常指同一个 estimator；penalized post-fit method 则可能不同：

- `post_selection_ols` 报告 active-set unpenalized refit，但 prediction 仍使用 penalized fit；
- `debiased` 报告 bias-corrected inferential parameter，但 prediction 仍使用 penalized fit；
- `bootstrap` 描述固定 tuning configuration 下重复 penalized refit 的分布。

不要只根据 coefficient table 的形状猜测 target。需要区分时，应查看 `inference_method_`、`inference_target_` 与相应模型页。

## 如何选择方法

可以按以下顺序判断：

1. **普通低维 Gaussian model**：使用模型本身的 classical/robust covariance。
2. **Smooth non-Gaussian L2/no-penalty GLM**：优先从 `inference_method="auto"` 开始；支持时解析为 fixed-penalty M-estimation。
3. **Sparse Gaussian L1/ElasticNet**：若高维 coefficient inference 的假设合理，使用 `debiased`；只有当目标本来就是 active-set diagnostic/refit 时才使用 `post_selection_ols`。
4. **需要 Gaussian fixed-tuning resampling 视角**：只在其公开范围内使用 `bootstrap`。
5. **SCAD/MCP active-set 解释**：仅在确实需要该条件 target 且模型支持时显式请求 `oracle`。

## 相关文档

- [Penalized GLM 推断](penalized-glm-inference.md) — 公式、支持矩阵与详细统计 target
- [交叉验证](cross-validation.md) — tuning selection 与 final refit
- [Node-wise Lasso inference tuning migration](nodewise-alpha-migration.md) — `nodewise_alpha`
- [设备与 GPU 内存](device-and-memory.md) — backend/device 语义
- [Lasso](../models/lasso.md) 与 [ElasticNet](../models/elastic-net.md) — sparse inference 的模型专属说明
