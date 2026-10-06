# 推断模式

> 语言：中文  
> 最后更新：2026-10-06  
> 页面定位：选择并解释系数推断方法  
> 切换：[English](../../en/guides/inference-modes.md)

## 本页解决什么问题

statgpu 提供多种推断方法，是因为经典低维回归、固定惩罚 GLM，以及经过 L1 正则化选择得到的稀疏模型，面对的是不同的统计问题。

本页主要回答两个问题：

1. **当前已拟合模型应该使用哪一种推断方法？**
2. **报告出的区间究竟对应原始拟合参数、偏差修正后的参数，还是另一次重拟合的参数？**

具体后端内核、内部结果存储方式以及验证证据不属于本用户指南。

## 方法总览

| 已拟合模型 / 场景 | 推断方法 | 解释 |
|---|---|---|
| `LinearRegression` / 共享平方误差 L2/Ridge | 经典或稳健协方差 | nonrobust 使用 t 参考分布，受支持的稳健协方差使用正态参考分布 |
| 普通 `GeneralizedLinearModel`，包括 Gaussian | `m_estimation` | 受支持的系数推断使用正态（z）参考分布 |
| 光滑的非 Gaussian L2 / 无惩罚 GLM | `m_estimation` | 固定惩罚强度下的估计方程推断 |
| Gaussian Lasso / ElasticNet | `debiased` | 去偏 / 去稀疏化系数推断 |
| Gaussian Lasso / ElasticNet | `post_selection_ols` | 在已选择的活跃集上做 OLS/WLS 诊断性重拟合 |
| 受支持的 Gaussian 惩罚模型 | `bootstrap` | 调参配置固定时的残差自助法分布 |
| Gaussian SCAD/MCP；非 Gaussian 重拟合存在下文所述限制 | 显式请求 `oracle` | 活跃集重拟合；普通区间不校正变量选择 |
| 不支持的损失函数 / 惩罚项 / 方法组合 | — | 直接报错，而不是静默换成另一种推断目标 |

完整支持矩阵见 [惩罚 GLM 推断](penalized-glm-inference.md) 与对应模型页。

## Gaussian 线性模型推断

`LinearRegression` 以及共享的惩罚平方误差 L2/Ridge 路径通过 `cov_type` 选择协方差估计方式和参考分布。

常用选项包括：

- `nonrobust`：经典协方差，使用 Student-t 参考分布；
- `hc0`、`hc1`、`hc2`、`hc3`：异方差稳健协方差，使用正态参考分布；
- `hac`：使用 Bartlett 核的 HAC 协方差，参考分布为正态分布。

数值推断在已拟合模型支持的后端上完成。数值阶段结束后，小型结果数组可以转换为 NumPy；这种用于结果整理的转换，并不表示显式 CUDA/Torch 拟合被静默搬回 CPU 重新计算。

普通 `GeneralizedLinearModel` 使用另一条推断路径：受支持的系数推断采用 `m_estimation` 和正态（z）参考分布，**即使设置 `family="gaussian"` 且 `C=0` 也是如此**。因此，即使 Gaussian 模型的系数和标准误相同，其 p 值和区间也可能与 nonrobust `LinearRegression` 不同，小样本时尤其明显。请检查 `model._inference_result.distribution` 及模型专属的协方差约定；选择 Gaussian 分布族并不等于请求 Student-t 推断。

模型专属的协方差选择见对应模型页。

## 惩罚 GLM 的固定惩罚 M-估计

对于受支持的光滑非 Gaussian L2 / 无惩罚模型，

```python
inference_method="auto"
```

会解析为 `m_estimation`。

当 L2 惩罚强度为正时，推断目标是**给定惩罚强度下的惩罚估计方程**；无惩罚拟合对应普通的无惩罚参数。当前非 Gaussian 固定惩罚协方差支持 `nonrobust`、`hc0`、`hc1`；在不支持的路径上请求 HC2、HC3 或 HAC 会直接报错。

### 分析权重

当所选光滑求解器支持分析权重 `sample_weight` 时，拟合与对应的 M-估计使用同一个归一化带权目标：

$$
L_w(\beta)
=
\frac{\sum_i w_i\,\ell_i(\beta)}{\sum_i w_i}.
$$

因此，把所有正的分析权重同时乘上一个常数不会改变统计推断目标。如果某个损失函数并没有定义相应的带权拟合，例如某些不支持带权的 Cox 路径，statgpu 会拒绝请求，而不是静默丢弃权重。

协方差公式和精确的损失函数/惩罚项支持范围见 [惩罚 GLM 推断](penalized-glm-inference.md)。

## 稀疏 Gaussian 推断

对于 Gaussian `Lasso`、`ElasticNet` 以及等价的平方误差 L1/ElasticNet 惩罚模型，三个公开模式对应不同的推断目标。

### `debiased`

去偏（de-biased / de-sparsified）推断先估计设计 Gram 矩阵或协方差矩阵的近似逆，即精度矩阵，再进行一步修正，以减小惩罚估计量的一阶正则化偏差。

当目标是高维模型中的边际系数推断，并且相关理论假设对应用场景合理时，可以使用这一模式。逐节点精度矩阵估计的调参与主模型惩罚参数分开控制；见 [逐节点 Lasso 推断调参迁移](nodewise-alpha-migration.md)。

支持分析权重时，去偏推断与稀疏 Gaussian 拟合使用同一个带权中心化统计问题，因此统一缩放全部正权重不会改变其统计推断目标。

如果开启同时推断，statgpu 使用 max-|Z| 校准，而不是把普通边际区间直接当作同时置信区间。相关控制见 Lasso/ElasticNet 模型文档。

### `post_selection_ols`

`post_selection_ols` 先取得惩罚模型选出的活跃集，然后在**完全相同的活跃集上重新拟合无惩罚 OLS 或 WLS**。

因此必须区分：

- 预测仍使用原来的惩罚 `coef_` / `intercept_`；
- 这一推断模式的系数表属于活跃集上的 OLS/WLS 重拟合；
- 在同一份数据上先选择变量，再计算普通 OLS/WLS 区间，**并不会自动得到一般意义上的选择后推断置信区间**。

因此，更合适的理解是“选择后的诊断性重拟合”或“条件重拟合”，而不是自动校正了变量选择不确定性的推断方法。

<!-- inference-example: post-selection -->
```python
import numpy as np
from statgpu.linear_model import Lasso

rng = np.random.default_rng(7)
X = rng.standard_normal((80, 3))
y = 1.0 + X @ np.array([1.5, 0.0, -0.8]) + rng.normal(scale=0.4, size=80)
model = Lasso(
    alpha=0.1,
    solver="fista",
    compute_inference=True,
    inference_method="post_selection_ols",
    device="cpu",
)
model.fit(X, y)

# 预测参数仍属于原惩罚拟合
prediction_coef = model.coef_
print(model._inference_result.method, prediction_coef.shape)
```

输出为 `post_selection_ols (3,)`。推断结果的 `params` 和 `conf_int` 包含截距及活跃集重拟合结果；预测仍使用 `coef_` 和 `intercept_`。未选中变量在推断结果中以零和 p 值一占位，不能把其零宽区间理解为总体系数确实为零的证据。

历史上的 `cpu_ols` 与 `gpu_ols` 只是 `post_selection_ols` 的弃用兼容别名，并不负责选择计算设备。

### `bootstrap`

Gaussian 残差自助法会保持拟合设计矩阵与调参配置不变。每次重抽样依次执行：

1. 根据已拟合值计算残差；
2. 对残差进行有放回抽样；
3. 构造新的自助法响应变量；
4. 使用相同的调参配置重新拟合同一个惩罚模型；
5. 汇总自助法样本得到的系数分布。

它并不是适用于所有 GLM 分布族的通用自助法。当前支持的残差自助法要求 `sample_weight=None` 且 `cov_type="nonrobust"`；带权残差自助法、稳健/HAC 分块自助法、非 Gaussian 自助法和 Cox 自助法不由这一模式提供。

因此，该结果描述的是固定设计、固定调参配置下的残差自助法不确定性，并不会自动调整调参或变量选择带来的额外不确定性。

<a id="scadmcp-active-set-inference"></a>

## SCAD/MCP 活跃集推断

专用的 `SCADRegression` 与 `MCPRegression` 构造函数不接受
`inference_method`；仅设置 `compute_inference=True` 后拟合会报错。请通过
`PenalizedLinearRegression` 显式选择方法，例如：

<!-- inference-example: gaussian-nonconvex-oracle -->
```python
import numpy as np
from statgpu.linear_model import PenalizedLinearRegression

rng = np.random.default_rng(31)
X = rng.normal(size=(80, 2))
y = 0.5 + X @ np.array([1.5, -0.8]) + rng.normal(scale=0.4, size=80)
model = PenalizedLinearRegression(
    penalty="scad", penalty_kwargs={"a": 3.7}, alpha=0.1,
    device="cpu", solver="fista", compute_inference=True,
    inference_method="oracle",
).fit(X, y)
print(model._inference_result.method)
print(np.round(model._inference_result.params, 3))
```

输出为 `oracle` 和约 `[0.522, 1.497, -0.759]`，首项为截距。MCP 可改用
`penalty="mcp", penalty_kwargs={"gamma": 3.0}`。示例使用高斯响应，
并未消除下文的变量选择和子模型设备限制。

在支持 `inference_method="oracle"` 的模型上，statgpu 以非凸惩罚拟合所选择的活跃集为条件进行推断。`auto` 不会静默选择这种解释，因为“以已选择的变量集合为条件”本身就是一个实质性的推断假设。

当前非 Gaussian oracle 重拟合可能重置原分布族参数，或保留默认正则化，因此成功返回的 `oracle` 结果不一定对应预期模型。不要使用这些结果表进行推断。[oracle 限制与显式重拟合示例](penalized-glm-inference.md#current-non-gaussian-oracle-limitation)说明了如何按指定配置创建独立的诊断模型；它仍不会校正在同一数据上选择变量的影响。oracle 接口会拒绝 GPU 父模型，但子模型默认使用 `device="auto"`；自行重拟合时应显式指定设备。

## 交叉验证后的推断

CV 选择与系数推断分成两个阶段：

```text
在各数据折上拟合候选模型
    -> 选择调参值
    -> 在全部观测上重拟合所选模型
    -> 只对最终重拟合执行一次推断
```

因此，除非某种方法明确说明做了额外修正，否则报告的不确定性都应解释为**以 CV 已经选定的调参配置为条件**，不会自动调整调参选择带来的额外不确定性。

选择与最终重拟合的语义见 [交叉验证](cross-validation.md)。

## 推断方法不负责选择硬件

`inference_method` 选择统计推断程序，`device` 决定该程序在支持范围内使用的计算硬件。

- `device="cpu"` 请求 NumPy CPU；
- `device="cuda"` 要求使用 CuPy CUDA 路径；
- `device="torch"` 要求使用 Torch CUDA 路径；
- `device="auto"` 可以在可用后端之间自动选择。

不支持的显式后端/推断方法组合会报错，而不会把 CPU 结果伪装成用户请求的加速器结果。某些推断方法支持的后端范围可能比其基础估计器更窄；如果计算设备很重要，应查看对应方法的专属文档。

## 如何理解拟合参数与推断参数

在普通无惩罚模型中，两者通常指向同一个估计量；惩罚模型中的拟合后推断方法则可能不同：

- `post_selection_ols` 报告活跃集上的无惩罚重拟合结果，但预测仍使用原惩罚拟合；
- `debiased` 报告偏差修正后的推断参数，但预测仍使用原惩罚拟合；
- `bootstrap` 描述固定调参配置下重复惩罚重拟合所形成的分布。

不要只根据系数表的形状猜测推断目标。`inference_method_` 和 `inference_target_` 在有值时记录实际方法与目标，但当前 `post_selection_ols` 会把这两项保留为 `None`。此时应查看 `model._inference_result.method` 及其元数据，了解实际方法、所选特征索引与重拟合信息。公开字段为空并不表示未执行推断；统计含义仍以相应模型页为准。

## 如何选择方法

可以按以下顺序判断：

1. **普通低维 Gaussian 模型**：使用模型本身的经典或稳健协方差。
2. **光滑非 Gaussian L2 / 无惩罚 GLM**：优先从 `inference_method="auto"` 开始；受支持时会解析为固定惩罚 M-估计。
3. **稀疏 Gaussian L1/ElasticNet**：若高维系数推断的假设合理，使用 `debiased`；只有当目标本来就是活跃集诊断/重拟合时才使用 `post_selection_ols`。
4. **需要 Gaussian 固定调参配置下的重抽样视角**：只在公开支持范围内使用 `bootstrap`。
5. **SCAD/MCP 活跃集解释**：只有在确实需要这一条件推断目标且模型支持时，才显式请求 `oracle`。

## 相关文档

- [惩罚 GLM 推断](penalized-glm-inference.md) — 公式、支持矩阵与详细统计推断目标
- [交叉验证](cross-validation.md) — 调参选择与最终重拟合
- [逐节点 Lasso 推断调参迁移](nodewise-alpha-migration.md) — `nodewise_alpha`
- [设备与 GPU 内存](device-and-memory.md) — 后端/设备语义
- [Lasso](../models/lasso.md) 与 [ElasticNet](../models/elastic-net.md) — 稀疏推断的模型专属说明
