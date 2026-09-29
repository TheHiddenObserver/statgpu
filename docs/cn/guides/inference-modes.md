# 推断模式

> 语言：中文  
> 最后更新：2026-09-17  
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
| Gaussian 线性模型 / Ridge | 经典或稳健协方差 | 对已拟合线性模型的系数做推断 |
| 光滑的非 Gaussian L2 / 无惩罚 GLM | `m_estimation` | 固定惩罚强度下的估计方程推断 |
| Gaussian Lasso / ElasticNet | `debiased` | 去偏 / 去稀疏化系数推断 |
| Gaussian Lasso / ElasticNet | `post_selection_ols` | 在已选择的活跃集上做 OLS/WLS 诊断性重拟合 |
| 受支持的 Gaussian 惩罚模型 | `bootstrap` | 调参配置固定时的残差自助法分布 |
| 受支持的 SCAD/MCP | 显式请求 `oracle` | 以已选择的活跃集为条件进行推断 |
| 不支持的损失函数 / 惩罚项 / 方法组合 | — | 直接报错，而不是静默换成另一种推断目标 |

完整支持矩阵见 [惩罚 GLM 推断](penalized-glm-inference.md) 与对应模型页。

## Gaussian 线性模型推断

普通 Gaussian 线性模型以及共享的平方误差 L2/Ridge 路径通过 `cov_type` 选择协方差估计方式和参考分布。

常用选项包括：

- `nonrobust`：经典协方差，使用 Student-t 参考分布；
- `hc0`、`hc1`、`hc2`、`hc3`：异方差稳健协方差，使用正态参考分布；
- `hac`：使用 Bartlett 核的 HAC 协方差，参考分布为正态分布。

数值推断在已拟合模型支持的后端上完成。数值阶段结束后，小型结果数组可以转换为 NumPy；这种用于结果整理的转换，并不表示显式 CUDA/Torch 拟合被静默搬回 CPU 重新计算。

模型专属的协方差选择见对应模型页。

## 惩罚 GLM 的固定惩罚 M-估计

对于受支持的光滑非 Gaussian L2 / 无惩罚模型，

```python
inference_method="auto"
```

会解析为 `m_estimation`。

当 L2 惩罚强度为正时，推断目标是**给定惩罚强度下的惩罚估计方程**；无惩罚拟合对应普通的无惩罚参数。当前非 Gaussian 固定惩罚协方差支持 `nonrobust`、`hc0`、`hc1`；在不支持的路径上请求 HC2、HC3 或 HAC 会直接报错。

### 解析权重

当所选光滑求解器支持解析 `sample_weight` 时，拟合与对应的 M-估计使用同一个归一化带权目标：

$$
L_w(\beta)
=
\frac{\sum_i w_i\,\ell_i(\beta)}{\sum_i w_i}.
$$

因此，把所有正的解析权重同时乘上一个常数不会改变统计推断目标。如果某个损失函数并没有定义相应的带权拟合，例如某些不支持带权的 Cox 路径，statgpu 会拒绝请求，而不是静默丢弃权重。

协方差公式和精确的损失函数/惩罚项支持范围见 [惩罚 GLM 推断](penalized-glm-inference.md)。

## 稀疏 Gaussian 推断

对于 Gaussian `Lasso`、`ElasticNet` 以及等价的平方误差 L1/ElasticNet 惩罚模型，三个公开模式对应不同的推断目标。

### `debiased`

去偏（de-biased / de-sparsified）推断通过估计设计矩阵精度矩阵的近似逆，并进行一步修正，来减小惩罚估计量的一阶正则化偏差。

当目标是高维模型中的边际系数推断，并且相关理论假设对应用场景合理时，可以使用这一模式。逐节点精度矩阵估计的调参与主模型惩罚参数分开控制；见 [逐节点 Lasso 推断调参迁移](nodewise-alpha-migration.md)。

支持解析权重时，去偏推断与稀疏 Gaussian 拟合使用同一个带权中心化统计问题，因此统一缩放全部正权重不会改变其统计推断目标。

如果开启同时推断，statgpu 使用 max-|Z| 校准，而不是把普通边际区间直接当作同时置信区间。相关控制见 Lasso/ElasticNet 模型文档。

### `post_selection_ols`

`post_selection_ols` 先取得惩罚模型选出的活跃集，然后在**完全相同的活跃集上重新拟合无惩罚 OLS 或 WLS**。

因此必须区分：

- 预测仍使用原来的惩罚 `coef_` / `intercept_`；
- 这一推断模式的系数表属于活跃集上的 OLS/WLS 重拟合；
- 在同一份数据上先选择变量，再计算普通 OLS/WLS 区间，**并不会自动得到一般意义上的选择后推断置信区间**。

因此，更合适的理解是“选择后的诊断性重拟合”或“条件重拟合”，而不是自动校正了变量选择不确定性的推断方法。

```python
from statgpu.linear_model import Lasso

model = Lasso(
    alpha=0.1,
    solver="fista",
    compute_inference=True,
    inference_method="post_selection_ols",
)
model.fit(X, y)

# 预测参数仍属于原惩罚拟合
prediction_coef = model.coef_
```

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

## SCAD/MCP 活跃集推断

在支持 `inference_method="oracle"` 的模型上，statgpu 以非凸惩罚拟合所选择的活跃集为条件进行推断。`auto` 不会静默选择这种解释，因为“以已选择的变量集合为条件”本身就是一个实质性的推断假设。

请求 `oracle` 前，请查看 [惩罚 GLM 推断](penalized-glm-inference.md) 中当前模型和后端的支持范围。

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

不要只根据系数表的形状猜测推断目标。需要区分时，应查看 `inference_method_`、`inference_target_` 与相应模型页。

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
