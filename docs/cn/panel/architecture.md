# 面板模型架构

> 语言：中文  
> 最后更新：2026-09-13  
> 切换：[英文版](../../en/panel/architecture.md)

本页只描述 **statgpu 当前已经实现的面板模型架构**。模型选择、识别假设与用户入口见 [面板模型总览](../models/panel.md)；具体估计器的统计定义与 API 见各模型页面。

## 1. 架构原则

面板模型面向用户的公共入口是估计器，而不是损失函数。用户构造 `PanelOLS`、`RandomEffects`、`PooledOLS`、`BetweenOLS`、`FirstDifferenceOLS` 或 `FamaMacBeth`，然后调用 `.fit()`。

六个估计器都以 `BasePanelModel(BaseEstimator)` 为共享基础，但 **`BasePanelModel` 只提供统计上中性的共享基础设施，并不决定某个面板估计器应采用哪一种经济计量变换，也不替具体估计器定义拟合空间。**

当前实现也**没有 `PanelLoss` 这一层，面板拟合并不通过通用的 `LossBase + Penalty + Solver` 流程组织**。当前面板模型的核心分工是：

1. 共享输入处理、元数据、计算后端、数值稳定性和结果生命周期；
2. 具体估计器定义自己的统计拟合空间；
3. 在该拟合空间中调用共享或模型专用的数值估计组件；
4. 再进入适用的协方差、推断、诊断和模型特有的拟合后处理。

## 2. 当前运行职责图

```text
用户
  │
  │  model = PanelEstimator(...)
  │  model.fit(...)
  ▼
具体面板估计器
  │
  ├── BasePanelModel 共享基础设施
  │     ├── 事务式拟合与拟合状态生命周期
  │     ├── 公式解析与附属数组对齐
  │     ├── 计算后端 / 设备的数值准备辅助函数
  │     ├── PanelIndexInfo：个体 / 时间 / 平衡性 / 原始顺序元数据
  │     ├── 共享线性预测辅助函数
  │     └── 共享摘要构造与基于残差 OLS 的推断收尾
  │
  ▼
估计器特定的拟合空间构造
  │
  ├── PooledOLS          → 原始堆叠设计
  ├── PanelOLS           → 个体 / 时间 / 双向去均值
  ├── BetweenOLS         → 个体均值
  ├── FirstDifferenceOLS → 个体内一阶差分
  ├── RandomEffects      → 辅助回归 + 方差分量 + 准去均值
  └── FamaMacBeth        → 分期横截面回归设计
  │
  ▼
数值估计
  │
  ├── 共享面板数值策略
  │     ├── `_linalg.py`：SVD / 秩判定 / 最小二乘策略
  │     ├── `_intercept.py`：常数列与响应整体水平的数值保护
  │     └── `_reductions.py`：稳定的分组归约
  │
  └── 具体估计器围绕这些基础组件组织自己的计算步骤
  │
  ▼
拟合后统计层
  │
  ├── 基于残差 OLS 的协方差调度（`_covariance.py`，适用时）
  ├── 系数推断收尾（`BasePanelModel`，适用时）
  ├── 拟合统计量与模型设定诊断
  │     （`_diagnostic_context.py`、`_diagnostics.py`）
  ├── 模型特有状态与效应恢复
  └── predict() / summary()
```

这是一张**职责图**，并不表示每个估计器都必须按完全相同的顺序调用每一个辅助函数。具体估计器会按需要复用共享基础组件。

尤其是：

- `FamaMacBeth` 保留自己的计算后端准备、分期结果聚合和系数序列协方差；
- `PanelOLS` 在自己的拟合路径中完成固定效应和时间效应的恢复；
- `RandomEffects` 自己组织组间/组内辅助回归、Swamy–Arora 方差分量估计和准去均值变换。

最重要的边界是：**共享层复用“怎样处理输入和元数据、怎样稳定求解面板最小二乘问题、怎样组织共同的推断结果”；具体估计器决定“要在哪一个统计拟合空间中估计什么”。**

## 3. 共享组件的职责

| 组件 | 当前职责 |
|---|---|
| `BasePanelModel` | 事务式拟合生命周期、公式和附属数组对齐、计算后端数值准备辅助函数、面板元数据、共享预测、基于残差 OLS 的推断收尾以及摘要构造。 |
| `_formula.py` | 标准 R 公式、fixest 管道语法、`EntityEffects`/`TimeEffects` 标记、附属数组对齐以及预测设计矩阵重建。 |
| `_results.py` | `PanelIndexInfo`、`PanelFitStatistics`、`PanelTestResult` 等结构化元数据和结果容器。 |
| `_linalg.py` / `_intercept.py` / `_reductions.py` | 按秩处理的最小二乘、常数列与响应整体水平的稳定化、分批的分期求解以及稳定的分组归约。 |
| `_covariance.py` | 非稳健、HC、聚类、HAC、Driscoll–Kraay 等协方差估计的共享实现与调度。 |
| `_diagnostic_context.py` / `_diagnostics.py` | 拟合统计量、自由度定义，以及 Hausman、pooling F、Breusch–Pagan LM 等面板模型诊断。 |
| 具体估计器模块 | 定义各估计器的统计变换、辅助估计、模型特有状态，以及哪些共享推断和协方差计算适用于该模型。 |

因此，`BasePanelModel` 不是一个“万能面板算法”。它不会自动执行组内变换，也不会替 `RandomEffects` 估计方差分量。它提供可复用的基础组件；`PanelOLS.fit()`、`RandomEffects.fit()` 等具体实现决定这些组件在什么统计结构下被调用。

## 4. 六类估计器的当前计算路径

| 估计器 | 拟合空间 / 核心变换 | 数值估计 | 推断路径 |
|---|---|---|---|
| `PooledOLS` | 原始堆叠设计，并自动加入截距 | 合并 OLS | 基于残差 OLS 的协方差 + 共享推断；可计算合并模型的拟合统计量与 BP-LM 诊断 |
| `PanelOLS` | 无效应时为原始层级回归；有个体/时间效应时进行单向或双向去均值 | 变换后的 OLS | 在变换后拟合空间中计算协方差和共享推断；效应恢复、面板拟合统计量与 pooling-F 相关计算保留在模型实现中 |
| `BetweenOLS` | 对每个个体分别计算 $X$、$y$ 的均值 | 个体均值 OLS | 在个体均值拟合空间中计算协方差和共享推断 |
| `FirstDifferenceOLS` | 个体内按时间排序后取一阶差分 | 差分 OLS | 在差分后的拟合空间中计算协方差和共享推断 |
| `RandomEffects` | 组间/组内辅助回归 → Swamy–Arora 方差分量 → 准去均值 | 可行 GLS，可表示为准去均值数据上的 OLS | 在准去均值拟合空间上使用共享的基于残差 OLS 的协方差和推断，同时保留 `theta_` 与方差分量 |
| `FamaMacBeth` | 每个时期单独构造一次横截面回归 | 分期 OLS / 分批 OLS，再聚合 $\hat\beta_t$ | **不使用基于残差 OLS 的协方差注册表**；协方差由分期系数序列 $\{\hat\beta_t\}$ 决定，因此保留模型专用实现 |

这个表解释了为什么不能把六个估计器简化成“一次通用 OLS 调用”。它们共享大量数值和推断基础设施，但**拟合空间的定义本身就是估计器统计含义的一部分。**

## 5. 固定效应示例：统计变换发生在求解之前

例如个体固定效应模型

$$
y_{it}=x_{it}^\top\beta+\alpha_i+\varepsilon_{it}
$$

先通过组内变换构造

$$
\widetilde y_{it}=y_{it}-\bar y_i,
\qquad
\widetilde x_{it}=x_{it}-\bar x_i,
$$

然后当前 `PanelOLS` 在变换后的拟合空间中求解

$$
\hat\beta
=
\arg\min_\beta
\sum_{i,t}
\left(\widetilde y_{it}-\widetilde x_{it}^\top\beta\right)^2.
$$

求得斜率后，面板模型的计算还没有结束：实现仍需要恢复个体/时间效应、确定效应空间的秩和残差自由度，再在变换后的设计矩阵上计算所选协方差、系数推断和面板模型特有的拟合统计量。

因此，当前完整的估计流程同时包含**拟合空间构造、数值求解和拟合后面板推断**。

## 6. 计算后端与数值策略

六类估计器都支持通过 `device` 选择 NumPy CPU、CuPy CUDA 或 Torch CUDA。显式请求 `device="cuda"` 或 `device="torch"`，但对应计算后端不可用时会直接报错，而不是切换到 CPU。

共享的面板最小二乘数值策略对极端 float64 系数尺度采用保守失败策略。若响应投影存在消去误差或动态范围风险，会使用维护中的稳定归约方法；当设计矩阵满秩且存在可安全利用的精确常数列时，可以先移除响应变量中的公共整体水平。若某个非常数系数已经低于 float64 投影能够可靠分辨的尺度，并且候选解显著违反最小二乘的一阶最优性条件，statgpu 会抛出 `FloatingPointError`，而不是发布一个数值有限但不可靠的系数。

`FamaMacBeth` 对每个时期使用同样的数值可靠性原则，并将单期系数精度不足与真正的秩亏分开报告。

共享辅助函数是可复用的基础组件，而不是强制调用链。例如 `FamaMacBeth` 有专用的计算后端准备逻辑；`RandomEffects` 与精确常数列路径还会调用 `_intercept.py` 中的数值稳定化辅助函数。因此，前面的架构图表达的是职责边界，而不是逐函数调用轨迹。

## 7. 协方差、诊断与结果发布

对于适用的、基于残差 OLS 的变换后拟合空间，`BasePanelModel._panel_store_ols_inference()` 使用 `_covariance.py` 中的统一调度计算协方差，并统一发布系数标准误、统计量、p 值和置信区间。

面板模型特有的拟合统计量和模型设定诊断由 `_diagnostic_context.py` 与 `_diagnostics.py` 提供。`FamaMacBeth` 是重要例外：它的协方差基于分期系数序列，而不是基于残差的 OLS sandwich 形式，因此保持模型专用实现。

结构化结果的基础容器由 `_results.py` 提供，包括：

- `PanelIndexInfo`：个体/时间编码、标签、观测数、平衡/非平衡状态以及原始观测顺序等元数据；
- `PanelFitStatistics`：组内/组间/总体 $R^2$、调整 $R^2$、模型 F 统计量等；
- `PanelTestResult`：诊断统计量、p 值、参考分布、自由度、适用性以及附加元数据。

## 8. 拟合生命周期

Panel 的 `fit()` 采用事务式生命周期。新的拟合尝试会先使上一轮的拟合与推断状态失效；如果新拟合在任何阶段抛出异常，已经部分写入的新结果会被清除，然后重新抛出原异常。

因此，重新拟合失败后，`predict()` 与 `summary()` 会把估计器视为未拟合状态，而不会继续暴露上一份数据的系数或推断结果，也不会暴露本次未完成拟合留下的中间结果。

基于公式的预测同样要求保持输入行数。如果建模变量缺失导致 Patsy 删除某个预测行，或者公式变换产生 NaN/Inf，预测会明确报错，而不会返回行数变少或包含非有限值的结果。

## 9. 与通用优化框架的当前边界

当前面板模型不通过通用的 `LossBase + Penalty + Solver` 流程组织拟合。通用优化框架的当前接口与职责见 [损失函数 × 惩罚项 × 求解器框架](../guides/loss-penalty-solver-framework.md)。

本页只记录当前实现，不定义未来的面板优化架构。