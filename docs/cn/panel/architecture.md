# 面板模型架构

> 语言：中文  
> 最后更新：2026-09-13  
> 切换：[英文版](../../en/panel/architecture.md)

本页说明 **statgpu 当前面板模型的实现架构**：公共模型类如何共享 `BasePanelModel` 基础设施，各估计量如何构造用于估计的数据，以及数值求解、协方差、推断和诊断如何衔接。模型选择、识别假设与用户入口见 [面板模型总览](../models/panel.md)；各估计量的统计定义与 API 见相应模型页面。

## 1. 总体架构

用户通过 `PanelOLS`、`RandomEffects`、`PooledOLS`、`BetweenOLS`、`FirstDifferenceOLS` 或 `FamaMacBeth` 构造具体模型对象并调用 `.fit()`。

这六类模型共享 `BasePanelModel(BaseEstimator)`。`BasePanelModel` 负责公式与输入对齐、面板索引元数据、计算后端准备、拟合状态管理、共享预测、部分推断收尾和摘要构造；具体模型类负责定义相应估计量的数据变换、辅助估计和模型特有结果。

当前计算流程可以概括为：

1. 解析输入、公式、个体/时间索引和计算后端；
2. 根据估计量构造用于估计的设计矩阵与响应；
3. 调用共享数值组件或模型专用求解步骤完成 OLS、GLS 或分期回归；
4. 计算适用的协方差、系数推断、拟合统计量和模型诊断；
5. 发布模型特有状态、预测和摘要结果。

## 2. 当前运行职责图

```text
用户
  │
  │  model = PanelEstimator(...)
  │  model.fit(...)
  ▼
具体面板模型类
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
各估计量所需的数据变换与回归问题构造
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
  └── 具体模型类围绕这些基础组件组织自己的计算步骤
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

共享基础组件由各模型按需要组合使用。例如：

- `PanelOLS` 负责组内/双向变换以及固定效应和时间效应恢复；
- `RandomEffects` 负责组间/组内辅助回归、Swamy–Arora 方差分量估计和准去均值；
- `FamaMacBeth` 负责分期回归、分期结果聚合和基于系数序列的协方差。

因此，共享层主要解决输入、元数据、数值线性代数和通用推断结果的复用，而具体模型决定采用怎样的数据变换，以及最终在哪一组设计矩阵与响应上估计参数。

## 3. 共享组件的职责

| 组件 | 当前职责 |
|---|---|
| `BasePanelModel` | 事务式拟合生命周期、公式和附属数组对齐、计算后端数值准备辅助函数、面板元数据、共享预测、基于残差 OLS 的推断收尾以及摘要构造。 |
| `_formula.py` | 标准 R 公式、fixest 管道语法、`EntityEffects`/`TimeEffects` 标记、附属数组对齐以及预测设计矩阵重建。 |
| `_results.py` | `PanelIndexInfo`、`PanelFitStatistics`、`PanelTestResult` 等结构化元数据和结果容器。 |
| `_linalg.py` / `_intercept.py` / `_reductions.py` | 按秩处理的最小二乘、常数列与响应整体水平的稳定化、分批的分期求解以及稳定的分组归约。 |
| `_covariance.py` | 非稳健、HC、聚类、HAC、Driscoll–Kraay 等协方差估计的共享实现与调度。 |
| `_diagnostic_context.py` / `_diagnostics.py` | 拟合统计量、自由度定义，以及 Hausman、pooling F、Breusch–Pagan LM 等面板模型诊断。 |
| 具体模型模块 | 定义各估计量的数据变换、辅助估计、模型特有状态，以及相应的推断与协方差计算。 |

`BasePanelModel` 提供跨模型共享的生命周期和统计基础设施；`PanelOLS.fit()`、`RandomEffects.fit()`、`FamaMacBeth.fit()` 等具体实现则组合这些组件，形成各自的估计流程。

## 4. 六类估计量的当前计算路径

| 估计量 | 用于估计的数据 / 核心变换 | 数值估计 | 推断路径 |
|---|---|---|---|
| `PooledOLS` | 原始堆叠设计，并自动加入截距 | 合并 OLS | 基于残差 OLS 的协方差 + 共享推断；同时计算合并模型的拟合统计量与 BP-LM 诊断 |
| `PanelOLS` | 无效应时使用原始层级数据；有个体/时间效应时进行单向或双向去均值 | 变换后的 OLS | 基于变换后的设计矩阵和残差计算协方差与共享推断；模型同时完成效应恢复、面板拟合统计量与 pooling-F 相关计算 |
| `BetweenOLS` | 对每个个体分别计算 $X$、$y$ 的均值 | 个体均值 OLS | 基于个体均值回归的设计矩阵和残差计算协方差与共享推断 |
| `FirstDifferenceOLS` | 个体内按时间排序后取一阶差分 | 差分 OLS | 基于差分后的设计矩阵和残差计算协方差与共享推断 |
| `RandomEffects` | 组间/组内辅助回归 → Swamy–Arora 方差分量 → 准去均值 | 可行 GLS，可表示为准去均值数据上的 OLS | 基于准去均值后的设计矩阵和残差使用共享协方差与推断，同时发布 `theta_` 与方差分量 |
| `FamaMacBeth` | 每个时期单独构造一次横截面回归 | 分期 OLS / 分批 OLS，再聚合 $\hat\beta_t$ | 根据分期系数序列 $\{\hat\beta_t\}$ 计算协方差，由 `FamaMacBeth` 路径专门完成 |

六类估计量共享数值线性代数和部分推断基础设施，同时通过各自的数据变换和辅助估计实现不同的统计定义。

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

然后 `PanelOLS` 基于变换后的 $\widetilde X$ 与 $\widetilde y$ 求解

$$
\hat\beta
=
\arg\min_\beta
\sum_{i,t}
\left(\widetilde y_{it}-\widetilde x_{it}^\top\beta\right)^2.
$$

斜率估计完成后，实现继续恢复个体/时间效应、确定效应空间的秩和残差自由度，并基于变换后的设计矩阵和残差计算所选协方差、系数推断和面板模型特有的拟合统计量。

因此，当前完整的估计流程包含**数据变换与估计问题构造、数值求解和拟合后面板推断**。

## 6. 计算后端与数值策略

这六类模型都支持通过 `device` 选择 NumPy CPU、CuPy CUDA 或 Torch CUDA。显式请求 `device="cuda"` 或 `device="torch"` 时，数值计算使用相应后端；对应后端不可用时会直接报错。

共享的面板最小二乘策略包含针对极端 float64 尺度的数值可靠性检查。响应投影存在消去误差或动态范围风险时，会使用维护中的稳定归约方法；当设计矩阵满秩且存在可安全利用的精确常数列时，可以先移除响应变量中的公共整体水平。若某个非常数系数已经低于 float64 投影能够可靠分辨的尺度，并且候选解显著违反最小二乘的一阶最优性条件，statgpu 会抛出 `FloatingPointError`，避免发布无法可靠验证的系数结果。

`FamaMacBeth` 对每个时期使用同样的数值可靠性原则，并分别识别单期系数精度不足和真正的秩亏。

各模型按需要组合共享辅助函数。例如 `FamaMacBeth` 使用专用的计算后端准备逻辑；`RandomEffects` 与精确常数列路径会调用 `_intercept.py` 中的数值稳定化辅助函数。

## 7. 协方差、诊断与结果发布

对于经过数据变换后仍可写成残差 OLS 的模型，`BasePanelModel._panel_store_ols_inference()` 使用 `_covariance.py` 中的统一调度计算协方差，并发布系数标准误、统计量、p 值和置信区间。

面板模型特有的拟合统计量和模型设定诊断由 `_diagnostic_context.py` 与 `_diagnostics.py` 提供。`FamaMacBeth` 则直接基于分期系数序列计算协方差。

结构化结果的基础容器由 `_results.py` 提供，包括：

- `PanelIndexInfo`：个体/时间编码、标签、观测数、平衡/非平衡状态以及原始观测顺序等元数据；
- `PanelFitStatistics`：组内/组间/总体 $R^2$、调整 $R^2$、模型 F 统计量等；
- `PanelTestResult`：诊断统计量、p 值、参考分布、自由度、适用性以及附加元数据。

## 8. 拟合生命周期

Panel 的 `fit()` 采用事务式生命周期。每次拟合开始时都会建立新的拟合状态；若拟合中途发生异常，已经写入的部分结果会被清理。成功拟合后，`predict()`、`summary()` 和推断属性统一读取当前这一次拟合发布的状态。

基于公式的预测保持输入行对齐：如果建模变量缺失导致 Patsy 删除预测行，或者公式变换产生 NaN/Inf，预测会明确报错，从而避免返回与输入行不一致的结果。

## 9. 与通用优化框架的关系

当前面板模型以**面板数据变换 + OLS/GLS/分期回归 + 面板专用推断**组织计算，并直接复用上述数值与统计组件。通用的 `LossBase + Penalty + Solver` 架构用于另一类基于显式目标函数组合的模型路径，见 [损失函数 × 惩罚项 × 求解器框架](../guides/loss-penalty-solver-framework.md)。

当前 Panel 路径直接组织面板变换和相应回归计算，因此无需额外构造 `PanelLoss`。