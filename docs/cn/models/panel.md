# 面板模型

> 语言：中文  
> 最后更新：2026-09-13  
> 切换：[英文版](../../en/models/panel.md)

`statgpu.panel` 提供六类常用面板数据估计量，覆盖合并回归、固定效应、组间估计、一阶差分、随机效应和 Fama–MacBeth 横截面回归。这些估计量可以作用于相同或相关的基础面板模型，但通过不同的数据变换、识别来源和附加假设得到各自的参数估计。

可以按下面的统计结构来区分：

| 估计量 | 统计视角 | 主要识别来源 / 额外条件 |
|---|---|---|
| [PanelOLS](../panel/panel-ols.md) | 个体效应和/或时间效应被视为固定但未知的干扰参数。 | 使用去除所选固定效应后剩余的变异；不要求固定效应与解释变量的历史路径正交。 |
| [FirstDifferenceOLS](../panel/first-difference-ols.md) | 从含固定个体效应的模型出发，通过差分消除不随时间变化的个体效应。 | 使用同一个体在相邻已观测时期之间的变化。 |
| [RandomEffects](../panel/random-effects.md) | 个体效应被建模为随机成分。 | 经典随机效应解释要求随机效应与解释变量历史正交，例如 $E(a_i\mid X_i)=0$。 |
| [BetweenOLS](../panel/between-ols.md) | 在个体内对 $X$ 和 $y$ 取均值，使每个个体对应一条均值观测。 | 使用个体之间的变异；若要恢复基础面板模型中相同的结构斜率，需要均值后的复合误差与均值后的解释变量正交。 |
| [PooledOLS](../panel/pooled-ols.md) | 所有堆叠观测共享同一个条件均值关系。 | 使用全部堆叠变异；合并回归误差必须对解释变量外生。 |
| [FamaMacBeth](../panel/fama-macbeth.md) | 每个时期分别进行一次横截面回归。 | 对各时期的斜率估计取平均，并根据这些斜率的时间序列变异进行推断。 |

## 文档导航

- [面板模型架构](../panel/architecture.md) — `BasePanelModel`、各估计量的数据变换与回归问题、共享数值线性代数、协方差/推断、诊断以及拟合生命周期。
- [协方差](../panel/covariance.md) — 非稳健、HC、聚类、HAC 与 Driscoll–Kraay 等协方差估计的定义。
- [拟合统计量](../panel/fit-statistics.md) — 组内、组间、总体 $R^2$，调整 $R^2$ 与模型 F 统计量等。
- [模型诊断](../panel/diagnostics.md) — Hausman、pooling F、Breusch–Pagan LM 等诊断方法。

每个模型页面分别说明**统计模型与识别假设**以及**数值估计方法**。这些识别假设决定所报告系数的经济计量解释，而数值部分说明 statgpu 如何计算相应估计量。

这六类模型都可以通过 `device` 使用 NumPy CPU、CuPy CUDA 或 Torch CUDA。每个模型页面都给出了 CPU/GPU 和公式接口示例；显式指定 `device="cuda"` 或 `device="torch"` 时，模型会使用相应计算后端。

面板模型之间共享的实现结构、数据变换、数值组件和推断层见 [面板模型架构](../panel/architecture.md)。