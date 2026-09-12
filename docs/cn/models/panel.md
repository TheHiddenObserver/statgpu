# 面板模型

> 语言：中文  
> 最后更新：2026-09-13  
> 切换：[英文版](../../en/models/panel.md)

`statgpu.panel` 提供六类面板数据估计器。它们不应理解成六个彼此无关的数据生成过程：多个估计器可以从同一个基础面板模型出发，只是对未观测异质性的处理方式、用于识别系数的变异来源或目标参数有所不同。

可以按下面的统计结构来区分：

| 估计器 | 统计视角 | 主要识别来源 / 额外条件 |
|---|---|---|
| [PanelOLS](../panel/panel-ols.md) | 个体效应和/或时间效应被视为固定但未知的干扰参数。 | 使用去除所选固定效应后剩余的变异；不要求固定效应与解释变量的历史路径正交。 |
| [FirstDifferenceOLS](../panel/first-difference-ols.md) | 可以从同一个含固定个体效应的模型出发，通过差分消除不随时间变化的个体效应。 | 使用同一个体在相邻已观测时期之间的变化。 |
| [RandomEffects](../panel/random-effects.md) | 个体效应被建模为随机成分。 | 经典随机效应解释要求随机效应与解释变量历史正交，例如 $E(a_i\mid X_i)=0$。 |
| [BetweenOLS](../panel/between-ols.md) | 先在个体内取均值，使每个个体只保留一条观测。 | 使用个体之间的变异；若要恢复基础面板模型中相同的结构斜率，需要均值后的复合误差与均值后的解释变量正交。 |
| [PooledOLS](../panel/pooled-ols.md) | 所有堆叠观测共享同一个条件均值关系。 | 使用全部堆叠变异；合并回归误差必须对解释变量外生。 |
| [FamaMacBeth](../panel/fama-macbeth.md) | 每个时期分别进行一次横截面回归。 | 目标是对各时期保留下来的斜率取平均，并根据这些斜率的时间序列变异进行推断。 |

## 文档导航

- [面板模型架构](../panel/architecture.md) — 当前 `BasePanelModel`、各估计器拟合空间、共享数值线性代数、协方差/推断、诊断以及拟合生命周期之间的职责边界。
- [协方差](../panel/covariance.md) — 非稳健、HC、聚类、HAC 与 Driscoll–Kraay 等协方差估计的定义。
- [拟合统计量](../panel/fit-statistics.md) — 组内、组间、总体 $R^2$，调整 $R^2$ 与模型 F 统计量等。
- [模型诊断](../panel/diagnostics.md) — Hausman、pooling F、Breusch–Pagan LM 等诊断方法。

每个估计器页面都会把**统计模型与识别假设**和**数值估计方法**分开说明。前者回答“在什么条件下，所报告的系数具有通常的面板经济计量解释”；软件本身仍然可以机械地计算某个估计量，因此这些统计假设需要由用户结合实际问题判断，而不是由 `.fit()` 自动验证。

六类估计器都可以通过 `device` 使用 NumPy CPU、CuPy CUDA 或 Torch CUDA。每个模型页面都给出了 CPU/GPU 和公式接口示例。若显式指定 `device="cuda"` 或 `device="torch"`，但对应计算后端不可用，statgpu 会直接报错，而不是自动切换到 CPU。

若需要理解不同面板估计器之间共享了哪些实现、哪些统计步骤必须保留在具体估计器中，以及当前面板模型为什么不属于通用的 `LossBase + Penalty + Solver` 拟合路径，请直接阅读 [面板模型架构](../panel/architecture.md)。