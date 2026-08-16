# 面板模型

> 语言：中文  
> 最后更新：2026-08-16  
> 切换：[English](../../en/models/panel.md)

`statgpu.panel` 提供六类面板估计器。它们不应理解成六个彼此无关的数据生成过程。多个 estimator 可以从同一个基础 panel model 出发，只是在对未观测异质性的假设、用于识别 coefficient 的 variation，或目标参数上有所不同。

可以按下面的统计结构来区分：

| 估计器 | 统计视角 | 主要识别来源 / 额外条件 |
|---|---|---|
| [PanelOLS](../panel/panel-ols.md) | entity 和/或 time effects 被视为 fixed but unknown nuisance parameters。 | 使用去除所选 fixed effects 后剩余的 variation；不要求 fixed effects 与 regressor history 正交。 |
| [FirstDifferenceOLS](../panel/first-difference-ols.md) | 可以从同一个 fixed-parameter entity-effect model 出发，通过差分消除 time-invariant entity effect。 | 使用同一 entity 在相邻已观测时期之间的变化。 |
| [RandomEffects](../panel/random-effects.md) | entity effect 被建模为随机成分。 | 经典 RE 解释要求 random effect 与 regressor history 正交，例如 $E(a_i\mid X_i)=0$。 |
| [BetweenOLS](../panel/between-ols.md) | 将 panel 在 entity 内取平均，得到每个 entity 一条观测。 | 使用 entity 之间的 variation；若要恢复基础 panel model 中相同的 structural slope，需要 averaged composite error 与 averaged regressors 正交。 |
| [PooledOLS](../panel/pooled-ols.md) | 所有 stacked observations 共享一个公共 conditional-mean relationship。 | 使用全部 stacked variation；combined regression error 必须对 regressors 外生。 |
| [FamaMacBeth](../panel/fama-macbeth.md) | 每个 time period 有自己的 cross-sectional regression。 | 目标是保留时期的 period-specific slope 平均值，并根据这些 slope 的 time-series variation 做 inference。 |

每个 estimator 页面现在都会把 **statistical model 与 identification assumptions** 和 **numerical estimator** 分开说明。前者回答“什么条件下 coefficient 具有通常的 panel-econometric interpretation”；软件本身仍然可以机械地计算 estimator，因此这些统计假设需要由用户结合实际问题判断，而不是由 `.fit()` 自动验证。

共享统计定义见 [covariance](../panel/covariance.md)、[fit statistics](../panel/fit-statistics.md) 与 [diagnostics](../panel/diagnostics.md)。

六类 estimator 都可通过 `device` 使用 NumPy CPU、CuPy CUDA 或 Torch CUDA。每个模型页面都给出了 CPU/GPU 与 formula 示例。若显式指定 `device="cuda"` 或 `device="torch"`，但对应 backend 不可用，statgpu 会直接报错，而不是静默切换到 CPU。

Panel 的 `fit()` 采用事务式生命周期。每次新的拟合尝试都会先失效上一轮 fitted/inference state；如果新拟合在任何阶段抛出异常，已经部分写入的新结果也会被清除后再重新抛出该异常。因此 failed refit 之后，`predict()` 与 `summary()` 会把 estimator 视为未拟合状态，而不会继续暴露上一份数据的 coefficient/inference，也不会暴露本次未完成拟合留下的中间结果。
