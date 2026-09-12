# 面板模型

> 语言：中文  
> 最后更新：2026-09-12  
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

## Panel estimator 的运行架构与 `LossBase` 边界

Panel 的公共接口仍然是 estimator：用户构造 `PanelOLS`、`RandomEffects`、`PooledOLS`、`BetweenOLS`、`FirstDifferenceOLS` 或 `FamaMacBeth`，然后调用 `.fit()`。当前实现**没有 `PanelLoss` 这一层**；Panel 的核心抽象也不是“定义一个新的逐样本 loss”，而是 panel 结构、数据变换、效应恢复和 panel-specific inference。

当前运行路径可以概括为：

```text
User
  │
  │  model = PanelEstimator(...)
  │  model.fit(...)
  ▼
Panel estimator / BasePanelModel
  │
  ├── formula + entity/time metadata
  ├── backend / device preparation
  ├── panel structure validation
  │
  ▼
Panel-specific transformation / intermediate state
  │
  ├── within / two-way demeaning          (PanelOLS)
  ├── entity means                        (BetweenOLS)
  ├── first differences                   (FirstDifferenceOLS)
  ├── variance components + quasi-demean (RandomEffects)
  ├── stacked design                      (PooledOLS)
  └── period-specific regressions         (FamaMacBeth)
  │
  ▼
OLS / GLS / repeated cross-sectional solve / specialized path
  │
  ▼
coefficient estimates
  │
  ├── fixed/random-effect recovery
  ├── covariance / inference
  ├── diagnostics / fit statistics
  └── prediction / summary
```

这与通用 [Loss × Penalty × Solver 框架](../guides/loss-penalty-solver-framework.md) 是两个不同层次的问题。通用优化框架中，Loss 与 Penalty 并列定义

$$
F(\beta)=L(\beta)+P(\beta),
$$

再由 Solver 消费 `value`、`gradient`、`Hessian`、`proximal` 等 primitive。Panel 当前最特殊的步骤则发生在进入数值求解之前或之外：先根据 panel 结构构造 transformed design/response，或者像 `FamaMacBeth` 那样形成一组 period-specific coefficient estimates。因此，把 `BasePanelModel` 继承到 `LossBase`，或者为了统一类树而引入一个泛化的 `PanelLoss`，都不能准确表达当前统计计算。

例如 fixed-effects 模型

$$
y_{it}=x_{it}^\top\beta+\alpha_i+\varepsilon_{it}
$$

在 within estimator 中先变换为

$$
\widetilde y_{it}=y_{it}-\bar y_i,
\qquad
\widetilde x_{it}=x_{it}-\bar x_i,
$$

随后才求解 transformed least-squares 问题

$$
\min_\beta\sum_{i,t}
\left(\widetilde y_{it}-\widetilde x_{it}^\top\beta\right)^2.
$$

这里真正可复用的 loss 仍然是 squared error；Panel 特有的是 transformation 以及拟合后的 effect/inference 语义，而不是另一个 `PanelLoss`。

### 未来 penalized Panel 应如何复用通用优化层

如果以后加入 penalized fixed effects、panel Lasso/ElasticNet 等模型，更自然的架构是 **composition**，而不是让 Panel estimator 继承 `LossBase`：

```text
Panel estimator
      │
      ▼
Panel transformation
      │
      ▼
   (X*, y*)
      │
      ├── LossBase object
      ├── Penalty object
      ▼
     Solver
      │
      ▼
    beta-hat
      │
      ▼
Panel inference / effects / diagnostics / prediction
```

也就是说，Panel estimator 继续拥有 panel 数据结构与统计语义；只有在某个 transformed optimization stage 确实满足通用 objective contract 时，才复用 `LossBase + Penalty + Solver`。`RandomEffects` 的 variance-component estimation、`FamaMacBeth` 的逐期回归与 coefficient-time-series covariance 等部分仍可以保持 estimator-specific，而不应为了接口统一被强行塞入通用 loss。

每个 estimator 页面现在都会把 **statistical model 与 identification assumptions** 和 **numerical estimator** 分开说明。前者回答“什么条件下 coefficient 具有通常的 panel-econometric interpretation”；软件本身仍然可以机械地计算 estimator，因此这些统计假设需要由用户结合实际问题判断，而不是由 `.fit()` 自动验证。

共享统计定义见 [covariance](../panel/covariance.md)、[fit statistics](../panel/fit-statistics.md) 与 [diagnostics](../panel/diagnostics.md)。

六类 estimator 都可通过 `device` 使用 NumPy CPU、CuPy CUDA 或 Torch CUDA。每个模型页面都给出了 CPU/GPU 与 formula 示例。若显式指定 `device="cuda"` 或 `device="torch"`，但对应 backend 不可用，statgpu 会直接报错，而不是静默切换到 CPU。

共享的 panel least-squares policy 对极端 float64 coefficient scale 也采用 fail-closed 策略。若 response projection 存在 cancellation/dynamic-range risk，会使用维护中的 magnitude-tiered reduction；full-rank 且首列为精确常数时，可以先沿 constant direction 安全移除公共 response level。若某个非 constant coefficient 已低于 float64 projection 可可靠分辨的尺度，并且候选解显著违反 least-squares stationarity，statgpu 会抛出 `FloatingPointError`，而不是发布一个有限但不可靠的 coefficient。Fama-MacBeth 对每个 period 使用同样原则，并将 period coefficient-resolution failure 与真正的 rank deficiency 分开报告。普通、可可靠分辨的拟合仍保留现有 SVD/Gram 路径。

Panel 的 `fit()` 采用事务式生命周期。每次新的拟合尝试都会先失效上一轮 fitted/inference state；如果新拟合在任何阶段抛出异常，已经部分写入的新结果也会被清除后再重新抛出该异常。因此 failed refit 之后，`predict()` 与 `summary()` 会把 estimator 视为未拟合状态，而不会继续暴露上一份数据的 coefficient/inference，也不会暴露本次未完成拟合留下的中间结果。formula-based prediction 同样要求 row-preserving：如果 modeled value 缺失会导致 Patsy 删除 prediction row，或者 formula transformation 生成 NaN/Inf，prediction 会明确报错，而不会返回更短或非有限的结果。
