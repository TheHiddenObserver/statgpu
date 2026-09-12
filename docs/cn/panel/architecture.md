# Panel 架构

> 语言：中文  
> 最后更新：2026-09-12  
> 切换：[English](../../en/panel/architecture.md)

本页只描述 **statgpu 当前已经实现的 Panel 架构**。模型选择、识别假设与用户入口见 [面板模型总览](../models/panel.md)；具体 estimator 的统计定义与 API 见各模型页面。

## 1. 架构原则

Panel 的公共入口是 estimator，而不是 loss。用户构造 `PanelOLS`、`RandomEffects`、`PooledOLS`、`BetweenOLS`、`FirstDifferenceOLS` 或 `FamaMacBeth`，然后调用 `.fit()`。

六个 estimator 都以 `BasePanelModel(BaseEstimator)` 为共享基础，但 **`BasePanelModel` 只提供统计上中性的共享基础设施，并不决定某个 panel estimator 的经济计量变换或 fit space。**

当前实现也**没有 `PanelLoss` 层，并不通过通用 `LossBase + Penalty + Solver` pipeline 组织 Panel 拟合**。Panel 当前的核心抽象是：

1. 共享输入、metadata、backend、数值稳定性与结果生命周期；
2. concrete estimator 定义自己的统计 fit space；
3. 在该 fit space 中调用共享或 estimator-specific 的数值估计；
4. 再进入适用的 covariance、inference、diagnostics 与模型特有 post-fit 逻辑。

## 2. 当前运行职责图

```text
User
  │
  │  model = PanelEstimator(...)
  │  model.fit(...)
  ▼
Concrete Panel estimator
  │
  ├── reusable BasePanelModel infrastructure
  │     ├── transactional fit / fitted-state lifecycle
  │     ├── formula parsing and side-array alignment
  │     ├── backend / device numeric preparation helpers
  │     ├── PanelIndexInfo: entity/time/balance/order metadata
  │     ├── shared linear prediction helpers
  │     └── shared summary / residual-OLS inference finalization
  │
  ▼
Estimator-specific fit-space construction
  │
  ├── PooledOLS          → stacked level design
  ├── PanelOLS           → within / two-way demeaning
  ├── BetweenOLS         → entity means
  ├── FirstDifferenceOLS → within-entity first differences
  ├── RandomEffects      → auxiliary fits + variance components + quasi-demeaning
  └── FamaMacBeth        → period-specific cross-sectional designs
  │
  ▼
Numerical estimation
  │
  ├── shared panel numerical policies
  │     ├── `_linalg.py`: SVD/rank/least-squares policy
  │     ├── `_intercept.py`: guarded constant/response-level solves
  │     └── `_reductions.py`: stable grouped reductions
  │
  └── estimator-specific orchestration around those primitives
  │
  ▼
Post-fit statistical layer
  │
  ├── residual-OLS covariance dispatch (`_covariance.py`)，在适用模型中使用
  ├── coefficient inference finalization (`BasePanelModel`)，在适用模型中使用
  ├── fit statistics / specification diagnostics
  │     (`_diagnostic_context.py`, `_diagnostics.py`)
  ├── estimator-specific state/effect recovery
  └── predict() / summary()
```

这是一张**职责图**，不是说每个 estimator 都必须按完全相同的顺序调用每一个 helper。具体 estimator 会选择性复用共享 primitive。

尤其是：

- `FamaMacBeth` 保留自己的 backend preparation、period aggregation 和 beta-series covariance；
- `PanelOLS` 在自己的 fit path 中完成 fixed/time effect recovery；
- `RandomEffects` 自己组织 between/within auxiliary regressions、Swamy-Arora variance-component construction 与 quasi-demeaning。

最重要的边界是：**共享层复用“怎样处理输入和 metadata、怎样稳定求解 panel least-squares、怎样组织共同 inference result”；具体 estimator 决定“要在哪个统计 fit space 中估计什么”。**

## 3. 共享组件的职责

| 组件 | 当前职责 |
|---|---|
| `BasePanelModel` | 事务式拟合生命周期、formula/side-array 对齐、backend 数值准备 helper、panel metadata、共享预测、residual-OLS inference finalization、summary 构造。 |
| `_formula.py` | 标准 R formula、fixest pipe syntax、`EntityEffects`/`TimeEffects` token、side-array 对齐与 prediction design 重建。 |
| `_results.py` | `PanelIndexInfo`、`PanelFitStatistics`、`PanelTestResult` 等结构化 metadata/result 容器。 |
| `_linalg.py` / `_intercept.py` / `_reductions.py` | rank-aware least-squares、constant/response-level 稳定化、batched period solve 与稳定 grouped reduction。 |
| `_covariance.py` | nonrobust、HC、cluster、HAC、Driscoll-Kraay 等 covariance 的共享实现与 dispatch。 |
| `_diagnostic_context.py` / `_diagnostics.py` | fit statistics、自由度定义、Hausman / pooling F / Breusch-Pagan LM 等 panel diagnostics。 |
| concrete estimator modules | 定义各 estimator 的统计变换、辅助估计、模型特有 state，以及哪些共享 inference/covariance contract 适用。 |

`BasePanelModel` 因此不是一个“万能 Panel 算法”。它不会自动执行 within transformation，也不会估计 RandomEffects 的 variance components。它提供可复用 primitive；`PanelOLS.fit()`、`RandomEffects.fit()` 等具体实现决定这些 primitive 在什么统计结构下被调用。

## 4. 六类 estimator 的当前计算路径

| Estimator | fit-space / 核心变换 | 数值估计 | 推断路径 |
|---|---|---|---|
| `PooledOLS` | 原始 stacked level design，并自动加入 intercept | pooled OLS | residual-OLS covariance + shared inference；可计算 pooled fit statistics 与 BP-LM diagnostic |
| `PanelOLS` | 无 effects 时为 level regression；有 effects 时做 entity/time/two-way demeaning | transformed OLS | transformed-fit-space covariance + shared inference；effect recovery 与 Panel fit-statistic / pooling-F context 保持 estimator-specific |
| `BetweenOLS` | 对每个 entity 取 $X$、$y$ 均值 | entity-mean OLS | entity-mean fit-space covariance + shared inference |
| `FirstDifferenceOLS` | entity 内按 time 排序后取一阶差分 | differenced OLS | differenced fit-space covariance + shared inference |
| `RandomEffects` | between/within auxiliary regressions → Swamy-Arora variance components → quasi-demeaning | feasible GLS，可实现为 quasi-demeaned transformed OLS | 在 quasi-demeaned fit space 上使用 shared residual-OLS covariance/inference，同时保留 `theta_` 与 variance components |
| `FamaMacBeth` | 每个 time period 单独构造 cross-sectional regression | period-specific OLS / batched OLS，再聚合 $\hat\beta_t$ | **不走 residual-OLS covariance registry**；covariance 基于 period coefficient series $\{\hat\beta_t\}$，保持 estimator-specific |

这个表解释了为什么不能把六个 estimator 简化成“一次通用 OLS 调用”。它们共享大量数值和推断基础设施，但 **fit-space 的定义本身就是 estimator 的统计含义的一部分。**

## 5. Fixed Effects 示例：统计变换发生在求解之前

例如 entity fixed-effects 模型

$$
y_{it}=x_{it}^\top\beta+\alpha_i+\varepsilon_{it}
$$

先通过 within transformation 构造

$$
\widetilde y_{it}=y_{it}-\bar y_i,
\qquad
\widetilde x_{it}=x_{it}-\bar x_i,
$$

然后当前 `PanelOLS` 在 transformed fit space 上求解

$$
\hat\beta
=
\arg\min_\beta
\sum_{i,t}
\left(\widetilde y_{it}-\widetilde x_{it}^\top\beta\right)^2.
$$

求得 slope 后，Panel 逻辑还没有结束：实现仍需要恢复 entity/time effects、确定 effect rank 和 residual degrees of freedom，再在 transformed design 上计算所选 covariance、coefficient inference 与 panel-specific fit statistics。

因此当前完整 estimator pipeline 同时包含 **fit-space construction、numerical solve 与 post-fit Panel inference**。

## 6. Backend 与数值策略

六类 estimator 都支持通过 `device` 选择 NumPy CPU、CuPy CUDA 或 Torch CUDA。显式请求 `device="cuda"` 或 `device="torch"` 但对应 backend 不可用时会直接报错，而不是切换到 CPU。

共享 panel least-squares policy 对极端 float64 coefficient scale 采用 fail-closed 策略。若 response projection 存在 cancellation/dynamic-range risk，会使用维护中的稳定化 reduction；full-rank 且存在可安全利用的精确常数列时，可先移除公共 response level。若非 constant coefficient 已低于 float64 projection 可可靠分辨的尺度，并且候选解显著违反 least-squares stationarity，statgpu 抛出 `FloatingPointError`，而不是发布有限但不可靠的 coefficient。

`FamaMacBeth` 对每个 period 使用同样的数值可靠性原则，并将 period coefficient-resolution failure 与真正的 rank deficiency 分开报告。

共享 helper 是可复用 primitive，而不是强制调用链。例如 `FamaMacBeth` 有专用 backend preparation；`RandomEffects` 与精确常数路径还会调用 `_intercept.py` 的稳定化 helper。因此架构图表达的是职责边界，不是函数调用的逐行 trace。

## 7. Covariance、diagnostics 与结果发布

对于适用的 transformed residual-OLS fit space，`BasePanelModel._panel_store_ols_inference()` 使用 `_covariance.py` 的 registry/dispatcher 计算 covariance，并统一发布 coefficient-level standard errors、统计量、p-values 与 confidence intervals。

Panel-specific fit statistics 与 specification diagnostics 由 `_diagnostic_context.py` 和 `_diagnostics.py` 提供。`FamaMacBeth` 是重要例外：其 covariance 基于 period coefficient series，而不是 residual-OLS sandwich，因此保持 estimator-specific。

结构化结果基础由 `_results.py` 提供，包括：

- `PanelIndexInfo`：entity/time codes、labels、counts、balanced/unbalanced 与 observation-order metadata；
- `PanelFitStatistics`：within/between/overall $R^2$、adjusted $R^2$、model F 等结构；
- `PanelTestResult`：diagnostic statistic、p-value、distribution、df、applicability 与 metadata。

## 8. 拟合生命周期

Panel 的 `fit()` 采用事务式生命周期。新的拟合尝试会先失效上一轮 fitted/inference state；如果新拟合在任何阶段抛出异常，已经部分写入的新结果会被清除后再重新抛出异常。

因此 failed refit 之后，`predict()` 与 `summary()` 会把 estimator 视为未拟合状态，而不会继续暴露上一份数据的 coefficient/inference，也不会暴露本次未完成拟合留下的中间结果。

formula-based prediction 同样要求 row-preserving：如果 modeled value 缺失导致 Patsy 删除 prediction row，或 formula transformation 生成 NaN/Inf，prediction 会明确报错，而不会返回更短或非有限的结果。

## 9. 与通用优化框架的当前边界

当前 Panel 实现不通过通用 `LossBase + Penalty + Solver` pipeline 组织拟合。通用优化框架的当前接口与职责见 [Loss × Penalty × Solver 框架](../guides/loss-penalty-solver-framework.md)。

本页只记录当前实现，不定义未来 Panel 优化架构。
