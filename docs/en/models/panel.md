# Panel Models

> Language: English  
> Last updated: 2026-09-12  
> Switch: [Chinese](../../cn/models/panel.md)

`statgpu.panel` provides six panel-data estimators. These estimators should not be read as six unrelated data-generating processes. Several of them can be applied to the same underlying panel model but use different assumptions or different sources of variation to identify the coefficient of interest.

A useful way to distinguish them is:

| Estimator | Statistical view | Main source of identification / extra condition |
|---|---|---|
| [PanelOLS](../panel/panel-ols.md) | Entity and/or time effects are fixed but unknown nuisance parameters. | Uses variation remaining after the selected fixed effects are removed; no orthogonality between the fixed effects and regressor history is required. |
| [FirstDifferenceOLS](../panel/first-difference-ols.md) | The same fixed-parameter entity-effect model can be differenced to eliminate a time-invariant entity effect. | Uses within-entity changes between consecutive observed periods. |
| [RandomEffects](../panel/random-effects.md) | The entity effect is modeled as a random component. | Classical RE interpretation requires the random effect to be orthogonal to the regressor history, for example $E(a_i\mid X_i)=0$. |
| [BetweenOLS](../panel/between-ols.md) | Averages a panel model to one observation per entity. | Uses between-entity variation; recovering the same structural slope requires the averaged composite error to be orthogonal to the averaged regressors. |
| [PooledOLS](../panel/pooled-ols.md) | One common conditional-mean relationship is fitted to all stacked observations. | Uses all stacked variation; the combined regression error must be exogenous with respect to the regressors. |
| [FamaMacBeth](../panel/fama-macbeth.md) | Each time period has its own cross-sectional regression. | Targets the average of the retained period-specific slopes and bases uncertainty on their time-series variation. |

## Panel estimator runtime architecture and the `LossBase` boundary

The user-facing interface for Panel is still the estimator: users construct `PanelOLS`, `RandomEffects`, `PooledOLS`, `BetweenOLS`, `FirstDifferenceOLS`, or `FamaMacBeth` and call `.fit()`. The current implementation has **no `PanelLoss` layer**. The central Panel abstraction is panel structure, data transformation, effect recovery, and panel-specific inference rather than a new per-observation loss.

The maintained runtime pipeline is:

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
  ├── panel-structure validation
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

This is a different architectural concern from the generic [Loss × Penalty × Solver framework](../guides/loss-penalty-solver-framework.md). In that optimization framework, Loss and Penalty jointly define

$$
F(\beta)=L(\beta)+P(\beta),
$$

and a Solver consumes primitives such as value, gradient, Hessian, and proximal operators. Panel's defining work occurs before or outside that stage: the estimator first constructs a transformed design/response from the panel structure, or—in the case of `FamaMacBeth`—constructs a sequence of period-specific coefficient estimates. Making `BasePanelModel` inherit `LossBase`, or introducing a generic `PanelLoss` only to force a single class hierarchy, would therefore misrepresent the maintained computation.

For example, the fixed-effects model

$$
y_{it}=x_{it}^\top\beta+\alpha_i+\varepsilon_{it}
$$

is transformed by the within estimator to

$$
\widetilde y_{it}=y_{it}-\bar y_i,
\qquad
\widetilde x_{it}=x_{it}-\bar x_i,
$$

and only then solves the transformed least-squares problem

$$
\min_\beta\sum_{i,t}
\left(\widetilde y_{it}-\widetilde x_{it}^\top\beta\right)^2.
$$

The reusable loss here is still squared error. What is specific to Panel is the transformation and the post-fit effect/inference semantics, not a separate `PanelLoss`.

### How future penalized Panel estimators should reuse the generic optimization layer

If penalized fixed effects, panel Lasso/ElasticNet, or related estimators are added later, the natural design is **composition**, not inheritance from `LossBase`:

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

The Panel estimator therefore continues to own panel structure and statistical semantics, while reusing `LossBase + Penalty + Solver` only for a transformed optimization stage that genuinely satisfies that generic objective contract. Estimator-specific stages such as RandomEffects variance-component estimation and Fama-MacBeth period regressions / coefficient-time-series covariance can remain specialized instead of being forced into the generic loss abstraction.

Each estimator page separates the **statistical model and identification assumptions** from the **numerical estimator**. The assumptions describe when the reported coefficient has the usual panel-econometric interpretation; the software can evaluate an estimator mechanically even when those substantive assumptions are not credible in a particular application.

Shared statistical definitions are collected in [covariance](../panel/covariance.md), [fit statistics](../panel/fit-statistics.md), and [diagnostics](../panel/diagnostics.md).

All six estimators support NumPy CPU, CuPy CUDA, and Torch CUDA through the `device` parameter. Each estimator page includes CPU/GPU and formula examples. If `device="cuda"` or `device="torch"` is requested explicitly but that backend is unavailable, statgpu raises an error instead of silently switching to CPU.

The shared panel least-squares policy is also fail-closed at extreme float64 coefficient scales. Cancellation-sensitive response projections use the maintained magnitude-tiered reducer, and a full-rank exact leading constant can remove a common response level before solving. If a non-constant coefficient is below the numerically certifiable resolution of the float64 projection and the resulting candidate materially violates least-squares stationarity, statgpu raises `FloatingPointError` rather than publishing a finite but unreliable coefficient. Fama-MacBeth applies the same principle period by period; a period coefficient-resolution failure is reported separately from genuine rank deficiency. Ordinary well-resolved fits retain the existing SVD/Gram paths.

Panel fits are transactional. A new `fit()` attempt invalidates the previous fitted and inference state before work begins, and any exception during the new fit clears partially written outputs before it is re-raised. After a failed refit, `predict()` and `summary()` therefore report the estimator as unfitted rather than exposing coefficients or inference from either the previous data or an incomplete new fit. Formula-based prediction is also row-preserving: if Patsy would drop a prediction row because a modeled value is missing, or if a formula transformation produces NaN/Inf, prediction fails clearly instead of returning a shorter or non-finite result.
