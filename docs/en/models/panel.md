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

## Current Panel Architecture

The public Panel interface is estimator-centric, not loss-centric. Users construct `PanelOLS`, `RandomEffects`, `PooledOLS`, `BetweenOLS`, `FirstDifferenceOLS`, or `FamaMacBeth` and call `.fit()`. All six estimators share `BasePanelModel(BaseEstimator)`, but **`BasePanelModel` contains statistically neutral shared infrastructure; it does not choose which econometric transformation defines a particular estimator.**

The current implementation can be read as the following runtime chain:

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
  ├── residual-OLS covariance dispatch (`_covariance.py`), where applicable
  ├── coefficient inference finalization (`BasePanelModel`), where applicable
  ├── fit statistics / specification diagnostics
  │     (`_diagnostic_context.py`, `_diagnostics.py`)
  ├── estimator-specific state/effect recovery
  └── predict() / summary()
```

This is a **responsibility map**, not a claim that every estimator calls every helper in exactly this order. Concrete estimators selectively reuse shared primitives. In particular, `FamaMacBeth` retains specialized backend preparation, period aggregation, and beta-series covariance; `PanelOLS` performs effect recovery inside its own fit path; and `RandomEffects` owns its auxiliary between/within regressions and Swamy-Arora variance-component construction.

The important boundary is that **the shared layer reuses how inputs and metadata are handled, how panel least-squares problems are solved stably, and how common inference results are organized; the concrete estimator decides which statistical fit space is being estimated.**

### Responsibilities of the Shared Components

| Component | Current responsibility |
|---|---|
| `BasePanelModel` | Transactional fit lifecycle, formula/side-array alignment, backend numeric preparation helpers, panel metadata, shared prediction, residual-OLS inference finalization, and summary construction. |
| `_formula.py` | Standard R formulas, fixest pipe syntax, `EntityEffects`/`TimeEffects` tokens, side-array alignment, and prediction-design reconstruction. |
| `_results.py` | Structured metadata/result containers such as `PanelIndexInfo`, `PanelFitStatistics`, and `PanelTestResult`. |
| `_linalg.py` / `_intercept.py` / `_reductions.py` | Shared numerical policy: rank-aware least-squares, guarded constant/response-level handling, batched period solves, and stable grouped reductions. |
| `_covariance.py` | Shared covariance implementations and dispatch for nonrobust, HC, cluster, HAC, and Driscoll-Kraay paths. |
| `_diagnostic_context.py` / `_diagnostics.py` | Fit statistics, degrees-of-freedom definitions, and Panel diagnostics such as Hausman, pooling F, and Breusch-Pagan LM. |
| concrete estimator modules | Define each estimator's statistical transformation, auxiliary estimation, model-specific state, and which shared inference/covariance contracts apply. |

`BasePanelModel` is therefore not a universal Panel algorithm. It does not automatically perform within transformation and it does not estimate RandomEffects variance components. It provides reusable primitives; concrete implementations such as `PanelOLS.fit()` and `RandomEffects.fit()` decide the statistical structure in which those primitives are used. Not every estimator uses every shared helper: `FamaMacBeth`, for example, preserves specialized backend and covariance logic because its inferential object is the time series of period coefficients rather than one residual-OLS fit space.

### Current Computation Path by Estimator

| Estimator | Fit space / core transformation | Numerical estimation | Inference path |
|---|---|---|---|
| `PooledOLS` | Original stacked level design with an automatically added intercept | pooled OLS | residual-OLS covariance + shared inference; pooled fit statistics and BP-LM diagnostic when applicable |
| `PanelOLS` | Level regression without effects; entity/time/two-way demeaning when effects are requested | transformed OLS | transformed-fit-space covariance + shared inference; fixed-effect recovery and Panel fit-statistic / pooling-F context remain estimator-specific |
| `BetweenOLS` | Entity means of $X$ and $y$ | entity-mean OLS | entity-mean fit-space covariance + shared inference |
| `FirstDifferenceOLS` | Within-entity first differences after time ordering when available | differenced OLS | differenced fit-space covariance + shared inference |
| `RandomEffects` | Between/within auxiliary regressions → Swamy-Arora variance components → quasi-demeaning | feasible GLS represented as quasi-demeaned transformed OLS | shared residual-OLS covariance/inference on the quasi-demeaned fit space, while retaining `theta_` and variance components |
| `FamaMacBeth` | Separate cross-sectional regression design for each time period | period-specific OLS / batched OLS followed by aggregation of $\hat\beta_t$ | **does not use the residual-OLS covariance registry**; covariance is based on the period coefficient series $\{\hat\beta_t\}$ and remains estimator-specific |

This table also explains why the six estimators cannot be reduced to “one generic OLS call.” They share substantial numerical and inference infrastructure, but **the definition of the fit space is itself part of the statistical meaning of the estimator.**

### Fixed Effects Example: the Statistical Transformation Precedes the Solve

For the entity fixed-effects model

$$
y_{it}=x_{it}^\top\beta+\alpha_i+\varepsilon_{it},
$$

the within estimator first constructs

$$
\widetilde y_{it}=y_{it}-\bar y_i,
\qquad
\widetilde x_{it}=x_{it}-\bar x_i,
$$

and current `PanelOLS` then solves the transformed least-squares problem

$$
\hat\beta
=
\arg\min_\beta
\sum_{i,t}
\left(\widetilde y_{it}-\widetilde x_{it}^\top\beta\right)^2.
$$

The Panel estimator is not finished once the slope is solved: the implementation still recovers entity/time effects, determines effect rank and residual degrees of freedom, and computes the selected covariance, coefficient inference, and Panel-specific fit statistics on the transformed design. The maintained estimator pipeline therefore contains **pre-fit transformation, numerical solve, and post-fit Panel inference**.

### Current Boundary with `LossBase`

The current Panel implementation has **no `PanelLoss` layer and does not organize fitting through the generic `LossBase + Penalty + Solver` pipeline**. Its current abstraction is estimator-specific fit-space construction plus shared Panel numerical/inference infrastructure. The current generic optimization interface is documented separately in [Loss × Penalty × Solver Framework](../guides/loss-penalty-solver-framework.md).

Each estimator page separates the **statistical model and identification assumptions** from the **numerical estimator**. The assumptions describe when the reported coefficient has the usual panel-econometric interpretation; the software can evaluate an estimator mechanically even when those substantive assumptions are not credible in a particular application.

Shared statistical definitions are collected in [covariance](../panel/covariance.md), [fit statistics](../panel/fit-statistics.md), and [diagnostics](../panel/diagnostics.md).

All six estimators support NumPy CPU, CuPy CUDA, and Torch CUDA through the `device` parameter. Each estimator page includes CPU/GPU and formula examples. If `device="cuda"` or `device="torch"` is requested explicitly but that backend is unavailable, statgpu raises an error instead of silently switching to CPU.

The shared panel least-squares policy is also fail-closed at extreme float64 coefficient scales. Cancellation-sensitive response projections use the maintained magnitude-tiered reducer, and a full-rank exact leading constant can remove a common response level before solving. If a non-constant coefficient is below the numerically certifiable resolution of the float64 projection and the resulting candidate materially violates least-squares stationarity, statgpu raises `FloatingPointError` rather than publishing a finite but unreliable coefficient. Fama-MacBeth applies the same principle period by period; a period coefficient-resolution failure is reported separately from genuine rank deficiency. Ordinary well-resolved fits retain the existing SVD/Gram paths.

Panel fits are transactional. A new `fit()` attempt invalidates the previous fitted and inference state before work begins, and any exception during the new fit clears partially written outputs before it is re-raised. After a failed refit, `predict()` and `summary()` therefore report the estimator as unfitted rather than exposing coefficients or inference from either the previous data or an incomplete new fit. Formula-based prediction is also row-preserving: if Patsy would drop a prediction row because a modeled value is missing, or if a formula transformation produces NaN/Inf, prediction fails clearly instead of returning a shorter or non-finite result.
