# Panel Architecture

> Language: English  
> Last updated: 2026-09-13  
> Switch: [Chinese](../../cn/panel/architecture.md)

This page describes **the Panel architecture implemented in statgpu today**: how the public model classes share `BasePanelModel` infrastructure, how each estimator constructs the data used for estimation, and how numerical estimation, covariance, inference, and diagnostics are connected. For model choice, identification assumptions, and user-facing entry points, see [Panel Models](../models/panel.md). For the statistical definition and API of an individual estimator, see its dedicated model page.

## 1. Overall Architecture

Users construct `PanelOLS`, `RandomEffects`, `PooledOLS`, `BetweenOLS`, `FirstDifferenceOLS`, or `FamaMacBeth` and call `.fit()`.

All six model classes share `BasePanelModel(BaseEstimator)`. `BasePanelModel` owns formula/input alignment, panel-index metadata, backend preparation, fit-state management, shared prediction, common inference finalization, and summary construction. Each concrete model class defines the data transformation, auxiliary estimation, and model-specific outputs required by its statistical estimator.

The current computation can be summarized as:

1. parse inputs, formulas, entity/time indices, and the execution backend;
2. construct the design matrix and response used by the selected estimator;
3. run shared numerical primitives or model-specific steps for OLS, GLS, or period-wise regressions;
4. compute the applicable covariance, coefficient inference, fit statistics, and diagnostics;
5. publish model-specific state, prediction, and summary results.

## 2. Current Runtime Responsibility Map

```text
User
  │
  │  model = PanelEstimator(...)
  │  model.fit(...)
  ▼
Concrete Panel model class
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
Estimator-specific data transformation and regression construction
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
  └── model-specific orchestration around those primitives
  │
  ▼
Post-fit statistical layer
  │
  ├── residual-OLS covariance dispatch (`_covariance.py`), where applicable
  ├── coefficient inference finalization (`BasePanelModel`), where applicable
  ├── fit statistics / specification diagnostics
  │     (`_diagnostic_context.py`, `_diagnostics.py`)
  ├── model-specific state/effect recovery
  └── predict() / summary()
```

Shared infrastructure is composed differently by each model. For example:

- `PanelOLS` owns within/two-way transformations and fixed/time-effect recovery;
- `RandomEffects` owns the between/within auxiliary regressions, Swamy-Arora variance-component estimation, and quasi-demeaning;
- `FamaMacBeth` owns period-wise regressions, period aggregation, and covariance based on the coefficient series.

The shared layer therefore concentrates reusable input handling, metadata, numerical linear algebra, and common inference publication, while each concrete model determines how the data are transformed and which design matrix and response enter estimation.

## 3. Responsibilities of Shared Components

| Component | Current responsibility |
|---|---|
| `BasePanelModel` | Transactional fit lifecycle, formula/side-array alignment, backend numeric preparation helpers, panel metadata, shared prediction, residual-OLS inference finalization, and summary construction. |
| `_formula.py` | Standard R formulas, fixest pipe syntax, `EntityEffects`/`TimeEffects` tokens, side-array alignment, and prediction-design reconstruction. |
| `_results.py` | Structured metadata/result containers including `PanelIndexInfo`, `PanelFitStatistics`, and `PanelTestResult`. |
| `_linalg.py` / `_intercept.py` / `_reductions.py` | Rank-aware least-squares, guarded constant/response-level handling, batched period solves, and stable grouped reductions. |
| `_covariance.py` | Shared implementations and dispatch for nonrobust, HC, cluster, HAC, and Driscoll-Kraay covariance paths. |
| `_diagnostic_context.py` / `_diagnostics.py` | Fit statistics, degrees-of-freedom definitions, and Panel diagnostics such as Hausman, pooling F, and Breusch-Pagan LM. |
| concrete model modules | Define each estimator's data transformation, auxiliary estimation, model-specific state, and corresponding inference/covariance calculations. |

`BasePanelModel` provides the shared lifecycle and statistical infrastructure; concrete implementations such as `PanelOLS.fit()`, `RandomEffects.fit()`, and `FamaMacBeth.fit()` compose those pieces into their estimator-specific workflows.

## 4. Current Computation Path by Estimator

| Estimator | Data used for estimation / core transformation | Numerical estimation | Inference path |
|---|---|---|---|
| `PooledOLS` | Original stacked level design with an automatically added intercept | pooled OLS | residual-OLS covariance + shared inference, together with pooled fit statistics and the BP-LM diagnostic |
| `PanelOLS` | Original level data without effects; entity/time/two-way demeaning when effects are requested | transformed OLS | covariance and shared inference from the transformed design and residuals, together with effect recovery, Panel fit statistics, and pooling-F calculations |
| `BetweenOLS` | Entity means of $X$ and $y$ | entity-mean OLS | covariance and shared inference from the entity-mean regression design and residuals |
| `FirstDifferenceOLS` | Within-entity first differences after time ordering when available | differenced OLS | covariance and shared inference from the differenced design and residuals |
| `RandomEffects` | Between/within auxiliary regressions → Swamy-Arora variance components → quasi-demeaning | feasible GLS represented as quasi-demeaned transformed OLS | shared covariance/inference from the quasi-demeaned design and residuals, together with `theta_` and variance components |
| `FamaMacBeth` | Separate cross-sectional regression design for each time period | period-specific OLS / batched OLS followed by aggregation of $\hat\beta_t$ | covariance from the period coefficient series $\{\hat\beta_t\}$, computed by the dedicated `FamaMacBeth` path |

The six estimators reuse common numerical linear algebra and parts of the inference infrastructure while implementing their statistical definitions through estimator-specific data transformations and auxiliary estimation.

## 5. Fixed Effects Example: the Statistical Transformation Precedes the Solve

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

and `PanelOLS` then solves the least-squares problem based on the transformed design and response:

$$
\hat\beta
=
\arg\min_\beta
\sum_{i,t}
\left(\widetilde y_{it}-\widetilde x_{it}^\top\beta\right)^2.
$$

After the slope estimate is obtained, the implementation recovers entity/time effects, determines effect rank and residual degrees of freedom, and computes the selected covariance, coefficient inference, and Panel-specific fit statistics from the transformed design and residuals.

The maintained workflow therefore contains **data transformation and estimation-problem construction, numerical solution, and post-fit Panel inference**.

## 6. Backend and Numerical Policy

All six model classes support NumPy CPU, CuPy CUDA, and Torch CUDA through the `device` parameter. An explicit `device="cuda"` or `device="torch"` request runs numerical work on the corresponding backend; an unavailable requested backend raises an error.

The shared panel least-squares policy includes numerical-reliability checks for extreme float64 scales. Cancellation- or dynamic-range-sensitive response projections use maintained stable reductions; when a full-rank exact constant column can be used safely, a common response level may be removed before solving. If a non-constant coefficient falls below the numerically certifiable resolution of the float64 projection and the candidate materially violates least-squares stationarity, statgpu raises `FloatingPointError` rather than publishing a coefficient that cannot be validated reliably.

`FamaMacBeth` applies the same numerical-reliability principle period by period and distinguishes coefficient-resolution failures from genuine rank deficiency.

Models compose the shared helpers as needed. `FamaMacBeth`, for example, uses specialized backend preparation, while `RandomEffects` and exact-constant paths also use stabilization helpers in `_intercept.py`.

## 7. Covariance, Diagnostics, and Result Publication

For regressions obtained after transformation that retain the residual-OLS form, `BasePanelModel._panel_store_ols_inference()` uses the covariance dispatch in `_covariance.py` and publishes coefficient-level standard errors, statistics, p-values, and confidence intervals.

Panel-specific fit statistics and specification diagnostics are implemented in `_diagnostic_context.py` and `_diagnostics.py`. `FamaMacBeth` computes covariance directly from the period coefficient series.

The structured result substrate in `_results.py` includes:

- `PanelIndexInfo`: entity/time codes, labels, counts, balanced/unbalanced state, and observation-order metadata;
- `PanelFitStatistics`: within/between/overall $R^2$, adjusted $R^2$, model F, and related metadata;
- `PanelTestResult`: diagnostic statistic, p-value, distribution, degrees of freedom, applicability, and metadata.

## 8. Fit Lifecycle

Panel `fit()` uses a transactional lifecycle. Each fit attempt establishes a new fit state; if fitting raises, partially written outputs are cleared. After a successful fit, `predict()`, `summary()`, and inference properties read from the state published by that fit.

Formula-based prediction preserves row alignment: if Patsy would drop a prediction row because a modeled value is missing, or a formula transformation produces NaN/Inf, prediction raises clearly instead of returning output misaligned with the input rows.

## 9. Relationship to the Generic Optimization Framework

Current Panel computation is organized around **panel-data transformations + OLS/GLS/period-wise regressions + Panel-specific inference**, reusing the numerical and statistical components described above. The generic `LossBase + Penalty + Solver` architecture serves model paths built around explicit objective composition; see [Loss × Penalty × Solver Framework](../guides/loss-penalty-solver-framework.md).

The current Panel path organizes the transformations and corresponding regressions directly, so it does not require an additional `PanelLoss` object.