# Panel Architecture

> Language: English  
> Last updated: 2026-09-12  
> Switch: [Chinese](../../cn/panel/architecture.md)

This page documents **only the Panel architecture that is implemented in statgpu today**. For model choice, identification assumptions, and user-facing entry points, see [Panel Models](../models/panel.md). For the statistical definition and API of an individual estimator, see its dedicated model page.

## 1. Architectural Principle

Panel is estimator-centric, not loss-centric. Users construct `PanelOLS`, `RandomEffects`, `PooledOLS`, `BetweenOLS`, `FirstDifferenceOLS`, or `FamaMacBeth` and call `.fit()`.

All six estimators share `BasePanelModel(BaseEstimator)`, but **`BasePanelModel` supplies statistically neutral infrastructure; it does not choose the econometric transformation or fit space that defines a concrete estimator.**

The current implementation also has **no `PanelLoss` layer and does not organize Panel fitting through the generic `LossBase + Penalty + Solver` pipeline**. The maintained abstraction is instead:

1. shared input, metadata, backend, numerical-stability, and lifecycle infrastructure;
2. estimator-specific construction of the statistical fit space;
3. shared or estimator-specific numerical estimation in that fit space;
4. covariance, inference, diagnostics, and estimator-specific post-fit logic where applicable.

## 2. Current Runtime Responsibility Map

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

This is a **responsibility map**, not a claim that every estimator calls every helper in exactly this order. Concrete estimators selectively reuse shared primitives.

In particular:

- `FamaMacBeth` retains specialized backend preparation, period aggregation, and beta-series covariance;
- `PanelOLS` performs fixed/time-effect recovery inside its own fit path;
- `RandomEffects` owns its between/within auxiliary regressions, Swamy-Arora variance-component construction, and quasi-demeaning.

The important boundary is that **the shared layer reuses how inputs and metadata are handled, how panel least-squares problems are solved stably, and how common inference results are organized; the concrete estimator decides which statistical fit space is being estimated.**

## 3. Responsibilities of Shared Components

| Component | Current responsibility |
|---|---|
| `BasePanelModel` | Transactional fit lifecycle, formula/side-array alignment, backend numeric preparation helpers, panel metadata, shared prediction, residual-OLS inference finalization, and summary construction. |
| `_formula.py` | Standard R formulas, fixest pipe syntax, `EntityEffects`/`TimeEffects` tokens, side-array alignment, and prediction-design reconstruction. |
| `_results.py` | Structured metadata/result containers including `PanelIndexInfo`, `PanelFitStatistics`, and `PanelTestResult`. |
| `_linalg.py` / `_intercept.py` / `_reductions.py` | Rank-aware least-squares, guarded constant/response-level handling, batched period solves, and stable grouped reductions. |
| `_covariance.py` | Shared implementations and dispatch for nonrobust, HC, cluster, HAC, and Driscoll-Kraay covariance paths. |
| `_diagnostic_context.py` / `_diagnostics.py` | Fit statistics, degrees-of-freedom definitions, and Panel diagnostics such as Hausman, pooling F, and Breusch-Pagan LM. |
| concrete estimator modules | Define each estimator's statistical transformation, auxiliary estimation, model-specific state, and which shared inference/covariance contracts apply. |

`BasePanelModel` is therefore not a universal Panel algorithm. It does not automatically perform within transformation, and it does not estimate RandomEffects variance components. It exposes reusable primitives; concrete implementations such as `PanelOLS.fit()` and `RandomEffects.fit()` decide the statistical structure in which those primitives are used.

## 4. Current Computation Path by Estimator

| Estimator | Fit space / core transformation | Numerical estimation | Inference path |
|---|---|---|---|
| `PooledOLS` | Original stacked level design with an automatically added intercept | pooled OLS | residual-OLS covariance + shared inference; pooled fit statistics and BP-LM diagnostic when applicable |
| `PanelOLS` | Level regression without effects; entity/time/two-way demeaning when effects are requested | transformed OLS | transformed-fit-space covariance + shared inference; effect recovery and Panel fit-statistic / pooling-F context remain estimator-specific |
| `BetweenOLS` | Entity means of $X$ and $y$ | entity-mean OLS | entity-mean fit-space covariance + shared inference |
| `FirstDifferenceOLS` | Within-entity first differences after time ordering when available | differenced OLS | differenced fit-space covariance + shared inference |
| `RandomEffects` | Between/within auxiliary regressions → Swamy-Arora variance components → quasi-demeaning | feasible GLS represented as quasi-demeaned transformed OLS | shared residual-OLS covariance/inference on the quasi-demeaned fit space, while retaining `theta_` and variance components |
| `FamaMacBeth` | Separate cross-sectional regression design for each time period | period-specific OLS / batched OLS followed by aggregation of $\hat\beta_t$ | **does not use the residual-OLS covariance registry**; covariance is based on the period coefficient series $\{\hat\beta_t\}$ and remains estimator-specific |

This is why the six estimators cannot be reduced to “one generic OLS call.” They share substantial numerical and inference infrastructure, but **the definition of the fit space is itself part of the statistical meaning of the estimator.**

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

and current `PanelOLS` then solves the transformed least-squares problem

$$
\hat\beta
=
\arg\min_\beta
\sum_{i,t}
\left(\widetilde y_{it}-\widetilde x_{it}^\top\beta\right)^2.
$$

The estimator is not finished once the slope has been solved. The implementation still recovers entity/time effects, determines effect rank and residual degrees of freedom, and computes the selected covariance, coefficient inference, and Panel-specific fit statistics on the transformed design.

The maintained estimator pipeline therefore contains **fit-space construction, numerical solve, and post-fit Panel inference**.

## 6. Backend and Numerical Policy

All six estimators support NumPy CPU, CuPy CUDA, and Torch CUDA through the `device` parameter. If `device="cuda"` or `device="torch"` is requested explicitly but the requested backend is unavailable, statgpu raises an error instead of silently switching to CPU.

The shared panel least-squares policy is fail-closed at extreme float64 coefficient scales. Cancellation- or dynamic-range-sensitive response projections use maintained stable reductions; when a full-rank exact constant column can be used safely, a common response level may be removed before solving. If a non-constant coefficient is below the numerically certifiable resolution of the float64 projection and the candidate materially violates least-squares stationarity, statgpu raises `FloatingPointError` instead of publishing a finite but unreliable coefficient.

`FamaMacBeth` applies the same numerical-reliability principle period by period and distinguishes a period coefficient-resolution failure from genuine rank deficiency.

Shared helpers are reusable primitives rather than a mandatory call chain. `FamaMacBeth`, for example, has specialized backend preparation; `RandomEffects` and exact-constant paths also use stabilization helpers in `_intercept.py`. The architecture diagram describes responsibility boundaries, not a line-by-line call trace.

## 7. Covariance, Diagnostics, and Result Publication

For applicable transformed residual-OLS fit spaces, `BasePanelModel._panel_store_ols_inference()` uses the registry/dispatcher in `_covariance.py` and publishes coefficient-level standard errors, statistics, p-values, and confidence intervals through the shared inference result path.

Panel-specific fit statistics and specification diagnostics are implemented in `_diagnostic_context.py` and `_diagnostics.py`. `FamaMacBeth` is the important exception: its covariance is based on the period coefficient series rather than a residual-OLS sandwich, so that logic remains estimator-specific.

The structured result substrate in `_results.py` includes:

- `PanelIndexInfo`: entity/time codes, labels, counts, balanced/unbalanced state, and observation-order metadata;
- `PanelFitStatistics`: within/between/overall $R^2$, adjusted $R^2$, model F, and related metadata;
- `PanelTestResult`: diagnostic statistic, p-value, distribution, degrees of freedom, applicability, and metadata.

## 8. Fit Lifecycle

Panel `fit()` uses a transactional lifecycle. A new fit attempt invalidates the previous fitted/inference state before work begins; if the new fit raises at any stage, partially written outputs are cleared before the exception is re-raised.

After a failed refit, `predict()` and `summary()` therefore treat the estimator as unfitted rather than exposing coefficients or inference from either the previous fit or the incomplete new fit.

Formula-based prediction is also row-preserving: if Patsy would drop a prediction row because a modeled value is missing, or a formula transformation produces NaN/Inf, prediction fails clearly rather than returning a shorter or non-finite result.

## 9. Current Boundary with the Generic Optimization Framework

Current Panel fitting is not organized through the generic `LossBase + Penalty + Solver` pipeline. The current generic optimization interface and responsibilities are documented in [Loss × Penalty × Solver Framework](../guides/loss-penalty-solver-framework.md).

This page records the current implementation only; it does not specify a future Panel optimization architecture.
