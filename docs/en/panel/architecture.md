# Panel Architecture

> Language: English  
> Last updated: 2026-09-17  
> This page: public architecture of the Panel model family  
> Switch: [Chinese](../../cn/panel/architecture.md)

This page explains how statgpu's Panel estimators are organized: which responsibilities are shared, where estimator-specific data transformations enter, and how estimation connects to covariance, inference, diagnostics, prediction, and summaries.

For model choice, identification assumptions, formulas, and statistical interpretation, start with [Panel Models](../models/panel.md) and the dedicated model pages. Internal module ownership and numerical implementation details belong in the repository's `dev/` architecture documentation.

## 1. Overall structure

Users fit one of the public Panel estimators:

- `PanelOLS`
- `RandomEffects`
- `PooledOLS`
- `BetweenOLS`
- `FirstDifferenceOLS`
- `FamaMacBeth`

The family shares `BasePanelModel` infrastructure for common input, state, prediction, and reporting behavior, while each concrete estimator owns the statistical transformation or auxiliary estimation that defines its model.

At a high level:

```text
input / formula / panel indices
        |
        v
shared Panel input and metadata handling
        |
        v
estimator-specific transformation or regression construction
        |
        v
OLS / GLS / period-wise numerical estimation
        |
        v
covariance + inference + diagnostics
        |
        v
fitted state + predict() + summary()
```

The important boundary is that the shared infrastructure does **not** define one generic Panel estimator. The model class still determines what data are transformed and what statistical estimating problem is solved.

## 2. Shared responsibilities

`BasePanelModel` and the shared Panel layer provide reusable behavior such as:

- formula/input alignment;
- entity/time index metadata;
- backend/device preparation;
- fitted-state lifecycle;
- common prediction behavior;
- shared result/summary plumbing;
- covariance and coefficient-inference integration where the estimator has an OLS-style residual representation.

Shared result objects also provide structured representations for panel-index metadata, fit statistics, and diagnostic-test results.

## 3. Estimator-specific construction

The six estimators reuse common infrastructure but construct different estimation problems:

| Estimator | Main data construction | Numerical form |
|---|---|---|
| `PooledOLS` | stacked level data | pooled OLS |
| `PanelOLS` | level data or entity/time/two-way within transformation | transformed OLS |
| `BetweenOLS` | entity means | OLS on entity-level means |
| `FirstDifferenceOLS` | within-entity first differences | OLS on differenced data |
| `RandomEffects` | auxiliary regressions, variance components, quasi-demeaning | feasible GLS represented through transformed regression |
| `FamaMacBeth` | one cross-sectional regression per period | period-wise OLS followed by coefficient aggregation |

This table describes the architecture of the estimation pipeline. The statistical derivations and assumptions of those transformations belong to the corresponding model documentation.

## 4. Estimation and inference layers

Most Panel estimators eventually produce a regression design, response, coefficients, and residuals. The post-fit layer then combines the pieces appropriate to that estimator:

- covariance estimation;
- coefficient standard errors, statistics, p-values, and confidence intervals;
- degrees of freedom and fit statistics;
- model-specific diagnostics;
- effect recovery or model-specific state where applicable.

`FamaMacBeth` is structurally different from residual-OLS models because its covariance is based on the period coefficient series rather than only on one stacked residual regression. That statistical distinction is preserved even though it shares surrounding Panel infrastructure.

For covariance definitions and diagnostic interpretation, use the Panel covariance/diagnostic documentation rather than treating this architecture page as the statistical reference.

## 5. Backend boundary

Panel estimators that support NumPy, CuPy, and Torch use the common `device` vocabulary described in [Device and GPU Memory](../guides/device-and-memory.md).

An explicit accelerator request remains explicit: unavailable requested backends raise rather than being silently replaced by CPU computation. Formula parsing or metadata preparation may still occur on CPU before numerical arrays are prepared for the selected backend.

Detailed linear-algebra stabilization, rank detection, grouped reductions, and implementation-specific numerical checks are internal numerical policy; users should rely on the documented failure behavior and model outputs rather than private helper structure.

## 6. Fit lifecycle

Panel `fit()` behaves transactionally from the user's point of view. A failed fit does not leave partially published results that appear to belong to a successful model. After a successful fit, prediction, summary, and inference properties refer to that successful fitted state.

Formula-based prediction also preserves row alignment. If formula processing would drop or invalidate prediction rows, statgpu raises rather than returning outputs whose rows no longer correspond to the caller's input.

## 7. Relationship to the generic loss/penalty/solver framework

Panel models are organized around **panel-data construction + OLS/GLS/period-wise regression + Panel-specific post-fit statistics**.

That is different from the generic `LossBase + Penalty + Solver` composition used by penalized objective-based estimators. See [Loss × Penalty × Solver Framework](../guides/loss-penalty-solver-framework.md) for that architecture.

The absence of a generic `PanelLoss` layer is therefore not a missing public feature: Panel estimators express their statistical definitions through their model-specific transformations and regression constructions.

## 8. Where to look next

- [Panel Models](../models/panel.md) — model choice and family overview
- dedicated Panel model pages — formulas, assumptions, parameters, examples, interpretation
- [Device and GPU Memory](../guides/device-and-memory.md) — device semantics
- Panel covariance/diagnostic pages — covariance estimators and specification tests
- [Loss × Penalty × Solver Framework](../guides/loss-penalty-solver-framework.md) — architecture of objective-composed penalized models
