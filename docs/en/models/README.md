<a id="models-overview"></a>

# Models and Statistical Methods

> Language: English  
> Last updated: 2026-10-13  
> Switch: [Chinese](../../cn/models/README.md)

Choose a family by the question you want to answer, then use its overview to
choose a tutorial. For a first fit, follow the [beginner path](../usage.md).
Current solver, penalty, backend, and inference coverage is maintained in
[Implemented Methods](../guides/implemented-methods.md) and the linked method pages.

<a id="regression-and-glm"></a>
<a id="specialized-statistical-modules"></a>
<a id="unsupervised-learning"></a>

## Choose a family

| Your question | Start here | What to choose next |
|---|---|---|
| How does a response vary with predictors? | [Regression](regression/README.md) | Response type, shrinkage, quantiles, or robust loss |
| Do group means differ, or how should many tests be handled? | [Statistical testing](testing/README.md) | ANOVA design, post-hoc comparisons, or p-value adjustment |
| How do variables vary together? | [Covariance estimation](covariance/README.md) | Empirical, shrinkage, robust, or sparse-precision estimation |
| Do you need a density, a smooth response curve, or nonlinear features? | [Nonparametric methods](nonparametric/README.md) | KDE/local regression, kernel methods, splines, or GAM |
| Which predictors should be retained? | [Feature selection](feature-selection/README.md) | AIC/BIC stepwise search or knockoff selection and its assumptions |
| Is the response time to an event? | [Survival analysis](coxph.md) | Compare Cox estimators [below](#choosing-a-cox-estimator) |
| Are observations repeated across entities and time? | [Panel data](../panel/README.md) | Choose among panel estimators |
| Do you want structure or a representation without a response? | [Unsupervised learning](../unsupervised/README.md) | Decomposition, clustering, mixture, or manifold methods |

The family pages explain method choice and link to every tutorial in that family.
For constructors, methods, and fitted attributes, use the
[complete API references](../README.md#complete-api-references).

## Survival Analysis

- [Cox Proportional Hazards](coxph.md)

### Choosing a Cox estimator

| Need | Estimator | Import | Contract |
|---|---|---|---|
| Full Cox fitting, baseline hazards, survival prediction, formula input, and inference | `CoxPH` | `from statgpu.survival import CoxPH` | Breslow/Efron/Exact ties, delayed entry, `(start, stop]`, strata, robust/cluster covariance, NumPy/CuPy/Torch |
| Select a non-negative L2 penalty by held-out partial likelihood | `CoxPHCV` | `from statgpu.survival import CoxPHCV` | Uses the canonical Cox semantics during CV and performs a final `CoxPH` refit |
| Estimate with L1, L2, ElasticNet, SCAD, or MCP | `PenalizedCoxPHModel` | `from statgpu.linear_model import PenalizedCoxPHModel` | Broad penalty and generic solver path; currently estimation-only and rejects `compute_inference=True` |

`CoxPH(penalty=...)` and `PenalizedCoxPHModel` are not interchangeable aliases.
Use the canonical `CoxPH`/`CoxPHCV` path when counting-process features,
stratification, baseline prediction, or statistical inference are required. Use
`PenalizedCoxPHModel` when the broader penalty family is the primary requirement
and estimation-only output is sufficient.

The [Cox model page](coxph.md) is the authoritative user-facing source for
Breslow/Efron/Exact ties, delayed-entry and `(start, stop]` data, strata,
robust/cluster inference, subject-grouped CV, prediction boundaries, and the
NumPy/CuPy/Torch support matrix.

## Core Framework

| Page | Content |
|---|---|
| [Loss Functions](../reference/losses.md) | Loss definitions and per-sample formulas |
| [Solver Algorithms](../guides/solver-algorithms.md) | Solver update formulas and applicability |
| [Loss × Penalty × Solver Framework](../guides/loss-penalty-solver-framework.md) | Dispatch logic and compatibility |
| [Solver × Penalty Matrix](../guides/solver-penalty-matrix.md) | Explicit solver routing and restrictions |
| [Inference API](../guides/inference-api.md) | Covariance, resampling, and inference interfaces |

## Current Coverage Principles

- NumPy, CuPy, and Torch are distinct array backends. Check the [device guide](../guides/device-and-memory.md) and model-specific limitations before relying on accelerator placement.
- Backend support may differ by solver, penalty, inference method, and optional
  dependency. Consult the detailed compatibility matrix instead of relying on a single
  global count.
- Consult the model page for its statistical definition, input restrictions,
  prediction, inference, and CV behavior.
