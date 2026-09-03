# Models Overview

> Language: English  
> Last updated: 2026-07-30  
> Switch: [Chinese](../../cn/models/README.md)

This page is a navigation overview. Current solver, penalty, backend, and inference
coverage is maintained in [Implemented Methods](../guides/implemented-methods.md) and
the linked model pages.

## Core Framework

| Page | Content |
|---|---|
| [Loss Functions](losses.md) | Loss definitions and per-sample formulas |
| [Solver Algorithms](../guides/solver-algorithms.md) | Public and internal solver implementations |
| [Loss × Penalty × Solver Framework](../guides/loss-penalty-solver-framework.md) | Dispatch logic and compatibility |
| [Solver × Penalty Matrix](../guides/solver-penalty-matrix.md) | Explicit solver routing and restrictions |
| [Inference API](../guides/inference-api.md) | Covariance, resampling, and inference interfaces |

## Regression and GLM

- [Linear Regression](linear-regression.md)
- [Ridge](ridge.md)
- [Lasso](lasso.md)
- [Elastic Net](elastic-net.md)
- [Adaptive Lasso](adaptive-lasso.md)
- [SCAD](scad.md)
- [MCP](mcp.md)
- [Logistic Regression](logistic-regression.md)
- [Poisson Regression](poisson-regression.md)
- [Generalized Linear Models](generalized-linear-model.md)
- [Ordered Logit/Probit](ordered.md)
- [Quantile Regression](quantile.md)
- [Robust Regression](robust.md)

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
NumPy/CuPy/Torch support matrix. Internal module ownership and extension rules
are documented in [`dev/design/ARCHITECTURE.md`](https://github.com/TheHiddenObserver/statgpu/blob/master/dev/design/ARCHITECTURE.md#5-survival--cox-architecture).

## Specialized Statistical Modules

- [ANOVA](anova.md)
- [Covariance Estimation](covariance.md)
- [Panel Data](panel.md)
- [Nonparametric Methods](nonparametric.md)
- [Kernel Methods](kernel-methods.md)
- [Spline Basis Functions](splines.md)
- [GAM / Semiparametric Models](semiparametric.md)
- [Feature Selection](feature-selection.md)
- [Knockoffs](knockoff.md)
- [Multiple Testing](multiple-testing.md)

## Unsupervised Learning

- [Unsupervised Overview](unsupervised.md)
- [PCA](../unsupervised/pca.md)
- [Truncated SVD](../unsupervised/truncated-svd.md)
- [Incremental PCA](../unsupervised/incremental-pca.md)
- [NMF](../unsupervised/nmf.md)
- [MiniBatch NMF](../unsupervised/minibatch-nmf.md)
- [DBSCAN](../unsupervised/dbscan.md)
- [UMAP](../unsupervised/umap.md)
- [t-SNE](../unsupervised/tsne.md)

## Current Coverage Principles

- NumPy, CuPy, and Torch are distinct execution backends; explicit device requests do
  not silently select another backend.
- Backend support may differ by solver, penalty, inference method, and optional
  dependency. Consult the detailed compatibility matrix instead of relying on a single
  global count.
- Validation claims are scoped to the exact model, backend, hardware, and commit tested.
- Historical release and benchmark records are evidence snapshots, not current support
  matrices.
