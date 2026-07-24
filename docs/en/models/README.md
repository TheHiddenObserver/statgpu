# Models Overview

> Language: English  
> Last updated: 2026-07-24  
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

The Cox page contains the authoritative ties, delayed-entry, robust/cluster inference,
optional dependency, and backend support matrix for `CoxPH`, `CoxPHCV`, and related
penalized paths.

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
