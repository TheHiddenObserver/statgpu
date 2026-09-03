# Models Overview

> Language: English
> Last updated: 2026-09-03
> Current package release: **0.2.5**
> Switch: [Chinese](../../cn/models/README.md)

This catalog reflects the current public model families. Use the model directory in
the left sidebar for direct access, and consult
[Implemented Methods](../guides/implemented-methods.md) for backend and solver
coverage.

## Regression and generalized linear models

| Family | Public API | Documentation |
|---|---|---|
| Ordinary least squares | `LinearRegression` | [Linear regression](linear-regression.md) |
| Gaussian regularization | `Ridge`, `Lasso`, `ElasticNet` and CV variants | [Ridge](ridge.md), [Lasso](lasso.md), [Elastic Net](elastic-net.md) |
| Structured/non-convex regularization | `AdaptiveLasso`, `SCADRegression`, `MCPRegression` | [Adaptive Lasso](adaptive-lasso.md), [SCAD](scad.md), [MCP](mcp.md) |
| Generalized linear models | `GeneralizedLinearModel`, `GammaRegression`, `InverseGaussianRegression`, `NegativeBinomialRegression`, `TweedieRegression` | [GLM](generalized-linear-model.md) |
| Classification and counts | `LogisticRegression`, `LogisticRegressionCV`, `PoissonRegression` | [Logistic](logistic-regression.md), [Poisson](poisson-regression.md) |
| Ordered outcomes | `OrderedLogitRegression`, `OrderedProbitRegression` | [Ordered models](ordered.md) |
| Quantile and robust fitting | `QuantileRegression`, `PenalizedQuantileRegression`, `PenalizedRobustRegression` | [Quantile](quantile.md), [Robust](robust.md) |

The unified penalized API also exposes Gaussian, logistic, Poisson, Gamma,
Inverse-Gaussian, Negative-Binomial, Tweedie, robust, quantile, and Cox families
through `statgpu.linear_model`.

## Survival analysis

| Need | Estimator | Contract |
|---|---|---|
| Full Cox fitting, prediction, formula input, and inference | `CoxPH` | Breslow/Efron/Exact ties, delayed entry, stratification, robust/cluster covariance |
| L2 selection by held-out partial likelihood | `CoxPHCV` | Subject-preserving CV and final `CoxPH` refit |
| Broad convex/non-convex penalties | `PenalizedCoxPHModel` | Estimation-oriented generic penalty path |

See [Cox proportional hazards](coxph.md) for the precise feature and backend
matrix.

## Specialized statistical modules

| Area | Documentation |
|---|---|
| ANOVA and post-hoc tests | [ANOVA](anova.md) |
| Covariance estimators | [Covariance](covariance.md) |
| Panel and longitudinal models | [Panel data](panel.md) |
| Nonparametric estimation | [Nonparametric methods](nonparametric.md) |
| Kernel models | [Kernel methods](kernel-methods.md) |
| Spline bases | [Splines](splines.md) |
| GAM and semiparametric models | [Semiparametric models](semiparametric.md) |
| Feature selection and diagnostics | [Feature selection](feature-selection.md) |
| Knockoff filters | [Knockoffs](knockoff.md) |
| Multiple testing | [Multiple testing](multiple-testing.md) |
| Loss definitions | [Loss functions](losses.md) |

## Unsupervised learning

| Family | Models |
|---|---|
| Matrix decomposition | [PCA](../unsupervised/pca.md), [Incremental PCA](../unsupervised/incremental-pca.md), [Truncated SVD](../unsupervised/truncated-svd.md), [NMF](../unsupervised/nmf.md), [MiniBatch NMF](../unsupervised/minibatch-nmf.md) |
| Clustering | [K-Means](../unsupervised/kmeans.md), [MiniBatch K-Means](../unsupervised/minibatch-kmeans.md), [Agglomerative clustering](../unsupervised/agglomerative-clustering.md), [DBSCAN](../unsupervised/dbscan.md), [Gaussian mixture](../unsupervised/gaussian-mixture.md) |
| Manifold learning | [t-SNE](../unsupervised/tsne.md), [UMAP](../unsupervised/umap.md) |

Open the [unsupervised overview](../unsupervised/) for shared API conventions.

## Panel data

The panel API now includes the complete estimator, covariance, fit-statistics, and
diagnostics documentation:

- [Pooled OLS](../panel/pooled-ols.md), [fixed-effects Panel OLS](../panel/panel-ols.md),
  [Between OLS](../panel/between-ols.md), [Random Effects](../panel/random-effects.md),
  [First-Difference OLS](../panel/first-difference-ols.md), and
  [Fama-MacBeth](../panel/fama-macbeth.md)
- [Covariance estimators](../panel/covariance.md),
  [fit statistics](../panel/fit-statistics.md), and
  [diagnostics](../panel/diagnostics.md)

## Core framework

- [Solver algorithms](../guides/solver-algorithms.md)
- [Loss × penalty × solver framework](../guides/loss-penalty-solver-framework.md)
- [Solver × penalty matrix](../guides/solver-penalty-matrix.md)
- [Inference API](../guides/inference-api.md)

Backend support may differ by model, solver, penalty, inference method, and optional
dependency. Benchmark and validation claims remain scoped to the exact commit,
hardware, backend, and dataset recorded by the corresponding evidence.
