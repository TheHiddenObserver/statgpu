# StatGPU Documentation

> Language: English  
> Switch: [Chinese](../cn/README.md)

## Getting Started

- [Quickstart](getting-started/quickstart.md) — install, first model, device selection

## Guides

- [Inference API](guides/inference-api.md) — distributions, multiple testing, permutation test, bootstrap
- [Implemented Methods](guides/implemented-methods.md) — full method list with solvers, penalties, link functions
- [Cross-Validation](guides/cross-validation.md) — CV API, architecture, GPU acceleration, caching
- [Solver × Penalty Matrix](guides/solver-penalty-matrix.md) — loss × penalty × solver compatibility
- [Device and GPU Memory](guides/device-and-memory.md) — device selection, memory cleanup
- [PyTorch Backend](guides/pytorch-backend.md) — torch backend guide, torch.compile
- [Distribution API](guides/distribution-api.md) — statistical distribution functions
- [Inference Modes](guides/inference-modes.md) — Lasso inference (debiased, bootstrap)
- [Multiple Testing](guides/multiple-testing-combine-pvalues.md) — p-value adjustment and combination
- [Benchmarks](guides/benchmarks.md) — performance benchmarks and comparisons

## Models

### Linear Family
- [LinearRegression](models/linear-regression.md) — OLS with inference
- [Ridge](models/ridge.md) — Ridge regression + RidgeCV
- [Lasso](models/lasso.md) — Lasso + LassoCV + debiased inference
- [ElasticNet](models/elastic-net.md) — ElasticNet + ElasticNetCV
- [SCAD](models/scad.md) — non-convex penalty with oracle property
- [MCP](models/mcp.md) — non-convex penalty with oracle property
- [AdaptiveLasso](models/adaptive-lasso.md) — adaptive L1 penalty

### Loss Functions (v0.2.1)
- [Loss Functions (LossBase)](models/losses.md) — architecture for 12 loss types
- [Quantile Regression](models/quantile.md) — pinball loss + PenalizedQuantileRegression
- [Robust Regression](models/robust.md) — Huber, Bisquare, Fair + PenalizedRobustRegression

### Generalized Linear Models
- [GeneralizedLinearModel](models/generalized-linear-model.md) — GLM + PenalizedGLM base
- [LogisticRegression](models/logistic-regression.md) — logistic classification
- [PoissonRegression](models/poisson-regression.md) — count regression
- [Ordered Models](models/ordered.md) — ordered logit/probit

### Survival
- [CoxPH](models/coxph.md) — Cox proportional hazards + penalized

### Unsupervised
- [Unsupervised Overview](models/unsupervised.md) — 13 algorithms: PCA, KMeans, DBSCAN, GMM, UMAP, NNDescent, t-SNE, NMF, Agglomerative, TruncatedSVD, IncrementalPCA, MiniBatchKMeans, MiniBatchNMF

### Panel
- [Panel](models/panel.md) — six panel estimators including pooled, between, first-difference, and Fama–MacBeth

### Nonparametric
- [Nonparametric Overview](models/nonparametric.md) — kernel methods and splines
- [Kernel Methods](models/kernel-methods.md) — KDE, kernel regression, KRR
- [Splines](models/splines.md) — B/natural/cyclic/thin-plate splines and SplineTransformer
- [Semiparametric (GAM)](models/semiparametric.md) — generalized additive models

### Inference
- [ANOVA](models/anova.md) — one/two-way, Welch, post-hoc, and effect sizes
- [Covariance](models/covariance.md) — empirical/shrinkage, robust MCD, and sparse precision
- [Multiple Testing](models/multiple-testing.md) — p-value adjustment (BH, Holm, Bonferroni) and combination (Fisher, Cauchy, Stouffer)
- [Knockoff](models/knockoff.md) — knockoff feature selection
- [Feature Selection](models/feature-selection.md) — stepwise selection and knockoff overview
- [Regression Diagnostics](guides/regression-diagnostics.md) — residuals, leverage, Cook’s distance, and VIF

## Reference

- [Solver Algorithms](guides/solver-algorithms.md) — 10 solvers: algorithm details
- [Loss × Penalty × Solver Framework](guides/loss-penalty-solver-framework.md) — dispatch logic
- [Changelog](changelog.md) — version history
