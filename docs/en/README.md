# StatGPU Documentation

> Language: English  
> Switch: [Chinese](../README.md)

## Getting Started

- [Quickstart](getting-started/quickstart.md) — install, first model, device selection

## Guides

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

### Generalized Linear Models
- [GeneralizedLinearModel](models/generalized-linear-model.md) — GLM + PenalizedGLM base
- [LogisticRegression](models/logistic-regression.md) — logistic classification
- [PoissonRegression](models/poisson-regression.md) — count regression
- [Ordered Models](models/ordered.md) — ordered logit/probit

### Survival
- [CoxPH](models/coxph.md) — Cox proportional hazards

### Panel
- [Panel](models/panel.md) — fixed/random effects panel models

### Nonparametric
- [Nonparametric Overview](models/nonparametric.md) — kernel methods and splines
- [Kernel Methods](models/kernel-methods.md) — KDE, kernel regression, KRR
- [Splines](models/splines.md) — B-spline basis, penalized splines
- [Semiparametric (GAM)](models/semiparametric.md) — generalized additive models

### Inference
- [ANOVA](models/anova.md) — analysis of variance
- [Covariance](models/covariance.md) — covariance estimation, shrinkage
- [Knockoff](models/knockoff.md) — knockoff feature selection

## Reference

- [Changelog](changelog.md) — version history
