# statgpu Documentation Portal (English)

> Language: English  
> Last updated: 2026-10-06  
> Switch: [Chinese](../cn/usage.md)

Start with the Quickstart for a first fit. Use the model and reference pages to check supported options, defaults, and limitations.

## Getting started

- [Complete API references](README.md#complete-api-references) — constructors, methods, parameters, and result attributes

- [Quickstart](getting-started/quickstart.md) — installation, first fit, prediction, and device selection
- [Implemented Methods](guides/implemented-methods.md) — public estimator and method inventory
- [Device and GPU Memory](guides/device-and-memory.md) — CPU/CUDA/Torch behavior and memory controls
- [Cross-Validation](guides/cross-validation.md) — folds, tuning grids, selection, and final refit
- [How statgpu Cross-Validation Works](guides/cross-validation-design.md) — public CV execution model and acceleration semantics
- [Inference Modes](guides/inference-modes.md) — choose and interpret coefficient-inference methods
- [Inference API](guides/inference-api.md) — distributions, multiple testing, permutation tests, and bootstrap utilities
- [Changelog](changelog.md) — release history

For CuPy, install `statgpu[gpu11]` or `statgpu[gpu12]` for the matching CUDA major version. Use `statgpu[torch]` for the PyTorch backend. The base installation contains the Cox implementation; `statgpu[survival]` adds optional external comparison dependencies.

## Model families

- [Models Overview](models/README.md)
- [Linear Regression](models/linear-regression.md)
- [Ridge](models/ridge.md)
- [Lasso](models/lasso.md)
- [ElasticNet](models/elastic-net.md)
- [Generalized Linear Models](models/generalized-linear-model.md)
- [Quantile Regression](models/quantile.md)
- [Robust Regression](models/robust.md)
- [Cox Proportional Hazards](models/coxph.md)
- [Panel Models](models/panel.md)
- [ANOVA](models/anova.md)
- [Covariance Estimation](models/covariance.md)
- [Nonparametric Methods](models/nonparametric.md)
- [Unsupervised Learning](models/unsupervised.md)
- [Feature Selection](models/feature-selection.md)
- [Regression Diagnostics](guides/regression-diagnostics.md)

## Optimization and compatibility references

- [Loss Functions](models/losses.md) — low-level loss definitions and numerical properties
- [Loss × Penalty × Solver Framework](guides/loss-penalty-solver-framework.md) — how the pieces compose
- [Solver × Penalty Matrix](guides/solver-penalty-matrix.md) — supported and unsupported combinations
- [Solver Algorithms](guides/solver-algorithms.md) — algorithm definitions
- [Penalized Solver API Migration](guides/penalized-solver-api-migration.md) — migration from legacy solver controls

## Statistical utilities

- [Distribution API](guides/distribution-api.md)
- [Multiple Testing](guides/multiple-testing-combine-pvalues.md)
- [ANOVA](models/anova.md)
- [Covariance Estimation](models/covariance.md)

To contribute code, documentation, or tests, see the [Contributor Guide](../../CONTRIBUTING.md).
