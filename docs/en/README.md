# StatGPU Documentation

> Language: English  
> Last updated: 2026-10-13  
> Switch: [Chinese](../cn/README.md)

## Getting Started

- [Beginner path](usage.md) — the short route from installation to choosing and checking a method
- [Quickstart](getting-started/quickstart.md) — installation, first fit, prediction, and device selection

<a id="models"></a>
<a id="linear-family"></a>
<a id="generalized-linear-models"></a>
<a id="survival"></a>
<a id="unsupervised"></a>
<a id="panel"></a>
<a id="nonparametric"></a>
<a id="inference"></a>

## Models and Statistical Methods

Use the [family chooser](models/README.md) to match a method to your question.
Each family entry leads to its tutorials and complete API reference.

- Response models: [regression](models/regression/README.md) for response types and penalties; [survival](models/coxph.md) for time-to-event data
- Statistical questions: [testing](models/testing/README.md) for group comparisons and multiple tests; [covariance](models/covariance/README.md) for joint variation and precision matrices
- Flexible fitting and selection: [nonparametric methods](models/nonparametric/README.md) for densities, curves, and kernel features; [feature selection](models/feature-selection/README.md) for stepwise and knockoff methods
- Data structure: [panel data](panel/README.md) for entity/time observations; [unsupervised learning](unsupervised/README.md) for structure without a response

## Guides

- [Inference API](guides/inference-api.md) — distributions, multiple testing, permutation test, bootstrap
- [Implemented Methods](guides/implemented-methods.md) — public model and method inventory
- [Cross-Validation](guides/cross-validation.md) — folds, tuning grids, selection, and final refit
- [How statgpu Cross-Validation Works](guides/cross-validation-design.md) — public CV execution model, acceleration concepts, and statistical invariants
- [CoxPHCV Experimental Screening Safety](guides/cox-cv-staged-safety.md) — user-visible behavior of experimental screening controls
- [Solver × Penalty Matrix](guides/solver-penalty-matrix.md) — loss × penalty × solver compatibility
- [Device and GPU Memory](guides/device-and-memory.md) — device selection, memory cleanup
- [PyTorch Backend](guides/pytorch-backend.md) — PyTorch execution, inference boundaries, and troubleshooting
- [Distribution API](guides/distribution-api.md) — statistical distribution functions
- [Regression Diagnostics](guides/regression-diagnostics.md) — residuals, leverage, Cook’s distance, and VIF
- [Inference Modes](guides/inference-modes.md) — choosing and interpreting coefficient-inference methods
- [Multiple Testing overview](models/testing/multiple-testing.md) — choose a family, error criterion, or global test
- [P-value adjustment and combination API](guides/multiple-testing-combine-pvalues.md) — complete signatures, shapes, and numerical limits
- [Benchmarks](guides/benchmarks.md) — benchmark methodology and comparison notes
- [Interactive Benchmark Dashboard](guides/statgpu_benchmark_dashboard.md) — filters, charts, metric panels, and data provenance

## Reference

### Complete API references

- [Complete covariance API](reference/covariance-api.md) — estimators, fitted matrices, scoring, and backend limits
- [ANOVA API](reference/anova-api.md) — seven functions, five result types, defaults and input/device limits
- [Complete kernel API](reference/kernel-methods-api.md) — four estimators, pairwise kernels, shapes and numerical/backend limits
- [Shared estimator API](reference/estimator-api.md) — parameter management and inherited inference helpers
- [Linear-model API](reference/linear-model-api.md) — constructors, fit/predict methods, CV, and fitted results
- [Feature-selection API](reference/feature-selection-api.md) — stepwise and knockoff interfaces
- [Survival and smoothing API](reference/survival-smoothing-api.md) — Cox, kernels, splines, bandwidth selection, and GAM
- [Unsupervised API](unsupervised/api-reference.md) — all twelve estimator interfaces and fitted attributes
- [Distribution API](guides/distribution-api.md) — supported families, parameters, and array semantics
- [Inference API](guides/inference-api.md) — result objects, resampling, and multiple-testing functions

<a id="loss-functions"></a>

### Algorithms and release history

- [Loss Functions](reference/losses.md) — low-level loss definitions and numerical properties
- [Penalized Solver API Migration](guides/penalized-solver-api-migration.md) — migration from legacy solver controls
- [Solver Algorithms](guides/solver-algorithms.md) — optimization algorithm details
- [Loss × Penalty × Solver Framework](guides/loss-penalty-solver-framework.md) — component composition and dispatch
- [L-BFGS Float32 Numerical Behavior](guides/lbfgs-float32-precision-contract.md) — interpreting native float32 backend differences
- [Changelog](changelog.md) — version history
