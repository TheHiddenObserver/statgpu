# statgpu Documentation Portal (English)

> Language: English  
> Last updated: 2026-07-24  
> Switch: [Chinese](../cn/usage.md)

This portal links to maintained capability inventories instead of duplicating
version-sensitive support tables.

## Getting Started

- [Quickstart](getting-started/quickstart.md)
- [Implemented Methods](guides/implemented-methods.md)
- [Device and GPU Memory](guides/device-and-memory.md)
- [PyTorch Backend](guides/pytorch-backend.md)
- [Cross-Validation](guides/cross-validation.md)
- [Inference API](guides/inference-api.md)
- [Changelog](changelog.md)

Use `pip install statgpu[gpu11]` or `statgpu[gpu12]` for the matching CuPy
CUDA major version, `statgpu[torch]` for the PyTorch backend, and
`statgpu[survival]` for optional CPU survival-analysis dependencies.

## Model Families

- [Models Overview](models/README.md)
- [Generalized Linear Models](models/generalized-linear-model.md)
- [Cox Proportional Hazards](models/coxph.md)
- [Panel Models](models/panel.md)
- [ANOVA](models/anova.md)
- [Covariance Estimation](models/covariance.md)
- [Nonparametric Methods](models/nonparametric.md)
- [Unsupervised Learning](models/unsupervised.md)
- [Feature Selection](models/feature-selection.md)
- [Regression Diagnostics](guides/regression-diagnostics.md)

The CV classes `RidgeCV`, `LassoCV`, `ElasticNetCV`,
`LogisticRegressionCV`, `PenalizedGLM_CV`, and `CoxPHCV` are implemented.
Exact loss, penalty, inference, and backend coverage is listed in
[Implemented Methods](guides/implemented-methods.md) and the relevant model page.

## Validation and Evidence

Validation claims are scoped to the model, backend, hardware, and commit tested.
Hosted CI, physical-GPU campaigns, historical benchmarks, and release evidence are
recorded in their corresponding workflow, model, changelog, `results/`, or `dev/`
artifacts. A skipped GPU test is not treated as physical-GPU evidence.

## Contributor Checklist

Follow [`dev/AGENTS.md`](../../dev/AGENTS.md) and
[`.claude/workflows/new-module-dev.md`](../../.claude/workflows/new-module-dev.md):
preserve explicit device semantics, verify objective normalization before external
comparisons, add architecture-specific tests, and synchronize README, English/Chinese
docs, and all three changelogs for user-visible changes.
