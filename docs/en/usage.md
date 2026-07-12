# statgpu Documentation Portal (English)

> Language: English  
> Last updated: 2026-07-12  
> Switch: [Chinese](../cn/usage.md)

This portal points to the maintained capability inventories rather than duplicating
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
CUDA major version, and `statgpu[torch]` for the PyTorch backend.

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
Their exact loss/penalty/backend coverage is listed in
[Implemented Methods](guides/implemented-methods.md).

## Validation Boundary

Hosted CI covers Python 3.9–3.12, the full CPU test tree, static contracts, and
NumPy/Torch-CPU parity for the affected native-backend paths. Physical CuPy CUDA
and Torch CUDA convergence, transfer, memory, runtime, and repeated-fit validation
remains `PARTIAL_REMOTE_PENDING`; documentation does not claim otherwise.

## Contributor Checklist

Follow `dev/AGENTS.md` and `.claude/workflows/new-module-dev.md`: preserve explicit
device semantics, verify objective normalization before external comparisons, add
architecture-specific tests, and synchronize README, English/Chinese docs, and all
three changelogs for user-visible changes.
