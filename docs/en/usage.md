# statgpu Documentation Portal (English)

> Language: English  
> Last updated: 2026-10-13  
> Switch: [Chinese](../cn/usage.md)

A short path to a first fit. The [documentation index](README.md) organizes the
full guides and references.

## Getting started

1. Follow the [Quickstart](getting-started/quickstart.md) to install statgpu, fit a model, and predict.
2. Check the [device guide](guides/device-and-memory.md) before choosing CPU, CUDA, or Torch. For CuPy, install `statgpu[gpu11]` or `statgpu[gpu12]` for the matching CUDA major version; use `statgpu[torch]` for the PyTorch backend. The base installation contains the Cox implementation; `statgpu[survival]` adds optional external comparison dependencies.

<a id="model-families"></a>

3. Choose a question in [Models and Statistical Methods](models/README.md), then work through that family's tutorial. Check its input assumptions, outputs, and limitations before changing the example.
4. Use [cross-validation](guides/cross-validation.md) for tuning and [inference modes](guides/inference-modes.md) when you need coefficient uncertainty. See [regression diagnostics](guides/regression-diagnostics.md) for checking a fitted regression.

## Optimization and compatibility references

When a fit needs more configuration, consult the [solver × penalty matrix](guides/solver-penalty-matrix.md)
and [loss × penalty × solver framework](guides/loss-penalty-solver-framework.md).
For older code, use the [penalized solver migration guide](guides/penalized-solver-api-migration.md).
The [CV design guide](guides/cross-validation-design.md) explains execution and acceleration semantics.

## Statistical utilities

For group comparisons or many p-values, start with [statistical testing](models/testing/README.md).
For distribution functions, resampling, and result objects, see the
[distribution API](guides/distribution-api.md) and [inference API](guides/inference-api.md).

Use [Implemented Methods](guides/implemented-methods.md) to check coverage and the
[complete API references](README.md#complete-api-references) for signatures and fitted attributes.
Changes are recorded in the [changelog](changelog.md).
To contribute code, documentation, or tests, see the [Contributor Guide](../../CONTRIBUTING.md).
