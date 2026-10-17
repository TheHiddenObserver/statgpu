---
layout: home

hero:
  name: statgpu
  text: Statistical computing, accelerated
  tagline: GPU-accelerated statistical methods with an sklearn-style API, beginner-oriented guides, and reproducible benchmark evidence.
  image:
    light: /images/statgpu-compute-hero-light.jpg
    dark: /images/statgpu-compute-hero.webp
    alt: GPU computing grid connected to statistical curves and data paths
  actions:
    - theme: brand
      text: Get started
      link: /en/getting-started/quickstart
    - theme: alt
      text: Browse models
      link: /en/models/README
    - theme: alt
      text: Open benchmark dashboard
      link: /dashboard/
      target: _self

features:
  - title: Familiar interfaces, a range of statistical methods
    details: Use sklearn-style modeling interfaces, with R-style formula input supported by multiple regression models. Explanations and examples help you choose methods and interpret results.
  - title: Choose CPU or GPU for your task
    details: Use NumPy, CuPy, or PyTorch backends. Choose a device to suit your data size, computational task, and the methods you need.
  - title: Explore GPU acceleration at scale
    details: Compare performance across data sizes to explore potential GPU benefits for large samples, high-dimensional problems, and repeated computation.
---

## Practical guides

Start with the statistical question, choose a method and computing device, then fit, validate, and interpret the results.

| Your next step | Start here |
|---|---|
| 1. Choose a method for your question | [Models and methods](models/README.md) |
| 2. Decide whether you need a GPU | [GPU starter guide](guides/when-to-use-gpu.md)<br>Advanced: [Device and memory configuration](guides/device-and-memory.md) |
| 3. Prepare data and fit a model | [Array input and a first fit](getting-started/quickstart.md) · [R-style formula input](reference/linear-model-api.md#formula-inputs) |
| 4. Tune parameters with cross-validation | [Cross-validation guide](guides/cross-validation.md) |
| 5. Interpret estimates and statistical inference | [Coefficient inference methods](guides/inference-modes.md) · [Statistical tests and result objects](guides/inference-api.md) |

## Methods and references

- Method families: [Regression](models/regression/README.md) · [Survival analysis](models/survival/README.md) · [Statistical testing](models/testing/README.md) · [Nonparametric methods](models/nonparametric/README.md) · [Panel data](panel/README.md) · [Unsupervised learning](unsupervised/README.md)
- Interfaces and coverage: [Complete API references](README.md#complete-api-references) · [Implemented methods](guides/implemented-methods.md)
- Solvers: [Algorithms](guides/solver-algorithms.md) · [Compatibility matrix](guides/solver-penalty-matrix.md)
- Performance: [Reading benchmark results](benchmarks.md) · [Interactive benchmark dashboard](guides/statgpu_benchmark_dashboard.md#open-dashboard)

## Contact and contribute

Email: [linziqian@fzu.edu.cn](mailto:linziqian@fzu.edu.cn).

Report problems or suggest features through [GitHub Issues](https://github.com/TheHiddenObserver/statgpu/issues). Read the [contribution guide](https://github.com/TheHiddenObserver/statgpu/blob/master/CONTRIBUTING.md) before contributing code, documentation, or tests.

[Acknowledgements](acknowledgements.md) ·
[Changelog](changelog.md) ·
[Releasing](https://github.com/TheHiddenObserver/statgpu/blob/master/RELEASING.md) ·
[GitHub](https://github.com/TheHiddenObserver/statgpu)
