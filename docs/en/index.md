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
      link: /en/models/
    - theme: alt
      text: Open benchmark dashboard
      link: /dashboard/
      target: _self

features:
  - title: Learn the method, not just the API
    details: Model guides explain when to use a method, its intuition and assumptions, a runnable example, result interpretation, and common pitfalls.
  - title: CPU and GPU with explicit behavior
    details: Use NumPy on CPU or CuPy and PyTorch on CUDA. Explicit GPU requests never silently fall back to CPU.
  - title: Evidence you can inspect
    details: Explore versioned benchmarks, validation results, hardware metadata, and source provenance in the dashboard.
---

## Explore statgpu

| I want to... | Start here |
|---|---|
| run my first model | [Quickstart](getting-started/quickstart.md) |
| choose a statistical method | [Model catalog](models/) |
| check what is implemented | [Implemented methods](guides/implemented-methods.md) |
| choose or understand a solver | [Solver algorithms](guides/solver-algorithms.md) and [compatibility matrix](guides/solver-penalty-matrix.md) |
| configure CPU, CUDA, or Torch | [Device and memory guide](guides/device-and-memory.md) |
| understand standard errors and tests | [Inference API](guides/inference-api.md) |
| compare measured performance | [Interactive benchmark dashboard](/dashboard/) |

## Main areas

- [Regression and statistical models](models/)
- [Unsupervised learning](unsupervised/)
- [Panel-data models](panel/)
- [Cross-validation](guides/cross-validation.md)
- [Changelog](changelog.md)

[Contributing](https://github.com/TheHiddenObserver/statgpu/blob/master/CONTRIBUTING.md) ·
[Releasing](https://github.com/TheHiddenObserver/statgpu/blob/master/RELEASING.md) ·
[GitHub](https://github.com/TheHiddenObserver/statgpu)
