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
  - title: Learn the method, not just the API
    details: Model guides explain when to use a method, its intuition and assumptions, a runnable example, result interpretation, and common pitfalls.
  - title: CPU and GPU with explicit behavior
    details: Use NumPy on CPU or CuPy and PyTorch on CUDA. Explicit GPU fits require the corresponding CUDA backend; method-specific host and reporting boundaries are explained in the device guide.
  - title: Evidence you can inspect
    details: Explore versioned benchmarks, validation results, hardware metadata, and source provenance in the dashboard.
---

## Explore statgpu

| I want to... | Start here |
|---|---|
| run my first model | [Quickstart](/en/getting-started/quickstart) |
| choose a statistical method | [Model catalog](/en/models/README) |
| use DataFrame formulas | [Usage guide](/en/usage) |
| configure CPU, CUDA, or Torch | [Device and memory](/en/guides/device-and-memory) |
| understand standard errors and tests | [Inference API](/en/guides/inference-api) |
| compare measured performance | [Interactive benchmark dashboard](/dashboard/) |

English is the default documentation language. Use **Change language** in the navigation bar when you need [简体中文](/cn/).

## Main areas

- [Regression and statistical models](/en/models/README)
- [Unsupervised learning](/en/unsupervised/README)
- [Panel-data models](/en/panel/README)
- [Cross-validation](/en/guides/cross-validation)
- [Changelog](/en/changelog)

[Contributing](https://github.com/TheHiddenObserver/statgpu/blob/master/CONTRIBUTING.md) ·
[Releasing](https://github.com/TheHiddenObserver/statgpu/blob/master/RELEASING.md) ·
[GitHub](https://github.com/TheHiddenObserver/statgpu)
