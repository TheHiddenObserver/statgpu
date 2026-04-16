# Nonparametric Methods

> Language: English  
> Last updated: 2026-04-15  
> This page: Nonparametric overview  
> Switch: [Chinese](../../models/nonparametric.md)

Language switch: [Chinese](../../models/nonparametric.md)

## Overview

statgpu currently groups its nonparametric features into two tracks:

- KDE: `fit_kde` / `kde_pdf` / `kde_bootstrap_confidence_interval`
- Kernel Regression: `fit_kernel_regression` / `kernel_regression_predict`

## KDE

KDE supports NumPy/CuPy backends, weighted samples, bandwidth selection, and bootstrap intervals.

### Main APIs

- `fit_kde(samples, bandwidth='scott', weights=None, kernel='gaussian', backend='auto')`
- `kde_pdf(samples, points, ...)`
- `kde_bootstrap_confidence_interval(samples, points, ...)`
- sklearn-style: `KernelDensityEstimator(...).fit(X).predict(points)` / `score_samples(points)`

### Supported kernels

- `gaussian`
- `rectangular`
- `triangular`
- `epanechnikov`
- `biweight`
- `triweight`
- `cosine`
- `optcosine`

### Bandwidth rules

- Standard rules: `scott`, `silverman`
- R-style rules: `nrd0`, `nrd`
- Selectors: `ucv`, `bcv`, `sj`, `sj-ste`, `sj-dpi`

### Example

```python
import numpy as np
from statgpu.nonparametric import fit_kde, kde_pdf

x = np.random.randn(500)
grid = np.linspace(-4, 4, 200)

kde = fit_kde(x, bandwidth="scott", kernel="gaussian")
density = kde_pdf(x, grid, bandwidth="scott")
```

## Kernel Regression

Kernel Regression supports `nw` and `local_linear`, plus a diagonal metric mode aligned with statsmodels.

### Main APIs

- `fit_kernel_regression(samples, targets, regression='nw', kernel='gaussian', kernel_metric='full'|'diagonal', ...)`
- `kernel_regression_predict(samples, targets, points, ...)`
- sklearn-style: `KernelRegressionRegressor(...).fit(X, y).predict(points)`

### Key options

- `regression='nw'|'local_linear'`
- `kernel_metric='full'|'diagonal'`
- `bandwidth_per_feature`: per-feature bandwidth control for diagonal mode

### Validated results

- `run_id=20260415_103036`: diagonal metric mode matches statsmodels at machine precision
- `run_id=20260415_120903`: multidimensional local-linear batching improved runtime while preserving parity

## Related pages

- [Models Overview](README.md)
- [Benchmark Index](../benchmarks.md)

## Benchmark Suggestions

- Python comparisons:
	- `dev/benchmarks/benchmark_kde_vs_scipy.py`
	- `dev/benchmarks/benchmark_kernel_regression_vs_statsmodels.py`
- R comparison:
	- `dev/benchmarks/benchmark_nonparametric_vs_r.py`
- One-shot suite (Python + R):
	- `dev/benchmarks/benchmark_nonparametric_comparison_suite.py`
