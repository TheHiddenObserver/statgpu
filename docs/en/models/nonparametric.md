# Nonparametric Methods

> Language: English  
> Last updated: 2026-04-17  
> This page: Nonparametric overview  
> Switch: [Chinese](../../models/nonparametric.md)

Language switch: [Chinese](../../models/nonparametric.md)

## Overview

The nonparametric module currently has two main families:
- KDE: density estimation via `fit_kde`, `kde_pdf`, and bootstrap confidence intervals.
- Kernel Regression: Nadaraya-Watson (`nw`) and local-linear (`local_linear`) regression via functional APIs and sklearn-style wrappers.

Both families support NumPy/CuPy execution paths and are used in dedicated SciPy/statsmodels/R comparison benchmarks.

## Path

KDE:
- `statgpu.nonparametric.fit_kde`
- `statgpu.nonparametric.kde_pdf`
- `statgpu.nonparametric.kde_bootstrap_confidence_interval`
- `statgpu.nonparametric.KernelDensityEstimator`

Kernel Regression:
- `statgpu.nonparametric.fit_kernel_regression`
- `statgpu.nonparametric.kernel_regression_predict`
- `statgpu.nonparametric.KernelRegressionRegressor`

## Objective Function

- KDE estimates a smooth density \(\hat f(x)\) from sample points and kernel weights.
- Kernel regression estimates \(m(x)=E[Y|X=x]\) with kernel-weighted local averaging (`nw`) or local linear correction (`local_linear`).

## Estimating Equation

- KDE:
\[
\hat f(x)=\frac{1}{nh}\sum_{i=1}^n K\left(\frac{x-X_i}{h}\right)
\]
with selected kernel and bandwidth policy.
- Kernel regression (`nw`):
\[
\hat m(x)=\frac{\sum_i K_h(x-X_i)Y_i}{\sum_i K_h(x-X_i)}
\]
with optional diagonal/full kernel metric behavior.

## Covariance/Inference

Nonparametric APIs do not expose a unified `cov_type` table like parametric models.
- KDE confidence intervals are available through bootstrap (`kde_bootstrap_confidence_interval`).
- Kernel regression focuses on prediction consistency and cross-framework parity rather than coefficient-level covariance reporting.

## Parameters

Common nonparametric controls:
- `backend`: `auto` / `numpy` / `cupy`
- `kernel`: `gaussian`, `rectangular`, `triangular`, `epanechnikov`, `biweight`, `triweight`, `cosine`, `optcosine`
- `bandwidth`: `scott`, `silverman`, `nrd0`, `nrd`, `ucv`, `bcv`, `sj`, `sj-ste`, `sj-dpi`, or numeric
- Kernel regression specific: `regression='nw'|'local_linear'`, `kernel_metric='full'|'diagonal'`, `bandwidth_per_feature`

## CPU+GPU Examples

```python
import numpy as np
from statgpu.nonparametric import fit_kde, kde_pdf, fit_kernel_regression, kernel_regression_predict

# CPU KDE
x = np.random.randn(500)
grid = np.linspace(-4, 4, 200)
kde = fit_kde(x, bandwidth="scott", kernel="gaussian", backend="numpy")
density = kde_pdf(x, grid, bandwidth="scott", backend="numpy")

# GPU kernel regression
kr = fit_kernel_regression(X_gpu, y_gpu, regression="local_linear", kernel_metric="diagonal", backend="cupy")
y_hat = kernel_regression_predict(X_gpu, y_gpu, Xq_gpu, regression="local_linear", kernel_metric="diagonal", backend="cupy")
```

## strict/approx difference

For kernel regression, `kernel_metric="diagonal"` is often preferred for strict parity checks against statsmodels diagonal-kernel configurations. Full-kernel settings and broader bandwidth selectors can provide flexibility but may trade exact parity for broader modeling choices.

## Outputs

- KDE: fitted estimator object, density values, and optional bootstrap interval bounds.
- Kernel regression: fitted regressor object and predictions at query points.
- sklearn-style wrappers expose `fit`, `predict`, and scoring-compatible interfaces.

## FAQ

- Which bandwidth rule should I start with? `scott` or `silverman` is a reliable baseline; move to `sj`/CV selectors for harder distributions.
- When should I use `local_linear` over `nw`? `local_linear` usually reduces boundary bias at higher compute cost.
- How do I match external frameworks closely? Align kernel type, bandwidth rule/value, and use diagonal metric where required by the comparison target.

## External Validation

- Python benchmarks:
  - `dev/benchmarks/benchmark_kde_vs_scipy.py`
  - `dev/benchmarks/benchmark_kernel_regression_vs_statsmodels.py`
- R benchmark:
  - `dev/benchmarks/benchmark_nonparametric_vs_r.py`
- Combined suite:
  - `dev/benchmarks/benchmark_nonparametric_comparison_suite.py`
- Representative artifacts:
  - `results/kde_vs_scipy_*.json`
  - `results/kernel_regression_vs_statsmodels_*.json`

## References

- Rosenblatt, M. (1956). Remarks on some nonparametric estimates of a density function. *Annals of Mathematical Statistics*, 27(3), 832-837. [https://doi.org/10.1214/aoms/1177728190](https://doi.org/10.1214/aoms/1177728190)
- Parzen, E. (1962). On estimation of a probability density function and mode. *Annals of Mathematical Statistics*, 33(3), 1065-1076. [https://doi.org/10.1214/aoms/1177704472](https://doi.org/10.1214/aoms/1177704472)
- Nadaraya, E. A. (1964). On estimating regression. *Theory of Probability and Its Applications*, 9(1), 141-142. [https://doi.org/10.1137/1109020](https://doi.org/10.1137/1109020)
- Watson, G. S. (1964). Smooth regression analysis. *Sankhya: The Indian Journal of Statistics, Series A*, 26(4), 359-372.
- Fan, J., & Gijbels, I. (1996). *Local Polynomial Modelling and Its Applications*. Chapman & Hall.
