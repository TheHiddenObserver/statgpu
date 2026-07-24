# statgpu

[![PyPI version](https://img.shields.io/pypi/v/statgpu.svg)](https://pypi.org/project/statgpu/)
[![Python versions](https://img.shields.io/pypi/pyversions/statgpu.svg)](https://pypi.org/project/statgpu/)
[![License](https://img.shields.io/github/license/TheHiddenObserver/statgpu.svg)](https://github.com/TheHiddenObserver/statgpu/blob/master/LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/TheHiddenObserver/statgpu.svg)](https://github.com/TheHiddenObserver/statgpu/stargazers)
[![Downloads](https://img.shields.io/pypi/dm/statgpu.svg)](https://pypi.org/project/statgpu/)

GPU-accelerated statistical methods with an sklearn-style API.

## Core Features

- 🚀 **Three backends**: NumPy (CPU), CuPy (CUDA), and PyTorch (CUDA), with automatic device selection
- 🧭 **Explicit backend semantics**: core numerical arrays remain on the selected backend where supported; explicit device requests do not silently switch backend, and model-specific metadata, control-flow, and scalar boundaries are documented per method
- 🔧 **sklearn-style estimators**: familiar `fit`/`predict`/`score` methods and parameter conventions
- 📊 **GLM + robust + quantile + Cox**: Gaussian and non-Gaussian regression, robust losses, quantile regression, and survival analysis
- 🔥 **Penalty framework**: L1, L2, Elastic Net, SCAD, MCP, adaptive, and grouped penalties
- ⚡ **Solver framework**: exact, IRLS, Newton, L-BFGS, FISTA-family, proximal IRLS, proximal Newton, and ADMM implementations where supported
- 🧮 **Inference**: covariance, standard errors, hypothesis tests, confidence intervals, penalized sandwich/oracle inference where supported, debiased Lasso, bootstrap, and simultaneous inference
- 📈 **Nonparametric**: KDE, kernel regression, kernel approximation, B-splines, and GAM
- 🧬 **Unsupervised**: PCA, KMeans, DBSCAN, GMM, UMAP, t-SNE, NNDescent, and related methods
- 📐 **Distributions**: backend-aware distribution functions through `get_distribution()`
- 🧪 **Multiple testing**: `adjust_pvalues`, `combine_pvalues`, and `permutation_test`
- 🔁 **Cross-validation**: `PenalizedGLM_CV`, `RidgeCV`, `LassoCV`, `ElasticNetCV`, `LogisticRegressionCV`, and `CoxPHCV`

## Implemented Methods

> **[Full method list with solvers, penalties, and link functions →](docs/en/guides/implemented-methods.md)**

| Category | Highlights |
|---|---|
| **Regression & GLM** | LinearRegression, Ridge, Lasso, ElasticNet, Logistic, Poisson, Gamma, Inverse Gaussian, Negative Binomial, Tweedie, QuantileRegression, and ordered models |
| **Penalized models** | Unified penalized GLM, typed family wrappers, penalized quantile/robust regression, and PenalizedCoxPHModel |
| **Cross-validation** | RidgeCV, LassoCV, ElasticNetCV, LogisticRegressionCV, PenalizedGLM_CV, and CoxPHCV |
| **ANOVA** | One-way, two-way, Welch ANOVA, post-hoc comparisons, and effect sizes |
| **Covariance** | Empirical and shrinkage covariance, MinCovDet, GraphicalLasso, and GraphicalLassoCV |
| **Panel data** | PanelOLS, RandomEffects, PooledOLS, BetweenOLS, FirstDifferenceOLS, and FamaMacBeth |
| **Nonparametric** | KDE, kernel regression, KernelRidge/CV, KernelPCA, Nystroem, spline bases, and SplineTransformer |
| **Semiparametric** | GAM with penalized B-splines and GCV |
| **Unsupervised** | PCA, SVD, NMF, UMAP, t-SNE, KMeans, DBSCAN, GMM, and AgglomerativeClustering |
| **Survival** | CoxPH and PenalizedCoxPHModel |
| **Feature selection** | Stepwise selection plus fixed-X and model-X knockoff filters and wrappers |
| **Diagnostics** | RegressionDiagnostics and `diagnose_model` |
| **Multiple testing** | `adjust_pvalues`, `combine_pvalues`, and `permutation_test` |

## Documentation

- **English docs**: [docs/en/](docs/en/) — full documentation index
- **Chinese docs**: [docs/cn/](docs/cn/) — 中文文档
- **Quickstart**: [Quickstart](docs/en/getting-started/quickstart.md)
- **Implemented methods**: [Method Inventory](docs/en/guides/implemented-methods.md)
- **GLM + Penalty**: [Generalized Linear Model](docs/en/models/generalized-linear-model.md)
- **Cross-validation**: [Cross-Validation Guide](docs/en/guides/cross-validation.md)
- **Loss × Penalty × Solver Framework**: [Framework Guide](docs/en/guides/loss-penalty-solver-framework.md)
- **Solver-Penalty Matrix**: [Solver × Penalty](docs/en/guides/solver-penalty-matrix.md)
- **Survival analysis**: [Cox Proportional Hazards](docs/en/models/coxph.md)
- **Panel models**: [Panel Data Models](docs/en/models/panel.md)
- **Device & memory**: [Device and GPU Memory](docs/en/guides/device-and-memory.md)
- **PyTorch backend**: [PyTorch Backend](docs/en/guides/pytorch-backend.md)
- **Distribution API**: [Distribution API](docs/en/guides/distribution-api.md)
- **Multiple testing**: [Multiple Testing](docs/en/guides/multiple-testing-combine-pvalues.md)
- **Contributing**: [Contributor Guide](CONTRIBUTING.md)
- **Releasing**: [PyPI Release Guide](RELEASING.md)
- **Changelog**: [Changelog](docs/en/changelog.md)

## Installation

```bash
# CPU only
pip install statgpu

# CuPy backend — choose the CUDA major version that matches your environment
pip install "statgpu[gpu11]"
pip install "statgpu[gpu12]"

# PyTorch backend
pip install "statgpu[torch]"

# Formula/dataframe interfaces
pip install "statgpu[formula]"

# CPU delayed entry and exact Efron robust Cox inference
pip install "statgpu[survival]"

# Development environment
pip install -e ".[dev,validation,formula]"
```

Choose CuPy and PyTorch builds compatible with the installed CUDA driver and runtime.

## Quick Start

The default example runs after the base `pip install statgpu` installation. Use an
explicit GPU device only after installing the corresponding CuPy or PyTorch extra.

```python
import numpy as np
from statgpu import adjust_pvalues, combine_pvalues
from statgpu.inference import norm, poisson
from statgpu.linear_model import LinearRegression, PenalizedGLM_CV

# Generate data using statgpu distributions
X = norm.rvs(size=(10000, 100))
y = X @ norm.rvs(size=100) + norm.rvs(size=10000) * 0.5

# Portable linear regression: selects an available backend
model = LinearRegression(device="auto")
model.fit(X, y)
print(f"R²: {model.score(X, y):.4f}")

# Penalized GLM with cross-validation
y_pois = poisson.rvs(
    mu=np.exp(X[:, :5] @ np.ones(5) * 0.1),
    size=X.shape[0],
)
cv_model = PenalizedGLM_CV(
    loss="poisson",
    penalty="elasticnet",
    l1_ratio=0.5,
    cv=5,
    device="cpu",
)
cv_model.fit(X[:, :5], y_pois)
print(f"Best alpha: {cv_model.alpha_:.4f}")

# Multiple-testing correction
reject, pvals_adj = adjust_pvalues(
    np.array([0.003, 0.02, 0.5]),
    method="bh",
)

# Global p-value combination
stat, p_global = combine_pvalues(
    np.array([0.01, 0.07, 0.03, 0.40]),
    method="fisher",
)
```

## Device Control

```python
import statgpu as sg

# Global setting
sg.set_device("cuda")
sg.set_device("cpu")
sg.set_device("auto")

# Per-model setting; requires the matching GPU extra and runtime
from statgpu.linear_model import LinearRegression
model = LinearRegression(device="cuda", n_jobs=4)
```

## Benchmark Results (RTX 4090)

Full reports: `results/unsupervised_bench_2026-06-27.md`, `results/glm_solver_benchmark_2026-06-23.md`

Test environment: RTX 4090 (24GB), CuPy 14.1.0, PyTorch 2.8.0+cu128,
scikit-learn 1.8.0, statsmodels 0.14.6, lifelines 0.30.3.
These are environment-specific benchmark results, not installation requirements or
universal speed guarantees.

### Selected Benchmark Results

| Module | Dataset | n | p | Best Speedup | Precision |
|---|---|---:|---:|---:|---|
| Poisson GLM | freMTPL2 | 678K | 42 | 196.9x vs sklearn | coef_corr=1.000000 |
| Gamma GLM | synthetic | 678K | 42 | 97.9x vs sklearn | coef_corr=0.9995 |
| CoxPH | synthetic | 1.9K | 500 | 1.2x vs CPU | coef_corr=1.000 |
| adjust_pvalues (BH) | synthetic | — | 1M | 0.55x | 100% agreement |
| PenalizedPoisson (L1) | freMTPL2 | 678K | 42 | — | OK |
| PenalizedCoxPH (L2) | synthetic | 1.9K | 500 | — | C-index match |

### Precision Summary

| Module | Metric | Result |
|---|---|---|
| Poisson GLM | coefficient correlation vs sklearn | 1.000000 |
| Gamma GLM | coefficient correlation vs sklearn | 0.9995 |
| CoxPH | coefficient correlation vs lifelines | 1.000 |
| adjust_pvalues (BH) | rejection agreement vs statsmodels | 100% |
| Penalized models | self-consistency | validated across supported penalties |

## Contributing

Contributions are welcome, including bug fixes, documentation, tests, statistical
validation, GPU performance work, and new methods.

1. Read the [Contributor Guide](CONTRIBUTING.md) before making a substantial change.
2. Open an issue first for new estimators, public API changes, inference methods, solvers, penalties, or large refactors.
3. Install development and validation dependencies with `python -m pip install -e ".[dev,validation,formula]"`.
4. Add focused tests and run the relevant CPU and physical-GPU checks.
5. Update English and Chinese documentation and changelogs for user-visible behavior changes.

Maintainers preparing a package release should follow the [PyPI Release Guide](RELEASING.md).

## Requirements

- Python >= 3.9
- NumPy >= 1.20
- CuPy optional, using the wheel matching the CUDA major version
- PyTorch optional, using a CUDA-compatible build for GPU execution
- CUDA runtime compatible with the selected CuPy or PyTorch build

## License

Apache License 2.0 — see [LICENSE](LICENSE).
