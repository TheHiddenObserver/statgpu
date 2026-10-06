# statgpu

[![PyPI version](https://img.shields.io/pypi/v/statgpu.svg)](https://pypi.org/project/statgpu/)
[![Python versions](https://img.shields.io/pypi/pyversions/statgpu.svg)](https://pypi.org/project/statgpu/)
[![License](https://img.shields.io/github/license/TheHiddenObserver/statgpu.svg)](https://github.com/TheHiddenObserver/statgpu/blob/master/LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/TheHiddenObserver/statgpu.svg)](https://github.com/TheHiddenObserver/statgpu/stargazers)
[![Downloads](https://img.shields.io/pypi/dm/statgpu.svg)](https://pypi.org/project/statgpu/)

GPU-accelerated statistical methods with an sklearn-style API.

## Core Features

- 🚀 **Three backends**: NumPy (CPU), CuPy (CUDA), and PyTorch (CUDA), with automatic device selection
- 🧭 **Explicit backend semantics**: core numerical arrays remain on the selected backend where supported; the device convention requires explicit requests to be respected, with [current kernel/spline routing exceptions](docs/en/guides/device-and-memory.md#current-smoothing-and-spline-exceptions) requiring actual array-placement checks; model-specific metadata, control-flow, and scalar boundaries are documented per method
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
| **Survival** | CoxPH/CoxPHCV with Breslow, Efron, or Exact ties, `(start, stop]` data, strata, robust covariance, and subject-grouped CV; PenalizedCoxPHModel |
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

# Optional statsmodels dependency for external Cox validation
pip install "statgpu[survival]"

# Development environment
pip install -e ".[dev,validation,formula]"
```

Choose CuPy and PyTorch builds compatible with the installed CUDA driver and runtime.

## Quick Start

The default example runs after the base `pip install statgpu` installation. Use an
explicit GPU device only after installing the corresponding CuPy or PyTorch extra.

<!-- api-example: readme-quick-start -->
```python
import numpy as np
from statgpu import adjust_pvalues, combine_pvalues
from statgpu.inference import norm
from statgpu.linear_model import LinearRegression, PenalizedGLM_CV

# Native distribution sampling uses NumPy's global seed.
np.random.seed(42)
X = norm.rvs(size=(300, 8), backend="numpy")
y = X @ norm.rvs(size=8, backend="numpy") + norm.rvs(size=300, backend="numpy") * 0.5

model = LinearRegression(device="cpu")
model.fit(X, y)
print(f"R²: {model.score(X, y):.4f}")

# Native poisson.rvs takes a scalar mean. NumPy supports one mean per row.
rng = np.random.default_rng(42)
y_pois = rng.poisson(np.exp(X[:, :5] @ np.ones(5) * 0.1))
cv_model = PenalizedGLM_CV(
    loss="poisson",
    penalty="elasticnet",
    l1_ratio=0.5,
    alpha_grid=np.logspace(-2, 0, 8),
    cv=3,
    device="cpu",
    compute_inference=False,
)
cv_model.fit(X[:, :5], y_pois)
print(f"Best alpha: {cv_model.alpha_:.4f}")

reject, pvals_adj = adjust_pvalues(
    np.array([0.003, 0.02, 0.5]), method="bh",
)
stat, p_global = combine_pvalues(
    np.array([0.01, 0.07, 0.03, 0.40]), method="fisher",
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

## Benchmark evidence

Historical reports are available for [GLM solvers (2026-06-23)](results/glm_solver_benchmark_2026-06-23.md)
and [unsupervised methods (2026-06-27)](results/unsupervised_bench_2026-06-27.md).
Both reports identify Tesla P100 hardware. They are dated snapshots, not
measurements of the current source or guarantees for another GPU or workload.
Earlier README tables had inconsistent hardware/report attribution and are
retained in the [historical attribution record](dev/references/readme-historical-benchmarks.md).

Check correctness and align the statistical objective before comparing speed.
A useful benchmark records its source commit, hardware and package versions,
workload, timing/synchronization and transfer boundaries. Coefficient correlation
alone does not prove coefficient accuracy, calibrated inference, or backend parity.

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
