# statgpu

[![PyPI version](https://img.shields.io/pypi/v/statgpu.svg)](https://pypi.org/project/statgpu/)
[![Python versions](https://img.shields.io/pypi/pyversions/statgpu.svg)](https://pypi.org/project/statgpu/)
[![License](https://img.shields.io/github/license/TheHiddenObserver/statgpu.svg)](https://github.com/TheHiddenObserver/statgpu/blob/master/LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/TheHiddenObserver/statgpu.svg)](https://github.com/TheHiddenObserver/statgpu/stargazers)
[![Downloads](https://img.shields.io/pypi/dm/statgpu.svg)](https://pypi.org/project/statgpu/)

GPU-accelerated statistical methods with an sklearn-style API.

**[Website & Documentation](https://statgpu.github.io/) · [English](https://statgpu.github.io/en/README) · [中文文档](https://statgpu.github.io/cn/README)**

## Core Features

- 🚀 **Three backends**: NumPy (CPU), CuPy (CUDA), and PyTorch (CUDA), with automatic device selection
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

> **[Full method list with solvers, penalties, and link functions →](https://statgpu.github.io/en/guides/implemented-methods)**

| Category | Highlights |
|---|---|
| [**Regression & GLM**](https://statgpu.github.io/en/models/regression/README) | [LinearRegression](https://statgpu.github.io/en/models/regression/linear-regression), [Ridge](https://statgpu.github.io/en/models/regression/ridge), [Lasso](https://statgpu.github.io/en/models/regression/lasso), [ElasticNet](https://statgpu.github.io/en/models/regression/elastic-net), [Logistic](https://statgpu.github.io/en/models/regression/logistic-regression), [Poisson](https://statgpu.github.io/en/models/regression/poisson-regression), [Gamma](https://statgpu.github.io/en/reference/linear-model-api#gammaregression), [Inverse Gaussian](https://statgpu.github.io/en/reference/linear-model-api#inversegaussianregression), [Negative Binomial](https://statgpu.github.io/en/reference/linear-model-api#negativebinomialregression), [Tweedie](https://statgpu.github.io/en/reference/linear-model-api#tweedieregression), [QuantileRegression](https://statgpu.github.io/en/models/regression/quantile), and [ordered models](https://statgpu.github.io/en/models/regression/ordered) |
| [**Penalized models**](https://statgpu.github.io/en/guides/implemented-methods#penalized-models) | [Unified penalized GLM](https://statgpu.github.io/en/reference/linear-model-api#penalizedgeneralizedlinearmodel), [typed family wrappers](https://statgpu.github.io/en/guides/implemented-methods#penalized-models), penalized [quantile](https://statgpu.github.io/en/models/regression/quantile)/[robust regression](https://statgpu.github.io/en/models/regression/robust), and [PenalizedCoxPHModel](https://statgpu.github.io/en/guides/implemented-methods#survival-analysis) |
| [**Cross-validation**](https://statgpu.github.io/en/guides/cross-validation) | [RidgeCV](https://statgpu.github.io/en/reference/linear-model-api#ridgecv), [LassoCV](https://statgpu.github.io/en/reference/linear-model-api#lassocv), [ElasticNetCV](https://statgpu.github.io/en/reference/linear-model-api#elasticnetcv), [LogisticRegressionCV](https://statgpu.github.io/en/reference/linear-model-api#logisticregressioncv), [PenalizedGLM_CV](https://statgpu.github.io/en/reference/linear-model-api#penalizedglm_cv), and [CoxPHCV](https://statgpu.github.io/en/models/coxph) |
| [**ANOVA**](https://statgpu.github.io/en/models/testing/anova) | [Classical one-way](https://statgpu.github.io/en/models/testing/anova-oneway), [Welch](https://statgpu.github.io/en/models/testing/anova-welch), [balanced two-way](https://statgpu.github.io/en/models/testing/anova-twoway), [pairwise comparisons](https://statgpu.github.io/en/models/testing/anova-posthoc), and effect sizes; [complete API](https://statgpu.github.io/en/reference/anova-api) |
| [**Covariance**](https://statgpu.github.io/en/models/covariance/README) | [EmpiricalCovariance](https://statgpu.github.io/en/models/covariance/covariance-empirical), [ShrunkCovariance](https://statgpu.github.io/en/models/covariance/covariance-shrunk), [LedoitWolf](https://statgpu.github.io/en/models/covariance/covariance-ledoit-wolf), [OAS](https://statgpu.github.io/en/models/covariance/covariance-oas), [MinCovDet](https://statgpu.github.io/en/models/covariance/covariance-min-cov-det), [GraphicalLasso](https://statgpu.github.io/en/models/covariance/covariance-graphical-lasso), [GraphicalLassoCV](https://statgpu.github.io/en/models/covariance/covariance-graphical-lasso-cv); [complete API](https://statgpu.github.io/en/reference/covariance-api) |
| [**Panel data**](https://statgpu.github.io/en/panel/README) | [PanelOLS](https://statgpu.github.io/en/panel/panel-ols), [RandomEffects](https://statgpu.github.io/en/panel/random-effects), [PooledOLS](https://statgpu.github.io/en/panel/pooled-ols), [BetweenOLS](https://statgpu.github.io/en/panel/between-ols), [FirstDifferenceOLS](https://statgpu.github.io/en/panel/first-difference-ols), and [FamaMacBeth](https://statgpu.github.io/en/panel/fama-macbeth) |
| [**Nonparametric**](https://statgpu.github.io/en/models/nonparametric/README) | [KDE](https://statgpu.github.io/en/models/nonparametric/nonparametric), [kernel regression](https://statgpu.github.io/en/reference/survival-smoothing-api#kernel-regression), [spline bases](https://statgpu.github.io/en/models/nonparametric/splines), and [SplineTransformer](https://statgpu.github.io/en/models/nonparametric/splines) |
| [**Kernel methods**](https://statgpu.github.io/en/models/nonparametric/kernel-methods) | [KernelRidge](https://statgpu.github.io/en/models/nonparametric/kernel-ridge), [KernelRidgeCV](https://statgpu.github.io/en/models/nonparametric/kernel-ridge-cv), [KernelPCA](https://statgpu.github.io/en/models/nonparametric/kernel-pca), [Nystroem](https://statgpu.github.io/en/models/nonparametric/kernel-nystroem); [complete API](https://statgpu.github.io/en/reference/kernel-methods-api) |
| [**Semiparametric**](https://statgpu.github.io/en/models/nonparametric/semiparametric) | [GAM](https://statgpu.github.io/en/models/nonparametric/semiparametric) with penalized B-splines and GCV |
| [**Unsupervised**](https://statgpu.github.io/en/unsupervised/README) | [PCA](https://statgpu.github.io/en/unsupervised/pca), [SVD](https://statgpu.github.io/en/unsupervised/truncated-svd), [NMF](https://statgpu.github.io/en/unsupervised/nmf), [UMAP](https://statgpu.github.io/en/unsupervised/umap), [t-SNE](https://statgpu.github.io/en/unsupervised/tsne), [KMeans](https://statgpu.github.io/en/unsupervised/kmeans), [DBSCAN](https://statgpu.github.io/en/unsupervised/dbscan), [GMM](https://statgpu.github.io/en/unsupervised/gaussian-mixture), and [AgglomerativeClustering](https://statgpu.github.io/en/unsupervised/agglomerative-clustering) |
| [**Survival**](https://statgpu.github.io/en/models/coxph) | [CoxPH/CoxPHCV](https://statgpu.github.io/en/models/coxph) with Breslow, Efron, or Exact ties, `(start, stop]` data, strata, robust covariance, and subject-grouped CV; [PenalizedCoxPHModel](https://statgpu.github.io/en/guides/implemented-methods#survival-analysis) |
| [**Feature selection**](https://statgpu.github.io/en/models/feature-selection/README) | [Stepwise selection](https://statgpu.github.io/en/models/feature-selection/README) plus [fixed-X](https://statgpu.github.io/en/models/feature-selection/knockoff-fixed-x) and [model-X](https://statgpu.github.io/en/models/feature-selection/knockoff-model-x) [knockoff filters and wrappers](https://statgpu.github.io/en/models/feature-selection/knockoff); [complete API](https://statgpu.github.io/en/reference/feature-selection-api); 中文：[总览](https://statgpu.github.io/cn/models/feature-selection/knockoff) · [Fixed-X 教程](https://statgpu.github.io/cn/models/feature-selection/knockoff-fixed-x) · [Model-X 教程](https://statgpu.github.io/cn/models/feature-selection/knockoff-model-x) · [完整 API](https://statgpu.github.io/cn/reference/feature-selection-api) |
| [**Diagnostics**](https://statgpu.github.io/en/guides/regression-diagnostics) | [RegressionDiagnostics](https://statgpu.github.io/en/guides/regression-diagnostics) and [`diagnose_model`](https://statgpu.github.io/en/guides/implemented-methods#feature-selection-and-diagnostics) |
| [**Multiple testing**](https://statgpu.github.io/en/models/testing/multiple-testing) | [`adjust_pvalues`](https://statgpu.github.io/en/guides/multiple-testing-combine-pvalues), [`combine_pvalues`](https://statgpu.github.io/en/guides/multiple-testing-combine-pvalues), and [`permutation_test`](https://statgpu.github.io/en/guides/inference-api) |

## Documentation

- **English docs**: [Online documentation](https://statgpu.github.io/en/README) · [Source](docs/en)
- **Chinese docs**: [在线文档](https://statgpu.github.io/cn/README) · [源码](docs/cn)
- **Quickstart**: [Quickstart](docs/en/getting-started/quickstart.md)
- **Implemented methods**: [Method Inventory](docs/en/guides/implemented-methods.md)
- **GLM + Penalty**: [Generalized Linear Model](docs/en/models/regression/generalized-linear-model.md)
- **Cross-validation**: [Cross-Validation Guide](docs/en/guides/cross-validation.md)
- **Loss × Penalty × Solver Framework**: [Framework Guide](docs/en/guides/loss-penalty-solver-framework.md)
- **Solver-Penalty Matrix**: [Solver × Penalty](docs/en/guides/solver-penalty-matrix.md)
- **Survival analysis**: [Cox Proportional Hazards](docs/en/models/coxph.md)
- **Panel models**: [Panel Data Models](docs/en/panel/README.md)
- **Device & memory**: [Device and GPU Memory](docs/en/guides/device-and-memory.md)
- **PyTorch backend**: [PyTorch Backend](docs/en/guides/pytorch-backend.md)
- **Distribution API**: [Distribution API](docs/en/guides/distribution-api.md)
- **Multiple testing**: [Overview and tutorials](docs/en/models/testing/multiple-testing.md) · [Adjustment and combination API](docs/en/guides/multiple-testing-combine-pvalues.md)
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
See the [historical attribution record](dev/references/readme-historical-benchmarks.md)
for the limitations of older comparison tables.

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
