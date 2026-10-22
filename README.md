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
- 🧭 **Explicit backend semantics**: choose CPU or GPU execution per model; [current kernel/spline routing exceptions](docs/en/guides/device-and-memory.md#current-smoothing-and-spline-exceptions) can return CPU results despite a GPU request, so check array placement when GPU execution is required. Input, output, and scalar conversions are documented per method
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
| [**Regression & GLM**](docs/en/models/regression/README.md) | [LinearRegression](docs/en/models/regression/linear-regression.md), [Ridge](docs/en/models/regression/ridge.md), [Lasso](docs/en/models/regression/lasso.md), [ElasticNet](docs/en/models/regression/elastic-net.md), [Logistic](docs/en/models/regression/logistic-regression.md), [Poisson](docs/en/models/regression/poisson-regression.md), [Gamma](docs/en/reference/linear-model-api.md#gammaregression), [Inverse Gaussian](docs/en/reference/linear-model-api.md#inversegaussianregression), [Negative Binomial](docs/en/reference/linear-model-api.md#negativebinomialregression), [Tweedie](docs/en/reference/linear-model-api.md#tweedieregression), [QuantileRegression](docs/en/models/regression/quantile.md), and [ordered models](docs/en/models/regression/ordered.md) |
| [**Penalized models**](docs/en/guides/implemented-methods.md#penalized-models) | [Unified penalized GLM](docs/en/reference/linear-model-api.md#penalizedgeneralizedlinearmodel), [typed family wrappers](docs/en/guides/implemented-methods.md#penalized-models), penalized [quantile](docs/en/models/regression/quantile.md)/[robust regression](docs/en/models/regression/robust.md), and [PenalizedCoxPHModel](docs/en/guides/implemented-methods.md#survival-analysis) |
| [**Cross-validation**](docs/en/guides/cross-validation.md) | [RidgeCV](docs/en/reference/linear-model-api.md#ridgecv), [LassoCV](docs/en/reference/linear-model-api.md#lassocv), [ElasticNetCV](docs/en/reference/linear-model-api.md#elasticnetcv), [LogisticRegressionCV](docs/en/reference/linear-model-api.md#logisticregressioncv), [PenalizedGLM_CV](docs/en/reference/linear-model-api.md#penalizedglm_cv), and [CoxPHCV](docs/en/models/coxph.md) |
| [**ANOVA**](docs/en/models/testing/anova.md) | [Classical one-way](docs/en/models/testing/anova-oneway.md), [Welch](docs/en/models/testing/anova-welch.md), [balanced two-way](docs/en/models/testing/anova-twoway.md), [pairwise comparisons](docs/en/models/testing/anova-posthoc.md), and effect sizes; [complete API](docs/en/reference/anova-api.md) |
| [**Covariance**](docs/en/models/covariance/README.md) | [EmpiricalCovariance](docs/en/models/covariance/covariance-empirical.md), [ShrunkCovariance](docs/en/models/covariance/covariance-shrunk.md), [LedoitWolf](docs/en/models/covariance/covariance-ledoit-wolf.md), [OAS](docs/en/models/covariance/covariance-oas.md), [MinCovDet](docs/en/models/covariance/covariance-min-cov-det.md), [GraphicalLasso](docs/en/models/covariance/covariance-graphical-lasso.md), [GraphicalLassoCV](docs/en/models/covariance/covariance-graphical-lasso-cv.md); [complete API](docs/en/reference/covariance-api.md) |
| [**Panel data**](docs/en/panel/README.md) | [PanelOLS](docs/en/panel/panel-ols.md), [RandomEffects](docs/en/panel/random-effects.md), [PooledOLS](docs/en/panel/pooled-ols.md), [BetweenOLS](docs/en/panel/between-ols.md), [FirstDifferenceOLS](docs/en/panel/first-difference-ols.md), and [FamaMacBeth](docs/en/panel/fama-macbeth.md) |
| [**Nonparametric**](docs/en/models/nonparametric/README.md) | [KDE](docs/en/models/nonparametric/nonparametric.md), [kernel regression](docs/en/reference/survival-smoothing-api.md#kernel-regression), [spline bases](docs/en/models/nonparametric/splines.md), and [SplineTransformer](docs/en/models/nonparametric/splines.md) |
| [**Kernel methods**](docs/en/models/nonparametric/kernel-methods.md) | [KernelRidge](docs/en/models/nonparametric/kernel-ridge.md), [KernelRidgeCV](docs/en/models/nonparametric/kernel-ridge-cv.md), [KernelPCA](docs/en/models/nonparametric/kernel-pca.md), [Nystroem](docs/en/models/nonparametric/kernel-nystroem.md); [complete API](docs/en/reference/kernel-methods-api.md) |
| [**Semiparametric**](docs/en/models/nonparametric/semiparametric.md) | [GAM](docs/en/models/nonparametric/semiparametric.md) with penalized B-splines and GCV |
| [**Unsupervised**](docs/en/unsupervised/README.md) | [PCA](docs/en/unsupervised/pca.md), [SVD](docs/en/unsupervised/truncated-svd.md), [NMF](docs/en/unsupervised/nmf.md), [UMAP](docs/en/unsupervised/umap.md), [t-SNE](docs/en/unsupervised/tsne.md), [KMeans](docs/en/unsupervised/kmeans.md), [DBSCAN](docs/en/unsupervised/dbscan.md), [GMM](docs/en/unsupervised/gaussian-mixture.md), and [AgglomerativeClustering](docs/en/unsupervised/agglomerative-clustering.md) |
| [**Survival**](docs/en/models/coxph.md) | [CoxPH/CoxPHCV](docs/en/models/coxph.md) with Breslow, Efron, or Exact ties, `(start, stop]` data, strata, robust covariance, and subject-grouped CV; [PenalizedCoxPHModel](docs/en/guides/implemented-methods.md#survival-analysis) |
| [**Feature selection**](docs/en/models/feature-selection/README.md) | [Stepwise selection](docs/en/models/feature-selection/README.md) plus [fixed-X](docs/en/models/feature-selection/knockoff-fixed-x.md) and [model-X](docs/en/models/feature-selection/knockoff-model-x.md) [knockoff filters and wrappers](docs/en/models/feature-selection/knockoff.md); [complete API](docs/en/reference/feature-selection-api.md); 中文：[总览](docs/cn/models/feature-selection/knockoff.md) · [Fixed-X 教程](docs/cn/models/feature-selection/knockoff-fixed-x.md) · [Model-X 教程](docs/cn/models/feature-selection/knockoff-model-x.md) · [完整 API](docs/cn/reference/feature-selection-api.md) |
| [**Diagnostics**](docs/en/guides/regression-diagnostics.md) | [RegressionDiagnostics](docs/en/guides/regression-diagnostics.md) and [`diagnose_model`](docs/en/guides/implemented-methods.md#feature-selection-and-diagnostics) |
| [**Multiple testing**](docs/en/models/testing/multiple-testing.md) | [`adjust_pvalues`](docs/en/guides/multiple-testing-combine-pvalues.md), [`combine_pvalues`](docs/en/guides/multiple-testing-combine-pvalues.md), and [`permutation_test`](docs/en/guides/inference-api.md) |

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
