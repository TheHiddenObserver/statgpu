# statgpu

[![PyPI version](https://img.shields.io/pypi/v/statgpu.svg)](https://pypi.org/project/statgpu/)
[![Python versions](https://img.shields.io/pypi/pyversions/statgpu.svg)](https://pypi.org/project/statgpu/)
[![License](https://img.shields.io/github/license/TheHiddenObserver/statgpu.svg)](https://github.com/TheHiddenObserver/statgpu/blob/master/LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/TheHiddenObserver/statgpu.svg)](https://github.com/TheHiddenObserver/statgpu/stargazers)
[![Downloads](https://img.shields.io/pypi/dm/statgpu.svg)](https://pypi.org/project/statgpu/)

GPU-accelerated statistical methods with an sklearn-compatible API.

## Documentation

- **English documentation**: [docs/en/](docs/en/)
- **中文文档**: [docs/cn/](docs/cn/)
- **Quickstart**: [docs/en/getting-started/quickstart.md](docs/en/getting-started/quickstart.md)
- **Implemented methods**: [docs/en/guides/implemented-methods.md](docs/en/guides/implemented-methods.md)
- **GLM + penalty framework**: [docs/en/models/generalized-linear-model.md](docs/en/models/generalized-linear-model.md)
- **Cross-validation**: [docs/en/guides/cross-validation.md](docs/en/guides/cross-validation.md)
- **Loss × penalty × solver framework**: [docs/en/guides/loss-penalty-solver-framework.md](docs/en/guides/loss-penalty-solver-framework.md)
- **Device and memory**: [docs/en/guides/device-and-memory.md](docs/en/guides/device-and-memory.md)
- **CoxPH contract**: [docs/en/models/coxph.md](docs/en/models/coxph.md)
- **Panel models**: [docs/en/models/panel.md](docs/en/models/panel.md)
- **Contributor guide**: [CONTRIBUTING.md](CONTRIBUTING.md)
- **Release guide**: [RELEASING.md](RELEASING.md)
- **PR #79 final validation**: [dev/reviews/pr79_physical_gpu_validation.md](dev/reviews/pr79_physical_gpu_validation.md)
- **Changelog**: [docs/en/changelog.md](docs/en/changelog.md)

## Core Features

- **Three backends**: NumPy CPU, CuPy CUDA, and Torch CUDA.
- **Explicit device semantics**: `device="cuda"` and `device="torch"` do not silently fall back to CPU; `device="auto"` is the only automatic-selection mode.
- **sklearn-style estimators**: `fit`, `predict`, `score`, fitted attributes, and cloning-oriented parameter contracts.
- **Statistical inference**: covariance, standard errors, test statistics, p-values, confidence intervals, likelihood criteria, bootstrap, permutation, and multiple testing where supported.
- **Penalized models**: L1, L2, Elastic Net, SCAD, MCP, adaptive and group penalties.
- **Cross-validation**: generic penalized GLM CV plus model-specific Ridge, Lasso, Elastic Net, Logistic, and CoxPH CV.
- **Formula interfaces**: patsy-based interfaces with explicit intercept, missing-row, and side-array alignment contracts.
- **Backend transparency**: core numerical array paths remain on the selected backend; intentional CPU boundaries are restricted to formula/label metadata and unsupported scalar distribution operations.

## Implemented Method Families

| Category | Representative interfaces |
|---|---|
| Regression and GLM | LinearRegression, Ridge, Lasso, ElasticNet, Logistic, Poisson, Gamma, InvGauss, Negative Binomial, Tweedie, QuantileRegression |
| Penalized GLM | PenalizedGLM, family wrappers, PenalizedQuantileRegression, PenalizedRobustRegression, PenalizedCoxPHModel |
| Cross-validation | RidgeCV, LassoCV, ElasticNetCV, LogisticCV, PenalizedGLM_CV, CoxPHCV |
| Survival | CoxPH with Breslow/Efron ties, delayed-entry support matrix, strict robust-inference contract, backend-native prediction |
| Panel data | PanelOLS, RandomEffects, PooledOLS, BetweenOLS, FirstDifferenceOLS, FamaMacBeth |
| ANOVA and inference | one-way/two-way/Welch ANOVA, post-hoc methods, effect sizes, diagnostics |
| Covariance | empirical/shrinkage covariance, MinCovDet, GraphicalLasso, GraphicalLassoCV |
| Nonparametric | KDE, kernel regression, KernelRidge/CV, KernelPCA, Nystroem, B-splines, SplineTransformer |
| Semiparametric | GAM |
| Unsupervised | PCA, SVD, NMF, UMAP, t-SNE, KMeans, DBSCAN, GMM, AgglomerativeClustering |
| Feature selection | stepwise selection and fixed-X/model-X knockoff interfaces |
| Multiple testing | `adjust_pvalues`, `combine_pvalues`, `permutation_test` |

## Important Statistical Contracts

### CoxPH delayed entry and inference

Delayed entry is supported subject to the documented matrix in the [CoxPH guide](docs/en/models/coxph.md). In particular:

- delayed entry plus robust or cluster covariance and `compute_inference=True` raises `NotImplementedError`;
- the same model with `compute_inference=False` is allowed as estimation-only, with `_bse` and `_conf_int` left unset;
- CPU delayed entry with a nonzero penalty is not implemented;
- robust inference is strict by default, and approximate Efron inference requires explicit opt-in.

### Rank-deficient PooledOLS

For exactly rank-deficient designs:

- prediction, fitted values, residuals, RSS, effective rank, and fitted-space comparisons remain valid;
- residual degrees of freedom use `nobs - rank(X)`;
- individual coefficients and coefficient-level inference are not uniquely identified and are classified as `NOT_COMPARABLE`, not as runtime errors or unique successful inference results.

### Canonical validation artifacts

PR79 reports use a fail-closed evidence pipeline:

```text
run_accuracy
    -> aggregate_results
    -> validated exact-head artifact
    -> emit_final_report
```

A canonical PASS requires a clean exact-head repository, matching embedded provenance, finite complete evidence, and zero unresolved checks. Old hard-coded PASS files are not authoritative.

## Installation

```bash
# CPU only
pip install statgpu

# CuPy CUDA 11.x
pip install "statgpu[gpu11]"

# CuPy CUDA 12.x
pip install "statgpu[gpu12]"

# Torch backend
pip install "statgpu[torch]"

# Formula/dataframe support
pip install "statgpu[formula]"

# CPU delayed entry and exact Efron robust Cox inference
pip install "statgpu[survival]"

# Development environment
pip install -e ".[dev,validation,formula]"
```

Choose CuPy and Torch builds compatible with the installed CUDA driver/runtime.

## Quick Start

```python
import numpy as np
from statgpu.linear_model import LinearRegression, PenalizedGLM_CV
from statgpu.inference import norm, poisson

X = norm.rvs(size=(10000, 100))
y = X @ norm.rvs(size=100) + norm.rvs(size=10000) * 0.5

model = LinearRegression(device="cuda")
model.fit(X, y)
print(f"R²: {model.score(X, y):.4f}")

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
```

## Device Control

```python
import statgpu as sg

sg.set_device("cuda")
sg.set_device("cpu")
sg.set_device("auto")
```

Per-estimator `device=` overrides follow the same explicit no-silent-fallback contract.

## PR #79 Final Validation

Final reviewed production head before documentation synchronization:

```text
c85750d63d4e6dbc9d988847566c20f5fa862e91
```

Verified results:

- GitHub Actions Tests run #545: PASS;
- Python 3.9–3.12 regression matrix: PASS;
- full CPU suite: 1074 passed, 275 skipped, 0 failed;
- clean-head canonical smoke: PASS with `canonical_eligible=True`;
- maintained Tesla P100 suite: 33 passed, 2 expected skips, 0 failed;
- CoxPH full maintained parity: PASS;
- Panel GPU prediction and rank-deficient contracts: PASS.

The earlier complete P100 campaigns and performance measurements remain historical regression evidence. See the [auditable final report](dev/reviews/pr79_physical_gpu_validation.md) for evidence boundaries and exact SHAs.

Non-blocking follow-ups:

- [Issue #81](https://github.com/TheHiddenObserver/statgpu/issues/81): backend-native NaN/Inf validation;
- [Issue #82](https://github.com/TheHiddenObserver/statgpu/issues/82): old scikit-learn clone compatibility;
- [Issue #83](https://github.com/TheHiddenObserver/statgpu/issues/83): legacy GPU diagnostic-script cleanup.

## Benchmark Notes

Performance measurements are hardware-, workload-, dtype-, and synchronization-specific. Repository benchmark reports under `results/` and `dev/benchmarks/` should be treated as regression evidence, not universal speed guarantees.

Examples include:

- `results/unsupervised_bench_2026-06-27.md`;
- `results/glm_solver_benchmark_2026-06-23.md`;
- the PR79 Tesla P100 evidence referenced in the final validation report.

## Contributing

Contributions are welcome for statistical methods, correctness fixes, documentation, tests, external validation, and GPU performance.

1. Read [CONTRIBUTING.md](CONTRIBUTING.md) and `dev/AGENTS.md`.
2. Open an issue first for new estimators, public API changes, inference methods, solvers, penalties, or large refactors.
3. Preserve NumPy, CuPy, and Torch behavior unless an explicit limitation is approved and documented.
4. Add focused tests and run the relevant CPU and physical-GPU checks.
5. Update English and Chinese documentation and changelogs for user-visible behavior changes.

## Requirements

- Python >= 3.9;
- NumPy >= 1.20;
- optional CuPy wheel matching the CUDA major version;
- optional Torch CUDA build;
- optional extras for formula, survival, development, and validation workflows.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
