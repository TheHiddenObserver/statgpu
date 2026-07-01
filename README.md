# statgpu

[![PyPI version](https://img.shields.io/pypi/v/statgpu.svg)](https://pypi.org/project/statgpu/)
[![Python versions](https://img.shields.io/pypi/pyversions/statgpu.svg)](https://pypi.org/project/statgpu/)
[![License](https://img.shields.io/github/license/TheHiddenObserver/statgpu.svg)](https://github.com/TheHiddenObserver/statgpu/blob/master/LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/TheHiddenObserver/statgpu.svg)](https://github.com/TheHiddenObserver/statgpu/stargazers)
[![Downloads](https://img.shields.io/pypi/dm/statgpu.svg)](https://pypi.org/project/statgpu/)

GPU-accelerated statistical methods with sklearn-compatible API.

## Documentation

- **English docs**: [docs/en/](docs/en/) — full documentation index
- **Chinese docs**: [docs/](docs/) — 中文文档
- **Quickstart**: [Quickstart](docs/en/getting-started/quickstart.md)
- **GLM + Penalty**: [Generalized Linear Model](docs/en/models/generalized-linear-model.md) — 7 families × 10 penalties × 3 backends
- **Cross-Validation**: [Cross-Validation Guide](docs/en/guides/cross-validation.md) — PenalizedGLM_CV, LassoCV, RidgeCV
- **Solver-Penalty Matrix**: [Solver × Penalty](docs/en/guides/solver-penalty-matrix.md) — solver dispatch and penalty routing
- **Device & Memory**: [Device and GPU Memory](docs/en/guides/device-and-memory.md)
- **PyTorch Backend**: [PyTorch Backend](docs/en/guides/pytorch-backend.md)
- **Distribution API**: [Distribution API](docs/en/guides/distribution-api.md) — 15 distributions across 3 backends
- **Multiple Testing**: [Multiple Testing](docs/en/guides/multiple-testing-combine-pvalues.md) — p-value adjustment and combination
- **Changelog**: [Changelog](docs/en/changelog.md)

## Features

- 🚀 **3 Backends**: NumPy (CPU), CuPy (CUDA), PyTorch (CUDA) — automatic device selection
- 🔧 **sklearn-compatible**: `fit`/`predict`/`score` API, `sklearn.base.clone()` supported
- 📊 **GLM + Robust + Quantile + Cox**: 10+ loss types (quantile, huber, bisquare, fair, cox_ph + 7 GLM families)
- 🔥 **10 Penalties**: l1, l2, elasticnet, scad, mcp, adaptive_l1, group_lasso, group_mcp, group_scad
- ⚡ **8 Solvers**: exact, newton, lbfgs, irls, fista, fista_bb, proximal_irls_cd, proximal_newton — `solver="auto"`
- 🧮 **Inference**: HC0-HC3/HAC robust SE, debiased Lasso, bootstrap, simultaneous CI
- 📈 **Nonparametric**: KDE, kernel regression, B-splines, GAM
- 🧬 **Unsupervised**: PCA, KMeans, DBSCAN, GMM, UMAP, t-SNE, NNDescent (12+ classes)
- 📐 **Distributions**: 15 distributions across 3 backends via `get_distribution()` — [API docs](docs/en/guides/distribution-api.md)
- 🧪 **Multiple Testing**: `adjust_pvalues` + `combine_pvalues` + `permutation_test`
- 🔥 **Cross-Validation**: PenalizedGLM_CV (all losses × 10 penalties), RidgeCV, LassoCV, ElasticNetCV

## Implemented Methods

> **[Full method list with solvers, penalties, link functions →](docs/en/guides/implemented-methods.md)**

| Category | Classes | Highlights |
|---|---|---|
| **Regression & GLM** | 12 classes | LinearRegression, Ridge, Lasso, ElasticNet, Logistic, Poisson, Gamma, InvGauss, NB, Tweedie, Ordered models |
| **Penalized GLM** | 11 classes | PenalizedGLM + 7 family wrappers + PenalizedQuantileRegression, PenalizedRobustRegression, PenalizedCoxPHModel × 10 penalties × 8 solvers |
| **Cross-Validation** | 6 classes | RidgeCV, LassoCV, ElasticNetCV, LogisticCV, PenalizedGLM_CV, CoxPHCV |
| **ANOVA** | 2 functions | `f_oneway`, `f_twoway` — GPU-accelerated |
| **Covariance** | 3 classes | EmpiricalCovariance, LedoitWolf, OAS |
| **Panel Data** | 2 classes | PanelOLS, RandomEffects |
| **Nonparametric** | 5 classes | KernelRidge, KernelRidgeCV, pairwise_kernels, bspline_basis, natural_cubic_spline_basis |
| **Semiparametric** | 1 class | GAM (penalized B-splines + GCV) |
| **Unsupervised** | 12 classes | PCA, SVD, NMF, UMAP, t-SNE, KMeans, DBSCAN, GMM, AgglomerativeClustering |
| **Survival** | 1 class | CoxPH (Breslow/Efron ties, robust SE) |
| **Feature Selection** | 2 functions | fixed-X / model-X knockoff filters |
| **Multiple Testing** | 3 functions | adjust_pvalues, combine_pvalues, permutation_test |

## Installation

```bash
# CPU only
pip install statgpu

# With GPU support (choose by CUDA major version)
# CUDA 11.x runtime:
pip install statgpu[gpu11]

# CUDA 12.x runtime:
pip install statgpu[gpu12]

# With PyTorch backend (CUDA 11.x)
pip install statgpu[torch]

# Development
pip install statgpu[dev]

# Formula interface
pip install statgpu[formula]
```

## Quick Start

```python
import numpy as np
from statgpu.inference import norm, poisson
from statgpu.linear_model import LinearRegression, PenalizedGLM_CV
from statgpu import adjust_pvalues, combine_pvalues

# Generate data using statgpu distributions (scipy-compatible API)
X = norm.rvs(size=(10000, 100))
y = X @ norm.rvs(size=100) + norm.rvs(size=10000) * 0.5

# Linear regression with GPU
model = LinearRegression(device='cuda')
model.fit(X, y)
print(f"R²: {model.score(X, y):.4f}")

# Penalized GLM with cross-validation
y_pois = poisson.rvs(mu=np.exp(X[:, :5] @ np.ones(5) * 0.1), size=X.shape[0])
cv_model = PenalizedGLM_CV(
    loss="poisson", penalty="elasticnet", l1_ratio=0.5,
    cv=5, device="cpu",
)
cv_model.fit(X[:, :5], y_pois)
print(f"Best alpha: {cv_model.alpha_:.4f}")

# Multiple-testing correction
reject, pvals_adj = adjust_pvalues(np.array([0.003, 0.02, 0.5]), method='bh')

# Global p-value combination
stat, p_global = combine_pvalues(np.array([0.01, 0.07, 0.03, 0.40]), method='fisher')
```

## Device Control

```python
import statgpu as sg

# Global setting
sg.set_device('cuda')  # Force GPU
sg.set_device('cpu')   # Force CPU
sg.set_device('auto')  # Auto-detect (default)

# Per-model setting
from statgpu.linear_model import LinearRegression
model = LinearRegression(device='cuda', n_jobs=4)
```

## Benchmark Results (RTX 4090)

Full reports: `results/unsupervised_bench_2026-06-27.md`, `results/glm_solver_benchmark_2026-06-23.md`

Test environment: RTX 4090 (24GB), CuPy 14.1.0, PyTorch 2.8.0+cu128, scikit-learn 1.8.0, statsmodels 0.14.6, lifelines 0.30.3<br/>
*Benchmark environment only; not installation requirements.*

### Real-Data Performance

| Module | Dataset | n | p | Best Speedup | Precision |
|--------|---------|---|---|-------------|-----------|
| Poisson GLM | freMTPL2 | 678K | 42 | 196.9x vs sklearn | coef_corr=1.000000 |
| Gamma GLM | synthetic | 678K | 42 | 97.9x vs sklearn | coef_corr=0.9995 |
| CoxPH | synthetic | 1.9K | 500 | 1.2x vs CPU | coef_corr=1.000 |
| adjust_pvalues (BH) | synthetic | — | 1M | 0.55x | 100% agreement |
| PenalizedPoisson(L1) | freMTPL2 | 678K | 42 | — | OK |
| PenalizedCoxPH(L2) | synthetic | 1.9K | 500 | — | C-index match |

### Precision Summary

| Module | Metric | Result |
|--------|--------|--------|
| Poisson GLM | coef correlation vs sklearn | 1.000000 (full freMTPL2) |
| Gamma GLM | coef correlation vs sklearn | 0.9995 |
| CoxPH | coef correlation vs lifelines | 1.000 |
| adjust_pvalues (BH) | reject agreement vs statsmodels | 100% (100K to 5M p-values) |
| Penalized (L1/L2) | self-consistency | C-index match across penalties |

## Requirements

- Python >= 3.8
- NumPy >= 1.20
- CuPy (optional, for GPU; choose wheel matching CUDA major version)
  - CUDA 11.x: `cupy-cuda11x`
  - CUDA 12.x: `cupy-cuda12x`
- CUDA runtime compatible with selected CuPy wheel

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
