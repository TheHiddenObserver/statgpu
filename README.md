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
- **Survival Analysis**: [Cox Proportional Hazards](docs/en/models/coxph.md) — CoxPH, CoxPHCV, and penalized Cox
- **Loss × Penalty × Solver Framework**: [Framework Guide](docs/en/guides/loss-penalty-solver-framework.md) — complete architecture, dispatch logic, coverage matrix
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
- 🔥 **Penalty framework**: 10 registered penalties; estimator-specific support varies (`PenalizedCoxPHModel` is validated for five)
- ⚡ **Solver framework**: 8 registered solvers with estimator-specific routing through `solver="auto"`
- 🧮 **Inference**: sandwich/oracle inference for supported penalized GLMs; analytical Hessian for ordered models; kernel + bootstrap for quantile regression; debiased Lasso + simultaneous CI; CoxPH model-based/robust inference. `PenalizedCoxPHModel` is estimation-only.
- 📈 **Nonparametric**: KDE, kernel regression, B-splines, GAM
- 🧬 **Unsupervised**: PCA, KMeans, DBSCAN, GMM, UMAP, t-SNE, NNDescent (12+ classes)
- 📐 **Distributions**: 15 distributions across 3 backends via `get_distribution()` — [API docs](docs/en/guides/distribution-api.md)
- 🧪 **Multiple Testing**: `adjust_pvalues` + `combine_pvalues` + `permutation_test`
- 🔥 **Cross-Validation**: PenalizedGLM_CV (supported GLM losses/penalties), RidgeCV, LassoCV, ElasticNetCV, CoxPHCV

## Implemented Methods

> **[Full method list with solvers, penalties, link functions →](docs/en/guides/implemented-methods.md)**

| Category | Classes | Highlights |
|---|---|---|
| **Regression & GLM** | 13 classes | LinearRegression, Ridge, Lasso, ElasticNet, Logistic, Poisson, Gamma, InvGauss, NB, Tweedie, QuantileRegression, Ordered models (logit/probit, GPU inference) |
| **Penalized GLM** | 11 classes | PenalizedGLM + 7 family wrappers + PenalizedQuantileRegression, PenalizedRobustRegression; PenalizedCoxPHModel validated for L1/L2/ElasticNet/SCAD/MCP (estimation-only) |
| **Cross-Validation** | 6 classes | RidgeCV, LassoCV, ElasticNetCV, LogisticCV, PenalizedGLM_CV, CoxPHCV |
| **ANOVA** | 2 functions | `f_oneway`, `f_twoway` — GPU-accelerated |
| **Covariance** | 3 classes | EmpiricalCovariance, LedoitWolf, OAS |
| **Panel Data** | 2 classes | PanelOLS, RandomEffects |
| **Nonparametric** | 5 classes | KernelRidge, KernelRidgeCV, pairwise_kernels, bspline_basis, natural_cubic_spline_basis |
| **Semiparametric** | 1 class | GAM (penalized B-splines + GCV) |
| **Unsupervised** | 12 classes | PCA, SVD, NMF, UMAP, t-SNE, KMeans, DBSCAN, GMM, AgglomerativeClustering |
| **Survival** | 2 classes | CoxPH and CoxPHCV: Breslow/Efron/Exact ties, delayed entry, `(start, stop]` data, strata, baseline survival, and subject-grouped CV; Exact covariance is nonrobust only |
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

## Benchmark Results

Full reports: `results/unsupervised_bench_2026-06-27.md`, `results/glm_solver_benchmark_2026-06-23.md`, `results/survival_completion_full_2026-07-12.json`

Results below come from different dated artifacts and machines; consult the linked artifact for its exact environment and timing scope. They are not installation requirements.

### Survival Phase-1 snapshot (RTX 5880 Ada)

The ratio below is `NumPy fit time / GPU fit time`, so values above 1 mean the GPU was faster. Fit timing used warm backend arrays and included optimization, inference, and baseline estimation; transfer was measured separately. The run used float64, one warm-up, and two timed repeats, so these figures are indicative rather than universal performance claims.

| Scale / scenario | Configuration | CuPy / NumPy | Torch / NumPy |
|---|---:|---:|---:|
| Quick delayed entry | 700 rows, 8 features | 0.647x | 0.959x |
| Full delayed entry | 2,500 rows, 16 features | 1.044x | 1.374x |
| Full stratified start-stop | 2,400 rows, 16 features, 4 strata | 0.241x | 0.411x |
| Full standard heavy ties | 20,000 rows, 32 features, Efron | 0.850x | 0.436x |
| Full Exact ties | 120 rows, 4 features | 0.069x | 0.095x |

Only delayed-entry fitting crossed 1x in parts of this benchmark. Standard, stratified start-stop, and Exact workloads were slower than NumPy at the measured scales; Exact currently prioritizes correctness through dynamic programming. The three CV backends selected the same penalty, and final-refit coefficient/SE differences were below `1e-16`. See `results/survival_completion_2026-07-12.json` and `results/survival_completion_full_2026-07-12.json` for precision, convergence, compatibility, and reproducibility metadata.

### Real-Data Performance

| Module | Dataset | n | p | Best Speedup | Precision |
|--------|---------|---|---|-------------|-----------|
| Poisson GLM | freMTPL2 | 678K | 42 | 196.9x vs sklearn | coef_corr=1.000000 |
| Gamma GLM | synthetic | 678K | 42 | 97.9x vs sklearn | coef_corr=0.9995 |
| adjust_pvalues (BH) | synthetic | — | 1M | 0.55x | 100% agreement |
| PenalizedPoisson(L1) | freMTPL2 | 678K | 42 | — | OK |

### Precision Summary

| Module | Metric | Result |
|--------|--------|--------|
| Poisson GLM | coef correlation vs sklearn | 1.000000 (full freMTPL2) |
| Gamma GLM | coef correlation vs sklearn | 0.9995 |
| adjust_pvalues (BH) | reject agreement vs statsmodels | 100% (100K to 5M p-values) |

## Requirements

- Python >= 3.9
- NumPy >= 1.20
- CuPy (optional, for GPU; choose wheel matching CUDA major version)
  - CUDA 11.x: `cupy-cuda11x`
  - CUDA 12.x: `cupy-cuda12x`
- CUDA runtime compatible with selected CuPy wheel

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
