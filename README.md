# statgpu

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
- 📊 **7 GLM Families**: squared_error, logistic, poisson, gamma, inverse_gaussian, negative_binomial, tweedie
- 🔥 **10 Penalties**: l1, l2, elasticnet, scad, mcp, adaptive_l1, group_lasso, group_mcp, group_scad
- ⚡ **6 Solvers**: exact, newton, lbfgs, irls, fista, fista_bb — `solver="auto"` selects optimal
- 🧮 **Inference**: HC0-HC3/HAC robust SE, debiased Lasso, bootstrap, simultaneous CI
- 📈 **Nonparametric**: KDE, kernel regression, B-splines, GAM
- 🧬 **Unsupervised**: PCA, KMeans, DBSCAN, GMM, UMAP, t-SNE (12 classes)
- 📐 **Distributions**: 15 distributions across 3 backends via `get_distribution()` — [API docs](docs/en/guides/distribution-api.md)
- 🧪 **Multiple Testing**: `adjust_pvalues` + `combine_pvalues` + `permutation_test`
- 🔥 **Cross-Validation**: PenalizedGLM_CV (all 7 losses × 10 penalties), RidgeCV, LassoCV, ElasticNetCV

## Implemented Methods

> **[Full method list with solvers, penalties, link functions →](docs/en/guides/implemented-methods.md)**

| Category | Classes | Highlights |
|---|---|---|
| **Regression & GLM** | 12 classes | LinearRegression, Ridge, Lasso, ElasticNet, Logistic, Poisson, Gamma, InvGauss, NB, Tweedie, Ordered models |
| **Penalized GLM** | 8 classes | PenalizedGLM + 7 family wrappers (Linear, Logistic, Poisson, Gamma, InvGauss, NB, Tweedie) × 10 penalties × 6 solvers |
| **Cross-Validation** | 6 classes | RidgeCV, LassoCV, ElasticNetCV, LogisticCV, PenalizedGLM_CV, CoxPHCV |
| **ANOVA** | 1 function | `f_oneway` — GPU-accelerated |
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
from statgpu.linear_model import LinearRegression, Lasso
from statgpu import adjust_pvalues, combine_pvalues, permutation_test

# Generate data
X = np.random.randn(10000, 100)
y = X @ np.random.randn(100) + 5

# Fit with GPU
model = LinearRegression(device='cuda')
model.fit(X, y)

# Predict
y_pred = model.predict(X)
print(f"R² score: {model.score(X, y):.4f}")

# Lasso with GPU-side inference and optional VRAM cleanup
lasso = Lasso(
    alpha=0.1,
    device='cuda',
    inference_method='gpu_ols_inference',
    gpu_memory_cleanup=True,
)
lasso.fit(X, y)

# Multiple-testing adjustment (BH/BY/Holm/Bonferroni/Hochberg)
reject, pvals_adj = adjust_pvalues(np.array([0.003, 0.02, 0.5]), method='bh')

# Global p-value combination (Fisher/Cauchy/Stouffer)
stat, p_global = combine_pvalues(np.array([0.01, 0.07, 0.03, 0.40]), method='fisher')

# Permutation test helper
p = permutation_test(
  lambda X_, y_: np.corrcoef(X_[:, 0], y_)[0, 1],
  X[:, :1],
  y,
  n_resamples=200,
  random_state=0,
).pvalue

# Knockoff feature selection with Torch GPU
from statgpu import fixed_x_knockoff_filter
import torch

X_knock = np.random.randn(1000, 50)
y_knock = X_knock[:, :10] @ np.ones(10) + np.random.randn(1000)

# Torch GPU backend for knockoff (faster on large datasets)
X_torch = torch.from_numpy(X_knock).to('cuda')
y_torch = torch.from_numpy(y_knock).to('cuda')

result = fixed_x_knockoff_filter(
    X_torch, y_torch,
    q=0.1, method='lasso_coef_diff',
    backend='torch', random_state=42
)
print(f"Selected features: {result.selected_features}")
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

Full report: `dev/tests/_bench_realdata_report.md`

Test environment: RTX 4090 (24GB), CuPy 14.1.0, PyTorch 2.8.0+cu128, scikit-learn 1.8.0, statsmodels 0.14.6, lifelines 0.30.3

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

MIT
