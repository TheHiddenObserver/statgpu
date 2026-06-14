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

- 🚀 **GPU Acceleration**: Automatic CUDA support via CuPy and PyTorch
- 🔧 **sklearn-compatible**: Familiar `fit`/`predict` API
- 🔄 **Auto Device Selection**: `device="auto"` can choose an available backend; explicit `cuda`/`torch` never silently falls back to CPU
- 📊 **Statistical Focus**: Methods from R that Python lacks
- 🧪 **Multiple Testing**: `adjust_pvalues` (`bh`/`by`/`holm`/`bonferroni`/`hochberg`) + `combine_pvalues` (`fisher`/`cauchy`/`stouffer`) across 3 backends (numpy/cupy/torch)
- 🧮 **Inference Support**:
  - `LinearRegression`, `Ridge`, `LogisticRegression`: `nonrobust` / `hc0` / `hc1` / `hc2` / `hc3` / `hac`
  - `Lasso`: `debiased` / `cpu_ols_inference` / `bootstrap` inference methods
- 📈 **Nonparametric Support**:
  - KDE: `fit_kde` / `kde_pdf` / `kde_bootstrap_confidence_interval`
  - KDE kernel options: `gaussian` / `rectangular` / `triangular` / `epanechnikov` / `biweight` / `triweight` / `cosine` / `optcosine`
  - Kernel regression: `fit_kernel_regression` / `kernel_regression_predict`
- 🧹 **GPU Memory Control**: `gpu_memory_cleanup` for all current models
- 🔥 **PyTorch Backend**: Optional Torch backend for GPU acceleration (PyTorch 2.0+)
  - All models support `device='torch'` for CUDA-accelerated PyTorch backend
  - **Knockoff filter**: `fixed_x_knockoff_filter`, `model_x_knockoff_filter` with `backend='torch'`
- 📐 **Unified Distribution Backend**: 15 distributions (norm, t, f, chi2, gamma, beta, uniform, expon, cauchy, laplace, logistic, weibull_min, lognorm, poisson, binom) across 3 backends (numpy/cupy/torch) via `get_distribution()`. GPU speedup 10-500x at 1M points. [API docs](docs/en/guides/distribution-api.md)

## Implemented Methods

> **[Full method list with solvers, penalties, link functions →](docs/en/guides/implemented-methods.md)**

| Category | Classes | Highlights |
|---|---|---|
| **Regression & GLM** | 12 classes | LinearRegression, Ridge, Lasso, ElasticNet, Logistic, Poisson, Gamma, InvGauss, NB, Tweedie, Ordered models |
| **Penalized GLM** | 4 classes | 7 families × 10 penalties × 6 solvers × 3 backends |
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

## GLM and Penalized GLM Notes

- Full model documentation: `docs/en/models/generalized-linear-model.md` / `docs/models/generalized-linear-model.md`
- `statgpu.glm_core` is the GLM-specific core layer; `statgpu.losses` is not a compatibility namespace.
- **7 GLM families**: squared_error, logistic, poisson, gamma, inverse_gaussian, negative_binomial, tweedie
- **10 penalties**: none, l1, l2, elasticnet, scad, mcp, adaptive_l1, group_lasso, group_mcp, group_scad
- **6 solvers**: exact, newton, lbfgs, irls, fista, fista_bb
- **3 backends**: CPU (NumPy), CuPy (CUDA), PyTorch (CUDA)
- `PenalizedGLM_CV` provides unified cross-validation for all loss × penalty combinations.
- `Ridge`, `Lasso`, and `ElasticNet` are thin sklearn-style wrappers over typed penalized gaussian regression.
- `Ridge` supports `solver="exact"` for the closed-form L2 solution.
- `solver="auto"` routes: smooth penalties → IRLS, non-smooth → FISTA, non-convex → LLA+FISTA.
- Explicit `device="cuda"` or `device="torch"` never silently falls back to CPU.
- `sklearn.base.clone()` is supported for all estimators (verified via `get_params` round-trip).
- Formula fitting is optional and uses patsy, for example `model.fit(formula="y ~ x1 + C(group)", data=df)`.
- Benchmark: **1043/1043 ALL PASS** (v23c full matrix, 7 families × 10 penalties × 3 backends).

### PyTorch Backend Requirements

- PyTorch 2.0+ (for `torch.special` functions)
- CUDA 11.x or 12.x driver
- Recommended: Use conda environment with pre-configured CUDA toolkit

```bash
# Example conda setup
conda create -n statgpu python=3.10
conda activate statgpu
conda install pytorch cudatoolkit=11.7 -c pytorch
pip install statgpu
```

**Torch Backend Usage**:

```python
from statgpu.linear_model import Ridge, LogisticRegression, Lasso

# Torch GPU backend
model = Ridge(alpha=1.0, device='torch')  # Force Torch backend

# GPU auto mode (prefers CuPy when available)
model = Ridge(alpha=1.0, device='cuda')

# Torch CPU backend (useful for debugging)
model = Ridge(alpha=1.0, device='cpu')

# All covariance types supported
model = LogisticRegression(device='torch', cov_type='hc3')
model.fit(X, y)
print(f"Std Errors: {model._bse}")
```

**Performance Notes**:
- **Small datasets (<10K)**: CuPy faster due to lower overhead
- **Moderate-large datasets (20K-100K)**: Torch GPU competitive with CuPy
- **Robust covariance (HC2/HC3)**: Torch GPU within 4-30% of CuPy, 60x faster than CPU
- See `dev/docs/torch_backend_full_feature_report.md` for detailed benchmarks

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

## GLM + Penalty Example

```python
import numpy as np
from statgpu.linear_model import (
    PenalizedGLM_CV,
    PenalizedLogisticRegression,
    PoissonRegression,
    GammaRegression,
    NegativeBinomialRegression,
)

# Generate Poisson data
rng = np.random.default_rng(42)
X = rng.standard_normal((2000, 20))
beta = np.zeros(20); beta[:5] = [2, -1.5, 1, -0.5, 0.3]
y = rng.poisson(np.exp(X @ beta))

# PenalizedGLM_CV: unified CV for any loss × penalty
model = PenalizedGLM_CV(
    loss="poisson",
    penalty="elasticnet",
    l1_ratio=0.5,
    n_alphas=50,
    cv=5,
    device="cpu",  # or "cuda" for GPU
)
model.fit(X, y)
print(f"Best alpha: {model.alpha_:.4f}")
print(f"Non-zero coefficients: {np.sum(np.abs(model.coef_) > 1e-6)}")
print(f"Score: {model.score(X, y):.4f}")

# Negative Binomial with custom dispersion
nb_model = PenalizedGLM_CV(
    loss="negative_binomial",
    penalty="l1",
    loss_kwargs={"alpha": 2.0},  # custom dispersion
    device="cpu",
)
nb_model.fit(X, y)

# Direct model usage (no CV)
poisson = PoissonRegression(alpha=0.1, device="cpu")
poisson.fit(X, y)
print(f"Poisson coef[:3]: {poisson.coef_[:3]}")

gamma = GammaRegression(alpha=0.05, device="cpu")
gamma.fit(X, np.abs(y) + 1)
print(f"Gamma coef[:3]: {gamma.coef_[:3]}")
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

## Benchmark Scripts

- Multiple-testing timing benchmarks (3 backends + statsmodels/scipy):
  - `dev/benchmarks/_bench_inference_timing.py` (p=100-10k)
  - `dev/benchmarks/_bench_inference_timing_large.py` (p=50k-1M)
- Large-scale all-method runtime benchmark:
  - `dev/benchmarks/benchmark_all_methods_large_scale.py`
- Multi-target LinearRegression benchmark (statgpu vs sklearn vs R):
  - `dev/benchmarks/benchmark_multitarget_sklearn_r.py`
- Lasso inference CPU/GPU comparison:
  - `dev/benchmarks/benchmark_lasso_inference_gpu_vs_cpu.py`
- GPU memory cleanup effect:
  - `dev/benchmarks/benchmark_gpu_memory_cleanup.py`

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
