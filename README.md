# statgpu

GPU-accelerated statistical methods with sklearn-compatible API.

## Documentation

- Primary usage portal (English): `USAGE.md`
- Chinese usage portal: `USAGE_CN.md`
- English docs root: `docs/en/`
- Chinese docs root: `docs/`
- Repo development layout: `dev/` (`tests/`, `benchmarks/`, `comparisons/`, `validation/`, `manual/`, `scripts/` for Cox data + R bench helpers)

## Features

- 🚀 **GPU Acceleration**: Automatic CUDA support via CuPy and PyTorch
- 🔧 **sklearn-compatible**: Familiar `fit`/`predict` API
- 🔄 **Auto Device Selection**: `device="auto"` can choose an available backend; explicit `cuda`/`torch` never silently falls back to CPU
- 📊 **Statistical Focus**: Methods from R that Python lacks
- 🧪 **Multiple Testing**: `adjust_pvalues` (`bh`/`by`/`holm`/`bonferroni`/`hochberg`) + `combine_pvalues` (`fisher`/`cauchy`/`stouffer`) across 3 backends (numpy/cupy/torch)
- 🧮 **Inference Support**:
  - `LinearRegression`: `nonrobust` / `hc0` / `hc1` / `hc2` / `hc3` / `hac`
  - `Ridge`: `nonrobust` / `hc0` / `hc1` / `hc2` / `hc3` / `hac` ✅ (Torch backend)
  - `Lasso`: `cpu_ols_inference` / `gpu_ols_inference` / `bootstrap` ✅ (Torch backend)
  - `LogisticRegression`: `nonrobust` / `hc0` / `hc1` / `hc2` / `hc3` / `hac` ✅ (Torch backend)
- 📈 **Nonparametric Support**:
  - KDE: `fit_kde` / `kde_pdf` / `kde_bootstrap_confidence_interval`
  - KDE kernel options: `gaussian` / `rectangular` / `triangular` / `epanechnikov` / `biweight` / `triweight` / `cosine` / `optcosine`
  - Kernel regression: `fit_kernel_regression` / `kernel_regression_predict`
- 🧹 **GPU Memory Control**: `gpu_memory_cleanup` for all current models
- 🔥 **PyTorch Backend**: Optional Torch backend for GPU acceleration (PyTorch 2.0+)
  - Supported models: `Ridge`, `LogisticRegression`, `Lasso`, `LassoCV`, `CoxPH`
  - **Knockoff filter**: `fixed_x_knockoff_filter`, `model_x_knockoff_filter` with `backend='torch'`
- 📐 **Unified Distribution Backend**: 15 distributions (norm, t, f, chi2, gamma, beta, uniform, expon, cauchy, laplace, logistic, weibull_min, lognorm, poisson, binom) across 3 backends (numpy/cupy/torch) via `get_distribution()`. GPU speedup 10-500x at 1M points. [API docs](docs/en/guides/distribution-api.md)

## Implemented Methods (Current)

### Linear / Gaussian Regression

- `statgpu.linear_model.LinearRegression`
- `statgpu.linear_model.PenalizedLinearRegression`
- `statgpu.linear_model.Ridge`
- `statgpu.linear_model.Lasso`
- `statgpu.linear_model.ElasticNet`

### Generalized Linear Models (GLM)

- `statgpu.linear_model.GeneralizedLinearModel`
- `statgpu.linear_model.PoissonRegression`
- `statgpu.linear_model.LogisticRegression`
- `statgpu.linear_model.PenalizedLogisticRegression`
- `statgpu.linear_model.PenalizedPoissonRegression`

### Ordered / Survival Models

- `statgpu.linear_model.OrderedLogitRegression`
- `statgpu.linear_model.OrderedProbitRegression`
- `statgpu.survival.CoxPH`

Exported CV classes:

- `statgpu.linear_model.RidgeCV`
- `statgpu.linear_model.LogisticRegressionCV`
- `statgpu.survival.CoxPHCV`

Backend support and feature parity are documented per model in the Model Docs Index below.

## Model Docs Index

- Linear regression: `docs/en/models/linear-regression.md` / `docs/models/linear-regression.md`
- Logistic regression: `docs/en/models/logistic-regression.md` / `docs/models/logistic-regression.md`
- Poisson regression: `docs/en/models/poisson-regression.md` / `docs/models/poisson-regression.md`
- Generalized linear model: `docs/en/models/generalized-linear-model.md` / `docs/models/generalized-linear-model.md`
- Ridge: `docs/en/models/ridge.md` / `docs/models/ridge.md`
- Lasso: `docs/en/models/lasso.md` / `docs/models/lasso.md`
- Elastic Net: `docs/en/models/elastic-net.md` / `docs/models/elastic-net.md`
- Ordered regression: `docs/en/models/ordered.md` / `docs/models/ordered.md`
- Cox proportional hazards: `docs/en/models/coxph.md` / `docs/models/coxph.md`
- Nonparametric methods: `docs/en/models/nonparametric.md` / `docs/models/nonparametric.md`
- Knockoff filter: `docs/en/models/knockoff.md` / `docs/models/knockoff.md`

## Installation

```bash
# Local editable install (current recommended path before PyPI release)
pip install -e .

# Direct install from GitHub
pip install "git+https://github.com/TheHiddenObserver/statgpu.git"

# With GPU support (choose by CUDA major version)
# CUDA 11.x runtime:
pip install -e ".[gpu11]"

# CUDA 12.x runtime:
pip install -e ".[gpu12]"

# With PyTorch backend
pip install -e ".[torch]"

# Development
pip install -e ".[dev]"

# Formula interface
pip install -e ".[formula]"
```

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
y = X @ np.random.randn(100) + np.random.randn(10000) * 0.5 + 5

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

## Requirements

- Python >= 3.8
- NumPy >= 1.20
- CuPy (optional, for GPU; choose wheel matching CUDA major version)
  - CUDA 11.x: `cupy-cuda11x`
  - CUDA 12.x: `cupy-cuda12x`
- CUDA runtime compatible with selected CuPy wheel

## License

MIT
