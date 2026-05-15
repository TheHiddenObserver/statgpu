# statgpu

GPU-accelerated statistical methods with sklearn-compatible API.

## Features

- 🚀 **GPU Acceleration**: CUDA acceleration for supported estimators
- 🔧 **sklearn-compatible**: Familiar `fit`/`predict` API
- 🔄 **Auto Device Selection**: `device="auto"` can choose an available backend; explicit devices never silently fall back
- 📊 **Statistical Focus**: Methods from R that Python lacks
- 🧪 **Multiple Testing**: `adjust_pvalues` (`bh`/`by`/`holm`/`bonferroni`/`hochberg`) + `combine_pvalues` (`fisher`/`cauchy`/`stouffer`)
- 🧮 **Inference Support**:
  - `LinearRegression`: `nonrobust` / `hc0` / `hc1` / `hc2` / `hc3` / `hac`
  - `Ridge`: `nonrobust` / `hc0` / `hc1` / `hc2` / `hc3` / `hac`
  - `Lasso`: `debiased` / `cpu_ols_inference` / `gpu_ols_inference` / `bootstrap`
  - `LogisticRegression`: `nonrobust` / `hc0` / `hc1` / `hc2` / `hc3` / `hac`
- 📈 **Nonparametric Support**:
  - KDE: `fit_kde` / `kde_pdf` / `kde_bootstrap_confidence_interval`
  - KDE kernel options: `gaussian` / `rectangular` / `triangular` / `epanechnikov` / `biweight` / `triweight` / `cosine` / `optcosine`
  - Kernel regression: `fit_kernel_regression` / `kernel_regression_predict`
- 🧬 **Feature Selection**: Stepwise regression plus fixed-X and model-X knockoff filters
- 🧭 **Unsupervised Learning**:
  - Dimensionality and factorization: `PCA`, `TruncatedSVD`, `IncrementalPCA`, `NMF`, `MiniBatchNMF`
  - Clustering and mixtures: `KMeans`, `MiniBatchKMeans`, `DBSCAN`, `GaussianMixture`, `AgglomerativeClustering`
  - Manifold embeddings: `UMAP`, `TSNE`
- 🧹 **GPU Memory Control**: `gpu_memory_cleanup` for all current models
- 🔥 **PyTorch Backend**: Optional Torch execution path for supported estimators
- 📐 **Unified Distribution Backend**: 15 distributions (norm, t, f, chi2, gamma, beta, uniform, expon, cauchy, laplace, logistic, weibull_min, lognorm, poisson, binom) via `get_distribution()`

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

### Unsupervised Learning

- `statgpu.unsupervised.PCA`
- `statgpu.unsupervised.KMeans`
- `statgpu.unsupervised.DBSCAN`
- `statgpu.unsupervised.GaussianMixture`
- `statgpu.unsupervised.NMF`
- `statgpu.unsupervised.AgglomerativeClustering`
- `statgpu.unsupervised.TruncatedSVD`
- `statgpu.unsupervised.MiniBatchKMeans`
- `statgpu.unsupervised.IncrementalPCA`
- `statgpu.unsupervised.MiniBatchNMF`
- `statgpu.unsupervised.UMAP`
- `statgpu.unsupervised.TSNE`

Exported CV classes:

- `statgpu.linear_model.RidgeCV`
- `statgpu.linear_model.LogisticRegressionCV`
- `statgpu.survival.CoxPHCV`

Backend support and feature parity are documented per model in the Model Docs Index below.

## Model Docs Index

General documentation:

- Primary usage portal: [English](USAGE.md) / [Chinese](USAGE_CN.md)
- Documentation root: [English](docs/en/) / [Chinese](docs/)
- Repo development layout: [dev/](dev/) (`tests/`, `benchmarks/`, `comparisons/`, `validation/`, `manual/`, `scripts/`)

Model documentation:

- Linear regression: [EN](docs/en/models/linear-regression.md) / [CN](docs/models/linear-regression.md)
- Logistic regression: [EN](docs/en/models/logistic-regression.md) / [CN](docs/models/logistic-regression.md)
- Poisson regression: [EN](docs/en/models/poisson-regression.md) / [CN](docs/models/poisson-regression.md)
- Generalized linear model: [EN](docs/en/models/generalized-linear-model.md) / [CN](docs/models/generalized-linear-model.md)
- Ridge: [EN](docs/en/models/ridge.md) / [CN](docs/models/ridge.md)
- Lasso: [EN](docs/en/models/lasso.md) / [CN](docs/models/lasso.md)
- Elastic Net: [EN](docs/en/models/elastic-net.md) / [CN](docs/models/elastic-net.md)
- Ordered regression: [EN](docs/en/models/ordered.md) / [CN](docs/models/ordered.md)
- Cox proportional hazards: [EN](docs/en/models/coxph.md) / [CN](docs/models/coxph.md)
- Nonparametric methods: [EN](docs/en/models/nonparametric.md) / [CN](docs/models/nonparametric.md)
- Knockoff filter: [EN](docs/en/models/knockoff.md) / [CN](docs/models/knockoff.md)
- Unsupervised models: [EN](docs/en/models/unsupervised.md) / [CN](docs/models/unsupervised.md)

## Installation

`statgpu` is currently installed from source or directly from GitHub. Start with the CPU-only install, then add the optional backend extras you need.

### 1. Create an environment

```bash
conda create -n statgpu python=3.10
conda activate statgpu
python -m pip install --upgrade pip
```

### 2. Install the package

```bash
# Local editable install from a cloned checkout
pip install -e .

# Or install directly from GitHub
pip install "git+https://github.com/TheHiddenObserver/statgpu.git"
```

### 3. Add optional extras

Choose only the extras that match your environment:

| Use case | Command | Notes |
| --- | --- | --- |
| CUDA 11.x via CuPy | `pip install -e ".[gpu11]"` | Installs `cupy-cuda11x`; do not combine with `gpu12`. |
| CUDA 12.x via CuPy | `pip install -e ".[gpu12]"` | Installs `cupy-cuda12x`; do not combine with `gpu11`. |
| PyTorch CUDA backend | `pip install -e ".[torch]"` | Installs PyTorch from PyPI. If you need a specific CUDA wheel, install PyTorch from the official PyTorch index first, then install `statgpu`. |
| Optional CPU extension build deps | `pip install -e ".[cpu_ext]"` | Adds `Cython>=3.0` for optional Cython CPU extensions. |
| Formula/dataframe interface | `pip install -e ".[formula]"` | Adds `patsy` and `pandas`. |
| Development tools | `pip install -e ".[dev]"` | Adds pytest, black, flake8, and mypy. |
| External validation | `pip install -e ".[validation]"` | Adds scikit-learn and statsmodels for comparison tests. |

For a non-editable GitHub install with extras, append the extra to the package name:

```bash
pip install "statgpu[gpu12] @ git+https://github.com/TheHiddenObserver/statgpu.git"
```

## Requirements

### Core requirements

- Python >= 3.8
- NumPy >= 1.20
- SciPy >= 1.7
- joblib >= 1.0

### Optional backend requirements

- **CPU (`device="cpu"`)**: no optional GPU dependency required.
- **CuPy CUDA (`device="cuda"`)**: install exactly one CuPy wheel that matches your CUDA major version:
  - CUDA 11.x: `cupy-cuda11x>=13.0` via `statgpu[gpu11]`
  - CUDA 12.x: `cupy-cuda12x>=13.0` via `statgpu[gpu12]`
- **PyTorch CUDA (`device="torch"`)**: PyTorch >= 2.0 with CUDA support is recommended. Explicit `device="torch"` requires Torch CUDA to be available; it does not silently fall back to Torch CPU.
- **Formula interface**: `patsy>=0.5.3` and `pandas>=1.5` via `statgpu[formula]`.

### Device selection notes

- `device="cpu"` uses NumPy.
- `device="cuda"` uses CuPy and raises an error if CuPy/CUDA is unavailable.
- `device="torch"` uses Torch CUDA and raises an error if Torch CUDA is unavailable.
- `device="auto"` is the only mode that automatically chooses an available backend, usually preferring CuPy, then Torch CUDA, then NumPy.

```python
from statgpu.linear_model import Ridge

cpu_model = Ridge(alpha=1.0, device="cpu")
cupy_model = Ridge(alpha=1.0, device="cuda")
torch_model = Ridge(alpha=1.0, device="torch")
auto_model = Ridge(alpha=1.0, device="auto")
```

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

## License

MIT
