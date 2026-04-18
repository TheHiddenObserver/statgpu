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
- 🔄 **Auto Device Selection**: CPU fallback when GPU unavailable
- 📊 **Statistical Focus**: Methods from R that Python lacks
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

## Implemented Methods (Current)

- `statgpu.linear_model.LinearRegression`
- `statgpu.linear_model.Ridge` ✅ (Torch backend)
- `statgpu.linear_model.Lasso` ✅ (Torch backend)
- `statgpu.linear_model.LassoCV`
- `statgpu.linear_model.LogisticRegression` ✅ (Torch backend)
- `statgpu.survival.CoxPH` ✅ (Torch backend)

Exported CV interface skeletons (pending full CV training/search implementation):

- `statgpu.linear_model.RidgeCV`
- `statgpu.linear_model.LogisticRegressionCV`
- `statgpu.survival.CoxPHCV`

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
from statgpu import adjust_pvalues, permutation_test

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

# Multiple-testing adjustment (BH/BY/Holm/Bonferroni)
reject, pvals_adj = adjust_pvalues(np.array([0.003, 0.02, 0.5]), method='bh')

# Permutation test helper
p = permutation_test(
  lambda X_, y_: np.corrcoef(X_[:, 0], y_)[0, 1],
  X[:, :1],
  y,
  n_resamples=200,
  random_state=0,
).pvalue
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
