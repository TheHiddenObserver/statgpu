# statgpu

GPU-accelerated statistical methods with sklearn-compatible API.

## Documentation

- Primary usage portal (English): `USAGE.md`
- Chinese usage portal: `USAGE_CN.md`
- English docs root: `docs/en/`
- Chinese docs root: `docs/`
- Repo development layout: `dev/` (`tests/`, `benchmarks/`, `comparisons/`, `validation/`, `manual/`, `scripts/` for Cox data + R bench helpers)

## Features

- 🚀 **GPU Acceleration**: Automatic CUDA support via CuPy
- 🔧 **sklearn-compatible**: Familiar `fit`/`predict` API
- 🔄 **Auto Device Selection**: CPU fallback when GPU unavailable
- 📊 **Statistical Focus**: Methods from R that Python lacks
- 🧮 **Inference Support**:
  - `LinearRegression`: `nonrobust` / `hc0` / `hc1`
  - `Ridge`: `nonrobust` / `hc0` / `hc1`
  - `Lasso`: `cpu_ols_inference` / `gpu_ols_inference` / `bootstrap`
  - `LogisticRegression`: `nonrobust` / `hc0` / `hc1`
- 🧹 **GPU Memory Control**: `gpu_memory_cleanup` for all current models

## Implemented Methods (Current)

- `statgpu.linear_model.LinearRegression`
- `statgpu.linear_model.Ridge`
- `statgpu.linear_model.Lasso`
- `statgpu.linear_model.LogisticRegression`
- `statgpu.survival.CoxPH`

## Installation

```bash
# CPU only
pip install statgpu

# With GPU support (choose by CUDA major version)
# CUDA 11.x runtime:
pip install statgpu[gpu11]

# CUDA 12.x runtime:
pip install statgpu[gpu12]

# Development
pip install statgpu[dev]
```

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
