# statgpu

GPU-accelerated statistical methods with sklearn-compatible API.

## Documentation

- Usage portal: `USAGE.md`
- Quickstart: `docs/getting-started/quickstart.md`
- Guides:
  - `docs/guides/device-and-memory.md`
  - `docs/guides/inference-modes.md`
- Model docs:
  - `docs/models/linear-regression.md`
  - `docs/models/ridge.md`
  - `docs/models/lasso.md`
  - `docs/models/logistic-regression.md`
  - `docs/models/coxph.md`
- Benchmarks: `docs/benchmarks.md`

## Features

- 🚀 **GPU Acceleration**: Automatic CUDA support via CuPy
- 🔧 **sklearn-compatible**: Familiar `fit`/`predict` API
- 🔄 **Auto Device Selection**: CPU fallback when GPU unavailable
- 📊 **Statistical Focus**: Methods from R that Python lacks

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

## Requirements

- Python >= 3.8
- NumPy >= 1.20
- CuPy (optional, for GPU; choose wheel matching CUDA major version)
  - CUDA 11.x: `cupy-cuda11x`
  - CUDA 12.x: `cupy-cuda12x`
- CUDA runtime compatible with selected CuPy wheel

## License

MIT
