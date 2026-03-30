# statgpu

GPU-accelerated statistical methods with sklearn-compatible API.

## Features

- 🚀 **GPU Acceleration**: Automatic CUDA support via CuPy
- 🔧 **sklearn-compatible**: Familiar `fit`/`predict` API
- 🔄 **Auto Device Selection**: CPU fallback when GPU unavailable
- 📊 **Statistical Focus**: Methods from R that Python lacks

## Installation

```bash
# CPU only
pip install statgpu

# With GPU support (CUDA 11.x)
pip install statgpu[gpu]

# Development
pip install statgpu[dev]
```

## Quick Start

```python
import numpy as np
from statgpu.linear_model import LinearRegression

# Generate data
X = np.random.randn(10000, 100)
y = X @ np.random.randn(100) + 5

# Fit with GPU
model = LinearRegression(device='cuda')
model.fit(X, y)

# Predict
y_pred = model.predict(X)
print(f"R² score: {model.score(X, y):.4f}")
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
- CuPy >= 9.0 (optional, for GPU)
- CUDA 11.1+ (for GPU)

## License

MIT
