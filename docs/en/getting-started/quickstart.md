# Quickstart

> Language: English  
> Last updated: 2026-10-07  
> This page: Getting started  
> Switch: [Chinese](../../cn/getting-started/quickstart.md)

Language switch: [Chinese](../../cn/getting-started/quickstart.md)

## Installation

With Python 3.9 or later, install from an existing repository checkout:

```bash
cd statgpu
python -m pip install -e .
```

This installs the base NumPy, SciPy, and joblib dependencies. The CPU example below
needs neither a GPU nor CuPy or PyTorch. Optional GPU setup is described separately
[below](#optional-gpu-execution).

## Minimal Example

```python
import numpy as np
from statgpu.linear_model import LinearRegression

rng = np.random.default_rng(42)
X = rng.normal(size=(1000, 20))
y = 1.0 + X @ rng.normal(size=20) + 0.1 * rng.normal(size=1000)

model = LinearRegression(device="cpu")
model.fit(X, y)
print(model.predict(X[:3]))
print(model.score(X, y))
```

The first output contains predictions for three rows. The second is the training
R-squared, close to 1 for this low-noise simulated example. It does not measure
performance on new data; use held-out data or cross-validation for that purpose.

## Device Control

```python
import statgpu as sg

sg.set_device("cpu")
print(sg.get_device().value)  # cpu
sg.set_device("auto")        # restore automatic selection
```

The first example explicitly selects CPU regardless of the global setting.
`sg.set_device("auto")` restores the global automatic policy, which resolves to
available CuPy CUDA, then Torch CUDA, then NumPy CPU. For GPU execution in this
Quickstart, set the estimator's device explicitly as shown below. See
[device selection](../guides/device-and-memory.md) for model-specific support
and exceptions.

## Optional GPU Execution

Keep `device="cpu"` unless you want GPU execution and have installed a compatible
backend. Both options below require an NVIDIA GPU and a compatible NVIDIA driver;
installing an optional dependency alone does not establish that CUDA works.

### CuPy

From the repository directory, choose **one** extra matching a working CUDA
Toolkit/runtime installation:

```bash
# CUDA 12.x:
python -m pip install -e ".[gpu12]"
```

For CUDA 11.2–11.8, use `python -m pip install -e ".[gpu11]"` instead, with a
compatible [CuPy 13 release](https://docs.cupy.dev/en/v13.6.0/install.html).
The extras select `cupy-cuda12x>=13.0` or `cupy-cuda11x>=13.0`; they do not install
the CUDA Toolkit. Do not install multiple CuPy distributions in the same environment.
Check the [CuPy installation requirements](https://docs.cupy.dev/en/stable/install.html)
for a compatible Python, CuPy, CUDA, and driver combination.

Once CuPy can allocate an array on your GPU, reuse `X` and `y` from the CPU example:

```python
model = LinearRegression(device="cuda")  # CuPy CUDA
model.fit(X, y)
print(model.score(X, y))
```

### PyTorch

Use the [official PyTorch installation selector](https://pytorch.org/get-started/locally/)
to install a CUDA-enabled build suitable for your platform and driver, then install
the optional dependency from the repository directory:

```bash
python -m pip install -e ".[torch]"
```

The extra requires PyTorch 2.0 or later; it does not choose the appropriate CUDA
build for you. Confirm that `torch.cuda.is_available()` is `True` before reusing
`X` and `y`:

```python
model = LinearRegression(device="torch")  # PyTorch CUDA
model.fit(X, y)
print(model.score(X, y))
```

For this estimator, an unavailable explicit `"cuda"` or `"torch"` request raises
an error rather than falling back to CPU. `"cuda"` selects CuPy, not PyTorch.
For more setup and supported-model details, see the
[PyTorch backend guide](../guides/pytorch-backend.md).

## Next Steps

- [Device and GPU Memory](../guides/device-and-memory.md)
- [Inference Modes (Lasso)](../guides/inference-modes.md)
- [Models Overview](../models/README.md)
- [Benchmark Index](../guides/benchmarks.md)
