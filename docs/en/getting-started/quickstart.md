# Quickstart

> Language: English  
> Last updated: 2026-04-11  
> This page: Getting started  
> Switch: [Chinese](../../getting-started/quickstart.md)

Language switch: [Chinese](../../getting-started/quickstart.md)

## Installation

```bash
cd statgpu
pip install -e .
```

## Minimal Example

```python
import numpy as np
from statgpu.linear_model import LinearRegression

X = np.random.randn(1000, 20)
y = X @ np.random.randn(20) + 0.1 * np.random.randn(1000)

model = LinearRegression(device="cuda")
model.fit(X, y)
print(model.score(X, y))
```

## Device Control

```python
import statgpu as sg

sg.set_device("auto")
sg.set_device("cuda")
sg.set_device("cpu")
print(sg.get_device().value)  # resolved device, e.g. cuda / cpu
```

See also:
- [Device and GPU Memory](../guides/device-and-memory.md)
- [Inference Modes (Lasso)](../guides/inference-modes.md)
- [Models Overview](../models/README.md)
- [Benchmark Index](../benchmarks.md)
