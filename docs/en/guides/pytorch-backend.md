# PyTorch Backend Guide

> Language: English  
> Last updated: 2026-07-24  
> Switch: [Chinese](../../cn/guides/pytorch-backend.md)

## Overview

StatGPU supports three execution backends:

| Device value | Numerical backend | Typical execution |
|---|---|---|
| `"cpu"` | NumPy | CPU |
| `"cuda"` | CuPy | NVIDIA CUDA |
| `"torch"` | PyTorch | NVIDIA CUDA |
| `"auto"` | Automatically selected | CuPy, Torch CUDA, or NumPy according to availability and input |

`device="torch"` is the explicit PyTorch request. `device="cuda"` selects CuPy;
it is not an alias for Torch. Explicit requests fail when the requested backend is
unavailable and do not silently execute on another backend.

Model, solver, cross-validation, and inference coverage can differ. Use the
[Implemented Methods](implemented-methods.md) inventory and the relevant model page
rather than assuming every public estimator has an identical Torch path.

## Installation

Install the optional Torch dependency with:

```bash
pip install "statgpu[torch]"
```

A compatible PyTorch CUDA build and NVIDIA driver are required for GPU execution.
Verify the environment before fitting a model:

```python
import torch

print(torch.__version__)
print(torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
```

Installing the base package alone does not install PyTorch:

```bash
pip install statgpu
```

## Basic Usage

### NumPy input with explicit Torch execution

```python
import numpy as np
from statgpu.linear_model import LinearRegression

rng = np.random.default_rng(42)
X = rng.normal(size=(1000, 20))
y = 1.0 + X @ rng.normal(size=20) + rng.normal(size=1000)

model = LinearRegression(device="torch")
model.fit(X, y)
print(model.score(X, y))
```

The estimator converts compatible NumPy input to the selected Torch CUDA backend.
Explicit Torch execution raises when Torch CUDA is unavailable.

### Torch CUDA tensors

```python
import torch
from statgpu.linear_model import LinearRegression

X = torch.randn(1000, 20, device="cuda", dtype=torch.float64)
y = torch.randn(1000, device="cuda", dtype=torch.float64)

model = LinearRegression(device="torch")
model.fit(X, y)
prediction = model.predict(X)
```

Backend-preserving output is method-specific. Consult the model page for whether an
output remains a Torch tensor or is intentionally exposed as CPU metadata or a scalar
statistical summary.

## Device Selection

### Per-estimator selection

```python
from statgpu.linear_model import Ridge

model = Ridge(alpha=1.0, device="torch")
```

### Global default

```python
import statgpu as sg

sg.set_device("torch")
```

A per-estimator `device=` argument takes precedence where the estimator exposes it.
Use `"auto"` only when automatic backend selection is intended.

## Statistical Inference

Torch execution does not by itself guarantee that every inference option is available.
Inference support depends on the estimator, covariance type, solver, data contract, and
optional dependencies. For an inference-capable model, inspect its documentation for:

- supported covariance estimators;
- standard errors, test statistics, p-values, and confidence intervals;
- strict versus explicitly requested approximate behavior;
- delayed-entry, clustering, ties, rank-deficiency, or formula restrictions;
- whether final summaries are Torch arrays, NumPy arrays, or scalar metadata.

Unsupported inference combinations should fail explicitly or operate in a documented
estimation-only mode; they should not silently produce approximate results.

## Execution Boundaries

Core numerical arrays should remain on the selected Torch backend where the method
supports Torch execution. Legitimate CPU boundaries may include:

- formula, label, feature-name, and small index metadata;
- fold definitions, convergence decisions, and scalar control flow;
- scalar distribution functions unavailable in Torch;
- user-facing summaries intentionally represented as NumPy or Python scalars;
- external validation libraries that only accept CPU arrays.

These boundaries are model-specific. A global claim that every intermediate remains on
GPU would be incorrect. Full-design transfers or backend changes must not occur as a
silent fallback.

## Dtype and Numerical Precision

Statistical inference commonly benefits from `float64`:

```python
X = torch.randn(2000, 50, device="cuda", dtype=torch.float64)
```

Use matching dtypes for predictors, responses, weights, offsets, and initialization
arrays. Differences across NumPy, CuPy, and Torch should be judged with tolerances that
reflect the algorithm, condition number, stopping rule, and dtype rather than requiring
bitwise equality.

## Randomness and Reproducibility

Set both model-level random-state parameters and Torch seeds when the algorithm uses
randomness:

```python
import torch

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
```

Cross-validation folds, landmark sampling, randomized decompositions, and stochastic
initialization may also use estimator-specific `random_state` parameters.

## Memory Management

GPU memory usage depends on the estimator and workload. Exact kernel methods and dense
Hessian or covariance calculations can require quadratic or larger intermediate storage.
Use problem-appropriate batching or approximation methods where documented.

For troubleshooting only, cached Torch memory can be released with:

```python
import torch

torch.cuda.empty_cache()
```

Some estimators expose `gpu_memory_cleanup=True`. This controls cache cleanup and does
not change the statistical objective or permit a CPU fallback.

## Performance and Validation Evidence

GPU performance depends on sample size, feature dimension, dtype, kernel or solver,
hardware, synchronization, and memory pressure. Small workloads may be faster on CPU.
Do not interpret a benchmark from one model or GPU as a universal speed guarantee.

Maintained evidence should record:

- exact commit SHA;
- Python, Torch, CUDA, and driver versions;
- GPU model;
- synchronized timing methodology;
- accuracy or statistical parity metrics;
- passed, failed, and skipped tests.

Current and historical benchmark artifacts live under `results/` and `dev/benchmarks/`.
The retained [Torch backend report](../../../dev/docs/torch_backend_final_report.md) is a
dated evidence snapshot, not a current support matrix.

## Troubleshooting

### Torch CUDA is unavailable

```python
import torch
print(torch.cuda.is_available())
```

Check the NVIDIA driver, the installed Torch build, and its bundled CUDA runtime. A
system CUDA toolkit version does not by itself determine which Torch wheel is usable.

### Explicit Torch execution raises

This is expected when Torch CUDA or a required Torch operation is unavailable. Use
`device="cpu"` or `device="auto"` only when that behavior matches the intended contract;
do not expect `device="torch"` to fall back silently.

### Out of memory

Reduce the problem size, use a documented batched or approximate method, lower the CV
grid or fold count, or choose a method with lower memory complexity. Calling
`torch.cuda.empty_cache()` cannot reduce the live tensor memory required by the
algorithm.

### Results differ from another framework

First align:

- objective normalization;
- regularization scale;
- intercept and feature encoding;
- solver and stopping tolerance;
- sample weights, offsets, ties, and covariance options;
- dtype and random seed.

A penalty parameter may require rescaling when another framework optimizes a summed
loss while StatGPU optimizes a mean loss.

## Related Documentation

- [Device and Memory Management](device-and-memory.md)
- [Implemented Methods](implemented-methods.md)
- [Cross-Validation](cross-validation.md)
- [Inference API](inference-api.md)
- [Models Overview](../models/README.md)
- [Quickstart](../getting-started/quickstart.md)

## References

- [PyTorch documentation](https://pytorch.org/docs/)
- [StatGPU Torch backend evidence snapshot](../../../dev/docs/torch_backend_final_report.md)
