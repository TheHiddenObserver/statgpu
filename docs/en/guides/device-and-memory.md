# Device and GPU Memory

> Language: English  
> Last updated: 2026-09-17  
> This page: device selection and user-visible GPU memory controls  
> Switch: [Chinese](../../cn/guides/device-and-memory.md)

## Device selection

For estimators that expose a `device` parameter, statgpu uses the following meanings:

- `device="cpu"` — request NumPy CPU computation;
- `device="cuda"` — request CuPy CUDA computation;
- `device="torch"` — request Torch CUDA computation;
- `device="auto"` — allow statgpu to choose among supported available backends.

An explicit accelerator request is authoritative. If the requested CuPy/Torch CUDA backend is unavailable, statgpu raises an error rather than silently replacing the fit with a CPU calculation.

Model-specific backend coverage can be narrower than the generic device vocabulary. Check the relevant model page or [Implemented Methods](implemented-methods.md) when a particular backend is required.

## Input conversion and preprocessing

Formula/DataFrame parsing and other metadata preparation may occur on CPU before numerical model computation. This does not change the selected numerical backend: arrays used by the model are converted to the requested backend before the supported numerical path runs.

Transfers between NumPy, CuPy, and Torch may use optimized mechanisms internally. Applications should rely on the resulting device semantics, not on a particular transfer implementation such as DLPack or pinned memory.

## Automatic device selection

`device="auto"` may choose a backend from availability, input/workload characteristics, and estimator-specific performance heuristics. Those size thresholds are implementation details and may change as kernels and benchmarks improve.

If reproducible hardware placement matters, use an explicit device rather than depending on an internal `auto` threshold.

## Solver compatibility is documented separately

Device support and solver compatibility are different questions. A backend can be available while a particular loss × penalty × solver combination is unsupported.

Use:

- [Solver × Penalty Matrix](solver-penalty-matrix.md) for compatibility;
- [Solver Algorithms](solver-algorithms.md) for algorithm definitions;
- the relevant model page for model-specific restrictions.

This page does not duplicate those matrices.

## GPU memory cleanup

Some GPU-capable estimators expose `gpu_memory_cleanup`.

- `gpu_memory_cleanup=False` (default where exposed) favors repeated-fit throughput by allowing backend memory pools/caches to retain reusable allocations.
- `gpu_memory_cleanup=True` asks the estimator to release reclaimable cached GPU memory at its documented cleanup points, which can reduce steady GPU-memory usage at the cost of some reuse.

Example:

```python
from statgpu.linear_model import Ridge

model = Ridge(
    alpha=1.0,
    device="cuda",
    gpu_memory_cleanup=True,
)
model.fit(X, y)
```

The option does not mean that fitted state needed for `predict()`, `score()`, or supported inference is discarded. Estimators only expose cleanup behavior at points compatible with their fitted-state contract.

## When to enable cleanup

`gpu_memory_cleanup=True` is useful when:

- several models share a GPU;
- memory pressure matters more than repeated-fit latency;
- a long-running process should return reclaimable pool memory between fits.

Leaving it disabled is often preferable when repeatedly fitting the same kind of model and maximum throughput matters.

## Related documentation

- [Implemented Methods](implemented-methods.md) — model/backend inventory
- [Cross-Validation](cross-validation.md) — device behavior during selection and refit
- [Solver × Penalty Matrix](solver-penalty-matrix.md) — solver compatibility
- [PyTorch Backend](pytorch-backend.md) — PyTorch-specific usage
