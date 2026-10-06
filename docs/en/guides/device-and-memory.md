# Device and GPU Memory

> Language: English  
> Last updated: 2026-10-06  
> This page: device selection and user-visible GPU memory controls  
> Switch: [Chinese](../../cn/guides/device-and-memory.md)

## Device selection

The intended convention for estimators with a `device` parameter is:

- `device="cpu"` — request NumPy CPU computation;
- `device="cuda"` — request CuPy CUDA computation;
- `device="torch"` — request Torch CUDA computation;
- `device="auto"` — allow statgpu to choose among supported available backends.

Under this convention, an explicit accelerator request should be authoritative: an unavailable CuPy/Torch CUDA backend should raise rather than silently run on CPU. The following estimators currently have exceptions, so their `device` setting alone is not proof of hardware placement.

<a id="current-smoothing-and-spline-exceptions"></a>

### Current smoothing, kernel-feature, and spline exceptions

- `KernelDensityEstimator` (including `KDE`) and `KernelRegression` (including `KernelRegressionRegressor`) can fit NumPy or Torch CPU input on Torch CPU even with `device="torch"` and `backend="auto"` or `"torch"`. Explicit `backend="torch"` can also run on CPU with `device="cuda"`; explicit `backend="numpy"` overrides either accelerator request and returns NumPy CPU arrays.
- `SplineTransformer` gives a supplied Torch tensor priority over `device`. Torch CPU inputs can therefore leave fitted knots and transformed features on CPU even with `device="torch"` or `"cuda"`. The same request with NumPy input can instead raise when the requested accelerator is unavailable.

- `KernelPCA` and `Nystroem` reject `device="torch"` when Torch CUDA is unavailable. After the availability check succeeds, however, NumPy or Torch CPU input can still produce Torch CPU features because these paths do not consistently move inputs to the requested device. Selecting the Torch library or resolving a CUDA backend does not establish tensor placement.

For these estimators, inspect actual arrays after fitting and prediction/transform. Check `samples_` and the returned density/prediction for KDE/regression; check each array in `knots_` and the transformed basis for `SplineTransformer`; check the returned `fit_transform`, `transform`, or `predict` features for `KernelPCA`/`Nystroem`. Their public fitted arrays are deliberately NumPy and cannot establish numerical-device placement. For Torch tensors, inspect `.device` and `.is_cuda`; for CuPy arrays, inspect `.device`; NumPy arrays are on CPU. Neither `model.device` nor a library name such as `backend_="torch"` confirms CUDA execution. If accelerator placement is required, reject a CPU result before using it. To choose a predictable CPU path, pass NumPy inputs with `device="cpu"` and, for KDE/regression, `backend="numpy"`.

These are current routing limitations, not alternative meanings of `device="torch"` or `device="cuda"`. See [kernel smoothing](../models/nonparametric.md#optional-gpu-execution-and-external-comparisons), [kernel features](../models/kernel-methods.md#backend-and-execution-boundaries), and [splines](../models/splines.md#backend-execution-and-extrapolation-boundary) for their other restrictions.

Model-specific backend coverage can be narrower than the generic device vocabulary. Check the relevant model page or [Implemented Methods](implemented-methods.md) when a particular backend is required.

## Input conversion and preprocessing

Formula/DataFrame parsing and other metadata preparation may occur on CPU before numerical model computation. Under the intended device convention, arrays are then converted to the requested numerical backend. The smoothing/kernel-feature/spline exceptions above do not consistently enforce that conversion; inspect their actual array placement.

Transfers between NumPy, CuPy, and Torch may use optimized mechanisms internally. Applications should rely on the resulting device semantics, not on a particular transfer implementation such as DLPack or pinned memory.

### Global settings and estimator settings

For estimators using the shared device routing, an explicit `device="cpu"`,
`"cuda"`, or `"torch"` overrides the global setting from `statgpu.set_device`.
An estimator's `device="auto"` instead inherits that global setting. Thus,
after `set_device("torch")`, passing `device="auto"` does not restore automatic
CPU fallback. Reset the global policy with `set_device("auto")`, or pass
`device="cpu"` when CPU execution is intended. This setting is process-wide;
restore it after a temporary change. The estimator-specific exceptions above
still apply.

## Automatic device selection

`device="auto"` may choose a backend from availability, input/workload characteristics, and estimator-specific performance heuristics. Those size thresholds are implementation details and may change as kernels and benchmarks improve.

If reproducible hardware placement matters, use an explicit device rather than depending on an internal `auto` threshold, and check actual fitted/output arrays for the smoothing/kernel-feature/spline exceptions above.

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

This standalone example requires a working CuPy/CUDA installation:

<!-- api-example: cuda-cleanup -->
```python
import numpy as np
from statgpu.linear_model import Ridge

rng = np.random.default_rng(42)
X = rng.normal(size=(200, 5))
y = 1.0 + X @ np.arange(1.0, 6.0) + rng.normal(size=200)
model = Ridge(
    alpha=1.0,
    device="cuda",
    gpu_memory_cleanup=True,
)
model.fit(X, y)
print(model.score(X, y))
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
