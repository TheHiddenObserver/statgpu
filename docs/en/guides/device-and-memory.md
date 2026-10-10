# Device and GPU Memory

> Language: English\
> Last updated: 2026-10-10\
> This page: device selection and GPU memory controls\
> Switch: [Chinese](../../cn/guides/device-and-memory.md)

## Choose a device

For estimators using the shared device handling, such as `Ridge`, the choices are:

- `device="cpu"`: compute with NumPy on CPU.
- `device="cuda"`: compute with CuPy on a CUDA GPU.
- `device="torch"`: compute with Torch on a CUDA GPU.
- `device="auto"`: let the estimator choose a backend, taking the global setting into account.

An explicit `"cuda"` or `"torch"` request on these estimators raises an error if the corresponding CUDA backend is unavailable. Use `device="cpu"` for CPU execution, or `device="auto"` when automatic selection is acceptable. Some kernel and spline estimators handle these settings differently; read the [exceptions and placement checks](#current-smoothing-and-spline-exceptions) before relying on GPU execution.

This example runs on CPU without CuPy or Torch:

<!-- api-example: device-cpu -->
```python
import numpy as np
from statgpu.linear_model import Ridge

rng = np.random.default_rng(42)
X = rng.normal(size=(200, 5))
y = 1.0 + X @ np.arange(1.0, 6.0) + rng.normal(size=200)
model = Ridge(alpha=1.0, device="cpu", compute_inference=False).fit(X, y)
predictions = model.predict(X[:3])
print(predictions)
```

Backend support depends on the model and operation. Check the model page or [Implemented Methods](implemented-methods.md) before choosing a backend for fitting, cross-validation, or inference. An available backend does not make every loss, penalty, and solver combination usable; see the [Solver × Penalty Matrix](solver-penalty-matrix.md).

### Estimator `device` versus function `backend`

`device="torch"` requests Torch CUDA on estimators with the shared device handling. A function's `backend="torch"` selects the Torch library and can support Torch CPU. For example, `get_backend(backend="torch", device="cpu")` returns a Torch CPU backend, while `get_backend(backend="torch")` chooses Torch CUDA when available and Torch CPU otherwise. Torch must be installed in either case. Check each function's backend documentation rather than treating these two parameters as interchangeable.

## Global settings and estimator settings

`statgpu.set_device` changes the process-wide device setting. For estimators using the shared device handling:

- An explicit estimator `device="cpu"`, `"cuda"`, or `"torch"` takes precedence over the global setting.
- An estimator's `device="auto"` consults the global setting. It does not reset that setting to automatic selection.

After `set_device("torch")`, for example, `device="auto"` can still raise when Torch CUDA is unavailable. Use `set_device("auto")` to restore automatic global selection, or set `device="cpu"` on the model for an explicit CPU fit. Setting an unavailable global GPU device emits a warning and retains the requested setting.

With an automatic global setting, `statgpu.get_device()` reports CuPy CUDA when available, then Torch CUDA, then CPU. This reports the resolved global choice, not the device used by every model. Models with their own automatic selection, including penalized GLMs, can choose CPU for some workloads when their `device` is `"auto"`, even with a global GPU setting. Use an explicit per-model device when hardware placement matters, subject to the exceptions below.

Prefer a per-model setting when only one fit needs a different device. If you change the global setting temporarily, restore the previously configured policy afterward, including on an error. Keep track of that policy yourself: `get_device()` resolves `"auto"` to a concrete device, so saving its result does not preserve an earlier `"auto"` policy.

<a id="current-smoothing-and-spline-exceptions"></a>

## Kernel and spline exceptions: check array placement

For the following estimators, a GPU setting alone does not establish GPU execution:

- `KernelDensityEstimator` (including `KDE`) and `KernelRegression` (including `KernelRegressionRegressor`) can fit NumPy or Torch CPU input on Torch CPU with `device="torch"` and `backend="auto"` or `"torch"`. Explicit `backend="torch"` can also run on CPU with `device="cuda"`; explicit `backend="numpy"` overrides either GPU request. With `device="cuda", backend="auto"`, these estimators can select Torch when CuPy is unavailable but Torch CUDA is available, and CPU input can still remain on CPU.
- `SplineTransformer` preserves a supplied Torch tensor's device. Torch CPU input can therefore leave both fitted knots and transformed features on CPU even with `device="torch"` or `"cuda"`. NumPy input with the same settings can instead raise when the requested accelerator is unavailable.
- `KernelPCA` and `Nystroem` reject `device="torch"` when Torch CUDA is unavailable. After that check succeeds, NumPy or Torch CPU input can still produce Torch CPU features. Passing the availability check does not establish tensor placement.

Check the arrays that these methods actually use and return:

- KDE/kernel regression: `samples_` and the returned density or prediction.
- `SplineTransformer`: each array in `knots_` and the transformed basis.
- `KernelPCA`/`Nystroem`: the features returned by `fit_transform`, `transform`, or `predict`. Their public fitted arrays are stored as NumPy arrays even after GPU computation and cannot identify the computation device.

For Torch tensors, inspect `.device` and `.is_cuda`; for CuPy arrays, inspect `.device`. NumPy arrays are on CPU. Neither `model.device` nor a library label such as `backend_="torch"` proves CUDA execution.

If GPU execution is required, do not accept a CPU result from these methods. Use a supported GPU input/backend combination and verify the arrays above before using the result. For a predictable CPU alternative, pass NumPy inputs with `device="cpu"`; for KDE/kernel regression, also set `backend="numpy"`.

Other restrictions and supported combinations are described in [kernel smoothing](../models/nonparametric.md#optional-gpu-execution-and-external-comparisons), [kernel features](../models/kernel-methods.md#backend-and-execution-boundaries), and [splines](../models/splines.md#backend-execution-and-extrapolation-boundary).

## Input conversion and returned arrays

Formula/DataFrame parsing and preparation of column information can run on CPU before numerical fitting. GPU fitting can also involve copying input arrays from CPU to GPU. Account for those transfers when comparing end-to-end runtime, and check the model's accepted input types before reusing GPU arrays.

Output types are method-specific. For example, `Ridge.predict(X)` returns a NumPy array by default, including after a GPU fit. `Ridge.predict(X, return_cpu=False)` returns predictions in the fitted model's numerical backend: NumPy for a CPU fit, CuPy or Torch for a GPU fit. Use this option when continuing with operations on the same GPU backend. `Ridge.coef_` remains a NumPy array and does not indicate where fitting ran.

A documented conversion to NumPy at the output is different from the CPU execution described in the kernel/spline exceptions. Check the method's return-value documentation before using an output's type to infer computation placement.

## GPU memory cleanup

Some GPU-capable estimators expose `gpu_memory_cleanup`. On `Ridge`, for example:

- `gpu_memory_cleanup=False` (the default) retains reusable backend allocations, which can help repeated-fit throughput.
- `gpu_memory_cleanup=True` requests release of unused GPU memory-pool allocations during fit cleanup. This can reduce reserved GPU memory, at the cost of allocating memory again for later work.

Cleanup does not discard fitted state needed for `predict()`, `score()`, or supported inference. It also cannot release memory still held by live arrays in your code or the fitted model, so it does not guarantee that all GPU memory is returned or that a fit will require less peak memory. The available option and cleanup timing depend on the estimator.

The following standalone example requires a working CuPy/CUDA installation:

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

Enable cleanup when several models share a GPU or returning unused memory matters more than repeated-fit latency. Leave it disabled when you want to reuse allocations across repeated fits and have enough GPU memory.

## Related documentation

- [Implemented Methods](implemented-methods.md) — model/backend inventory
- [Cross-Validation](cross-validation.md) — device behavior during selection and refit
- [Solver × Penalty Matrix](solver-penalty-matrix.md) — solver compatibility
- [Solver Algorithms](solver-algorithms.md) — algorithm definitions
- [PyTorch Backend](pytorch-backend.md) — PyTorch-specific usage
