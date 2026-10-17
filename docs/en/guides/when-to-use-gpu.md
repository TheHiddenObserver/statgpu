# When should I use a GPU?

> Language: English\
> Last updated: 2026-10-17\
> This page: understand the workload, choose a device, then try a small example\
> Switch: [Chinese](../../cn/guides/when-to-use-gpu.md)

A GPU is worth trying when most of the work is large, regular numerical operations, especially when the same data is used repeatedly. Start with CPU for small jobs, one-row predictions, or workflows dominated by data preparation. A GPU-capable model is not necessarily faster on your dataset.

You do not need to write CUDA code to use statgpu. First understand where the time goes; then select a supported backend with `device`.

<a id="why-matrix-work-can-benefit"></a>

## Why matrix work can benefit

Think of a data table as a matrix: each row is an observation and each column is a feature. A linear model predicts a value for each row by multiplying its features by the same learned weights and adding them up.

For example, with weights `[2, -1]` and no intercept:

- Row `[3, 1]` gives `3 × 2 + 1 × (-1) = 5`.
- Row `[0, 4]` gives `0 × 2 + 4 × (-1) = -4`.
- Row `[2, 2]` gives `2 × 2 + 2 × (-1) = 2`.

Those row calculations can happen independently. Matrix multiplication organizes many such operations into work a GPU can process in parallel. During fitting, related operations repeatedly combine predictions, errors, and features to find the model's parameters.

![A CPU handles flexible tasks with a few illustrated work lanes, while a GPU spreads matching numerical tasks across many lanes. The lanes represent parallel work rather than actual core counts.](/images/gpu-work-en.svg)

*Conceptual illustration, not a hardware diagram or a timing result. CPUs also perform parallel and vectorized calculations; GPUs are designed for high throughput across many similar operations. Neither picture determines which device wins for a particular model.*

For one short row, starting GPU work and moving data can take more effort than the arithmetic itself. For a large matrix, there can be enough arithmetic to outweigh that overhead. Fitting still has dependent steps: a solver may need one iteration's result before starting the next, even when the work within an iteration is parallel.

<a id="when-gpu-is-worth-trying"></a>

## When GPU is worth trying

Look for a combination of **substantial numerical work, repeated use, supported operations, and enough GPU memory**. These are concrete candidates to investigate, not measured speedups or minimum-size rules:

| Example workload | What the GPU may accelerate | What could limit the benefit |
|---|---|---|
| [Ridge regression](../models/regression/ridge.md) on a dense table with 100,000 rows and 200 features | Large matrix products and a linear solve | A strong CPU implementation may still win; fitting and inference have different costs |
| [PCA](../unsupervised/pca.md) on 50,000 rows and 500 numeric features | Matrix products and decomposition to summarize correlated features | Solver choice changes the work and memory use; randomized PCA is approximate |
| [KMeans](../unsupervised/kmeans.md) on 200,000 rows, 40 features, and 20 centers | Repeated point-to-center distances and center updates | Initialization, convergence, and intermediate arrays also cost time |
| [Cross-validation](cross-validation.md) over a substantial dataset | Repeated fits and model-specific opportunities to reuse work | Folds and candidates are not automatically all concurrent; a grid of tiny fits may remain slower |

The row and feature counts describe illustrative problems. They are not benchmark results and do not imply a common crossover size. Changing the feature count, algorithm, dtype, CPU thread count, or GPU can reverse the comparison.

Repeated work helps most when the implementation can reuse data or allocations. Merely calling many small fits in a Python loop does not turn them into a single parallel GPU operation.

<a id="when-cpu-is-the-better-start"></a>

## When CPU is the better start

- **Small or occasional jobs.** A short regression or a few predictions may finish before GPU setup and transfer costs are recovered.
- **Low latency for individual requests.** Predicting one row at a time is different from processing a large batch.
- **Frequent trips between CPU and GPU.** Repeatedly converting arrays, printing intermediate values, or returning large outputs to NumPy can erase a compute advantage.
- **Mostly parsing, Python control flow, or irregular decisions.** A GPU does not automatically accelerate file loading, DataFrame/formula preparation, Python loops, or branch-heavy logic.
- **Unsupported operations or insufficient memory.** Check the model, solver, input format, and inference support before choosing a GPU. CPU may be the simpler supported option.

There is no universal “use GPU above N rows” rule. Begin with a correct CPU result, then compare the actual workload you need.

<a id="count-the-whole-journey"></a>

## Count the whole journey

The time you wait can include reading and preparing data, copying inputs to the GPU, running numerical operations, and returning results to CPU memory. Accelerating only the middle part may leave most of the total unchanged.

![Data starts in CPU memory, moves to the GPU for numerical work that may be repeated there, and returns to CPU memory when a result is needed. Reusing GPU data avoids repeated bulk transfers.](/images/gpu-journey-en.svg)

*Conceptual data journey, not a measured timeline. Box widths and arrows do not represent elapsed time. Actual methods can also perform CPU work or transfer small reporting arrays between stages.*

“GPU-resident” means an array stays in GPU memory between operations. It can avoid repeatedly copying the same bulk data, if the receiving methods accept that array type. It does not promise an entirely CPU-free workflow. For example, `Ridge.coef_` is stored as a NumPy array even after a GPU fit.

### The working set must fit, not just the input

GPU memory must hold inputs, outputs, model state, and temporary work arrays. For the illustrative Ridge table above, the input matrix alone uses `100,000 × 200 × 8 = 160,000,000` bytes in float64, about 160 MB. That is **not** an estimate of peak fit memory. Centered copies, matrix products, solver workspaces, inference, and repeated fits can need more.

Float32 and float64 have different memory and precision requirements; double-precision performance also varies by GPU. Check what the estimator actually computes rather than assuming that a float32 input guarantees float32 execution. Choose precision for the numerical problem before comparing speed.

`gpu_memory_cleanup=True`, where supported, can release unused pooled allocations. It cannot free live arrays or make an oversized working set fit. See [device and memory controls](device-and-memory.md#gpu-memory-cleanup).

<a id="try-it-with-an-explicit-device"></a>

## Try it with an explicit device

These examples teach placement and output handling. They are deliberately small and **are not performance demonstrations**. Follow the [Quickstart installation instructions](../getting-started/quickstart.md) first.

### 1. Establish a small CPU result

Each row below has 20 features and one continuous response. `Ridge` fits a regularized linear prediction model. `compute_inference=False` skips coefficient uncertainty calculations because this example only needs predictions.

<!-- learner-example: gpu-decision-cpu -->
```python
import numpy as np
from statgpu.linear_model import Ridge

rng = np.random.default_rng(42)
X = rng.normal(size=(2000, 20))  # NumPy float64, in CPU memory
weights = rng.normal(size=20)
y = 1.0 + X @ weights + 0.1 * rng.normal(size=2000)

cpu_model = Ridge(
    alpha=0.1, solver="exact", device="cpu", compute_inference=False
).fit(X, y)
cpu_predictions = cpu_model.predict(X[:3])
print(cpu_predictions.shape)  # (3,)
print(isinstance(cpu_model.coef_, np.ndarray))  # True
```
<!-- example-end: gpu-decision-cpu -->

`cpu_predictions` contains predictions for the first three rows. The printed `(3,)` describes its shape; `True` confirms that the fitted coefficients are stored in a NumPy array. This is a usage check on training data, not a held-out assessment of predictive accuracy. For model interpretation and numerical cautions, see [Ridge regression](../models/regression/ridge.md).

### 2. Choose one available CUDA backend

For `Ridge`, `device="cuda"` means CuPy CUDA and `device="torch"` means PyTorch CUDA. Both need a working NVIDIA GPU installation. An unavailable explicit backend raises an error; `device="torch"` does not silently use Torch CPU.

**CuPy:** run the CPU setup above, then this example with a working CuPy/CUDA installation. Copying `X` and `y` explicitly makes their placement visible.

<!-- example-requires: gpu-decision-cpu -->
<!-- learner-example: gpu-decision-cupy -->
```python
import cupy as cp

X_gpu = cp.asarray(X)
y_gpu = cp.asarray(y)
gpu_model = Ridge(
    alpha=0.1, solver="exact", device="cuda", compute_inference=False
).fit(X_gpu, y_gpu)

predictions_gpu = gpu_model.predict(X_gpu, return_cpu=False)
print(isinstance(predictions_gpu, cp.ndarray))  # True
print(predictions_gpu.device)

# Keep the bulk predictions and residuals on the GPU; return one scalar.
training_mse = cp.mean((predictions_gpu - y_gpu) ** 2)
print(float(training_mse.get()))
```
<!-- example-end: gpu-decision-cupy -->

That last value is training mean squared error. It illustrates a chained calculation, not a speed comparison or a test of generalization. `Ridge.predict` normally returns NumPy; `return_cpu=False` keeps predictions in the fitted numerical backend. It does not change the NumPy storage of `coef_` or eliminate every internal transfer.

**PyTorch alternative:** after the same CPU setup, this example accepts the NumPy inputs and converts them for CUDA fitting. It requires `torch.cuda.is_available()` to be `True`.

<!-- example-requires: gpu-decision-cpu -->
<!-- learner-example: gpu-decision-torch -->
```python
torch_model = Ridge(
    alpha=0.1, solver="exact", device="torch", compute_inference=False
).fit(X, y)
torch_predictions = torch_model.predict(X[:3], return_cpu=False)
print(torch_predictions.is_cuda)  # True
print(torch_predictions.device)
```
<!-- example-end: gpu-decision-torch -->

### 3. Check the model-specific behavior

- `device="auto"` is a selection policy, not a speed guarantee. Under shared estimator handling it consults the global setting; after `set_device("torch")`, an automatic estimator can still require Torch CUDA. Some models have workload-specific automatic choices. Use [device selection](device-and-memory.md#global-settings-and-estimator-settings) for the details.
- A function's `backend="torch"` is not the same contract as an estimator's `device="torch"`; some functions support Torch CPU.
- Kernel smoothing, kernel features, and splines have [placement exceptions](device-and-memory.md#current-smoothing-and-spline-exceptions). Do not assume a GPU setting makes those methods run on a GPU.
- Backend support can differ for fitting, prediction, cross-validation, and inference. Check the [method inventory](implemented-methods.md) and the relevant model page.

<a id="measure-your-own-workload"></a>

## Measure your own workload

For a useful CPU/GPU comparison:

1. Use the **same data and statistical problem**: shapes, preprocessing, dtype, objective, penalty scale, solver settings, convergence criteria, and requested inference. Compare numerical results as well as elapsed time.
2. Decide whether you care about **one complete job** or **repeated work with arrays already on the GPU**. Include transfers and output conversion for the first; report excluded setup explicitly for the second.
3. Warm up each path, repeat the measurement, and report a central time and variation. If first-call latency matters, measure it separately rather than hiding it in a warmed-up result.
4. Synchronize the **CUDA device actually used** before and after the timed region. A CPU stopwatch around asynchronous GPU work can stop before the calculation finishes.
5. Record CPU/GPU identity, package versions, source version, data shape, dtype, and timing scope. A faster run with failed convergence or materially different results is not an equivalent solution.

The [benchmark dashboard](statgpu_benchmark_dashboard.md#open-dashboard) contains recorded measurements. Match the environment, workload, backend, and metric scope before comparing rows. A ratio against another library is not automatically a statgpu GPU-versus-CPU speedup; historical results do not guarantee current performance. See [how to interpret and reproduce benchmarks](../benchmarks.md).

## Further reading

- [Device and GPU memory](device-and-memory.md): exact device semantics, output arrays, and cleanup
- [Cross-validation](cross-validation.md): selection and final-refit behavior
- [NVIDIA CUDA Best Practices: choosing work for a GPU](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#what-runs-on-a-cuda-enabled-device): parallel work and transfer costs
- [CuPy Performance Best Practices](https://docs.cupy.dev/en/stable/user_guide/performance.html): asynchronous timing, warm-up, and one-time overheads
