# CPU/GPU acceleration internals

> Language: English
> Last updated: 2026-09-03
> Switch: [简体中文](../../cn/guides/acceleration-internals.md)

This is an advanced implementation guide. Model pages remain focused on statistical use; this page explains where computation runs, why a GPU can help, and where transfer, synchronization, precision, or memory can dominate.

## Execution contract

| `device` | Numerical backend | Contract |
|---|---|---|
| `"cpu"` | NumPy and CPU scientific Python | numerical model work stays on CPU |
| `"cuda"` | CuPy on NVIDIA CUDA | raises if CuPy/CUDA is unavailable |
| `"torch"` | PyTorch on NVIDIA CUDA | raises if Torch CUDA is unavailable |
| `"auto"` | capability- and input-aware selection | the only mode allowed to choose another available backend |

Backend coverage is estimator- and solver-specific. Check the [implemented-method inventory](implemented-methods.md) and model page instead of assuming every method has identical CPU, CuPy, and Torch paths.

## Data lifecycle

1. The public estimator validates shape, response domain, weights, and controls.
2. Formula/DataFrame input, labels, and category expansion are handled on CPU.
3. Numerical arrays are converted once to the selected backend where the method supports it.
4. Fitting, inference, or prediction uses backend-native linear algebra and kernels.
5. Large arrays should remain on that backend; small convergence decisions can synchronize a scalar.
6. Public coefficients, tables, or metadata may intentionally cross back to NumPy/Python at the reporting boundary.

A documented metadata transfer is not a silent device fallback. Moving an entire design to CPU to continue a requested GPU fit would be a fallback and is not permitted by the explicit-device contract.

## Main acceleration patterns

| Pattern | Representative methods | Dominant work | GPU consideration |
|---|---|---|---|
| dense linear algebra | linear regression, Ridge, PCA/SVD | matrix products, factorizations, solves | benefits when matrices are large enough to amortize transfer and launch overhead |
| iterative smooth likelihood | GLM IRLS, Newton, L-BFGS | repeated gradients, Hessians, weighted solves | device-resident iterations matter more than one fast kernel |
| proximal and non-convex optimization | Lasso, Elastic Net, SCAD, MCP | gradient/prox steps, LLA outer loops | fused operations and fewer host convergence checks reduce synchronization |
| distance and graph construction | DBSCAN, clustering | distance blocks, neighbor relations, connected components | `batch_size` trades peak memory against extra launches |
| resampling and model grids | cross-validation | many related fits | data reuse and avoiding repeated transfer can dominate single-fit speed |

Algorithm choice is part of performance. CPU and GPU paths should optimize the same declared objective, but may use different factorizations, search structures, kernels, or batching.

## Formula and preprocessing

Patsy Formula parsing is CPU preprocessing:

```python
model.fit(formula="y ~ x1 + C(group)", data=frame)
```

The resulting dense design is transferred to the requested numerical backend. This is convenient and preserves feature names; it is not zero-copy. For repeated large GPU fits, parse once or provide prebuilt CuPy/Torch arrays where the estimator accepts them. See the [Formula support matrix](formula-interface.md).

## Transfers and interoperability

- NumPy input must be copied to CUDA for CuPy or Torch execution.
- Existing CuPy and Torch CUDA arrays should be preserved where the public path accepts them.
- CuPy-to-Torch and Torch-to-CuPy conversions prefer DLPack sharing when available.
- NumPy-to-Torch CUDA can use pinned host memory when supported.
- Coefficients and structured inference summaries are often NumPy because they are reporting objects, even if numerical inference ran on GPU.

Measure end-to-end time, including transfer and required synchronization. Timing only an asynchronous kernel without synchronizing understates elapsed time.

## Precision and numerical parity

`float64` is the safer default for ill-conditioned designs and statistical inference. `float32` can reduce memory and improve throughput but changes rounding, convergence, and boundary decisions. Exact label numbers, selected variables near a threshold, or p-values near a cutoff can differ across backends without implying a different objective.

Compare with tolerances based on:

- design condition number and feature scale;
- dtype;
- solver stopping rule and iteration cap;
- stochastic seed;
- whether the result is invariant to label numbering.

Linear-regression HAC has a special large-CPU optimization: it benchmarks a small score-matrix probe and can cache a mixed-precision accumulation preference. Final covariance storage remains double precision. See the [linear inference reference](../models/linear-regression-inference.md).

## Memory behavior

GPU memory consists of live arrays plus reusable allocator caches. `gpu_memory_cleanup=False` usually improves repeated-fit speed; `True` asks supported estimators to release cached blocks after public work, but cannot reduce the live memory required by the algorithm.

Watch for:

- $O(np)$ design storage;
- $O(p^2)$ Gram, Hessian, or covariance matrices;
- $O(n^2)$ pairwise distance or kernel matrices;
- fold/grid multiplication in cross-validation;
- temporary copies caused by dtype or layout conversion.

Use a model's documented `batch_size`, randomized/incremental variant, smaller grid, or lower-memory algorithm where available. Cache cleanup is not a substitute for changing an intrinsically quadratic workload.

## Inference boundaries

Estimation and inference can have different cost profiles:

- OLS covariance adds a $p\times p$ inverse/solve and sandwich products.
- HC2/HC3 additionally compute leverage values.
- HAC adds score autocovariances through the chosen lag.
- GLM M-estimation adds bread/meat matrices after optimization.
- SCAD oracle inference currently refits the selected active set on CPU, even when selection ran on GPU.

Therefore report estimation-only and estimation-plus-inference benchmarks separately, and never assume `compute_inference=False` changes only a tiny constant.

## When CPU can be faster

CPU is often faster for small $n$ or $p$, one-off fits starting from NumPy, branch-heavy algorithms, or workloads whose GPU path requires many small synchronizations. GPU acceleration is most convincing when the numerical workload is large, device-resident, and repeated enough to amortize setup.

Use the [benchmark dashboard](/dashboard/) with its exact hardware and environment metadata; do not turn one benchmark into a universal speed claim.

## Reproducible backend validation

Record:

- commit SHA and statgpu version;
- CPU/GPU model, driver, CUDA, CuPy, Torch, Python, and NumPy versions;
- estimator, solver, dtype, dimensions, and all tolerances;
- whether input started on host or device;
- warm-up policy and synchronization points;
- statistical parity metric as well as elapsed time.

For stochastic algorithms, set both estimator random state and backend seeds. Compare objective, predictions, KKT/gradient residuals, or label-invariant clustering metrics—not just raw coefficient bits or cluster IDs.

## Source map

- backend selection and conversion: `statgpu/backends/` and `statgpu/_base.py`
- linear CPU/CuPy/Torch paths: `statgpu/linear_model/wrappers/_linear.py`
- ordinary GLM orchestration: `statgpu/linear_model/_glm_base.py`
- penalized dispatch and solvers: `statgpu/linear_model/penalized/_fit_mixin.py` and `statgpu/solvers/`
- DBSCAN paths: `statgpu/unsupervised/_dbscan.py`
- benchmark generation: `dev/benchmarks/`

Private helpers can change. The stable user contract is the public estimator behavior documented on each model page.
