# Native KDE bootstrap

> Language: English  
> Last updated: 2026-10-22  
> Switch: [Chinese](../../cn/guides/native-kde-bootstrap.md)

> **Version scope:** This guide describes unreleased source changes that are not included in the published 0.3.0 wheel or sdist. See the [changelog](../changelog.md#unreleased--working-tree-repairs) for release status. Updating the website does not publish a new package release.

Use `kde_bootstrap_confidence_interval` or
`kde_confidence_interval(method="bootstrap")` for pointwise percentile intervals
of a smoothed density. CuPy and Torch perform sampling, resampling, bandwidth
selection, refitting, density evaluation and percentiles on the selected device.
Torch CPU is also supported: selecting the Torch library alone does not prove
CUDA execution. The [API reference](../reference/survival-smoothing-api.md)
describes arguments and shapes.

## Requesting Torch CUDA

With the global device policy set to `"auto"`, `backend="torch"` preserves native Torch input placement and uses Torch CPU for host lists, tuples or NumPy arrays. It selects an array library, not an accelerator. `torch.cuda.device(...)` selects a CUDA device for CUDA operations; it does not turn these host inputs into a request for GPU computation.

The bootstrap functions have no separate `device` argument. To run host inputs on Torch CUDA, call `statgpu.set_device("torch")` before the bootstrap call with `backend="torch"`, then restore the previously configured global policy, including on error. Do not save `get_device()` as that policy: it resolves `"auto"` rather than preserving it. For direct estimator fits, use `KDE(device="torch", backend="torch")`. An explicit Torch CUDA request raises if the required accelerator is unavailable; it does not silently fall back to CPU.

## Inputs and statistical interpretation

The native paths support numeric bandwidth factors and `scott`, `silverman`,
`nrd`, `nrd0`, `ucv`, `bcv`, `sj`, `sj-ste`, and `sj-dpi`. Gaussian, rectangular,
triangular, Epanechnikov, biweight and triweight kernels support the existing
multivariate input shapes; cosine and optcosine remain one-dimensional only.
Convertible real numeric inputs, including float32 and noncontiguous views,
are converted to float64. Existing finite-input, positive-total-weight and
covariance requirements still apply; a selector may fail for constant or
sparse resamples rather than silently substituting another selector.

Rows are sampled with replacement according to normalized original weights.
Each refit uses the sampled original weights, normalized again. This double
use of weights is not a universal survey, frequency or importance-weight
bootstrap. Establish that the resampling scheme suits the study design.
Zero weights are permitted when the remaining weights define a valid fit.
Intervals are pointwise, not simultaneous, and do not correct smoothing bias.

A string bandwidth selector is recomputed for every native replicate. A numeric
factor remains fixed, but each replicate recomputes sample covariance and hence
its absolute kernel width. The NumPy one-dimensional Gaussian fast path instead
holds the initially selected factor fixed even for string selectors. Backend
comparisons with string selectors therefore need not describe identical
statistical workloads. Use a numeric factor when a fixed-factor comparison is
intended; that still does not make random draws identical across backends.

Weighted R-style selectors retain their quantile-resampling extension, and
multivariate R-style selectors retain their principal-axis projection. These
are not exact multivariate R equivalents. Kernel-regression CV selectors are
not KDE selector options.

## Native PCA and weighted selector differences

The NumPy selector algorithms retain their legacy behavior. Native selectors
use stable sorting for equal-valued samples and orient the principal eigenvector
so its first largest-absolute loading is positive. They stay on the native
backend rather than copying arrays to CPU to reproduce legacy results.

Two different effects matter. Even with a unique, well-separated largest
eigenvalue, an eigenvector and its negative describe the same PCA axis. But the
weighted linear-quantile resampling extension used here is not invariant to
reversing that axis. Reversing projected values also reverses the order of
unequal weights; interpolation through their cumulative probabilities need not
produce a reflected copy of the same resample. Consequently native weighted
multivariate UCV/BCV/SJ-family factors can differ materially from historical
NumPy factors. This is a selector-convention difference, not merely floating-point
noise, and it is not confined to repeated eigenvalues.

Separately, a repeated largest eigenvalue admits multiple principal directions;
a nearly repeated one makes the direction sensitive to numerical perturbations.
Fixing the vector's sign cannot resolve that basis ambiguity. Projection-based
selectors, including the nrd family, can then differ between backends or devices.
Use the projection diagnostics to interpret the chosen direction; no exact
legacy NumPy factor or cross-backend interval equality is promised. A numeric
bandwidth factor avoids PCA-based bandwidth selection when a common fixed-factor
workload is needed, without making independent RNG draws equal.

For native multivariate PCA selection, a `RuntimeWarning` reports a repeated
or numerically near-repeated largest eigenvalue, including a zero spectral
scale. The comparison is relative to the covariance eigenvalue scale and
float64 precision, not an absolute data-unit threshold. A warning does not
switch to CPU, reject the fit or restore legacy factors. Sign-sensitive weighted
resampling can differ even without a warning.

A successful public interval call combines warnings from its original fit and
bootstrap refits into at most one warning, rather than warning for every draw.
`metadata["projection_diagnostics"]` reports `status`, total ambiguous `count`,
`exact_degeneracy_count`, `near_degeneracy_count`, `zero_scale_count`, and
`min_relative_eigengap` among ambiguous projections. The three category counts
are mutually exclusive. With no ambiguity, the count is zero and the minimum
is `None`; this is not a count of all PCA fits. A standalone fit warns for its own projection.
Numeric bandwidth, Scott/Silverman, and one-dimensional selection do not use
this PCA extension and therefore do not emit its warning.

## Random state and reproducibility

Native bootstrap accepts the inputs accepted by `numpy.random.default_rng`,
including `None`, nonnegative integer seeds, `SeedSequence`, and `Generator`.
It consumes four uint32 words from that NumPy generator per invocation to seed
a local backend generator. Supplying a `Generator` advances its state by this
seed bridge, rather than by the historical host sampling sequence. A repeated
call with the same advancing generator is not a reset; pass the same integer
seed in separate calls when repeatability is intended.

The backend RNG generates the actual bootstrap indices. Global NumPy, CuPy and
Torch RNG state is not mutated. Repeatability concerns a fixed backend,
software and device environment, not equality of NumPy/CuPy/Torch bitstreams or
of historical bootstrap results. Query chunk size does not change RNG batching.
Sampling uses representable float64 cumulative probabilities; extremely small
positive masses can collapse through rounding.

## Return boundary and memory

`KDEBootstrapResult` remains a host-facing result: `points`, `estimate`, `lower`,
`upper`, and optional `bootstrap_samples` are NumPy arrays. The final transfer
is deliberate, not CPU fallback during native bootstrap. Input conversion,
scalar validation/control and the small seed bridge can also use the host.
The `backend` field names the array library, not necessarily a GPU device.
Inspect tensor placement when distinguishing Torch CPU from CUDA.

`batch_size` bounds query batches. Optional `workspace_bytes` is a positive
integer budget for estimated bootstrap-managed buffers and can reduce query
chunk size. It does not reduce the number of observations, queries or resamples.
If even one query and the retained buffers cannot fit the estimate, the call
raises `MemoryError`.

Exact percentiles retain a matrix of `n_resamples * n_query` values even if
`return_bootstrap_samples=False`. The estimate also allows for quantile copies,
one replicate's refit buffers, known selector buffers and query temporaries. It
excludes caller inputs, the original model, library scratch and allocator caches.
Consequently this is not a hard peak-memory or VRAM cap. NumPy bootstrap and
normal intervals do not apply this native workspace budget.

Native result metadata describes execution, selector refitting, the seed bridge,
probability precision, host-transfer boundaries, query chunk and estimated
managed bytes. It does not certify speed or physical GPU execution. Compare
synchronized end-to-end runtimes on the intended data sizes; small workloads
can favor the NumPy fast path.
