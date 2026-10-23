# Public native KDE bootstrap migration

## Target and scope

Baseline: `bf21587e4783801b84e1df7c98ce54dd5c840bb2` (merged PR #253).
New shared capability and public-contract repair: native CuPy and Torch KDE
bootstrap, including weighted Torch fitting. Active axes are API, backend/device,
numerical selectors, resampling, memory, benchmarks, documentation. No unrelated
inference or kernel-regression CV selector migration is claimed.

## Contract inventory and intended closure

| Consumer | Baseline | Intended change |
|---|---|---|
| `KernelDensityEstimator` / `KDE` / `fit_kde` / `kde_pdf` | NumPy/CuPy; Torch explicit weights and multivariate vector query broken | Repair Torch weighted fit and preserve all kernels and dimensions |
| `select_bandwidth`, KDE consumer | GPU arrays exported for selectors, including initial weight diagnostics | Native CuPy/Torch KDE selector arrays; CPU algorithm preserved |
| `kde_confidence_interval(method='bootstrap')` and alias wrapper | Host RNG/resampling/percentiles, Torch blocked by weights | Native RNG/index/refit/density/percentiles with final NumPy reporting |
| `kde_confidence_interval(method='normal')` | Host normal-CI arithmetic | Preserve method and regression-test shared fit changes |
| `KernelRegression` | Shares shape/weights/covariance helpers | Check shared helper regressions; no CV algorithm migration |
| Dev fixed-index/native-RNG candidates | Narrow historical experiments and frozen validators | Keep historical evidence identity; public production never imports dev |
| EN/CN reference/model/device docs | Describe earlier unsupported Torch paths | Reconcile only support actually implemented and tested |

Selectors: numeric, scott, silverman, nrd, nrd0, ucv, bcv, sj, sj-ste,
sj-dpi. Kernels: Gaussian, rectangular, triangular, Epanechnikov, biweight,
triweight in supported dimensions; cosine and optcosine remain 1D-only.
Weights: omitted, equal, unequal, zero entries with positive total and valid
covariance degrees of freedom. Convertible input types and layouts remain
supported, internally float64; no new device-float64-only interface.

## Statistical and RNG contract

Resample rows with replacement using normalized original weights, and refit
using those same original weights selected at sampled indices and renormalized.
A string selector is recomputed each native replicate. A numeric covariance
factor stays fixed but sample covariance changes. The existing NumPy 1D Gaussian
fast path fixes the originally selected factor; this remains an explicit
benchmark workload distinction. Intervals are pointwise percentile intervals,
not bias-corrected/simultaneous intervals or a remedy for arbitrary survey designs.

Native backends obtain exactly four uint32 words from
`numpy.random.default_rng(random_state)` once per invocation, derive one uint64
local backend seed with `SeedSequence`, then execute one device RNG vector per
replicate. None, nonnegative integer seeds beyond uint32, and existing accepted
NumPy Generator/SeedSequence inputs follow default_rng validation. Generator state
advances by the four-word bridge, unlike historical host-choice consumption.
Global RNG state is unchanged. Reproducibility is within a fixed backend/software/
device environment, not NumPy/CuPy/Torch bitstream equality. Query chunking does
not alter RNG call shape. Float64 cumulative rounding follows representable CDF
mass, without the experimental candidate's additional collapsed-mass rejection.

## Execution and resource boundaries

Native array computations include bandwidth/PCA/weighted-quantile selection,
RNG, indices, resampling, covariance/refit, density and percentiles. Host scalar
control/diagnostic extraction and the seed bridge are permitted. Input conversion
and final NumPy return are the only full numerical array transfer boundaries.
Torch CPU is an explicit supported library/device mode, not physical CUDA evidence.

One replicate's indices/refit buffers are live at a time. Exact percentiles retain
B by q draws even if callers do not request returning them. Optional
`workspace_bytes` plans bootstrap-managed arrays and reduces query chunks only;
it fails rather than dropping draws or samples. This is not a hard CUDA memory
cap: known selector arrays are included conservatively; caller inputs, original fitted
model, library scratch and allocator cache are excluded and disclosed. Separate physical memory diagnostic
must report actual peaks. Explicit accelerator failures must not turn into CPU
success.

## Verification and completion

- CPU existing regression suites and new Torch CPU numeric tests.
- Broad selector/kernel/dimension/weight/layout matrix with actual per-draw
  fixed-index references; same integer and advancing Generator state tests.
- Package import and public execution with dev excluded.
- CuPy and Torch CUDA physical tests with operation evidence and forbidden host
  conversion guards, not just output array residency.
- Synchronized full-public end-to-end benchmark with own realized-index oracle
  outside timing, two warmups/seven repeats, fair stated workload differences.
- Independent review against exact dirty fingerprint before commit/publication.

Physical GPU execution and new performance claims remain pending until an
explicitly authorized exact-source acceptance run. Historical PR #252/#253
artifacts do not validate the new public code.

## Approved native policy: weighted multivariate selector orientation

CPU selector algorithms are preserved exactly, including legacy eigensolver sign
and default sort behavior. Native selectors use a canonical eigenvector sign and
stable tie sorting. Weighted linear-quantile resampling is not reflection-invariant,
so ordinary (nondegenerate) weighted multivariate UCV/BCV/SJ factors can differ
materially from historical host-selected factors. Ten local diagnostic fixtures
showed maximum differences of 22.8%, 9.9%, 51.4% and 41.3% for UCV, BCV, SJ and
SJ-DPI respectively. These are not roundoff or performance evidence. Degenerate
principal eigenspaces can additionally change nrd-family projections.

Native operation validation conditions the independent NumPy oracle on the
actual native principal direction and stable interpolation convention; it also
checks the eigen residual. It must not rewrite the CPU oracle or relax all
numerical tolerances. The user approved retaining native capability without exact reproduction of
legacy results, with documentation of PCA differences and a warning for repeated
or near-repeated largest eigenvalues. The CPU legacy algorithm remains unchanged.
An orientation-invariant statistical redesign is not part of this migration.
This policy approval does not establish physical GPU acceptance; exact-source
correctness, warning behavior, device and resource checks remain required.
