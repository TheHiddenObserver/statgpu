# Native CuPy KDE: endogenous resampling experiment

Base: merged PR #252, `8417c200d2e0751c94caef698060419867f30a3f`.
Status: developer-only implementation; physical acceptance of this RNG extension
is pending. The earlier fixed-index P100 evidence does not validate this change.
No public dispatch, return type, defaults or package version changes.

## Contract reconnaissance and approved boundary

| Consumer | Existing behavior | This experiment |
| --- | --- | --- |
| Public `kde_bootstrap_confidence_interval` | NumPy RNG/host resampling; NumPy fast path or hybrid CuPy refits; final NumPy results | Unchanged, used as timed baselines |
| Fixed-index developer candidate | CuPy covariance/density/quantiles; caller-owned indices | Unchanged, deterministic oracle/control |
| New RNG candidate | Not previously present | CuPy-only float64 univariate Gaussian; fixed positive numeric bandwidth |
| Torch, string bandwidth, other kernels/multivariate, public opt-in/default | Outside the accepted candidate scope | Explicitly deferred; no new public support claim |

Active axes: resampling, RNG/reproducibility, backend/device, allocation ownership,
performance/evidence. This is the explicitly scoped CuPy-native experiment, not a
new three-backend public estimator. No CV, loss/penalty or formula change.

## Statistical and RNG contract

The resampling unit is an observation row, sampled independently with replacement.
Original nonnegative weights define sampling probabilities and again define the
fit weights of selected rows. Resampled weights are normalized before the fixed
numeric-factor covariance refit. This preserves the existing unusual weighted
procedure; it is not a replacement for a study-design-specific weighted bootstrap.
Intervals are linear pointwise percentiles conditional on the fixed factor, not
bias-corrected intervals, simultaneous bands or a coverage guarantee.

`native_kde_bootstrap_rng` creates a local CuPy `RandomState` on the input device
for every call. An explicit integer seed in [0, 2**32 - 1] is required (default 0);
bools, None, negative and larger seeds fail. A full B-by-n float64 uniform matrix
is generated in one call, then mapped by device `searchsorted(..., side='right')`
into a device CDF. This avoids CuPy 13.6 weighted `choice`'s size-by-n temporary.
The complete CDF is normalized by its terminal value; zero-mass plateaus remain
zero mass. A positive weight that underflows or collapses to zero CDF increment
is rejected, not silently made unreachable. This is a narrower finite-precision
probability envelope, not public API compatibility for every weight vector.

Reproducibility is within the same CuPy/CUDA/cuRAND version and concrete device.
Repeated seed/input gives repeated indices, independently of query chunk size;
there is no RNG batch-size knob or automatic sample-count downscale. Global
CuPy/NumPy RNG state is not used or modified. NumPy and CuPy seeds do not promise
identical streams; neither cross-version nor cross-device bitwise reproducibility
is claimed. This uint32 seed contract is intentionally narrower than the public
NumPy `random_state` contract.

The four result arrays remain on the input GPU. Optional `return_indices=True`
returns the device indices for an untimed diagnostic oracle; it is not enabled
in timed calls. Final host reporting is outside the candidate and included in
end-to-end benchmark timings. The public result type remains NumPy-backed.

## Resource and numerical limits

The managed workspace plan now reserves endogenous B*n indices, random uniforms,
search output and CDF/probability/validation arrays in addition to the inherited
KDE buffers. It checks the minimum before RNG construction. Inputs, CUDA context,
cuRAND internal state and unobservable library scratch remain outside this plan;
it is not a hard total-CUDA or process-RSS cap. Separate fresh-pool diagnostics
must include RNG allocations, while disclosing allocations that bypass the pool.
Full indices consume O(B*n), not O(B*n*n); larger requests fail the plan instead
of silently changing B, n or the seed schedule.

The fixed-index candidate's finite/conditioning envelope is inherited. A numerical
or invalid resample failure aborts the entire call; no partial intervals or host
numerical fallback are returned. Native public integration must separately
resolve broader bandwidth selector and numerical compatibility.

## Acceptance and performance

See [physical acceptance recipe](../benchmarks/NATIVE_KDE_RNG_ACCEPTANCE.md).
The fixed-index regression suite stays active. RNG tests additionally cover seed
validation, exact CDF boundaries/zero support, approximate weighted frequencies,
reproducibility, global RNG isolation, views/singletons, retained output lifetime,
nondefault stream, mixed-device rejection and preallocation budget failure.

The new benchmark times the actual public NumPy fast path, actual public CuPy
bootstrap and native RNG candidate with RNG construction/generation, required
transfers and all four host outputs included. Each implementation's own realized
indices are captured only during a separate diagnostic run and passed to the
independent Gaussian oracle. Timed calls are unpatched and uninstrumented, with
per-call checks outside the timer against that seeded diagnostic output. This
separates algebraic correctness from random-stream equality; it does not declare
cross-backend draws elementwise identical. Statistical sampling-law tests are a
separate gate, not a claim of Monte Carlo confidence-interval coverage.

Small and GPU-scale points are both retained. CPU wins at small scale do not
invalidate the native path; no scale-independent speedup is inferred. The old
same-index runner remains a separate controlled computation comparison, not an
end-to-end public-API benchmark. Torch, extended scales and extra CPU thread
profiles require their own future scope/resource decisions.

## Design audit and next stage

Preimplementation audit checked inverse-CDF support, weighted-choice allocation,
seed/global-state semantics, normalization, resource accounting and the public
NumPy fast-path comparison. Public API integration is a later explicit opt-in
stage: preserve old defaults/return conventions, define selector/refit and RNG
metadata, and revalidate compatibility before any default promotion.
