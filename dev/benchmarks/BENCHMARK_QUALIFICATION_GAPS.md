# Candidate benchmark qualification inventory

Source baseline: merge `48d03262e765fe276888c4781fd2819b16888bad` (PR #257).
This inventory is a contract audit, not a physical run. The retained PR #257
P100 evidence belongs to `62fcd9ab`; its successful representative scan does not
validate this changed harness or establish performance eligibility.

| Suite | Registered identities | Timing/evidence addressed here | Remaining bounded claim |
|---|---:|---|---|
| gaussian-refresh | 2 | Plain public estimators timed separately from observed ingress preflight | Ingress/output precision is not computed intermediate precision |
| classification-metrics | 7 | Shared qualification distinguishes output ownership from numerical execution | Native operation/dtype tracing remains missing; no promotion |
| model-families | 74 | Shared qualification retains per-algorithm traces and generic dispatch distinction | Generic dispatch arrays are not whole-algorithm proof |
| statistical-families | 138 | Host-only NA whitelist and diagnostic RNG/weak-oracle restrictions retained | Host RNG is not native CUDA RNG; resampling identities are not coverage/FDR proofs |
| learning-families | 44 | Algorithm-stage evidence and untimed tracing for helper/estimator paths | Explicit host/hybrid, invariant-only and unresolved stages remain noneligible |
| workflow-families | 4 | Mandatory correctness preflight; per-scope host/mixed placement | Ingress observation remains in some fit/selection samples; GPU sync does not make host diagnostics CUDA computation |
| six-family-cv | 6 | Separate clean synchronized public-fit samples from profiled selector/refit diagnostics | Float64 accuracy protocol is not mixed-precision default performance; no CV-only subtraction |

There are 275 identities, not 275 expanded parameter/backend rows. The campaign
constructs exact expanded inventories from selected shapes, dtypes, scenarios and
backends. Missing/duplicate rows and source drift must not be hidden by exit 0.

## Independent qualification lanes

The shared `benchmark_qualification.py` reads raw artifacts without modifying
or promoting them. Process exit, source identity, row dispositions and selected
identities govern the stored-result correctness lane. The performance lane is
always diagnostic in this campaign: `UNVERIFIED_EXCLUSIVITY`, no comparisons and
no canonical dashboard qualification. Producer eligibility flags cannot override
this. Dirty local source is partial evidence, never canonical evidence.

GPU output ownership is deliberately not treated as numerical execution proof.
Source-specific primitive/operator observations are finite, not transitive
whole-program certification. Runtime numerical operations must have actual
backend/device and computed dtype evidence; input/output dtype alone is not enough.
Host control, report conversion, host numerical computation and native numerical
computation are separate concepts. An observed hybrid workflow may be numerically
correct without qualifying as an entirely native implementation.

## Timing, precision and memory

Retain every raw repeat and its actual synchronization target. Correctness and
convergence must pass before accepting timing; a later bad repeat invalidates all
samples for that row. CV profiler totals remain diagnostic, not clean E2E.
Gaussian fit/predict and workflow stages have distinct transfer scopes and must
not be pooled or subtracted to manufacture a new measurement.

Process high-water RSS, allocator peaks, pool snapshots, estimated workspaces and
per-call memory requirements are different metrics. Missing GPU peak measurement
must remain `not_measured`; an estimate is not a hardware resource guarantee.
Separate input dtype from observed computed dtype and output dtype. Existing
float64 promotions and approximate `C=1e12` Logistic references are unchanged.

## Environment and promotion

Existing Docker/P100 may run correctness and diagnostic measurements without
host PID mapping. Unknown NVML PIDs are neither self nor proven competitors.
No mounts, security changes or killing unrelated processes are authorized by a
benchmark invocation. Controlled performance requires independently reviewable
allocation/process exclusivity evidence for the measured interval. No flag here
can assert that evidence into existence.

Canonical promotion remains a separate reviewed change: immutable source and
raw-data hashes, aligned objective/grid/folds/data/precision/transfer/cache
semantics, matched environment groups, appropriate independent numerical oracle,
source registration/parser/schema/coverage updates and dashboard tests. This PR
does not register new measurements or publish the dashboard.
