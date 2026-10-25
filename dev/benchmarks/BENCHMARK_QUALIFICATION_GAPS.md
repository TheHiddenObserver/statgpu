# Candidate benchmark qualification inventory

This inventory distinguishes registered representative coverage, independent
runner inventory, finite configuration axes and physical evidence. It is a
contract audit, not a physical run. Every execution must retain its actual source
identity and artifacts. The retained PR #257 P100 evidence belongs to `62fcd9ab`;
its representative scan does not validate changed runners/adapters or new matrix
points, and does not establish performance eligibility.

| Suite | Registered identities | Timing/evidence addressed here | Remaining bounded claim |
|---|---:|---|---|
| gaussian-refresh | 2 | Plain public estimators timed separately from observed ingress preflight | Ingress/output precision is not computed intermediate precision |
| classification-metrics | 7 | Shared qualification distinguishes output ownership from numerical execution | Native operation/dtype tracing remains missing; no promotion |
| model-families | 74 | Shared qualification retains per-algorithm traces and generic dispatch distinction | Generic dispatch arrays are not whole-algorithm proof |
| statistical-families | 138 | Host-only NA whitelist and diagnostic RNG/weak-oracle restrictions retained | Host RNG is not native CUDA RNG; resampling identities are not coverage/FDR proofs |
| learning-families | 44 | Algorithm-stage evidence and untimed tracing for helper/estimator paths | Explicit host/hybrid, invariant-only and unresolved stages remain noneligible |
| workflow-families | 4 | Mandatory correctness preflight; per-scope host/mixed placement | Ingress observation remains in some fit/selection samples; GPU sync does not make host diagnostics CUDA computation |
| six-family-cv | 6 | Separate clean synchronized public-fit samples from profiled selector/refit diagnostics | Float64 accuracy protocol is not mixed-precision default performance; no CV-only subtraction |

There are **275 registered identities**, not 275 expanded parameter/backend
rows. The unchanged default representative campaign selects **21 invocations and
1692 rows**, including **16 expected host-only CUDA not-applicable rows**. The
latter are dispositions, not successful GPU numerical operations. The campaign
constructs exact expanded inventories from selected shapes, dtypes, scenarios,
seeds, profiles and backends. Missing/duplicate rows and source drift must not be
hidden by exit 0.

## Inventory denominators and deduplication

The source audit snapshot covered 202 tracked Python files (158 top-level and 44
nested), of which seven were the registered campaign runners and 41 additional
files were parameterized timing runners. The remaining files include correctness
gates, collection runners, wrappers, diagnostics, legacy programs and helpers.
These are snapshot counts, not live file counts after adding infrastructure.
New support files are reported separately as `new_since_audit`.
**41 files is not 41 disjoint or uniformly executable suites.** Default remote
launchers, import-time workloads, hard-coded output paths and missing structured
artifacts need explicit dispositions; a directory-wide execution loop is not an
acceptable substitute for adapters.

The 41 independent entries currently split into 25 executable adapters (14 ready and 11 conditional on external dependencies),
nine blocked admissions, six unsupported contracts and one deduplicated CV-models entry. Bounded per-lane configurations are independently identified; their count is
not the file count. Admission is not execution or numerical acceptance; some admitted protocols are timing-only and must preserve
correctness/device-proof unavailable status.

The public coverage mapping contains 151 entries: 113 direct smoke mappings,
13 runner-smoke mappings, three direct mappings with unreleased repairs, three
indirect mappings, four aliases and 15 infrastructure exemptions. These describe
relationships, not 151 numerical passes, independent oracles or physical GPU
proofs. Wrapper/child relationships, aliases and overlapping scale points must
be deduplicated by their actual case/configuration/protocol identities.

Completion must report planned, executed, passed, failed, blocked,
not-applicable and not-run work separately. Selection completeness means every
selected executable contract is accounted for; it does not turn blocked axes
into covered axes. Preserve the exact requested denominator in the plan and raw
artifacts rather than deriving it only from successful results.

## Finite configuration axes and remaining gaps

| Axis / lane | Required distinction or unresolved evidence |
|---|---|
| Seeds | Schedule and preserve explicit seeds; repeated-seed smoke is not Monte Carlo coverage, power or FDR calibration |
| Scale and shape | Separate n/p/query/target/grid/bootstrap axes; honor pairwise/workspace and oracle bounds; overlapping default/medium/large rows are not additive unique coverage |
| Learning scales | All 44 cases are addressable at explicit shapes under per-algorithm policies; expanded candidate points beyond those bounds remain exclusions, not silently shrunk or passed |
| Quantile reference | Large n=2048 exceeds the existing dense-LP oracle limit n=2000; use a suitable independent oracle or retain blocked status, never loosen tolerances |
| Solver/family/penalty | Declare supported combinations and high-risk interactions per algorithm; fixed registry cases do not imply a full public parameter product |
| Covariance/inference | Align covariance, weights, clustering, ties/strata and reference semantics; numerical output does not establish inferential calibration |
| Weights/sparse | Distinguish no/uniform/unequal/zero-containing weights from CSR/CSC structure; mark unsupported and unadapted combinations explicitly |
| CV | Preserve folds, grid, scoring, selection and final refit in each profile; profiler totals cannot supply clean CV-only timings by subtraction |
| Precision | Record input, observed compute and output dtype separately; fixed float64 or promotion does not close native-float32 computation |
| Backend/device | Preserve host-only and host/hybrid dispositions; output placement and generic dispatch do not prove whole-algorithm native CUDA |
| Memory | Process RSS, allocator snapshots, estimated buffers and measured GPU peaks are different; per-algorithm GPU peak evidence remains a separate measurement lane |
| External/statistical/hardware | R and external-package dependencies, repeated-seed calibration, multi-GPU/nondefault-device and MPS need dedicated contracts and applicable environments |

Independent timing families include public KDE bootstrap and hybrid phases;
Cox ties/strata/cluster/stability; panel covariance and Fama–MacBeth scale;
inference, quantile and group/penalized GLM comparisons; unsupervised and
external-package comparisons; compile cold/steady-state and memory-cleanup
studies. A supported adapter must specify bounded argv, expected artifact/row
contracts, dependencies, scope and source identity. Unsupported producers remain
in the inventory with a reason rather than disappearing from the report.

The public KDE bootstrap, controlled-index developer kernel, native-RNG
experiment and hybrid-phase runner have different protocols. Do not sum them as
independent public API coverage or combine phase measurements into E2E. Kernel,
1D/multidimensional support, weight pattern, numeric/string bandwidth, query and
bootstrap counts and workspace are distinct axes. Numeric covariance factors and
selector refits have different semantics; shared seeds across backend RNGs are
not paired draws. A large KDE Cartesian product is not implicitly authorized or
safe just because its small components are individually valid.

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

## Configured correctness failures are retained

Local small-fixture review (Python 3.12 CPU, not P100 acceptance) observed:
- PCA randomized, n=64,p=32,seed=13: residual 8.441084122116452 versus
  exact rank-2 optimum 8.43893174893318, beyond the unchanged accuracy allowance.
  The CLI emits failed/no accepted timing. Approximate-solver quality is not
  silently relabelled exact-oracle parity.
- Float32 input, analytic weights, n=48,p=3,seed=19: Gaussian/newton and
  negative-binomial/lbfgs can hit max_iter even when coefficient errors pass.
  The convergence gate retains failure; this benchmark-only change does not
  modify production solvers or relax tolerances.
These are configuration-specific observations, not a claim that every backend
or environment must exhibit the same failure. Fresh physical results are pending.
