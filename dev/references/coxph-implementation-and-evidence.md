# CoxPH implementation details and historical validation

This developer reference preserves the implementation and validation material
previously included in the bilingual CoxPH model pages at source snapshot
`b58af74bd53e32c77d546a70785af6913f6dff08`. Moving these notes does not record a new
GPU, R, or performance run. Each historical result remains limited to its own
recorded source, validator, and environment.

For current user-facing concepts, examples, support, and failure behavior, see
[CoxPH (English)](../../docs/en/models/coxph.md) or
[CoxPH（中文）](../../docs/cn/models/coxph.md).

## Implementation snapshot

The details below describe the recorded implementation, not additional public
API guarantees. In particular, the Exact denominator sums over subsets but the
implementation uses elementary-symmetric dynamic programming rather than
explicitly enumerating every subset. For the ordinary right-censored nested
path, the prefix-state work is `O(n * max_ties)` for a fixed moment width;
derivative state width also depends on the number of features. General
counting-process risk sets use the corresponding non-nested dynamic program.

The public `CoxPH` and `CoxPHCV` estimators factorize one-dimensional labels:
host strings/objects and finite numeric CuPy/Torch labels are encoded internally
as consecutive int64 codes. Low-level counting-process primitives do not
factorize labels and therefore require finite integer-valued numeric codes
representable as signed int64.

For ordinary right-censored Exact fits, the risk sets are nested within each
stratum. StatGPU sorts rows by stratum and decreasing stop time, then reuses one
segmented elementary-symmetric prefix dynamic program across every failure group
on NumPy, CuPy, and Torch without a Python loop over strata.
This removes the repeated risk-set scan that made work grow with both sample
count and failure-group count. Failure numerators use backend-native grouped
reductions instead of a dense failure-group-by-sample mask. The prefix workspace
defaults to a 512 MiB ceiling
controlled by `STATGPU_EXACT_NESTED_MAX_BYTES` and is checked before allocation.

On Torch CUDA, long multidimensional `cumsum(dim=0)` calls in PyTorch 2.0 can
dominate this otherwise linear prefix DP. For at least 2,048 rows and at most 64
trailing moment channels, StatGPU therefore lays out each channel contiguously,
runs the efficient one-dimensional CUDA scan per channel, and stacks the results
back on device. `STATGPU_TORCH_EXACT_SCAN_MIN_ROWS` and
`STATGPU_TORCH_EXACT_SCAN_MAX_CHANNELS` control these conservative gates, and
`STATGPU_TORCH_EXACT_SCAN_STRATEGY` accepts `auto`, `native`, or `channelwise`.
`auto` enables the split scan only for the benchmarked Torch 2.0 + Pascal/P100
combination; unbenchmarked Torch/GPU combinations use the native scan. CPU,
small, or wide inputs also keep Torch's native multidimensional scan. The additional
transpose/output workspace is included in the existing nested-workspace check:
if the base DP fits but the channel-scan workspace does not, the nested
algorithm stays active and uses the native Torch scan rather than falling back
to the more expensive general Exact path.

Delayed entry prevents the nested-prefix shortcut. With at least eight strata,
GPU backends first try all eligible failure groups in one backend-native batch;
smaller GPU cases and NumPy use per-stratum batches to avoid empty cross-stratum
mask work. The separate 512 MiB ceiling is controlled by
`STATGPU_EXACT_BATCH_MAX_BYTES`. An oversized global batch is retried per
stratum before the memory-bounded per-group path; score-residual requests and
conservative numerical-range gates also retain the normalized implementation.
These are explicit algorithmic fallbacks, never implicit CPU fallbacks.

For Breslow/Efron delayed-entry objectives,
`STATGPU_COX_GROUP_MAX_BYTES` controls the dense failure-group workspace
(512 MiB by default). If even one failure group exceeds the ceiling,
the selected GPU backend uses a stable multi-pass row-streaming moment
calculation. This keeps an extreme single stratum/risk set bounded instead of
letting the minimum batch size allocate an unbounded `O(n)` mask.

Full-fit inference also constructs a Breslow baseline hazard. For ordinary
right-censored rows, StatGPU now sorts each stratum by decreasing stop time and
computes every risk denominator from one log-risk prefix. NumPy uses
`logaddexp.accumulate`, Torch uses `logcumsumexp`, and CuPy uses a shifted
exponential cumulative sum within a conservative predictor-range gate. Extreme
CuPy predictors and delayed-entry rows retain the stable backend-native
per-failure-group calculation. This removes the former
failure-group-by-sample risk-mask scan from the common right-censored path.

## Historical external validation

The maintained R baseline uses R 4.4.1 with `survival` 3.8.9 and aligns ties,
Newton `max_iter=80`, and `tol=1e-8`. At `n=3000`, `p=10`, the Breslow and
Efron comparisons use 3,000 independent HC1 units and 120 cluster units. The
maximum StatGPU-versus-R coefficient/SE/p-value differences were
`5.55e-16`/`1.39e-16`/`8.00e-19` for HC1 and
`5.55e-16`/`1.32e-16`/`2.22e-16` for cluster covariance. Unsupported
statsmodels covariance modes are recorded as unsupported rather than relabeled
as external evidence.

Machine-readable R comparison artifacts:

- `results/benchmark_frontend_sources/coxph_robust_inference_breslow_pr80_20260729_schema11.json`;
- `results/benchmark_frontend_sources/coxph_robust_inference_efron_pr80_20260729_schema11.json`.

These are fixed-source, shape-specific comparisons, not a universal accuracy or
performance guarantee. Exact-ties and performance conclusions remain bound to
their dedicated artifacts listed in `dev/reviews/pr80_review_fix.md`.

### Published Physical-GPU Validation

CuPy and Torch CUDA validation checks numerical agreement, device ownership,
and error boundaries. A stable [published GPU validation record](https://gist.github.com/TheHiddenObserver/ebbb7f2401f45b124069a30d3510c139)
([raw JSON](https://gist.githubusercontent.com/TheHiddenObserver/ebbb7f2401f45b124069a30d3510c139/raw/pr80_final_gpu_suite_schema3.json))
is pinned to source commit `a726937a39eb0ed5a370dd03362884b63a9e9818`, with artifact
SHA-256 `e01ad0bfec238d06167caeef9955e92b6cf84eea4ccc69a3056eb794ded6eccb`.
It certifies only the recorded source and runtime environment; later commits do
not automatically inherit that evidence. Source hashes, imported paths, hardware,
and validation provenance are recorded in the artifact and
[developer validation reference](../reviews/pr80_review_fix.md).

This record is not a new performance-crossover benchmark or an R-alignment run,
and is not a universal accuracy or performance guarantee for every dataset,
backend version, or GPU. Those comparisons remain tied to their own artifacts.

