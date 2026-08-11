# PR #126 rank/chronology re-review checkpoint

Status: **PARTIAL_REMOTE_PENDING / LOCAL REVIEW IN PROGRESS / NOT MERGE-READY**

Validation tier: `local-full` is the current target for active local gates; fresh `remote-full` physical CUDA correctness and performance evidence remains required.

## Impact classification and capability decisions

Active axes remain Public API, Inference, Backend, Formula/data alignment, Benchmark/performance, and Docs/artifacts. Loss, penalty, solver, and CV remain inactive because Stage C changes covariance/inference and metadata ordering rather than objectives, regularization, solvers, or tuning.

Touched public model-family capability decisions remain those in `dev/plans/panel_p1_stage_c_covariance_plan.md`: all panel families are three-backend and inference/formula supported; PooledOLS, PanelOLS, and RandomEffects require benchmark evidence; BetweenOLS, FirstDifferenceOLS, and FamaMacBeth are documented not-performance-sensitive for Stage C; CV is non-tunable/not applicable.

## Findings closed in the current auto-fix cycle

- `[CRITICAL][INFER][fixed] statgpu/panel/_covariance.py` — pseudoinverse and numerical rank now come from one backend-native SVD mask with explicit float64 cutoff `max(n,k) * eps * s_max`, eliminating independent NumPy/CuPy/Torch defaults near the numerical-rank boundary.
- `[CRITICAL][FORMULA][fixed] statgpu/panel/_first_diff.py` — FirstDifferenceOLS now derives chronology codes through `factorize_panel_metadata`, preserving pandas ordered-categorical time ordering before differencing.
- `[HIGH][MATRIX][fixed locally / needs remote GPU] dev/benchmarks/validate_panel_stage_c_gpu.py` — the maintained physical primitive matrix now adds rank-boundary nonrobust, HC0, HC2, HC3, cluster, and DK cases, expanding the physical target from 26+6 to 27+12 checks per backend.
- `[LOW][ARTIFACT][fixed] dev/benchmarks/validate_panel_stage_c_gpu.py` — CuPy version provenance probes CUDA-specific distribution names.
- `[MEDIUM][TEST][fixed] dev/tests/test_panel_stage_c_external.py` — the independent stable-HC reference now implements the reviewed explicit SVD cutoff instead of reintroducing NumPy's default `pinv` threshold.
- `[MEDIUM][DOC/ARTIFACT][fixed] plan/docs/changelogs/physical review record` — the explicit numerical-rank policy is documented and the old `ec511f53...` P100 run is consistently preserved as historical evidence rather than current acceptance.

## Validation completed before this checkpoint

- First repair focused covariance/formula/runner/inference suite passed before production commit `dc8abf00...`.
- Maintained `Panel Stage C Torch CPU` workflow passed on connector-authored checkpoint `e3a88cf2...`, including the new numerical-rank boundary parity test.
- Full `Tests` workflow passed on `e3a88cf2...`.
- The first hosted external run exposed only the stale NumPy-default reference helper; production definitions were not weakened to satisfy it.
- After reference alignment, **34/34** external-definition/estimator tests passed.
- The focused covariance/edge/physical-runner suite passed **53/53**.
- Syntax compilation and `git diff --check` passed.

## Evidence lifecycle

The earlier exact-clean P100 source `ec511f539adeaaedf310f92248200d0868577532` predates the repaired production source and is therefore immutable historical evidence only. Its canonical source IDs must not be overwritten.

Fresh `remote-full` acceptance requires an exact-clean run at the final local review head with:

- CuPy: 26 estimator cases + 12 public primitives = **39/39**;
- Torch: 26 estimator cases + 12 public primitives = **39/39**;
- requested backend equal to persisted/executed backend for every case;
- rank-boundary nonrobust/HC0/HC2/HC3/cluster/DK primitives present and passing;
- synchronized performance rerun using the maintained performance runner.

Fresh results must be registered as new immutable benchmark sources rather than replacing the historical `ec511...` canonical sources.

This connector-authored checkpoint intentionally triggers the full permanent hosted workflow matrix. A subsequent `.claude/skills/code-review.md` re-review will determine whether any new CRITICAL/HIGH/relevant-MEDIUM issue remains locally. The PR remains Draft and unmerged.