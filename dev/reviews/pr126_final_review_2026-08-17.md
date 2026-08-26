# PR #126 historical physical acceptance snapshot

Date: 2026-08-17

> **Historical evidence only (status updated 2026-08-18).** Subsequent PR #126 review/fix loops changed valid Fama-MacBeth, shared panel least-squares, covariance, and diagnostic numerical paths after the source recorded below. This snapshot must not be used as current-head merge acceptance. Fresh exact-head CuPy and Torch CUDA revalidation is required and remains pending until new physical artifacts are produced.

This file is the durable repository-side audit record for PR #126. It records
technical and physical-validation facts only. Transient lifecycle state such as
Draft, merge-ready, hosted-pending, or generated-assets-pending belongs in the
pull-request timeline and CI rather than in this file, so this record does not
become stale when a documentation/artifact-only descendant is created.

## Scope

PR #126 completes Panel Tier-1 Stage C covariance/inference work tracked by
Issue #93 and includes the later Fama-MacBeth correctness, backend-inference,
and performance review/fix work performed under `.claude/skills/code-review.md`.

Active review axes were BACKEND, INFER, FORMULA, MATRIX, PERF, TEST, DOC,
ARTIFACT, API lifecycle, and maintainability. CV/loss/penalty/regularization
solver gates are not applicable to the touched panel covariance capabilities.

The architectural/capability contract remains in
`dev/plans/panel_p1_stage_c_covariance_plan.md`.

## Historical validated numerical source

The historical numerical source validated on physical GPU in this snapshot is:

`8c60db00f5ea986aed96b1f1dce3f5c3b4f0bcd4`

At the time of this snapshot, production numerical code remained frozen after this source during its physical-evidence promotion, documentation reconciliation, artifact cleanup, and deterministic benchmark-data regeneration. Later review/fix loops intentionally changed that numerical code, so the evidence below is historical rather than current-head acceptance.

## Historical Fama-MacBeth solver contract at `8c60db00...`

At this historical source, NumPy remained the serial rank-revealing SVD statistical reference. GPU retained
periods are grouped by exact row count and first receive a conservative Gram
spectrum certificate. A period may use the batched normal-equation candidate
only when

`lambda_min(X'X) / lambda_max(X'X) > 1e-4`.

The certificate is a performance gate, not a replacement rank definition.
Uncertified periods fall back to the maintained SVD cutoff
`max(n_t, k) * eps * s_max`; Torch may use its documented stacked-SVD support
for an unsafe subset, while CuPy keeps supported two-dimensional SVD fallback
solves. Near-rank-boundary and rank-deficient behavior therefore remains
SVD-owned.

Maintained regression coverage includes both sides of the SVD rank boundary,
well-conditioned Gram/SVD coefficient parity, balanced and unbalanced period
bucketing, chronological rank rejection, no-intercept formula rejection,
backend-native inference, failed-refit invalidation, and direct-fit finite-input
validation ownership.

## Historical Tesla P100 physical acceptance

Four final physical runners passed from the same clean detached numerical source
`8c60db00...` on Tesla P100-SXM2-16GB with CuPy 13.6.0 and Torch 2.0.0:

1. `results/pr126_p100_fama_fix/panel_stage_c_correctness_8c60db00.json`
   - 35 estimator cases per GPU backend;
   - 12 public covariance primitives per GPU backend;
   - requested/executed backend and prediction contracts passed.
2. `results/pr126_p100_fama_fix/fama_macbeth_optimized_v2_8c60db00.json`
   - chronology, formula, rank, inference, reporting, and optimized solver
     provenance passed.
3. `results/pr126_p100_fama_fix/fama_macbeth_scaling_v2_8c60db00.json`
   - synchronized resident-array micro/medium/large scaling and numerical parity.
4. `results/pr126_p100_fama_fix/panel_hac_chronology_8c60db00.json`
   - ordered chronology, lexical negative control, and shared backend-native
     Student-t inference passed.

All four artifacts record the exact numerical SHA and clean-source provenance.

### P100 Fama-MacBeth scaling

`backend_over_numpy_median_ratio` (smaller than 1 means GPU faster):

| scale | CuPy | Torch |
| --- | ---: | ---: |
| micro, 64 x 128 x 4 | 0.549 | 0.343 |
| medium, 128 x 1024 x 8 | 0.204 | 0.168 |
| large, 128 x 4096 x 16 | 0.114 | 0.109 |

These correspond to approximately 1.82x/2.92x, 4.91x/5.97x, and
8.75x/9.16x CuPy/Torch speedups on the maintained P100 resident-array protocol.
Every measured GPU backend x scale reports one `gram-certified` solver batch,
one control synchronization, and zero SVD fallbacks. Coefficient, beta-series,
and prediction differences remain near machine precision; the largest recorded
scaling statistic difference is below `4e-11`.

These measurements are workload- and hardware-specific evidence, not a universal
GPU speedup guarantee.

## Benchmark and historical-evidence retention boundary

Historical Stage-C artifacts explicitly registered in
`dev/benchmarks/frontend_sources.json` remain immutable repository inputs and are
protected by manifest SHA256 identities.

The benchmark catalog also deliberately retains a small set of superseded,
unregistered Stage-C measurements as historical audit evidence. The maintained
`test_stage_c_superseded_artifacts_remain_historical` regression protects the
`5ed763be`, `9c0b3050`, `aad53587`, and `c151550a` correctness/performance pairs.
Those files therefore remain in the durable tree even though they are not
canonical dashboard sources.

The final `8c60db00` Fama-MacBeth/HAC physical artifacts are retained as direct
PR acceptance evidence. They are intentionally not canonical dashboard sources
unless a separate dated source-registration decision is made.

Pre-final Fama-MacBeth physical JSONs that are unregistered, undated, superseded
by the exact-source `8c60db00` evidence, and not protected by an explicit
repository retention contract were removed during final merge-tree cleanup.
Duplicate `.log` copies of the registered PR126 final Stage-C JSON evidence were
also removed; the JSON sources, environment metadata, and human-readable
validation summary remain. Removed material remains recoverable from Git history
and the PR timeline.

## Review-record policy

Intermediate PR126 review/fix checkpoint Markdown files were intentionally
removed from the durable merge tree after completion. They encoded transient
states such as physical-pending, hosted-pending, or Draft status and could
contradict later accepted evidence. The full checkpoint history remains in Git
history and the pull-request discussion.

This file replaces those checkpoint files as the single repository-side PR126
audit record. It deliberately avoids transient merge/readiness claims.

## External and hosted validation

The maintained hosted matrix covers:

- repository Tests;
- Benchmark Frontend deterministic data/build checks;
- Panel Stage C Torch CPU, including maintained Torch 2.0 coverage;
- Panel Stage C external covariance definitions, including pinned Python
  references and R `plm` / `sandwich` alignment;
- maintenance compatibility;
- release notes validation;
- release package validation.

Exact-head hosted results are authoritative in GitHub Actions and the PR timeline
rather than copied back into this repository record after every documentation or
artifact-only descendant.

## Separate follow-up

Legacy Gaussian linear-model inference backend debt is tracked separately in
Issue #127 and is not part of the PR126 numerical acceptance contract.
