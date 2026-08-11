# PR #126 post-promotion strict review checkpoint — 2026-08-11

Standard: `.claude/skills/code-review.md` (`auto-fix` mode)

## Status

**PHYSICAL_GPU_ACCEPTED / POST-PROMOTION REVIEW CLEAN / HOSTED_FINAL_PENDING**

Validation tier: `remote-full`.

Numerical measurement head:

`3dc7df19176f8fb881a8d37e9d75b4f75e71b058`

Raw evidence commit:

`be679c13357475b8d26db42661b557fbea2f327f`

Canonical promotion head reviewed here:

`d771a7432e3b3da3299b6af2b05c5171579092ce`

## Active strict-review axes

- public API / presentation contract;
- inference;
- backend / three-backend behavior;
- formula and panel-metadata alignment;
- benchmark/performance;
- docs/artifacts.

Loss, penalty, optimizer/solver framework, and tunable CV remain inactive for Stage C. Capability decisions remain those recorded in `dev/plans/panel_p1_stage_c_covariance_plan.md`: the touched panel estimators are three-backend and inference/formula supported; PooledOLS, PanelOLS, and RandomEffects require physical benchmark evidence; BetweenOLS and FirstDifferenceOLS are not-performance-sensitive for this Stage-C change; FamaMacBeth covariance remains frozen/out of scope.

## Physical evidence audit

The exact-clean Tesla P100-SXM2-16GB evidence passed a fail-closed raw audit before canonical promotion.

Correctness/backend provenance:

- artifact: `results/pr126_p100/panel_stage_c_gpu_validation_3dc7df19.json`;
- measurement SHA: `3dc7df19176f8fb881a8d37e9d75b4f75e71b058`;
- clean measurement tree: yes;
- CuPy package version recorded by the correctness runner: `13.6.0`;
- CuPy: **39/39** = 27 estimator integrations + 12 direct public covariance primitives;
- Torch: **39/39** = 27 estimator integrations + 12 direct public covariance primitives;
- every estimator and primitive records requested backend == executed backend;
- numerical-rank-boundary primitives `nonrobust`, `HC0`, `HC2`, `HC3`, `cluster`, and `DK` pass on both GPU backends;
- `panel_rank_boundary_dk` passes on both backends with `design_rank=2`, `design_columns=3`, and `rank_deficient_extension=true`;
- no numerical CPU fallback is accepted by the physical runner.

Immutable correctness identity:

- canonical source: `panel-stage-c-rank-policy-validation-pr126-20260811-c67ada7ec59f`;
- SHA-256: `c67ada7ec59ff85d6a652714dc45cdae81efc67a4c13fe853c85d0712fc689ad`;
- Git blob: `5de153023af751ff960969483c11fb77dc115456`.

Synchronized performance:

- artifact: `results/pr126_p100/panel_stage_c_performance_3dc7df19.json`;
- measurement SHA: `3dc7df19176f8fb881a8d37e9d75b4f75e71b058`;
- clean measurement tree: yes;
- **58/58** rows = 54 maintained base rows + 4 bounded high-T QS rows;
- high-T matrix: CuPy/Torch × PooledOLS/PanelOLS QS at `N=10,000`, `k=2`, `T=200`;
- timing scope: synchronized end-to-end estimator fit;
- every raw timing sample is finite and positive, and every stored median equals the median of its three raw samples;
- no speedup or CPU-baseline claim is encoded.

Immutable performance identity:

- canonical source: `panel-stage-c-rank-policy-performance-pr126-20260811-f27bef0b7c55`;
- SHA-256: `f27bef0b7c55db8a8d31f0574e3133760649c28a9089b3c7d3cfcc121009e152`;
- Git blob: `5e903cf0bd9a5f73e9273d9cc3488cb5e13219da`.

The performance artifact itself records `environment.packages.cupy=null` because that runner queried the unsuffixed `cupy` distribution name while the environment uses a CUDA-suffixed CuPy distribution. This is a non-numerical metadata limitation, not a backend-execution ambiguity: the paired required exact-head correctness artifact independently records CuPy `13.6.0`, while the performance runner imported CuPy and failed closed unless each timed estimator persisted the requested backend. The v2 performance parser deliberately does **not** synthesize a CuPy version or a cross-artifact backend check into timing rows; the relationship is documented only in immutable manifest provenance.

## Canonical promotion audit

The new evidence is appended through new immutable v2 parser/source identities. The historical v1 `ec511f53...` sources and parser contracts are not overwritten.

Promotion-local validation completed successfully:

- deterministic benchmark generator: **2120 runs**, **47 models**;
- manifest sources: **15/15 available and parsed**;
- generator validation errors: **0**;
- generator errors/warnings: **0 / 0** (existing method-unavailable notices remain informational only);
- maintained parser/catalog/inventory/domain suite: **59/59 passed**;
- strict `generate_benchmark_data.py --check --strict-sources`: success;
- frontend source/e2e/contract TypeScript checks: success;
- frontend production build: success;
- `git diff --check`: success.

The generated inventory reports 15 eligible sources, 15 registered sources, 15 available registered sources, 15 parsed registered sources, zero eligible-unregistered sources, and zero unclassified artifacts.

## Exact-source applicability

The raw evidence commit `be679c13...` differs from numerical measurement `3dc7df19...` only by the two immutable raw JSON artifacts. The promotion gate then explicitly rejected any change from `3dc7df19...` to the promoted tree under:

- `statgpu/panel/**`;
- `dev/benchmarks/validate_panel_stage_c_gpu.py`;
- `dev/benchmarks/benchmark_panel_stage_c_covariance.py`.

The gate passed. Promotion therefore changes parser/source metadata, maintained benchmark tests, docs/review records, and deterministic frontend assets only; it does not invalidate the exact-head physical numerical measurement.

## Post-promotion re-review findings

- `[MEDIUM][ARTIFACT][fixed] dev/reviews/pr126_post_promotion_review_2026-08-11.md` — this checkpoint still described the superseded `ec511f53...` 32/32 promotion. It is now refreshed to the current `3dc7df19...` 39/39 rank-policy evidence and `d771a743...` canonical promotion.
- `[MEDIUM][PROVENANCE][fixed during promotion review] dev/benchmarks/frontend_data/parsers/panel_stage_c_rank_policy.py` — the performance parser initially risked presenting paired correctness provenance as if it were timing-artifact-local metadata. The promoted parser now records only facts proven by the timing artifact; the CuPy `13.6.0` relationship is retained only in manifest/review provenance.
- `[MEDIUM][MATRIX][fixed during promotion] benchmark canonical fixtures` — adding two required sources legitimately changes the manifest from 13 to 15 sources and the deterministic bundle from 1984 to 2120 runs. Maintained source-count, inventory, coverage-list, and run-count contracts were updated and passed 59/59.

The post-promotion source/parser/manifest/generated-data/docs review found no remaining CRITICAL, HIGH, or relevant MEDIUM issue.

## Review-thread state

There are three historical inline review threads:

- stale CUDA-docs P2: resolved and outdated;
- ordered-categorical chronology P1: resolved and outdated;
- backend-provenance P2: unresolved but outdated.

The unresolved backend-provenance thread is intentionally retained as review history. Its underlying issue is closed: fitted estimators persist the backend selected at the numerical fit boundary, maintained CPU parity exercises the contract, and the current P100 correctness artifact records requested/executed backend equality for every estimator and primitive.

## Final gate

This review-only checkpoint does not change production source, physical runners, parser logic, manifest identities, or generated benchmark assets. Permanent hosted workflows must now complete green on the checkpoint head, followed by one final read-only `.claude/skills/code-review.md` review of the checkpoint delta and workflow results.

If that final review finds no new CRITICAL, HIGH, or relevant MEDIUM issue, the technical status may advance to:

**PHYSICAL_GPU_ACCEPTED / COMPLETE / MERGE-READY**

PR #126 remains Draft, open, and unmerged. It must not be marked Ready or merged without explicit user instruction.
