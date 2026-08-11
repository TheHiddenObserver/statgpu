# PR #126 post-promotion strict review checkpoint — 2026-08-11

Standard: `.claude/skills/code-review.md` (`auto-fix` mode)

## Terminal verdict

**PHYSICAL_GPU_ACCEPTED / COMPLETE / MERGE-READY**

Validation tier: `remote-full`.

Numerical measurement head:

`3dc7df19176f8fb881a8d37e9d75b4f75e71b058`

Raw evidence commit:

`be679c13357475b8d26db42661b557fbea2f327f`

Canonical promotion head:

`d771a7432e3b3da3299b6af2b05c5171579092ce`

Post-promotion hosted-review checkpoint:

`d38feab5aca8fa76b9be9f08d753be8bf85b7203`

Terminal review/status lineage:

`ad214a4c39ecf322dcc51173877152808a07c83f` → `8fc44ac9d886a74e7110f6e3f2cb450066cffa14` → `5d90950b96392e7a10f790e38f7e9fa6ee021f90`.

These terminal commits change review/status Markdown only; they do not change production source, parser logic, source manifests, deterministic benchmark assets, physical runners, or numerical evidence.

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

The raw evidence commit `be679c13...` differs from numerical measurement `3dc7df19...` only by the two immutable raw JSON artifacts. The promoted/checkpoint tree differs from the numerical measurement only by those raw artifacts plus parser/source metadata, benchmark tests, docs/review records, and deterministic benchmark assets. There is no change after the numerical measurement to:

- `statgpu/panel/**`;
- `dev/benchmarks/validate_panel_stage_c_gpu.py`;
- `dev/benchmarks/benchmark_panel_stage_c_covariance.py`.

The exact-source gate therefore remains satisfied: canonical promotion and terminal review/status updates do not invalidate the P100 numerical measurement.

## Auto-fix / re-review findings closed after physical return

- `[MEDIUM][PROVENANCE][fixed] dev/benchmarks/frontend_data/parsers/panel_stage_c_rank_policy.py` — paired correctness provenance is not presented as timing-artifact-local metadata. The timing parser records only timing-artifact facts; CuPy `13.6.0` remains cross-source manifest/review provenance.
- `[MEDIUM][MATRIX][fixed] canonical benchmark fixtures` — the two new required immutable sources legitimately move the manifest from 13 to 15 sources and the deterministic bundle from 1984 to 2120 runs. Maintained inventory/coverage/run-count fixtures were updated and pass.
- `[MEDIUM][ARTIFACT][fixed] dev/reviews/pr126_post_promotion_review_2026-08-11.md` — the stale `ec511f53...` 32/32 checkpoint was replaced with the current `3dc7df19...` 39/39 rank-policy promotion record.
- `[MEDIUM][ARTIFACT][fixed] dev/reviews/pr126_physical_gpu_validation.md` — the authoritative physical record was advanced from pending to the terminal acceptance state after the hosted/post-promotion gate completed.

No CRITICAL, HIGH, or relevant MEDIUM finding remains after the final review/fix loop.

## Final hosted gate

All seven permanent workflows completed successfully on the terminal review/status head `5d90950b96392e7a10f790e38f7e9fa6ee021f90`:

1. Tests — success;
2. Panel Stage C Torch CPU — success;
3. Panel Stage C external covariance — success, including `Run R plm and sandwich alignment`;
4. Maintenance compatibility — success;
5. Release notes validation — success;
6. Release package validation — success, including distribution validation plus Ubuntu, Windows, and macOS wheel smoke;
7. Benchmark Frontend CI — success, including deterministic data checks, frontend build, E2E, production QA, and staleness checks.

An earlier terminal-doc-only head briefly exhibited a GitHub Actions workflow/check-suite aggregation lag for Release package validation even though all four constituent checks had already completed successfully. The new terminal head completed the same workflow normally at workflow level, confirming that the earlier state was platform aggregation lag rather than an unexecuted or failed gate.

## Review-thread state

There are three historical inline review threads:

- stale CUDA-docs P2: resolved and outdated;
- ordered-categorical chronology P1: resolved and outdated;
- backend-provenance P2: unresolved but outdated.

The unresolved backend-provenance thread is intentionally retained as review history. Its underlying issue is closed: fitted estimators persist the backend selected at the numerical fit boundary, maintained CPU parity exercises the contract, and the current P100 correctness artifact records requested/executed backend equality for every estimator and primitive.

## Lifecycle boundary

The technical Stage-C gate is closed: **PHYSICAL_GPU_ACCEPTED / COMPLETE / MERGE-READY**.

PR #126 intentionally remains Draft, open, and unmerged. No Ready-for-review transition or merge may be performed without explicit user instruction.
