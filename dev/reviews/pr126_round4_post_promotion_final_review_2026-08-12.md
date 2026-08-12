# PR #126 post-promotion final review — 2026-08-12

Review standard: `.claude/skills/code-review.md` (`auto-fix` mode)

## Checkpoint verdict

`PHYSICAL_GPU_ACCEPTED / CANONICAL_PROMOTED / LOCAL_REVIEW_CLEAN / HOSTED_FINAL_PENDING / NOT MERGE-READY`

Technical candidate before this review-record-only checkpoint:

`a4f8c6e8e0ed6c198dddccda275e0eff746b1f83`

The only tree delta introduced by this checkpoint is this review Markdown file. Production source, physical runners, immutable raw P100 artifacts, v4 parser/source registrations, maintained tests, coverage metadata, and generated benchmark assets are unchanged.

## Fresh physical evidence

Exact-clean measurement source:

`a99726e19c535dfcd0a94711bbc8be6aac437584`

Immutable raw artifact commit:

`ccc46da6c5f2dee025c7715e39215db69b2872b8`

Correctness schema v2:

- Tesla P100-SXM2-16GB;
- CuPy 13.6.0: 35 estimator integrations + 12 public primitives = **47/47**;
- Torch 2.0.0: 35 estimator integrations + 12 public primitives = **47/47**;
- all nine rank-deficient estimator acceptance cases record `fit_rank < parameter_count`, `coefficient_inference_applicable=false`, and an explicit rank-deficiency reason;
- `panel_entity_hc0` and `random_effects_explicit_constant_hc0` persist the requested prediction backend;
- SHA-256 `2d929bccf1c7a0ade385c495bd6a3144cd607dec413c80857e455263c9f1f017`;
- Git blob `ca40d98e48e7747c080f7bf1868cf355ada048a5`.

Performance schema v3:

- **60/60** synchronized end-to-end fit rows;
- 54 base rows + four `N=10,000, k=2, T=200` QS rows + two `N=10,000, k=2, T=20` unbalanced two-way-FE rows;
- exactly three finite positive samples per row and an exactly persisted median;
- each row persists the requested backend; the exact immutable runner fails closed before returning elapsed time unless fitted `model._backend_name` equals that backend;
- SHA-256 `2238002d491fe9397890af1d5e87162458f0a98b293ecf41f6b8831e5a9152b6`;
- Git blob `e1b61e05ea93425947d1e6a7b35d38227d22c358`.

No row-local `executed_backend` field is synthesized for schema v3.

## Canonical v4 promotion

Canonical promotion commit:

`72bc21d3d0a1afd23467ecb1ff176d42df709cb4`

New immutable v4 source IDs:

- `panel-stage-c-identifiability-validation-pr126-20260812-2d929bccf1c7`;
- `panel-stage-c-identifiability-performance-pr126-20260812-2238002d491f`.

Historical v1/v2/v3 Stage-C parser/source identities remain frozen.

Promotion gates passed:

- dedicated v4 parser corruption tests;
- maintained parser/catalog/inventory/domain matrix: **81/81**, before and after generation;
- canonical manifest: **19/19** registered/available/parsed required sources;
- canonical bundle: **2426 runs / 47 models**;
- deterministic strict-source generator: 0 validation errors, 0 errors, 0 warnings;
- frontend typecheck/build;
- exact measurement-source applicability.

## Post-promotion review/fix

Independent review found and fixed two relevant artifact-level findings:

1. `[MEDIUM][PARSER][fixed]` — the initial v4 performance parser allowed any positive repeat count and tolerance-based median equality. It now requires `repeats == 3`, exactly three raw samples, and exact equality with `statistics.median(samples)`. Corruption tests cover a two-sample row and a one-ULP median drift.
2. `[MEDIUM][ARTIFACT][fixed]` — the prior review header still said canonical promotion was pending after promotion had completed. The durable review record now reflects canonical promotion completion.

Post-fix gates passed:

- maintained canonical matrix: **81/81**;
- strict deterministic generator check: **2426 runs / 47 models / 19 sources**, 0 validation errors, 0 errors, 0 warnings;
- compile/static checks;
- measurement-source applicability relative to `a99726e1...`;
- immutable raw-artifact identity relative to `ccc46da6...`.

Final read-only review of technical candidate `a4f8c6e8...` found no unresolved CRITICAL, HIGH, or relevant MEDIUM finding. No new inline review thread appeared. The sole unresolved inline P2 is the older **outdated** backend-provenance thread; its underlying issue is closed by persisted fit-time backend provenance plus the current physical evidence contract and it remains only as review history.

## Final hosted gate

All seven permanent workflows must complete successfully on the exact SHA containing this review-record-only checkpoint:

1. Tests;
2. Panel Stage C Torch CPU;
3. Panel Stage C external covariance, including R `plm` / `sandwich`;
4. Maintenance compatibility;
5. Release notes validation;
6. Release package validation;
7. Benchmark Frontend CI, including deterministic data checks, E2E, and production QA.

If all seven complete successfully and no new review finding appears, the technical state becomes `PHYSICAL_GPU_ACCEPTED / COMPLETE / MERGE-READY`.

PR #126 remains Draft, open, and unmerged. This review record does not authorize Ready-for-review or merge; either action requires explicit user instruction.
