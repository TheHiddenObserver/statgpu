# PR #126 post-promotion strict review checkpoint — 2026-08-11

Standard: `.claude/skills/code-review.md` (`auto-fix` mode)

## Status

**PHYSICAL_GPU_ACCEPTED / HOSTED_FINAL_PENDING**

Fresh physical measurement source:

`ec511f539adeaaedf310f92248200d0868577532`

Raw evidence commit:

`e1a155bf77b416e0873a037015aaafd22371ab11`

Canonical promotion checkpoint before this review-only commit:

`f21d6f1d01c2fc9c66fa475e68ce664122b39807`

## Physical evidence audit

Fresh Tesla P100-SXM2-16GB evidence passed the fail-closed promotion audit:

- CuPy: 32/32 = 26 estimator covariance integrations + 6 direct public covariance primitives;
- Torch: 32/32 = 26 estimator covariance integrations + 6 direct public covariance primitives;
- every estimator/primitive case records requested backend == executed backend;
- no numerical CPU fallback was observed;
- synchronized performance: 58/58 rows = 54 base + 4 high-T QS rows;
- high-T matrix: CuPy/Torch × PooledOLS/PanelOLS QS at N=10,000, k=2, T=200;
- every timing sample and median is finite and positive, and stored medians equal the medians of the raw samples;
- no speedup or CPU-baseline claim is made.

Canonical immutable identifiers:

- correctness source: `panel-stage-c-validation-pr126-20260811-af2227efe3cd`;
- correctness SHA-256: `af2227efe3cd0ab77472ff1d6584233d475f6ed5c4c4d36d318efc127d143f63`;
- correctness Git blob: `a02fcad0eefd5993d2ae05b8d00a55e5ca1d885f`;
- performance source: `panel-stage-c-performance-pr126-20260811-409974070022`;
- performance SHA-256: `4099740700221ffdae2770427a5ad0fca7dc3c1ec0f47173caf166aa56a1fca0`;
- performance Git blob: `76d6eabeefc6e04095270a5df8a231cb150ea220`.

Promotion-local validation passed:

- deterministic benchmark generator: 0 validation errors, 0 warnings;
- frontend production build: success;
- promotion contract tests: 61/61 passed;
- strict generator `--check --strict-sources`: success;
- `git diff --check`: success.

## Exact-source applicability audit

The complete diff from physical measurement `ec511f53...` through canonical promotion `f21d6f1d...` contains only:

- the two immutable raw evidence JSON files;
- Stage-C parser/source manifest and benchmark coverage metadata;
- benchmark/catalog/parser contract tests;
- EN/CN/root documentation and review records;
- deterministic frontend benchmark data/assets.

It contains **no change** to:

- `statgpu/panel/**`;
- `dev/benchmarks/validate_panel_stage_c_gpu.py`;
- `dev/benchmarks/benchmark_panel_stage_c_covariance.py`.

Therefore the `ec511f53...` physical measurement remains applicable to the post-promotion branch under `RELEASING.md`; no additional physical rerun is required for the promotion-only delta.

## Active strict-review axes

- public API / presentation contract;
- inference;
- backend / three-backend behavior;
- formula compatibility;
- benchmark/performance;
- docs/artifacts.

Loss, penalty, solver, and CV remain inactive for Stage C.

## Final gate

Permanent hosted workflows must complete green on this review-only checkpoint, followed by one final `.claude/skills/code-review.md` re-review of the promotion/checkpoint delta. If that review finds no new CRITICAL/HIGH/relevant-MEDIUM issue, the technical status may advance to `PHYSICAL_GPU_ACCEPTED / COMPLETE / MERGE-READY`.

The PR remains Draft and must not be marked Ready or merged without explicit user instruction.
