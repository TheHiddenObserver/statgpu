# PR #126 rank-df v3 physical promotion review — 2026-08-12

Standard: `.claude/skills/code-review.md` (`auto-fix` / post-promotion review)

## Current verdict

**PHYSICAL_GPU_ACCEPTED / PROMOTION REVIEW CLEAN / HOSTED CHECKPOINT PENDING / NOT MERGE-READY**

This file is the connector-authored review-record-only checkpoint. The promoted technical tree immediately before this checkpoint is `6eb59dbcefe93e0672cc884542fad37fd34afdd4`. This checkpoint must pass all seven permanent hosted workflows before the PR can exit as `COMPLETE / MERGE-READY`.

## Fresh physical evidence accepted

Measurement source: exact-clean `f154647665788df2570439a1cc154a43f509aa45` on Tesla P100-SXM2-16GB.

Raw evidence commit: `2cda842d24f38a1ba95b949215b5779556168a2a`.

Correctness artifact:

- path: `results/pr126_p100/panel_stage_c_gpu_validation_f1546476.json`;
- SHA-256: `0b4eb5810ad0b49adef9aa3089c7058683d0134efbbf6502aa3d57f3ff6c0766`;
- Git blob: `47a0ee156b57f783fcd420565c431c5439589b9f`;
- CuPy 13.6.0: 35 estimator integrations + 12 public primitives = **47/47**;
- Torch 2.0.0: 35 estimator integrations + 12 public primitives = **47/47**;
- requested backend equals persisted executed backend for every estimator and primitive;
- all eight new exact-collinearity nonrobust/HC1 estimator cases are present on both backends and record `0 < fit_rank < parameter_count`;
- numerical rank-boundary coverage remains present;
- no numerical CPU fallback was accepted.

Performance artifact:

- path: `results/pr126_p100/panel_stage_c_performance_f1546476.json`;
- SHA-256: `09337cc62c942cff040685ab3a667a743fea01047d9843849d567f53aa6d2b5e`;
- Git blob: `c5b8757fd6d43a8043caf55cadc8f9c481fbf8bb`;
- synchronized end-to-end estimator timing: **58/58 rows**;
- each row contains three finite positive raw samples and the stored median matches the raw-sample median;
- high-T quadratic-spectral cases are retained;
- CuPy provenance is recorded as 13.6.0 via CUDA-suffixed distribution discovery;
- no CPU-speedup claim is encoded.

## Immutable v3 canonical promotion

New environment: `remote-p100-pr126-20260812`.

New required immutable sources:

- `panel-stage-c-rank-df-validation-pr126-20260812-0b4eb5810ad0`;
- `panel-stage-c-rank-df-performance-pr126-20260812-09337cc62c94`.

The new parser is `dev/benchmarks/frontend_data/parsers/panel_stage_c_rank_df.py`, parser version 3.0. It fails closed on:

- measurement SHA drift from `f1546476...`;
- dirty measurement tree;
- P100 / CuPy / Torch provenance drift;
- estimator or primitive case-set drift;
- requested/executed backend mismatch;
- non-success physical cases;
- non-finite recorded NumPy differences;
- rank-boundary metadata drift;
- any of the eight new exact-collinearity cases failing `0 < fit_rank < parameter_count`;
- performance row-count, timing-scope, backend, raw-sample, or stored-median drift.

Historical Stage-C v1/v2 parsers and source registrations remain immutable. The promotion gate explicitly required zero diff in:

- `dev/benchmarks/frontend_data/parsers/panel_stage_c.py`;
- `dev/benchmarks/frontend_data/parsers/panel_stage_c_rank_policy.py`.

The canonical catalog now contains 17 required sources and deterministically generates 2272 runs. Promotion validation completed with 77/77 focused parser/catalog tests, strict-source generation, zero validation errors, zero errors/warnings, strict-source re-check, and `git diff --check`.

## Post-promotion review/fix finding

### [MEDIUM][PROVENANCE][fixed] v3 parser inherited stale v2 textual labels

Independent post-promotion review found that the newly generated v3 parser correctly enforced measurement SHA `f154...` and parser version 3.0, but its module docstring still described the fresh source as `3dc7df19` and its internal emitted parser labels retained `_v2` suffixes.

This did not change which raw artifacts were parsed or accepted, but it made canonical run provenance text internally inconsistent. The fix changes the stale text to `f1546476` and the emitted labels to `_v3`, regenerates canonical frontend assets, and reruns the focused parser/catalog/strict-source gates. Historical v1/v2 parsers remain untouched.

## Net promotion tree audit

The net diff from raw-evidence commit `2cda842d...` to promoted technical tree `6eb59dbc...` contains only:

- the new v3 parser plus parser registry/export wiring;
- new immutable source/environment/comparison registrations;
- benchmark coverage-matrix registration;
- maintained parser/catalog/source-count tests;
- regenerated canonical frontend benchmark data.

No `statgpu/` production numerical source, physical runner, or historical v1/v2 parser changes occur in that promotion diff. Temporary promotion/fix helpers and workflows self-delete and are absent from the promoted tree.

## Independent re-review result

After the provenance-label fix, the post-promotion review found no unresolved CRITICAL, HIGH, or relevant MEDIUM issue in the fresh physical evidence, v3 parser, immutable-source registration, benchmark coverage, or generated assets.

The only unresolved inline PR thread remains the older **outdated** P2 backend-provenance thread on the historical runner diff. Its underlying no-fallback provenance issue is already fixed by persisted fit backend identity and is directly exercised by the accepted physical evidence; the thread is retained as review history.

## Remaining hard gate

1. This exact review-record-only checkpoint must pass all seven permanent hosted workflows.
2. A final read-only strict review must find no new CRITICAL, HIGH, or relevant MEDIUM issue.
3. If both hold, the technical status may become **PHYSICAL_GPU_ACCEPTED / COMPLETE / MERGE-READY**.

PR #126 remains Draft, open, and unmerged. Ready-for-review and merge are explicit user lifecycle actions only.
