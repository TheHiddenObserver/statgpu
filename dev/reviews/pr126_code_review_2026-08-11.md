# PR #126 strict code-review checkpoint — 2026-08-11

Standard: `.claude/skills/code-review.md` (`auto-fix` mode)

## Hard-exit status

**PARTIAL_REMOTE_PENDING**

The local review/fix loop has no unresolved CRITICAL, HIGH, or relevant MEDIUM finding at this checkpoint. Fresh exact-head physical GPU correctness and synchronized performance evidence remains required because the strict re-audit changed `statgpu/panel/_fixed_effects.py` after the prior P100 measurement.

Validation target/tier: `remote-full`.

## Impact classification

Active axes:

- public API / presentation contract;
- inference;
- backend / three-backend behavior;
- formula compatibility;
- benchmark/performance;
- docs/artifacts.

Inactive axes:

- loss;
- penalty;
- solver;
- CV.

Reason: Stage-C covariance choices do not change an optimization objective, regularization path, solver, or tunable CV layer.

## Capability decisions

| Model | backend | CV | inference | formula | benchmark |
| --- | --- | --- | --- | --- | --- |
| `PanelOLS` | three-backend | non-tunable | supported | supported | required |
| `RandomEffects` | three-backend | non-tunable | supported | supported | required |
| `PooledOLS` | three-backend | non-tunable | supported | supported | required |
| `BetweenOLS` | three-backend | non-tunable | supported | supported | not-performance-sensitive |
| `FirstDifferenceOLS` | three-backend | non-tunable | supported | supported | not-performance-sensitive |

`FamaMacBeth` retains its estimator-specific covariance path and is frozen by Stage C; its pre-existing formula-summary naming limitation remains an out-of-scope observation rather than a Stage-C change.

## Findings closed in this re-audit

[HIGH][FORMULA][fixed] `statgpu/panel/_fixed_effects.py` — `PanelOLS.summary()` overwrote formula/Patsy coefficient names with generic `x1`, `x2`, ... labels.

Impact: categorical levels, interactions, and transforms could be associated with the wrong displayed term names even though numerical inference was correct.

Fix: use stored `_feature_names` for formula fits and retain historical generic names only for array fits. Added a categorical + interaction + transform regression that requires `summary().feature_names`, `_feature_names`, and `_inference_result.feature_names` to equal Patsy term order.

Evidence: focused formula/inference suite passed 56/56 after the fix.

[MEDIUM][PERF][fixed] `dev/plans/panel_p1_stage_c_covariance_plan.md` — the prior phrase “timing/memory evidence” implied a peak-memory metric that the maintained physical runner did not collect.

Impact: the reviewed completion contract and actual benchmark evidence were ambiguous.

Fix: define memory as an algorithmic/no-host-transfer gate: no `n x n` hat matrices, no full numerical host copies, grouped-score working sets bounded to `O(Gk)` / `O(Tk)`, and target-scale physical runs must complete without OOM. Synchronized timing remains the explicit recorded metric; no speedup claim is made.

[MEDIUM][MATRIX][fixed] `dev/plans/panel_p1_stage_c_covariance_plan.md` — per-model benchmark capability decisions were missing.

Fix: Pooled/Panel/RE are `required`; Between/FirstDifference are `not-performance-sensitive` for Stage C because they only reuse the already exercised HC primitive and add no grouped/lag covariance kernel.

[MEDIUM][ARTIFACT][fixed] `dev/reviews/pr126_physical_gpu_validation.md` and PR body — prior `5ed763be...` exact-source acceptance was still presented as current after the new production-source fix.

Fix: current state is `PARTIAL_REMOTE_PENDING`; the old P100 evidence is immutable historical evidence only.

[MEDIUM][DOC][fixed] `docs/en/models/panel.md`, `docs/cn/models/panel.md`, `docs/en/changelog.md`, `docs/cn/changelog.md`, `CHANGELOG.md` — user-facing pages still presented the superseded P100 run as current acceptance.

Fix: all surfaces now distinguish the historical successful `5ed763be...` run from the fresh exact-head physical validation still required for the current branch.

## Re-review result

Latest source/matrix re-review found no new CRITICAL, HIGH, or relevant MEDIUM issue.

- Formula: intercept behavior, categorical/reference encoding, interactions/transforms, missing-row alignment, term names/order, and explicit unsupported no-intercept behavior are covered by the maintained formula tests and Patsy comparisons.
- Inference: Stage-C covariance definitions, public BSE/statistic/p-value/CI outputs, unified inference result storage, formula feature names, external statsmodels/linearmodels/R definitions, and Torch/NumPy parity remain covered.
- Backend: numerical covariance paths remain NumPy/CuPy/Torch native with no silent fallback; the re-audit production change did not alter a numerical backend kernel. The previous physical execution evidence is historical because exact-source policy still requires a rerun.
- Performance: no speedup claim is made; synchronized target-scale timing remains the recorded physical metric and memory acceptance is the documented algorithmic/no-OOM contract.
- Docs/artifacts: current acceptance surfaces consistently state `PARTIAL_REMOTE_PENDING`.

One unresolved inline P2 thread about backend provenance remains **outdated**; its underlying implementation issue is fixed by persisting and physically validating the backend selected at the numerical fit boundary. It is intentionally not resolved here to preserve review history.

## Local evidence

Focused review-fix validation:

```text
56 passed
```

Files exercised directly:

- `dev/tests/test_panel_stage_c_api_formula.py`
- `dev/tests/test_panel_formula.py`
- `dev/tests/test_panel_stage_c_inference_guard.py`
- `git diff --check`

This checkpoint is intentionally a review/status-only commit. Permanent hosted workflows must complete green on this checkpoint before local/full hosted validation is considered closed.

## Remaining remote gate

After hosted CI is green, rerun on a physical CUDA machine using the exact clean final checkpoint:

```bash
python dev/benchmarks/validate_panel_stage_c_gpu.py \
  --expected-sha <FINAL_SHA> \
  --backends cupy,torch \
  --out ~/statgpu_pr126_physical/panel_stage_c_gpu_validation_<short>.json

python dev/benchmarks/benchmark_panel_stage_c_covariance.py \
  --expected-sha <FINAL_SHA> \
  --backends cupy,torch \
  --out ~/statgpu_pr126_physical/panel_stage_c_performance_<short>.json
```

Do not restore `PHYSICAL_GPU_ACCEPTED / COMPLETE / MERGE-READY` until replacement evidence is audited, canonically promoted, final hosted gates are green, and a post-promotion strict re-review finds no new issue.
