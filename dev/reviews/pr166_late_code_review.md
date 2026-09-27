# PR #166 late-delta code review — Quantile async/LLA closure

## Target

- target_kind: range (`A..B`, two-dot)
- base_sha: `5dec9fbc658046ace4d6cb159a5f6629923f9275`
- head_sha: `f039610090e074004382e7b133ddd64d222ab71b`
- path_filter: `statgpu/solvers/_fista.py`, `statgpu/solvers/_constants.py`,
  `statgpu/linear_model/penalized/{_fit_mixin,_penalized_cv,_quantile_solver_contract}.py`,
  `dev/benchmarks/{run_quantile_smooth_fista_gpu_gate.py, run_quantile_group_lla_gpu_gate.py,
  validate_quantile_smooth_fista_gpu.py, validate_quantile_scalar_lla_gpu.py}`
- working_tree: clean at `head_sha` in a dedicated worktree; remote tip verified equal
- resolution: explicit immutable range (`git rev-parse`, `git ls-remote`); no symbolic refs
- mode: audit (no file edits during the review pass)

## Change classification

Solver/stopping, CV routing, and physical-evidence repairs on the Quantile async
sparse-CV FISTA route and its gates. No public API shape changes.

## Findings and disposition

### 1. HIGH — physical gate comparisons failed open on non-finite values (fixed)

`validate_quantile_smooth_fista_gpu.py` guarded `np.max(np.abs(scores - reference_scores))`
and the L1 final-refit objective with `> tolerance` checks that are false for NaN. The
strict Quantile CV contract can rewrite a numerically failed alpha column to all-NaN,
so the gate could report success while recording `score_error: NaN`.
`validate_quantile_scalar_lla_gpu.py` had the same NaN-blind pattern for parameters and
objectives. Fix: fail-closed `np.isfinite` guards before every comparison in both
validators (commit `ee28f783`).

### 2. MEDIUM — scalar LLA "converged" oracle is the trivial zero fixed point (recorded, deferred)

The accepted scalar-LLA case is built around an exact zero fixed point, so its CPU/GPU
"converged" parity is essentially zero-versus-zero. A nontrivial LP oracle was
prototyped and uncovered an accuracy floor of the fixed-step inner solve used by
`fista_lla_path` for Quantile. Adding the strict oracle to this PR's gate would fail, so
the finding is recorded instead of silently weakening the claim:

- `dev/tests/test_pr166_quantile_lla_accuracy_floor.py` pins the current floor with an
  upper-bound regression contract and a `xfail(strict=True)` target-accuracy contract.
- Follow-up (separate task/PR): upgrade the non-smooth LLA inner solver (backtracking or
  an equivalent safe step strategy), then promote the LP oracle into
  `validate_quantile_scalar_lla_gpu.py`.

Reproduction (p=1 convex weighted-L1; LP validated against a 5e5-point grid):

| path | beta | objective gap vs HiGHS LP | n_iter |
| --- | --- | --- | --- |
| HiGHS LP oracle | 1.56007771 | 0 | - |
| `fista_solver` (backtracking control) | 1.56007771 | 2.0e-13 | 128 |
| `fista_lla_path` (fixed-step inner) | 1.54418909 | 2.9e-4 | 15000 |

More budget does not move the fixed-step floor (500k iterations: 3.1e-4); a
conservative step multiplier only reduces it (x8: 4.4e-6; p=4 orthogonal design: 3.9e-4).
Root-cause pointers: `statgpu/solvers/_fista_lla.py:655-764` (fixed step, coefficient-delta
stopping) and `statgpu/losses/_quantile.py:107-125` (the returned "lipschitz" is a step
parameter, not a smooth-FISTA guarantee).

### 3. LOW — gate control constants were static copies (fixed)

`ASYNC_MOMENTUM_BETA_CAP`, `ASYNC_STALL_CHECKS`, and `ASYNC_STEP_CONTRACTION_FACTOR` are
now imported from `statgpu/solvers/_constants.py`, with a hosted mirror test so the
recorded payload cannot drift from the implementation.

### 4. LOW — scalar validator had no finiteness guards; one assertion was vacuous (fixed)

`_run` and the periodic-refresh probe now fail closed on non-finite parameters; objective
comparisons are guarded; `cpu_intercept != 0.0` replaces the always-false
`abs(cpu_intercept) > 1e-12` under `fit_intercept=False`.

### 5. LOW — batched L1/L2 tracking width could diverge from `_tracking_penalty_value` (fixed)

`_feature_penalty_width()` now resolves `n_features -> _p -> solver n_features`, mirroring
the synchronized fallback; hosted tests cover all three wrapper shapes.

## Evidence

- Local: 6 targeted test files 157 passed after the fixes; 32 quantile/fista/issue163/pr166
  files 679 passed, 4 skipped, 1 failure also reproduced at the base head (environment, not
  this range).
- Physical re-sign at `ee28f783`: `GATE1_EXIT=0`, `GATE2_EXIT=0`; both artifacts are
  byte-identical to the previous ones except the `source_sha*` fields, confirming the fixes
  are behavior-preserving.
- Evidence commit: `add65ed7` (`test: record PR166 physical GPU acceptance at ee28f783`).

## Residual evidence

- Finding 2's solver upgrade is explicitly outside this PR's scope; no production numerics
  were changed for it.
- No performance measurement was part of this review.

## Review verdict

Findings 1, 3, 4, and 5 are closed on the fixed source and the physical artifacts were
re-signed for that source. Finding 2 is recorded with a bounded regression contract and a
deferred follow-up. No other actionable finding remains for the reviewed range.
