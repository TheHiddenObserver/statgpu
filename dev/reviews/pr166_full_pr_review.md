# PR #166 full source review — Quantile feature branch

## Target

- target_kind: range (two-dot), full PR vs default branch
- base_sha: `fe10c6e6915acffa6565ab500260655f5a265ad5` (merge-base with `origin/master`)
- reviewed head at first pass: `9a5dbf560dd0ddcc8d1463f729764ddb6d460a52`
- fix range: `08e8fdc0..33d6cbc1`; physical evidence commit `0f8ed24bf1831a1ddcb31cb274fd5ed3d14fefda` at source `33d6cbc1eb0f4b5b572a565cdae58281b3bf3403`
- worktree: clean, dedicated checkout; mode: audit

## Scope

Three independent segments covered the 40-file, ~6.9k-line production diff (excluding
already-reviewed late-delta and Issue #169 surfaces):

- Segment A: solver/engine core (`_quantile_continuation`, `_quantile_solver_guard`,
  `_quantile_proximal_public_contract`, `_quantile_group_proximal_irls_lla`,
  `_proximal_irls_quantile`, `_fista_lla` legacy portions, small solver changes).
- Segment B: losses/penalties/backends (`losses/_quantile*`, `penalties/_adaptive_l1`,
  group penalties + layout, `backends/_array_ops`, `glm_core` validation).
- Segment C: linear_model layer (`wrappers/_quantile`, penalized quantile contracts,
  `_penalized_quantile`, `_predict_mixin`, `_base`, quantile `_fit_mixin`/`_penalized_cv`
  sections, solver API contract).

## Findings and disposition

All actionable findings were fixed in `08e8fdc0..33d6cbc1`:

- MEDIUM: complex `AdaptiveL1Penalty` controls/weights and adaptive-group weights were
  silently truncated instead of rejected (`08e8fdc0`).
- MEDIUM: the PR-added torch/numpy bandwidth-parity test exposed a pre-existing Torch
  kernel-inference precision defect (sandwich weights built with `torch.where` defaulted
  to float32). The production root cause was fixed in `wrappers/_quantile.py` and the
  test tolerance was restored to its original bound.
- LOW: zero-feature `QuantileLoss.irls()` designs were accepted (`08e8fdc0`).
- LOW: `fista_lla_path` generic inner loop could raise `UnboundLocalError` with a
  zero internal iteration budget (`74dc5759`).
- LOW: single-step marked continuation paths never solved their target alpha
  (`74dc5759`).
- LOW: adaptive-group continuation-start branch was unreachable and its scale set
  omitted `adaptive_group_lasso` (`74dc5759`).
- LOW: re-emitted Quantile convergence warnings pointed inside statgpu instead of at
  the caller (`losses` kernel, `fista_lla` fallback, typed estimator) (`74dc5759`).
- LOW: dead branch in the scalar-CV penalty resolver; dead locals in
  `AdaptiveL1Penalty.proximal`; duplicated `_external_warning_stacklevel` helpers;
  inaccurate guard docstring; float32-unsafe intermediate weights in the unweighted
  adaptive-group path (`74dc5759`).
- LOW: complex object-dtype adaptive-group weights still truncated after the first
  pass; the object-array guard was completed with the production dtype fix.

No CRITICAL/HIGH finding remains. Covered by new tests: complex-control rejection,
zero-feature rejection, single-step marked path, zero-budget fused fallback.

## Pre-existing environment observations (not caused by this range)

- `test_pr80_exact_source_runtime_provenance.py::test_runtime_probe_*` fails
  identically at the pre-Issue-#169 base in this environment.
- `test_kernel_torch_cpu_uses_numpy_population_std_bandwidth` needed the tolerance
  fix above; it passes after `d58b0d32`.

## Validation

- Targeted suites after fixes: 139 passed (5 files); broad collateral sweep over
  113 affected files: 1687 passed / 327 skipped / only the pre-existing provenance
  failures.
- Physical re-sign at `33d6cbc1`: `GATE1_EXIT=0`, `GATE2_EXIT=0`; smooth schema v23,
  group v3, nested scalar v3, boundary v1; scalar LP fixed-point gap `4.586e-11`.
