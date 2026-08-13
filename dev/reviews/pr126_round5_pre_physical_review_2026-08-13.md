# PR #126 — Round 5 pre-physical review checkpoint

Date: 2026-08-13

## Scope

This is a deliberately **pre-physical / pre-terminal-hosted** checkpoint under
`.claude/skills/code-review.md`.  It records the reviewed technical candidate
without trying to mutate a tracked file after the hosted result is known.
Terminal exact-head hosted status is recorded in GitHub review/PR metadata so
that recording the result does not itself move the commit under review.

## Technical candidate

The reviewed production + physical-runner candidate immediately before this
review-record commit is:

`4c38748ee2fe40a7d3b82d40698765dae84e7970`

The Round-5 delta closes the independent review findings that appeared after the
previous `ccdd094d...` checkpoint:

- prediction shape handling now restores an omitted fitted constant only on a
  compatible path, preserving its fitted value and column position and rejecting
  ambiguous short array input;
- formula-generated prediction matrices use Patsy column identity, so a slope
  that is constant on a prediction batch is not mistaken for an omitted formula
  intercept;
- two-way fixed-effect projection factorizes entity/time metadata once, uses
  residual group-mean convergence checks with a scale-aware roundoff floor, and
  exposes fail-closed `demean_max_iter` / `demean_tol` controls;
- unbalanced two-way fitted effects are recovered jointly;
- disconnected two-way prediction rejects cross-component, one-sided-known, and
  known-plus-unknown fixed-effect combinations whose values depend on arbitrary
  component normalization, while preserving the historical linear-only fallback
  when both labels are unseen;
- the permanent R external gate installs exact `plm==2.6-7` and
  `sandwich==3.1-3` versions before asserting them;
- the Stage-C physical correctness runner now audits the newly changed public
  prediction paths instead of only old covariance paths: one-way FE prediction,
  two-way FE prediction, omitted-explicit-constant RandomEffects prediction, and
  a dedicated disconnected-two-way prediction contract with backend provenance
  and fail-closed guard checks.

## Review findings closed in this round

1. **HIGH / MATRIX+TEST — physical GPU acceptance did not cover the prediction
   behavior changed by this review cycle.**  Fixed in `4c38748e...`: the existing
   35-estimator + 12-public-primitive covariance matrix is retained, while
   prediction contracts are explicitly labeled in case payloads and the
   disconnected-two-way audit is recorded separately under
   `prediction_contracts`.  A hosted NumPy regression imports and executes the
   physical runner audit before P100 evidence can be accepted.
2. Earlier prediction correctness, two-way convergence, disconnected
   identifiability, backend-provenance, external-version, and generated-artifact
   findings are regression-covered and have no active review thread.

## Validation before this checkpoint

- Focused final-review regression after convergence/API fixes: 14 passed,
  1 skipped.
- Guarded prediction-identifiability follow-up regression passed before its
  production fix was committed.
- Guarded physical-prediction follow-up regression passed and its scope check
  proved that the final fix changed only
  `dev/benchmarks/validate_panel_stage_c_gpu.py` and
  `dev/tests/test_panel_stage_c_final_review_fixes.py`; temporary workflow/helper
  machinery was deleted in the same commit.
- On `a7789f71...`, the immediately preceding production candidate, all seven
  permanent hosted workflows succeeded: Tests, Panel Stage C Torch CPU, Panel
  Stage C external covariance (Python + R), Maintenance compatibility, Release
  notes validation, Release package validation, and Benchmark Frontend CI.

## Evidence boundary and exit status

The accepted `a99726e1...` Tesla P100 artifacts are immutable historical evidence
only.  Production numerical/prediction behavior and the physical correctness
runner changed afterward, so those artifacts cannot establish current physical
acceptance.

After this review-record commit, the required next gate is normal permanent
hosted CI on the exact review-record head.  If that is clean and a fresh
changed+adjacent read-only review finds no new CRITICAL/HIGH/relevant MEDIUM,
the technical exit is:

`HOSTED_GREEN / REVIEW_CLEAN / PARTIAL_REMOTE_PENDING`

The remaining remote gate is then a fresh exact-clean Tesla P100 run of:

- `dev/benchmarks/validate_panel_stage_c_gpu.py` for CuPy + Torch correctness,
  including the new prediction-contract audit; and
- `dev/benchmarks/benchmark_panel_stage_c_covariance.py` for synchronized
  performance.

Fresh artifacts must be audited/promoted with a new parser/source identity before
physical acceptance can return to COMPLETE.  PR #126 remains Draft; this record
does not authorize Ready-for-review or merge.
