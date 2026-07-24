# PR #79 Third Review/Fix Cycle

Date: 2026-07-14  
Branch: `agent/code-review-fixes`  
Base: `master`

## Scope

This cycle deliberately targeted paths not exercised by the previous review: Torch API
differences, formula/array boundaries, string metadata, rank-deficient linear algebra,
non-finite input behavior, repeated device conversions, and GPU-sensitive allocation.

## New findings and fixes

- **[HIGH][BACKEND] Shared Cholesky solve**: Torch `solve_triangular` requires a
  two-dimensional right-hand side. `xp_cholesky_solve` now promotes vector RHS values
  and restores the original shape, fixing PanelOLS, RandomEffects, and penalized spline
  callers.
- **[HIGH][BACKEND] Panel scalar operations**: Panel utilities and inference used
  `torch.maximum(tensor, scalar)`. They now use the shared `xp_maximum` helper.
- **[HIGH][BACKEND/API] Panel labels and formula boundary**: string entity/time labels
  could not be converted to Torch, RandomEffects did not align explicit labels after
  Patsy row deletion, and the shared array-mode formula helper converted complete X/y
  arrays to NumPy. Labels are now CPU-factorized metadata with device int64 codes;
  array inputs preserve their backend.
- **[HIGH][PERF] FirstDifferenceOLS**: the transform copied complete X and y to NumPy,
  looped by entity, then copied differences back. Only a CPU sort index is now created;
  sorting and differencing execute on the numerical backend. BetweenOLS group collapse
  was also changed from O(number of groups) masked means to O(number of columns) scatter
  reductions.
- **[HIGH][BACKEND] KernelPCA and RidgeCV**: Torch does not support negative-step slicing
  and requires tensor operands for `maximum`. KernelPCA now uses `torch.flip`; RidgeCV
  uses `xp_maximum` for rank-deficient Gram eigenvalues.
- **[HIGH][BACKEND] Thin-plate splines**: Torch lacked the used `power` module function,
  scalar maximum failed, and polynomial allocation ignored the input device. The basis
  now uses backend-neutral exponentiation and device-aware helpers.
- **[MEDIUM][API] Finite-input contracts**: shared checks now reject NaN/Inf before
  low-level operations in panel, covariance/shrinkage, unsupervised estimators,
  KernelPCA, Nystroem, and thin-plate splines.
- **[MEDIUM][BACKEND] Natural spline fallback**: QR fallback identity allocation now
  follows the constraint-matrix device.

## Validation

- `dev/tests/test_third_full_review.py`: 21 focused regressions.
- Panel/formula/covariance plus new tests: 90 passed locally.
- Kernel-method, smoothing/spline/GAM, unsupervised, RidgeCV, and third-review focused
  suites passed in isolated local runs; optional CUDA tests remain hardware-gated.
- The checksum-verified application workflow passed package compilation and the focused
  Torch-CPU/cross-module regression set before committing the source changes.
- Permanent GitHub Actions run **#367** passed the Python 3.9, 3.10, 3.11, and 3.12
  regression matrices, the complete Python 3.11 CPU test tree, maintained-script/package
  compilation, expanded Ruff checks, Cox structural assertions, and complete collection.
- Temporary snapshot/application workflows and patch-transfer files were removed before
  the final permanent run.

## `dev/AGENTS.md` compliance

- No complete numerical design is newly transferred to CPU; FirstDifference and panel
  array entry points remove existing transfers.
- CPU metadata boundaries are explicit and limited to labels/sort indices.
- Torch behavior is tested without silently reclassifying explicit GPU modes as CPU.
- Public behavior changes are synchronized in EN/CN model pages and all changelogs.
- Physical CuPy/Torch CUDA numerical, memory, synchronization, runtime, and cleanup
  evidence remains remote-pending.

## Status

`PARTIAL_REMOTE_PENDING`: no unresolved local CRITICAL/HIGH finding from this cycle
remains after focused retesting; physical GPU validation is still required.
