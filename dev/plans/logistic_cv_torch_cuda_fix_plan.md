# LogisticRegressionCV Torch strict-CUDA repair plan

Issue: #112  
Baseline: `master` at `37f643a68ded71a33f8eea5ed8217aab42c650e1`  
Scope: correctness repair before the next roadmap feature package (#93)

## 1. Why this is the next stage

The benchmark/dashboard sequence through PR #115 is complete. The roadmap normally moves next to Panel P1 (#93), but its own priority rules put correctness and public-contract risk ahead of feature breadth. Canonical CV evidence currently records one maintained backend defect: `LogisticRegressionCV` fails on Torch strict CUDA while the corresponding NumPy, CuPy, and sklearn rows succeed. Therefore #112 is treated as a bounded correctness blocker before starting #93.

This change does not reopen dashboard performance work and does not add a new statistical feature.

## 2. Impact classification

Active gates:

- backend/device/dtype locality;
- cross-validation;
- numerical correctness/convergence;
- tests and compatibility;
- user-facing documentation/changelog;
- physical-GPU validation.

Inactive gates:

- public constructor/API shape: no public arguments or return types should change;
- formula semantics: no formula/model-matrix path changes are planned;
- inference definition: final-refit inference contract is unchanged;
- benchmark/performance claims: no speedup claim is planned.

Capability decision:

- backend: `three-backend` (existing capability, Torch path repaired);
- CV: `supported`;
- inference: `supported` for the final refit exactly as before;
- formula: unchanged;
- benchmark: correctness evidence required, performance claim not required.

## 3. Root cause established from current source

`_select_logistic_c_cv()` defaults to `gpu_cv_mixed_precision=True`, so the GPU CV working design is `float32`.

In `_solve_logistic_path_gpu_from_batch()` the current Torch path then creates:

- `params = backend.zeros(...)` without a dtype, and
- `reg_diag = backend.full(...)` without a dtype.

`TorchBackend.zeros/full` default to `torch.float64`. The first matrix multiply therefore combines a `float32` design matrix with `float64` parameters. Torch requires matching matrix-multiplication dtypes, so strict Torch CUDA fails before completing the first candidate. CuPy is more permissive about mixed-dtype arithmetic, which explains the backend-specific disposition seen in the canonical benchmark.

The same helper also converts every fitted coefficient vector/intercept to NumPy inside the candidate/fold loop and then converts them back to the GPU backend for scoring. This does not cause the reported failure, but it is an unnecessary device round trip in the exact code being repaired and increases the risk of dtype drift.

## 4. Implementation plan

### A. Make the batched IRLS working dtype explicit

In `statgpu/linear_model/cv/_logistic_cv.py`:

- create `params` with `dtype=X_design.dtype`;
- create regularization diagonals with `dtype=XtWX.dtype`;
- keep coefficient/intercept batches in backend-native arrays through scoring;
- only transfer the per-fold loss vector to NumPy when populating the public `loss_path` result;
- preserve the current objective, intercept non-penalization rule, folds, C grid, sample-weight semantics, convergence tolerance, and strict no-CPU-fallback behavior.

No statistical definition changes are allowed in this repair.

### B. Add deterministic regression coverage that does not require hosted CUDA

Add a focused test using `TorchBackend(device="cpu")` to exercise the internal mixed-precision batched helper with `float32` tensors. The test should prove:

- the helper no longer raises a Float/Double dtype mismatch;
- returned coefficient/intercept batches remain Torch tensors on the selected backend and keep the working dtype;
- candidate losses are finite and have the expected shape;
- no internal `backend.to_numpy()` call is required by the path solver.

Add/extend public CV tests to preserve:

- CPU/CuPy/Torch selection semantics where the backend is available;
- explicit-device no-fallback behavior;
- sample-weight behavior;
- deterministic candidate selection and transactional final refit.

### C. Check compatibility and numerical parity

Run the strongest locally available evidence:

- targeted LogisticRegressionCV tests;
- Torch CPU internal mixed-precision regression;
- maintenance/static tests covering CV backend routing;
- full hosted CPU/compatibility workflows on the PR head.

Numerical checks should compare the same C grid/folds and ensure the repaired path does not alter the declared loss or selection semantics beyond normal mixed-precision tolerance.

### D. Documentation and issue state

Update the root/EN/CN changelogs with the repaired strict-Torch CV behavior once implementation is validated.

Do **not** edit the canonical `2026-08-07` benchmark source from `failed` to `success` without a new physical-GPU measurement. Historical measured evidence remains immutable.

Issue #112 should close only after a current-source physical Torch CUDA reproduction succeeds. If physical GPU access is unavailable in this run, finish with `PARTIAL_REMOTE_PENDING` and leave #112 open with the exact rerun command/evidence needed.

## 5. Physical-GPU acceptance

Required for `COMPLETE`:

- Torch CUDA on a real NVIDIA GPU, preferably the original Tesla P100 / Torch 2.0.x compatibility environment or an equally strict supported environment;
- run the canonical CV reproduction for `LogisticRegressionCV` with default `gpu_cv_mixed_precision=True`;
- confirm strict Torch execution succeeds without CPU fallback;
- compare selected C / CV loss against NumPy or CuPy within a documented mixed-precision tolerance;
- record software/hardware/source SHA and the exact command.

A fresh canonical benchmark artifact may be added only from an actual rerun. The old failed artifact remains historical evidence.

## 6. Review/fix protocol

Review the plan before implementation for:

1. whether the proposed fix changes any statistical definition;
2. whether it accidentally weakens strict device semantics;
3. whether it leaves another Torch mixed-dtype boundary in fit or scoring;
4. whether tests can catch the original failure on CPU-only CI;
5. whether the plan improperly rewrites historical benchmark evidence;
6. whether the change should remain bounded to #112 rather than becoming a general CV refactor.

After implementation, run `.claude/skills/code-review.md` in auto-fix mode and repeat targeted validation/re-review until no new CRITICAL/HIGH or in-scope MEDIUM issue is found.

## 7. Non-goals

- no Panel #93 implementation in the same PR;
- no general `_penalized_cv.py` decomposition;
- no solver rewrite;
- no new penalty, formula, or inference behavior;
- no dashboard optimization or schema change;
- no fabricated or reconstructed benchmark success row;
- no performance claim without a new measured artifact.

## 8. Expected exit

- `COMPLETE` only if local/hosted gates and physical Torch CUDA validation all pass;
- otherwise `PARTIAL_REMOTE_PENDING` is acceptable when the implementation/review is clean and only physical-GPU evidence is missing.
