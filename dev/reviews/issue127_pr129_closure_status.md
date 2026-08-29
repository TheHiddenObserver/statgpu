# Issue #127 / PR #129 closure status

Recorded: 2026-08-30

This checkpoint records the current `.claude/skills/code-review.md` closure state after repeated hosted and physical review/fix cycles for PR #129.

## Superseded physical candidates

The following candidates are historical/diagnostic evidence only and must not be promoted as final acceptance evidence:

1. `3180add336b41017f4cd5a5e6721a6470a797360` — physical execution exposed unsupported CuPy triangular-solve usage and incomplete concrete-device provenance.
2. `11343d87e810cb147bd0b9f821ac5f5e52976ea2` — physical execution exposed silent invalid CuPy Cholesky results on singular normal equations.
3. `7375b36117988e182fb74443a937e6b302eabb99` — physical execution exposed the remaining bare shared CuPy inverse site that could return invalid values without triggering pseudoinverse recovery.
4. `0c05d17cb7f7520fb703566b1e662d18bf2d4ee0` — both canonical Tesla P100 validators passed on that exact source, but a later fresh code review found additional backend-locality/cleanup issues and the physical host-transfer acceptance logic subsequently changed. Its artifacts therefore no longer prove the current acceptance contract.

Later source changes addressed concrete-device CuPy response/weight alignment, native fitted-parameter lifetime, failed-refit invalidation, success/failure allocator cleanup including exact/precomputed inference, and the physical host-transfer fixture. Any earlier artifact predating those changes remains superseded.

## Current review/fix closure

Reviewed implementation head: `0adcd63afef6e38ee757a6635b30839d2edfbab2`.

Closed findings include:

- supported `cupyx.scipy.linalg.solve_triangular` usage for CuPy `LinearRegression`;
- scoped `cupyx.errstate(linalg="raise")` at maintained CuPy Cholesky/solve/inverse boundaries, preserving classified pseudoinverse recovery while unrelated failures remain fatal;
- shared `_cupy_asarray_on_device(...)` relocation when native CuPy values live on a different concrete GPU;
- authoritative X-device alignment for `LinearRegression` response and sample weights;
- fail-closed executed-backend and concrete-device provenance;
- backend-native fitted parameters retained through ordinary GPU squared-error L2 inference, with reporting materialized only afterward;
- final GPU cleanup after post-fit inference and cleanup on backend-fit/exact-precomputed inference failure;
- native fitted references cleared and `_fitted=False` on failed fit/inference transactions;
- a physical host-transfer guard that verifies native fitted state at the inference boundary without rejecting unrelated solver warm-start host work;
- hosted behavioral/static coverage for each repaired boundary.

On `0adcd63afef6e38ee757a6635b30839d2edfbab2`, all seven triggered hosted PR workflows completed successfully:

1. `Tests` — complete CPU test tree, Python 3.9-3.12 regression matrix, static/docs contracts, Torch CPU regressions, and external panel alignment;
2. `Gaussian inference backend-native` — Python 3.9/3.12 CPU, Torch 2.0.1 CPU, and static-contract jobs;
3. `Release package validation`;
4. `Release notes validation`;
5. `Maintenance compatibility`;
6. `Benchmark Frontend CI`;
7. `Panel Stage C Torch CPU`.

A fresh Codex review explicitly reviewed commit `0adcd63afe` and reported: `Didn't find any major issues.` All inline review threads are resolved. The complete-diff review state is therefore:

- CRITICAL: **0 open**;
- HIGH: **0 open**;
- relevant actionable MEDIUM: **0 open**.

## Finalization and hard-exit state

The commit containing this checkpoint is documentation/evidence-only. It must not modify production numerical code, physical-validator cases, thresholds, backend/device checks, provenance checks, or pass/fail semantics. Because canonical physical evidence is exact-SHA evidence, the resulting final source SHA still requires both a fresh hosted run and new exact clean-head physical CUDA execution even though the reviewed numerical implementation is unchanged.

After the finalization head is hosted-green and no new review finding exists, the only unresolved acceptance gate is exact clean-head physical CUDA execution on that same final `HEAD`. At that point the correct workflow state is **`PARTIAL_REMOTE_PENDING`**, not `COMPLETE`.

## Canonical physical acceptance

Run both validators from the same exact clean final checkout. `results/` is git-ignored, so the first artifact does not invalidate the second validator's clean-tree precondition.

```bash
SHA="$(git rev-parse HEAD)"

python dev/benchmarks/validate_gaussian_inference_backend_native_gpu.py \
  --out "results/pr129_gaussian_inference_${SHA:0:8}.json" \
  --expected-sha "$SHA" \
  --validation-tier remote-full \
  --backends cupy,torch

python dev/benchmarks/validate_gaussian_inference_cupy_rank_recovery_gpu.py \
  --out "results/pr129_gaussian_inference_cupy_rank_${SHA:0:8}.json" \
  --expected-sha "$SHA" \
  --validation-tier remote-full
```

Acceptance requires both commands to succeed on the same exact final candidate SHA. The full artifact must report `status: success`, `validation_tier: remote-full`, both required CUDA backends, concrete CUDA-device provenance, clean-tree acceptance, and successful native-fit/reporting-boundary checks. The focused artifact must report `status: success`, the same exact SHA, `validation_tier: remote-full`, concrete CuPy/CUDA provenance, and success for all maintained rank-recovery cases.

PR #129 remains open/unmerged and Issue #127 remains open until both exact-head physical validators pass and the resulting evidence is reviewed against this frozen acceptance contract. No GPU speedup claim is made.
