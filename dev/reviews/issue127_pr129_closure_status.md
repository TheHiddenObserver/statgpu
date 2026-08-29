# Issue #127 / PR #129 closure status

Recorded: 2026-08-29

This checkpoint records the fourth `.claude/skills/code-review.md` closure cycle for PR #129 after exact-head physical CUDA validation invalidated three earlier candidates.

## Superseded physical candidates

The following candidates are permanently superseded as acceptance evidence:

1. `3180add336b41017f4cd5a5e6721a6470a797360` — exact-head Tesla P100 `remote-full` execution exposed unsupported `cp.linalg.solve_triangular` usage in CuPy `LinearRegression` and missing concrete `_selected_backend_device` provenance in the physical host-transfer fixture.
2. `11343d87e810cb147bd0b9f821ac5f5e52976ea2` — exact-head Tesla P100 `remote-full` execution verified those first two repairs, then exposed silent invalid CuPy Cholesky results on singular normal equations, bypassing exception-only rank recovery.
3. `7375b36117988e182fb74443a937e6b302eabb99` — exact-head Tesla P100 execution verified the prior solver-status fixes through the first nine CuPy full-matrix cases, then both canonical validators exposed the same remaining missed site: shared `statgpu/linear_model/_gaussian_inference.py::_inverse_or_pinv()` still called bare `cp.linalg.inv`, which could return NaN/Inf for a singular bread matrix without raising and therefore bypass the intended `cp.linalg.pinv` recovery.

Artifacts from all three candidates remain diagnostic evidence only. They must not be reused or promoted as canonical success evidence.

A diagnostic monkeypatch on the same clean `7375b361...` P100 checkout added only the missing scoped CuPy solver-status policy around `_inverse_or_pinv()` and then made both canonical validators complete successfully. That diagnostic result establishes the repair target but is not acceptance evidence because the repository source was not the frozen candidate.

## Fourth review/fix closure

All four physical/review findings are closed in the current implementation:

- `LinearRegression` uses supported `cupyx.scipy.linalg.solve_triangular` on the CuPy path;
- the physical host-transfer case supplies both executed backend and concrete executed-device provenance before invoking post-fit Gaussian inference;
- maintained CuPy Cholesky/solve/inverse boundaries in the Gaussian/linear-model execution graph enable scoped solver-status errors with `cupyx.errstate(linalg="raise")`, making rank/definiteness failures visible to the existing classified recovery paths rather than silently propagating invalid values;
- shared `_inverse_or_pinv()` now applies that same scoped policy specifically to `cp.linalg.inv(matrix)` and preserves the existing `_linalg_exception_is_rank_failure()` classifier plus `cp.linalg.pinv(matrix)` recovery;
- hosted regression coverage contains both a source-level lock and a dynamic fake-CuPy test proving `_inverse_or_pinv()` executes `inv` inside the scoped errstate, exits the context after the rank failure, and then reaches `pinv` recovery;
- unrelated CUDA/programming failures remain fail-closed and are not converted into pseudoinverse recovery;
- no covariance formula, Ridge/L2 normalization, validator case matrix, threshold, backend-routing rule, provenance rule, or pass/fail semantic was changed by the fourth repair.

Reviewed implementation head: `b1d8151f0a9c2415ed81d552ace61ab60828a65a`.

On that head all seven triggered hosted PR workflows passed:

1. `Tests` — including the complete CPU test tree, Python 3.9–3.12 regression matrix, static contracts, docs contracts, Torch CPU regression, and panel external alignment;
2. `Gaussian inference backend-native` — Python 3.9 CPU, Python 3.12 CPU, Torch 2.0.1 CPU, and static-contract jobs;
3. `Release package validation`;
4. `Release notes validation`;
5. `Maintenance compatibility`;
6. `Benchmark Frontend CI`;
7. `Panel Stage C Torch CPU`.

Fresh complete-diff review after the fourth repair found:

- CRITICAL: **0 open**;
- HIGH: **0 open**;
- relevant actionable MEDIUM: **0 open**.

No submitted PR review or unresolved inline review thread remains. The fourth repair is a single production call-site policy fix plus its hosted regression lock; no additional solver/inference consumer gap was found in the affected shared Gaussian file.

## Finalization and hard-exit state

Production numerical code, validator cases, thresholds, provenance logic, and pass/fail semantics are frozen at the reviewed implementation head above. The finalization commit containing this checkpoint may only update documentation/evidence text and the non-semantic focused-validator freeze annotation so the final exact source SHA receives a fresh hosted run. Any subsequent numerical or validator-semantic change invalidates this closure and requires another review/fix cycle plus new physical evidence.

After the finalization commit's hosted workflows pass, the only unresolved acceptance gate is exact clean-head physical CUDA execution on the final `HEAD`. Therefore the correct hard-exit status is **`PARTIAL_REMOTE_PENDING`**, not `COMPLETE`.

## Canonical physical acceptance

Run both validators from the same exact clean candidate checkout. `results/` is git-ignored, so the first artifact does not invalidate the second validator's clean-tree precondition.

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

Acceptance requires both commands to succeed on the same exact final candidate SHA. The full artifact must report `status: success`, `validation_tier: remote-full`, both required CUDA backends, concrete CUDA-device provenance, and clean-tree acceptance. The focused artifact must report `status: success`, the same exact SHA, `validation_tier: remote-full`, concrete CuPy/CUDA provenance, and success for all maintained rank-recovery cases.

PR #129 must remain Draft and Issue #127 must remain incomplete until both exact-head physical validators pass and the resulting evidence is reviewed against this frozen contract.
