# Issue #127 / PR #129 closure status

Recorded: 2026-08-29

This checkpoint records the third `.claude/skills/code-review.md` closure cycle for PR #129 after exact-head physical CUDA validation invalidated two earlier candidates.

## Superseded physical candidates

The following candidates are permanently superseded as acceptance evidence:

1. `3180add336b41017f4cd5a5e6721a6470a797360` — exact-head Tesla P100 `remote-full` execution exposed two candidate-intrinsic HIGH findings: unsupported `cp.linalg.solve_triangular` usage in CuPy `LinearRegression`, and missing concrete `_selected_backend_device` provenance in the physical host-transfer fixture.
2. `11343d87e810cb147bd0b9f821ac5f5e52976ea2` — exact-head Tesla P100 `remote-full` execution verified the first two repairs, then exposed a third candidate-intrinsic HIGH finding: singular CuPy normal equations could return non-finite Cholesky/solve results instead of raising, so exception-only rank recovery did not reach the intended pseudoinverse path.

Artifacts from those candidates remain diagnostic evidence only. They must not be reused or promoted as canonical success evidence.

## Third review/fix closure

All three physical findings are closed in the current implementation:

- `LinearRegression` uses supported `cupyx.scipy.linalg.solve_triangular` on the CuPy path;
- the physical host-transfer case supplies both executed backend and concrete executed-device provenance before invoking post-fit Gaussian inference;
- maintained CuPy solver boundaries now enable scoped solver-status errors with `cupyx.errstate(linalg="raise")` so rank/definiteness failures become visible to the existing classified recovery paths rather than silently propagating invalid values;
- the repair covers shared Gaussian inference, `LinearRegression` fit/rank fallback/robust covariance, exact-L2 penalized fit/inference, and the shared `xp_cholesky_solve` CuPy boundary;
- unrelated CUDA/programming failures remain fail-closed and are not converted into pseudoinverse recovery;
- the penalized class-level wrapper preserves the real CuPy>=13 solver-status policy while allowing the existing lightweight unit-test double, which intentionally omits `cupyx.errstate`, to reach and verify synthetic CUDA-OOM exception routing.

Reviewed implementation/validator head: `1ef402c33e0a9e1b4e81ef454fd94433ba437d23`.

On that head all seven triggered hosted PR workflows passed:

1. `Tests` — including the complete CPU test tree, regression matrix, static contracts, docs contracts, Torch CPU regression, and panel external alignment;
2. `Gaussian inference backend-native` — Python 3.9 CPU, Python 3.12 CPU, Torch 2.0.1 CPU, and static-contract jobs;
3. `Release package validation`;
4. `Release notes validation`;
5. `Maintenance compatibility`;
6. `Benchmark Frontend CI`;
7. `Panel Stage C Torch CPU`.

The fresh complete-diff review after those fixes found no new numerical/backend/inference correctness issue. The only remaining actionable finding was that this closure ledger still described the superseded second-cycle freeze; this checkpoint corrects that evidence-chain inconsistency. No submitted PR review or unresolved inline review thread remains.

Review result before finalization:

- CRITICAL: **0 open**;
- HIGH: **0 open**;
- relevant actionable MEDIUM: **0 open**.

## Finalization and hard-exit state

Production numerical code, validator cases, thresholds, provenance logic, and pass/fail semantics are frozen at the reviewed implementation above. The finalization commit containing this checkpoint may only make documentation/evidence updates and a non-semantic validator freeze annotation so the final exact source SHA receives a fresh hosted run. Any subsequent numerical or validator-semantic change invalidates this closure and requires another review/fix cycle plus new physical evidence.

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

Acceptance requires both commands to succeed on the same exact final candidate SHA. The full artifact must report `status: success`, `validation_tier: remote-full`, both required CUDA backends, concrete CUDA-device provenance, and clean-tree acceptance. The focused artifact must report `status: success`, the same exact SHA, `validation_tier: remote-full`, concrete CuPy/CUDA provenance, and success for all maintained rank-recovery cases, including the shared singular-solve visibility case.

PR #129 must remain Draft and Issue #127 must remain incomplete until both exact-head physical validators pass and the resulting evidence is reviewed against this frozen contract.
