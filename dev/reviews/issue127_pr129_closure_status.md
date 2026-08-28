# Issue #127 / PR #129 closure status

Recorded: 2026-08-28

This checkpoint records the second `.claude/skills/code-review.md` hard exit after exact-head physical CUDA validation invalidated the first frozen candidate.

## Superseded physical candidate

Frozen candidate `3180add336b41017f4cd5a5e6721a6470a797360` is permanently superseded as acceptance evidence. Exact-head `remote-full` execution on Tesla P100 produced canonical failure artifacts and exposed two candidate-intrinsic HIGH findings:

1. `LinearRegression` CuPy fitting called nonexistent `cp.linalg.solve_triangular`;
2. physical `_host_transfer_case` supplied backend provenance but omitted `_selected_backend_device` before directly invoking the fail-closed post-fit Gaussian inference router.

Those failure artifacts remain useful diagnostic evidence, but they do not satisfy acceptance and must not be promoted as canonical success evidence.

## Review/fix closure

Both HIGH findings are now closed:

- `statgpu/linear_model/wrappers/_linear.py` imports and uses `cupyx.scipy.linalg.solve_triangular` for the CuPy Cholesky solves; the invalid `cp.linalg.solve_triangular` calls are gone;
- `_host_transfer_case` now sets `model._selected_backend_device = concrete_device` together with `_selected_backend_name`, matching the provenance that a real fit records before the same post-fit inference router is invoked;
- hosted/static contract checks require the maintained CuPy path to use the supported triangular-solve API and require the physical host-transfer fixture to provide concrete device provenance.

Reviewed repaired implementation/validator head: `a1d94af4f7de02048645347cb95a902caa46cc2b`.

On that repaired head all six hosted PR workflows passed:

1. `Tests`;
2. `Gaussian inference backend-native` — Python 3.9 CPU, Python 3.12 CPU, Torch 2.0.1 CPU, and static-contract jobs all passed;
3. `Release package validation`;
4. `Release notes validation`;
5. `Maintenance compatibility`;
6. `Benchmark Frontend CI`.

Fresh complete-diff review after the physical fixes found:

- CRITICAL: **0 open**;
- HIGH: **0 open**;
- relevant actionable MEDIUM: **0 open**.

No submitted PR review or unresolved review thread remains. The failed one-shot repair workflow scaffold was removed from the branch and is not part of the final PR diff. The full implementation plan was restored; its live status is `implemented; final acceptance pending`.

## Re-frozen hard-exit state

Production numerical code, validator cases, thresholds, provenance logic, and pass/fail semantics are frozen again. This checkpoint is documentation-only relative to the reviewed repaired implementation/validator head above. Any subsequent production or validator semantic change requires another review/fix cycle and invalidates physical evidence collected before that change.

After hosted workflows pass on the commit containing this checkpoint, the only unresolved acceptance gate is exact clean-head CuPy + Torch CUDA execution of the re-frozen physical validator and persistence of a new canonical success artifact.

Therefore the correct status is **`PARTIAL_REMOTE_PENDING`**, not `COMPLETE`.

Canonical physical invocation on the final clean candidate head is:

```bash
SHA="$(git rev-parse HEAD)"
python dev/benchmarks/validate_gaussian_inference_backend_native_gpu.py \
  --out "results/pr129_gaussian_inference_${SHA:0:8}.json" \
  --expected-sha "$SHA" \
  --validation-tier remote-full \
  --backends cupy,torch
```

The new artifact must report `status: success`, the exact final candidate SHA, `validation_tier: remote-full`, both required CUDA backends, and clean-tree acceptance. PR #129 must remain Draft and Issue #127 must remain incomplete until that new physical run passes.
