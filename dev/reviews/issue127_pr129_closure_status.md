# Issue #127 / PR #129 closure status

Recorded: 2026-08-28

This checkpoint records the `.claude/skills/code-review.md` review-fix hard exit for draft PR #129.

## Reviewed source and validator contract

- reviewed implementation/validator head: `63af2473f1e84ca5a1d5c4e655d50c0ec7857e8a`;
- baseline: `master` at `84f8bc7e17f66466b3a325cbb007b6cb41843821`;
- subsequent source changes before this checkpoint are documentation/status-only and do not alter production numerics, validator cases, thresholds, provenance logic, or pass/fail semantics;
- PR remains draft because canonical physical CUDA evidence is still pending.

## Final review-fix findings closed

The final backend/device rounds closed the remaining maintained-surface findings, including:

- `LinearRegression` CuPy/Torch critical-value evaluation selects the executed numerical backend explicitly;
- `LinearRegression` and penalized Gaussian consumers record and use concrete executed-device provenance instead of ambient CUDA state;
- CuPy fit/inference allocations and numerical calls are held on the fitted `X.device`;
- Torch Gaussian inference accepts legitimate `cpu` execution as well as concrete `cuda:N`, while CuPy remains restricted to concrete CUDA provenance;
- missing or invalid executed backend/device provenance fails closed rather than silently falling back to NumPy;
- exact-L2/Ridge/RidgeCV inference records concrete backend/device provenance and preserves the maintained `n_eff * alpha`, weighting, intercept, and final-refit contracts;
- CuPy non-rank failures propagate instead of being hidden by pseudoinverse recovery;
- the physical validator schema 5 covers the maintained CuPy/Torch CUDA matrix, exact SHA, clean-tree state, covariance/statistic/CI parity, host-transfer provenance, RidgeCV final refit, and negative controls.

## Hosted acceptance

On reviewed implementation/validator head `63af2473f1e84ca5a1d5c4e655d50c0ec7857e8a`, all six hosted PR workflows passed:

1. `Tests`;
2. `Gaussian inference backend-native` — including Python 3.9 CPU, Python 3.12 CPU, Torch 2.0.1 CPU, and static-contract jobs;
3. `Release package validation`;
4. `Release notes validation`;
5. `Maintenance compatibility`;
6. `Benchmark Frontend CI`, including production browser QA.

The final complete-diff review found:

- CRITICAL: **0 open**;
- HIGH: **0 open**;
- relevant actionable MEDIUM: **0 open**.

No submitted PR review or unresolved review thread remains at this checkpoint.

## Frozen hard-exit state

The physical validator contract is frozen. Do not change production numerical code, validator cases, thresholds, provenance logic, or pass/fail semantics without returning the PR to review/fix and rerunning affected physical evidence.

This checkpoint is documentation-only and intentionally creates the final candidate head for one last hosted rerun. Once those hosted workflows pass on this docs-only head, the only unresolved acceptance gate is exact clean-head CuPy + Torch CUDA execution of the frozen physical validator and persistence of its canonical artifact.

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

The resulting artifact must report `status: success`, the exact final candidate SHA, `validation_tier: remote-full`, both required backends, and clean-tree acceptance before PR #129 can leave draft / Issue #127 can be marked `COMPLETE`.
