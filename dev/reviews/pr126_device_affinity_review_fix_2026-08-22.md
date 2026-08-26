# PR #126 device-affinity and GPU-numerics review-fix round — 2026-08-22

Review standard: `.claude/skills/code-review.md`, `auto-fix` mode.

This pass re-reviewed the PR head that added the CuPy device-affinity contract
(`6711e6f4...`) and closed four physical GPU gates found only by exact-head
fresh Tesla P100 execution. All fixes are committed as `6711e6f4..da3604ee`;
final physical acceptance at `da3604ee` is 12/12 runners PASS on
Tesla P100-SXM2-16GB (CuPy 13.6.0, Torch 2.0.0+cu117).

## Findings fixed

1. **CRITICAL — CuPy `maximum.at` / `cupyx.scatter_max` return `inf` for
   float64 magnitudes around 1e7..1e308.**

   Observed on CuPy 13.6 even with unique indices. Group min/max scatter in
   `statgpu/panel/_reductions.py` and `statgpu/survival/_risk_sets.py` fell
   back to the sequential host scatter and restored results on the reference
   device. Torch keeps `scatter_reduce_`, NumPy keeps the native `at` scatter.
   Without this, the tiered cancellation reducer looped for 128 rounds and
   raised "grouped score cancellation exceeded the float64 tier budget".

2. **CRITICAL — Torch CUDA SVD leaks ~1e-16 into structurally zero `U`
   entries.**

   cuSOLVER's default `gesvdj` driver produced `U[0] = -1.37e-16` for a rank-1
   design where NumPy/LAPACK returns exact zero; a 2^55 response amplified
   that into a spurious 3.48 coefficient in a one-way FE prediction. Panel
   code calls `torch.linalg` directly (the array module is `torch` itself), so
   the driver selection was moved into the shared `_panel_svd` helper used by
   every panel least-squares SVD path: CUDA uses `driver="gesvd"`, whose U/V
   factors match the NumPy reference exactly. Backend-wrapper-only fixes do
   not reach these call sites.

3. **HIGH — Failed panel fits cleared the executed backend provenance.**

   `transactional_fit` reset `_backend_name` on failure, so a fail-closed fit
   could not be attributed to the backend it actually executed on. Failed fits
   now retain the concrete executed backend name.

4. **HIGH — Physical parity asserts at 1e154 scales depended on NumPy-SVD
   roundoff accidents.**

   Structurally zero beta/covariance coordinates amplify unavoidable SVD
   roundoff (one eps factor), while the NumPy reference can be exactly zero by
   accident. The extreme-scale asserts in
   `validate_fama_macbeth_review_fix_gpu.py` now use reference-scale absolute
   tolerances instead of `atol=0`, without weakening relative parity on
   resolved entries.

5. **HIGH — Fama-MacBeth overflow fixture no longer forced SVD fallback.**

   The constant 6e307 response is absorbed by the constant anchor, so the
   normal-equation RHS stayed finite and the certified Gram path solved
   exactly (matching NumPy). The fixture now uses a constant-plus-slope
   response whose RHS overflows float64 after anchoring, restoring the
   mandatory SVD-fallback contract; the entity-aware scaled-R2 check reuses a
   constant response where every total-SS direction is genuinely degenerate.

6. **MEDIUM — Mixed-coefficient resolution fixture could not fail closed on
   the exact gesvd driver.**

   The well-conditioned fixture only tripped through a NumPy-SVD roundoff
   accident. The resolution fixture now uses zero-mean near-collinear columns
   (no constant anchor, intercept addition keeps full rank, condition ~1e12)
   so SVD roundoff is unavoidable on every backend. Fama-MacBeth gets a
   separate well-conditioned fixture whose certified Gram path applies a
   deterministic RHS-rounding error bound.

7. **MEDIUM — Residual CuPy allocations were not reference-device bound.**

   `_scatter_add`, `make_group_dummies`, and `_row_weights` created CuPy
   arrays on the current device instead of the reference device.
   `xp_arange` gained range-form support; all three call sites now use the
   device-aware helpers.

8. **MEDIUM — Optimized-wrapper provenance contract was stale.**

   The balanced 64x128 timing fixture is well conditioned, so NumPy now
   retains all periods on the same certified Gram path as the GPU backends
   (one exact-size bucket, one control transfer, zero SVD fallbacks) instead
   of the historical serial/64/64 expectation; `_expected_provenance` was
   updated and documented.

## Validation

Local: 3008 passed, 649 skipped (5 pre-existing failures reproduced on the
pre-fix head `6711e6f4` and are unrelated frontend/PR80/PR122 asset contracts).

Remote Tesla P100, clean detached worktree at `da3604ee`, CuPy 13.6.0 /
Torch 2.0.0+cu117 / NumPy 1.24.2, all 12 physical runners PASS:

- `validate_panel_stage_c_gpu.py` — 35 cases/backend, 12 primitives/backend
- `validate_fama_macbeth_optimized_gpu.py` — focused oracle + certified-Gram
  provenance (numpy/cupy/torch all gram-certified, 1 batch, 1 control sync,
  0 fallbacks)
- `validate_panel_hac_chronology_gpu.py`
- `validate_fama_macbeth_t2_tail_gpu.py`
- `validate_panel_cupy_device_affinity_gpu.py`
- `benchmark_fama_macbeth_scaling_gpu.py`
- `validate_fama_macbeth_rhs_cancellation_gpu.py` (cupy, torch)
- `validate_fama_macbeth_rank_precision_precedence_gpu.py` (cupy, torch)
- `validate_panel_intercept_cancellation_gpu.py` (cupy, torch)

Artifacts: `results/pr126_review_fix_da3604ee/` (12 JSON payloads).

## Review exit

- actionable CRITICAL: 0 open
- actionable HIGH: 0 open
- locally actionable MEDIUM: 0 open
- merge performed: no
- remaining hard exit: final exact-head hosted CI on the new head