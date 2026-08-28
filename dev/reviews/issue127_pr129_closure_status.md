# Issue #127 / PR #129 closure status

Recorded: 2026-08-28

This checkpoint records that the prior `PARTIAL_REMOTE_PENDING` hard exit was invalidated by exact-head physical CUDA execution on frozen candidate `3180add336b41017f4cd5a5e6721a6470a797360`.

## Physical acceptance result on frozen candidate

The remote-full physical validator was executed against exact clean head `3180add336b41017f4cd5a5e6721a6470a797360` on a Tesla P100 environment. The run produced canonical failure artifacts with the expected `git_sha` and `validation_tier=remote-full` metadata. The failures expose candidate/validator defects rather than an external environment outage.

Two review/fix findings are reopened:

1. **HIGH — CuPy LinearRegression triangular solve uses a nonexistent API.** `statgpu/linear_model/wrappers/_linear.py` calls `cp.linalg.solve_triangular`, but CuPy does not expose that symbol. The maintained repository already uses `cupyx.scipy.linalg.solve_triangular` for this operation. Physical execution reaches this path and fails before the CuPy matrix can complete.
2. **HIGH — physical host-transfer fixture violates the frozen provenance contract.** `_host_transfer_case` manually constructs `PenalizedGeneralizedLinearModel`, sets `_selected_backend_name`, and directly calls `_compute_post_fit_gaussian_inference`, but does not set `_selected_backend_device`. The production fail-closed router therefore correctly rejects the fixture with missing concrete device provenance. This is deterministic for both CuPy and Torch and requires a validator fix.

Because the frozen physical validator failed for candidate-intrinsic reasons, the previous validator freeze is revoked and PR #129 returns to review/fix. Do not claim `PARTIAL_REMOTE_PENDING` or `COMPLETE` from `3180add336b41017f4cd5a5e6721a6470a797360`.

## Required repair

- replace the invalid CuPy triangular solve call with the maintained `cupyx.scipy.linalg.solve_triangular` path (with the same narrow import fallback policy already used elsewhere if needed);
- in `_host_transfer_case`, set `model._selected_backend_device = concrete_device` together with `_selected_backend_name` before invoking the post-fit inference path;
- add hosted/static regression checks that forbid `cp.linalg.solve_triangular` in the maintained LinearRegression CuPy path and require the host-transfer fixture to provide concrete device provenance;
- rerun the complete hosted matrix on the repaired exact head;
- perform a fresh complete-diff review;
- freeze the repaired validator contract and rerun exact clean-head CuPy + Torch CUDA `remote-full` acceptance.

Current status: **REVIEW_FIX_REOPENED**.
