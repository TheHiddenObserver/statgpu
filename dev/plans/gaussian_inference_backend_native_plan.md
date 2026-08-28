# Gaussian Linear-Model Backend-Native Inference Plan

Issue: #127  
Baseline release: `0.2.5`  
Baseline `master`: `84f8bc7e17f66466b3a325cbb007b6cb41843821`  
Planning review status: **REVIEW_CLEAN**  
Production implementation status: **physical CUDA exposed candidate defects; review/fix reopened**

The detailed implementation plan remains unchanged from the reviewed PR #129 plan. The previous `PARTIAL_REMOTE_PENDING` checkpoint on frozen candidate `3180add336b41017f4cd5a5e6721a6470a797360` is revoked because exact-head `remote-full` physical validation exposed two candidate-intrinsic defects: an invalid `cp.linalg.solve_triangular` call in the maintained CuPy `LinearRegression` path, and a physical-validator host-transfer fixture that omitted `_selected_backend_device` while directly invoking the fail-closed post-fit inference router.

Required closure sequence is therefore reset to:

1. repair both defects with minimal production/validator changes;
2. add hosted/static regressions for the two physical findings;
3. rerun all hosted PR gates on the repaired exact head;
4. perform a fresh complete-diff review with CRITICAL/HIGH/actionable MEDIUM all at zero;
5. freeze the repaired validator contract;
6. rerun exact clean-head CuPy + Torch CUDA acceptance at `validation_tier=remote-full` and persist canonical evidence;
7. only after physical success may PR #129 / Issue #127 be marked `COMPLETE`.

Until steps 1-4 close, status is **REVIEW_FIX_REOPENED**. After those gates close, if only the new physical execution remains, the status may return to `PARTIAL_REMOTE_PENDING`.
