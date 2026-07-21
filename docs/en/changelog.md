# Changelog

> Language: English  
> Last updated: 2026-07-21  
> This page: Changelog  
> Switch: [Chinese](../cn/changelog.md)

## 2026-07

### Fixed (2026-07-21) — PR #79 final physical-GPU correctness pass

- **Panel inference and rank-deficient PooledOLS**:
  - Root cause: CPU distribution critical values were combined directly with CuPy/Torch
    arrays, categorical cluster labels were sent to numerical GPU constructors, and
    rank-deficient designs depended on unstable direct solves.
  - Impact: clustered inference could fail with device or object-dtype errors, while
    singular pooled designs could produce unstable coefficients and covariance results.
  - Fix: convert critical values with backend-aware helpers, factorize labels as CPU
    metadata before copying integer codes, and use a stable least-squares/pseudoinverse
    path where rank deficiency requires it.
  - Files: `statgpu/panel/_utils.py`, `statgpu/panel/_pooled.py`.

- **Cross-backend array construction and CuPy 13.x compatibility**:
  - Root cause: Torch-only `device=` arguments were forwarded to NumPy/CuPy `asarray`,
    and linear wrappers attempted implicit `np.asarray(cupy_array)` conversion.
  - Impact: valid explicit-CUDA inputs failed before model computation.
  - Fix: pass `device=` only on Torch paths, guard Nystroem construction by backend,
    and use explicit backend-to-NumPy conversion only at documented output boundaries.
  - Files: `statgpu/backends/_utils.py`,
    `statgpu/nonparametric/kernel_methods/_nystroem.py`,
    `statgpu/linear_model/wrappers/_linear.py`.

- **Debiased-Lasso post-fit diagnostics**:
  - Root cause: inference cleanup cleared `_resid`, `_X_design`, and `_y` although
    `rsquared`, AIC, BIC, and related diagnostics still require them.
  - Impact: a successful inference fit could leave the estimator unable to provide
    documented diagnostics.
  - Fix: preserve fitted inference state on NumPy, CuPy, and Torch paths.
  - File: `statgpu/linear_model/penalized/_inference_mixin.py`.

- **Weighted GLM fused loss/gradient recursion**:
  - Root cause: `_weighted_loss_and_grad()` called `loss.fused_value_and_gradient()` with
    weights, which dispatched back into `_weighted_loss_and_grad()`.
  - Impact: weighted smooth-penalty logistic fits could end in `RecursionError` after
    FISTA-BB correctly redirected to FISTA.
  - Fix: compute the weighted per-sample loss and score directly, keeping reductions on
    the selected backend.
  - File: `statgpu/glm_core/_fused.py`.

- **StepwiseSelector legacy sklearn clone behavior**:
  - Root cause: the constructor replaced public parameters with normalized or copied
    objects, violating the identity check used by scikit-learn <=1.2.
  - Impact: `sklearn.base.clone()` failed for StepwiseSelector.
  - Fix: preserve public constructor parameters and keep normalized runtime state private.
  - File: `statgpu/feature_selection/_stepwise.py`.

### Optimized (2026-07-21) — synchronized Tesla P100 baseline

Physical-GPU timings were measured after correctness passed, with warmup and backend
synchronization. These are environment-specific regression baselines, not portable
performance guarantees.

| Shape | CuPy median | Torch median |
|---:|---:|---:|
| 200 x 5 | 2.9 ms | 3.7 ms |
| 2000 x 20 | 3.2 ms | 3.8 ms |
| 10000 x 50 | 4.3 ms | 5.1 ms |

Environment: Tesla P100-SXM2-16GB, Python 3.9, CuPy 13.6.0,
PyTorch 2.0.0+cu117. Audit report:
`dev/reviews/pr79_physical_gpu_validation.md`.

### Improved (2026-07-21) — validation and release evidence

- Added a reproducible physical-GPU validation plan, remote orchestrator, shared GPU
  fixtures, result aggregation, device-transfer audit, memory checks, performance
  measurements, and external-reference comparisons.
- Added `dev/tests/test_pr79_physical_gpu.py` and supporting scripts under
  `dev/validation/`.
- Added the final review artifact at
  `dev/reviews/pr79_physical_gpu_validation.md` and bilingual user-facing summaries at
  `docs/en/releases/pr79-final-validation.md` and
  `docs/cn/releases/pr79-final-validation.md`.

### Validation (2026-07-21) — all gates passed

| Gate | Scope | Result |
|---|---|---|
| A | GPU smoke | 160 passed, 0 failed, 2 expected skips |
| B | NumPy/CuPy/Torch correctness | 1100 passed, 0 failed, 124 skipped, 1 strict XFAIL |
| C | Metamorphic properties | 10/10 passed; one known finite-input finding |
| D | Device purity | Zero full-design transfers; three model families audited |
| E | Memory | Zero leaks over 15 repeated CuPy and Torch cycles |
| F | Performance | Three synchronized scales recorded on both GPU backends |
| G | External references | Ridge versus scikit-learn; linear regression versus statsmodels |
| Final | Complete suites | CPU 1100 passed; GPU 1100 passed |

Gate B improved from **1036 passed / 40 failed / 159 skipped** to
**1100 passed / 0 failed / 124 skipped / 1 strict XFAIL**. The clone XFAIL under
scikit-learn <=1.2 reproduces for the same 26 estimators on base SHA `a4879fb`, so it
is not introduced by PR #79.

### Known non-blocking follow-ups

- [Issue #81](https://github.com/TheHiddenObserver/statgpu/issues/81): complete the
  shared backend-native NaN/Inf validation contract. Ridge currently has one path that
  does not reject non-finite input before a CUDA kernel.
- [Issue #82](https://github.com/TheHiddenObserver/statgpu/issues/82): refactor public
  estimator constructors to satisfy the scikit-learn <=1.2 clone identity contract.
- The Torch Cox Hessian still materializes an `O(n*p*p)` intermediate and remains a
  separate performance optimization item.

None of these findings blocks the finite-input paths validated in PR #79.

## Historical entries

Detailed entries through 2026-07-14 are retained in
[the archived changelog](changelog-history-through-2026-07-14.md).
