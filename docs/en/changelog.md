# Changelog

> Language: English  
> Last updated: 2026-07-21  
> This page: Changelog  
> Switch: [Chinese](../cn/changelog.md)

## 2026-07

### Fixed (2026-07-21) — PR #79 physical-GPU validation

The complete Tesla P100 campaign passed on code head
`2f18e5dec9195da1a12e5eea89ee2d832557b3ad`.

- Gate A: 160 passed, 0 failed, 2 expected skips.
- Gate B: 1100 passed, 0 failed, 124 skipped, 1 strict XFAIL.
- Gate C: 10/10 metamorphic checks passed.
- Gate D: no audited full-design GPU-to-CPU transfer.
- Gate E: no leak over 15 repeated CuPy and Torch cycles.
- Gate F: synchronized Tesla P100 baselines recorded at three scales.
- Gate G: Ridge/scikit-learn and linear-regression/statsmodels parity passed.
- Final complete suites: CPU 1100 passed; GPU 1100 passed.

Gate B improved from **1036 passed / 40 failed / 159 skipped** to
**1100 passed / 0 failed / 124 skipped / 1 strict XFAIL**. The version-limited clone
XFAIL reproduces on base SHA `a4879fb` and is tracked in issue #82.

Production fixes from that campaign included panel device mismatches, categorical cluster
factorization, rank-deficient PooledOLS, Torch-only `device=` leakage, CuPy 13.x and
Nystroem construction, debiased-Lasso fitted-state retention, weighted GLM fused recursion,
and StepwiseSelector legacy clone behavior.

### Fixed (2026-07-21) — post-validation review-fix loop

A further review → fix → test → re-review cycle was completed after the full GPU campaign.
The cleaned code head is `ff72424071ec7ca52399146dbd8a556534c9e6c3`.

Additional repairs:

- preserved backend-native `LinearRegression.fit` and `predict` inputs until backend
  resolution instead of performing eager NumPy conversion;
- made PooledOLS HAC ordering explicit through validated, stable `time_index` sorting;
- used effective design rank for PooledOLS residual degrees of freedom;
- hardened the remote validator with shell `pipefail`, exact required SHAs, immutable base
  worktrees, and reset/clean verification;
- separated formula-controlled intercept semantics from the public clone-visible
  `fit_intercept` constructor parameter;
- corrected weighted `LinearRegression` on CPU, CuPy, and Torch by weighting the intercept
  column, fixing multi-output broadcasting, validating weights, retaining raw and weighted
  residual states, and using stable least-squares fallback paths;
- aligned original-length formula sample weights after Patsy removes missing rows.

Permanent tests were added in `dev/tests/test_pr79_final_review_fixes.py`, including
scikit-learn/statsmodels parity, rank-deficient and HAC invariants, formula intercept and
missing-row behavior, invalid weight contracts, multi-output WLS, orchestrator exact-SHA
checks, pipeline failure propagation, and optional physical CuPy/Torch parity.

### Validation boundary for the latest head

GitHub Actions Tests run #477 passed on cleaned code head `ff72424`:

- Python 3.9, 3.10, 3.11, and 3.12 regression matrices;
- static contracts, compilation, and complete test collection;
- the complete CPU suite.

The post-validation changes touch weighted CuPy/Torch `LinearRegression` paths. Therefore,
one focused physical-GPU recheck on the exact cleaned code head is still required before
PR #79 is changed from Draft to Ready for review. The prior full P100 campaign remains
valid evidence for `2f18e5d`, but is not presented as exact-head evidence for later code.
See `dev/reviews/pr79_physical_gpu_validation.md` for the required command and acceptance
criteria.

### Performance baseline — Tesla P100

These measurements were recorded on the physically validated head and are
hardware/environment-specific regression baselines, not portable guarantees.

| Shape | CuPy median | Torch median |
|---:|---:|---:|
| 200 x 5 | 2.9 ms | 3.7 ms |
| 2000 x 20 | 3.2 ms | 3.8 ms |
| 10000 x 50 | 4.3 ms | 5.1 ms |

Environment: Tesla P100-SXM2-16GB, Python 3.9, CuPy 13.6.0,
PyTorch 2.0.0+cu117.

### Known non-blocking follow-ups

- [Issue #81](https://github.com/TheHiddenObserver/statgpu/issues/81): complete the
  shared backend-native NaN/Inf validation contract.
- [Issue #82](https://github.com/TheHiddenObserver/statgpu/issues/82): coordinated public
  constructor refactor for scikit-learn <=1.2 clone identity.
- Torch Cox Hessian `O(n*p*p)` intermediate allocation remains a separate performance item.

## Historical entries

Detailed entries through 2026-07-14 are retained in
[the archived changelog](changelog-history-through-2026-07-14.md).
