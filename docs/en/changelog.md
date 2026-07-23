# Changelog

> Language: English  
> Last updated: 2026-07-23
> This page: Changelog  
> Switch: [Chinese](../cn/changelog.md)

## 2026-07

### Fixed (2026-07-23) — PR #79 complete review closure

- Unified CoxPH final KKT, line-search, termination, and public result fields on
  CPU/CuPy/Torch; rejected unsupported delayed-entry penalty/robust combinations.
- Added strict-by-default robust inference with explicit approximate opt-in,
  provenance fields, and the `statgpu[survival]` optional dependency.
- Kept Cox prediction/scoring backend-native, vectorized baseline hazards, removed
  the Torch `O(n p^2)` Hessian tensor, and avoided unconditional GPU training-data
  host copies for nonrobust inference.
- Hardened PR79 diagnostics and canonical-report generation against missing,
  failed, duplicate, non-finite, and wrong-SHA evidence, with a CPU smoke gate.
- Required clean, stable, exact-head provenance for canonical evidence, removed
  the stale hard-coded PASS artifacts, and added an executable 576-case physical
  GPU Cox matrix with permutation and peak-memory gates.
- Added behavioral regressions and synchronized the bilingual Cox support matrix;
  physical CUDA acceptance remains an exact-head follow-up gate.

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
The exact cleaned acceptance head is
`786af9e2eb4742a56e5203b4380b03aec63a3ac8`.

Additional repairs:

- preserved backend-native `LinearRegression.fit` and `predict` inputs until backend
  resolution instead of performing eager NumPy conversion;
- made PooledOLS HAC ordering explicit through validated, stable `time_index` sorting;
- used effective design rank for PooledOLS residual degrees of freedom;
- hardened the remote validator with shell `pipefail`, exact required SHAs, immutable base
  worktrees, and reset/clean verification;
- separated formula-controlled intercept semantics from the public clone-visible
  `fit_intercept` constructor parameter;
- corrected weighted `LinearRegression` on CPU, CuPy, and Torch, including intercept
  weighting, multi-output broadcasting, validation, residual state, singular fallback,
  diagnostics, and weighted R-squared;
- aligned original-length formula sample weights after Patsy removes missing rows;
- aligned CuPy and Torch degenerate overall F-test semantics with the CPU contract.

Permanent coverage in `dev/tests/test_pr79_final_review_fixes.py` includes reference-library
parity, rank-deficient and HAC invariants, formula behavior, invalid weights, multi-output
WLS, exact-SHA validator checks, backend-to-NumPy transfer guards, and physical CuPy/Torch
F-statistic edge cases.

### Validation (2026-07-21) — exact-head acceptance passed

On a clean Tesla P100 worktree at exact SHA
`786af9e2eb4742a56e5203b4380b03aec63a3ac8`:

- `STATGPU_REQUIRE_PHYSICAL_GPU=1` forced both CUDA backends to execute;
- `dev/tests/test_pr79_final_review_fixes.py` completed with
  **17 passed, 0 failed, 0 skipped in 7.28 seconds**;
- CuPy and Torch weighted fit/predict parity passed;
- formula missing-row and original-length weight alignment passed;
- perfect non-constant fits return `(inf, 0.0)` for the overall F test;
- intercept-only and otherwise undefined overall F tests return `(nan, nan)`;
- the exact SHA and clean-worktree state were recorded.

Standard GitHub Actions Tests run #483 also passed on the cleaned head, including the
Python 3.9–3.12 regression matrices, static contracts, compilation, complete collection,
and full CPU suite. PR #79 is therefore ready for review and squash merge.

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
