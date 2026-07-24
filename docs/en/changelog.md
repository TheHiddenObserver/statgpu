# Changelog

> Language: English  
> Last updated: 2026-07-24  
> This page: Changelog  
> Switch: [Chinese](../cn/changelog.md)

## 2026-07

### Validation (2026-07-24) — PR #79 exact-head closure

The final reviewed production head is
`c85750d63d4e6dbc9d988847566c20f5fa862e91`.

- GitHub Actions Tests run #545 passed on the exact head.
- Python 3.9, 3.10, 3.11, and 3.12 regression jobs passed.
- The complete CPU suite passed with **1074 passed, 275 skipped, 0 failed**.
- The clean-head canonical smoke pipeline passed with `canonical_eligible=True` and a
  `PASS` verdict.
- The maintained Tesla P100 suite passed **33 executed checks**, with two expected skips
  and zero failures.
- Maintained CoxPH, Linear, and Panel paths passed their PR79 acceptance contracts.

The six ignored legacy GPU diagnostic scripts executed separately are not part of the
maintained pytest Gate. Their conversion, replacement, or retirement is tracked in
[Issue #83](https://github.com/TheHiddenObserver/statgpu/issues/83).

### Fixed (2026-07-24) — final public-contract synchronization

- Corrected the CoxPH delayed-entry support matrix. Robust or cluster covariance with
  `compute_inference=True` raises explicitly; the same fit with
  `compute_inference=False` is allowed as estimation-only and leaves inference fields
  unset.
- Documented `CoxPHCV` as applying the same inference guard during final refit.
- Documented PooledOLS backend-preserving prediction, stable HAC `time_index` ordering,
  and effective-rank residual degrees of freedom.
- Clarified rank-deficient PooledOLS behavior: fitted values, prediction, RSS, rank, and
  fitted-space checks remain valid, while coefficient-level inference is
  `NOT_COMPARABLE` because it is not uniquely identified.
- Synchronized README, English/Chinese CoxPH and Panel pages, release summaries, and the
  auditable PR79 report.
- Removed stale hard-coded final accuracy artifacts. A new full canonical report may be
  committed only after a full exact-head raw campaign is validated by the current
  aggregator and renderer.

### Fixed (2026-07-23) — PR #79 complete review closure

- Unified CoxPH final KKT, line search, termination, and public result fields on
  CPU/CuPy/Torch.
- Added strict-by-default robust inference with explicit approximate opt-in,
  provenance fields, and the `statgpu[survival]` optional dependency.
- Kept Cox prediction and scoring backend-native, vectorized baseline hazards, removed
  the affected Torch Hessian materialization, and avoided unconditional GPU training-data
  host copies for nonrobust inference.
- Hardened PR79 diagnostics and canonical-report generation against missing, failed,
  duplicate, non-finite, dirty, and wrong-SHA evidence.
- Added behavioral regressions and synchronized the bilingual Cox support matrix.

### Validation history (2026-07-21)

The earlier complete Tesla P100 campaign passed on code head
`2f18e5dec9195da1a12e5eea89ee2d832557b3ad`:

- Gate A: 160 passed, 0 failed, 2 expected skips;
- Gate B: 1100 passed, 0 failed, 124 skipped, 1 strict XFAIL;
- Gate C: 10/10 metamorphic checks passed;
- Gate D: no audited full-design GPU-to-CPU transfer;
- Gate E: no leak over 15 repeated CuPy and Torch cycles;
- Gate F: synchronized Tesla P100 baselines recorded at three scales;
- Gate G: Ridge/scikit-learn and linear-regression/statsmodels parity passed.

A subsequent exact-head campaign on `786af9e2eb4742a56e5203b4380b03aec63a3ac8`
passed 17/17 focused physical-GPU checks. These historical SHAs remain auditable evidence,
but the 2026-07-24 entry above is the final PR head closure.

### Performance baseline — Tesla P100

These hardware-specific measurements remain regression baselines, not portable guarantees.

| Shape | CuPy median | Torch median |
|---:|---:|---:|
| 200 x 5 | 2.9 ms | 3.7 ms |
| 2000 x 20 | 3.2 ms | 3.8 ms |
| 10000 x 50 | 4.3 ms | 5.1 ms |

Environment: Tesla P100-SXM2-16GB, Python 3.9, CuPy 13.6.0,
PyTorch 2.0.0+cu117.

### Known non-blocking follow-ups

- [Issue #81](https://github.com/TheHiddenObserver/statgpu/issues/81): shared
  backend-native NaN/Inf validation.
- [Issue #82](https://github.com/TheHiddenObserver/statgpu/issues/82): coordinated
  public-constructor refactor for scikit-learn <=1.2 clone identity.
- [Issue #83](https://github.com/TheHiddenObserver/statgpu/issues/83): convert or retire
  ignored legacy GPU diagnostic scripts.

## Historical entries

Detailed entries through 2026-07-14 are retained in
[the archived changelog](changelog-history-through-2026-07-14.md).
