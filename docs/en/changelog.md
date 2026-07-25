# Changelog

> Language: English<br>
> Last updated: 2026-07-25<br>
> This page: Changelog<br>
> Switch: [Chinese](../cn/changelog.md)

## 2026-07

### Improved (2026-07-25) — v0.2.2 release preparation

- **Version and packaging**:
  - Updated `pyproject.toml` and `statgpu/__init__.py` from 0.2.1 to 0.2.2.
  - Retained the tag-triggered PyPI workflow and `STATGPU_NO_EXT=1` build policy,
    which produces a universal `py3-none-any` wheel plus a source distribution.
  - Kept Python 3.9 through 3.12 in the maintained CI matrix.
- **Included maintained scope**:
  - Carries the PR #79 correctness, backend-contract, inference, and validation
    work summarized in the entries below and the linked auditable artifacts.
  - Includes PR #84's release-facing README, documentation portals, method
    inventory, bilingual model/backend guides, and deterministic docs contracts.
- **Release files**:
  - `pyproject.toml`
  - `statgpu/__init__.py`
  - `CHANGELOG.md`
  - `docs/en/changelog.md`
  - `docs/cn/changelog.md`

### Validation (2026-07-25) — v0.2.2 release candidate

- The version declarations agree at 0.2.2; live PyPI metadata reported 0.2.1 as
  the latest release, and the remote repository had no `v0.2.2` tag.
- The documentation link check and maintained-document contracts passed for
  all 122 maintained documentation files.
- The complete CPU-only suite passed with **1051 passed, 257 skipped, 0 failed**.
- `STATGPU_NO_EXT=1` produced `statgpu-0.2.2-py3-none-any.whl` and
  `statgpu-0.2.2.tar.gz`; both artifacts passed `twine check`.
- Wheel and sdist metadata, archive paths, and contents were audited, with no
  local configuration, credentials, caches, or unrelated result bundles found.
- Fresh wheel and sdist environments both imported statgpu 0.2.2 from their
  installed `site-packages` and passed a CPU `LinearRegression` smoke test.

### Added and fixed (2026-07-25) — PR #80 Cox Phase-1 completion

- Reconciled the PR #80 head, originally based on 0.2.1, with the 0.2.2 release
  tree while preserving the 0.2.2 version and PR #79 inference/KKT contracts.
- Added a shared counting-process risk-set engine for Breslow, Efron, and Exact
  ties, delayed entry, `(start, stop]` time-varying rows, strata, penalties,
  robust/cluster inference, and subject-aware concordance.
- Extended `CoxPHCV` with start/strata/subject propagation, subject-preserving
  folds, Exact held-out likelihood, backend-consistent refit, and inference-mode
  provenance.
- Fixed final-KKT convergence, the open-left `start < event_time` boundary,
  baseline-hazard construction, backend-native prediction/scoring, and
  synchronized GPU benchmark timing and source-version reporting.
- The 2026-07-25 local NumPy quick gate passed all executable correctness,
  inference, CV, schema, and external-comparison checks. Paramiko validation of
  the exact reviewed source in remote `myconda` on a Tesla P100 exposed and
  fixed Torch prediction, scikit-learn 1.2.2 cloning, and test-boundary issues.
  The final physical-GPU matrix passed with **379 passed, 2 expected skips, 0
  failed**; quick/full benchmark schemas passed without gate failures on NumPy,
  CuPy, and Torch.

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
