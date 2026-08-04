# Changelog

> Language: English<br>
> Last updated: 2026-08-04<br>
> This page: Changelog<br>
> Switch: [Chinese](../cn/changelog.md)

## 2026-08

### Fixed (2026-08-04) — PR #80 exact-source CV review follow-up

- Canonical physical-GPU suites now prepend the audited Git checkout to
  `PYTHONPATH`, disable the user site, verify that actual imported module paths
  remain inside that checkout, and record SHA-256 hashes for those imported
  files. Child and nested runners inherit the same controlled environment.
- Requested CoxPHCV two-stage and successive-halving controls now produce one
  explicit exhaustive full-precision candidate pass on NumPy, CuPy, and Torch.
  Public diagnostics report `staged_safety_strategy="single_pass_exhaustive"`;
  no candidate is screened out and CuPy no longer repeats the complete grid.
- One-shot `CoxPHCV.cv_splits` iterators are materialized privately once and
  reused for repeated fits, scikit-learn clone, legacy parameter reconstruction,
  and pickle. Fit retains the original public constructor object.
- Hosted workflow #943 passed on implementation commit
  `4c8f9493ee08e7ecf6ec88c7296c02070547cda2`: the full CPU suite reported
  1879 passed and 662 skipped, while static, documentation, and Python 3.9–3.12
  regression jobs all passed. A refreshed clean exact-head CuPy/Torch promotion
  suite is still required before this review can be promoted to COMPLETE.

## Earlier history

Detailed entries through 2026-08-03 are retained in
[the archived changelog](changelog-history-through-2026-08-03.md).
