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
- Hosted workflow #946 passed on exact head
  `a726937a39eb0ed5a370dd03362884b63a9e9818`: the full CPU suite reported
  1879 passed and 662 skipped, while static, documentation, and Python 3.9–3.12
  regression jobs all passed.
- The unchanged physical result for that head is now durably published as
  [the final promotion artifact](https://gist.github.com/TheHiddenObserver/ebbb7f2401f45b124069a30d3510c139).
  It records 134/134 passing checks, zero return codes, empty gate-failure arrays,
  and SHA-256
  `e01ad0bfec238d06167caeef9955e92b6cf84eea4ccc69a3056eb794ded6eccb`.
- This follow-up makes the final aggregation format truly machine schema 3,
  synchronizes the primary CoxPH model pages, and brings `.markdown` archives
  back under maintained documentation checks. Because these commits create a
  new head, the final exact-head physical suite must be rerun before approval;
  the published Gist remains valid evidence for `a726937...` only.

## Earlier history

Detailed entries through 2026-08-03 are retained in
[the archived changelog](changelog-history-through-2026-08-03.markdown).
