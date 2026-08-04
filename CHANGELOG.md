# Changelog

All notable changes to statgpu are documented here, organized by date and PR.

## 2026-08-04

### PR #80 — Exact-source CV review-fix follow-up
- Bound the canonical physical-GPU suites to the files actually imported from the audited checkout, including runtime module paths and SHA-256 hashes.
- Converted requested CoxPHCV two-stage/successive-halving execution into one explicit exhaustive full-precision pass on NumPy, CuPy, and Torch, eliminating the repeated CuPy full-grid fit.
- Made one-shot `CoxPHCV.cv_splits` iterators reusable across repeated fit, scikit-learn clone, parameter reconstruction, and pickle without rewriting the public constructor attribute during fit.
- Published the unchanged exact-head `a726937a39eb0ed5a370dd03362884b63a9e9818` physical artifact as a durable Gist: 134/134 checks passed, all return codes were zero, every gate-failure array was empty, and the artifact SHA-256 is `e01ad0bfec238d06167caeef9955e92b6cf84eea4ccc69a3056eb794ded6eccb`.
- Bumped the final promotion report's machine schema to 3, synchronized primary CoxPH documentation and review status, and returned `.markdown` changelog archives to maintained documentation checks. These follow-up commits create a new head, so final exact-head physical promotion must be rerun before approval.

## Earlier history

Entries through 2026-07-27 are retained in
[`CHANGELOG-history-through-2026-07-27.md`](CHANGELOG-history-through-2026-07-27.md).
