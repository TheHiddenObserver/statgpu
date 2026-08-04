# Changelog

All notable changes to statgpu are documented here, organized by date and PR.

## 2026-08-04

### PR #80 — Exact-source CV review-fix follow-up
- Bound the canonical physical-GPU suites to the files actually imported from the audited checkout, including runtime module paths and SHA-256 hashes.
- Converted requested CoxPHCV two-stage/successive-halving execution into one explicit exhaustive full-precision pass on NumPy, CuPy, and Torch, eliminating the repeated CuPy full-grid fit.
- Made one-shot `CoxPHCV.cv_splits` iterators reusable across repeated fit, scikit-learn clone, parameter reconstruction, and pickle without rewriting the public constructor attribute during fit.
- Added hosted provenance, lifecycle, concurrency, cache, and single-pass regressions; the refreshed exact-head CuPy/Torch promotion suite remains required before final approval.

## Earlier history

Entries through 2026-07-27 are retained in
[`CHANGELOG-history-through-2026-07-27.md`](CHANGELOG-history-through-2026-07-27.md).
