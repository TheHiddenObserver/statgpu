# PR168 natural-spline expected-failure guard repair

## Scope and original finding

This bounded follow-up starts from `dd719917e336a9e8dc93386e13c35f861237abe4`
and compares the complete PR with `3fba9af81db8624ab6e882cb153e7ede7ead7f63`.
The preceding read-only review found one MEDIUM / TEST issue: two cycle-4
natural-spline guards classified any constant or endpoint-curvature mismatch as
known issue [#224](https://github.com/TheHiddenObserver/statgpu/issues/224).

A large-range all-zero basis and a small-range unprojected ordinary B-spline
basis both left the prior smoothing suite at 233 passed / 62 xfailed. Those were
unrelated regressions, not acceptable instances of the tracked numerical defect.
The underlying numerical algorithm is outside this repair and remains unchanged.

The repair changes only two test modules, this developer record and the English
index's accidental Chinese word in its ordered-model description. It introduces
no public numerical capability, dependency or CI change. The earlier intentional
Torch model-X runtime repair and UMAP guard repair remain in the broader PR.

## Guard discrimination

`test_pr168_survival_smoothing_cycle4.py` now requires the expected shape, finite
values and full column rank before any known-defect sentinel can be raised.
It uses a basis-relative reconstruction tolerance, including for tiny columns,
and reconstructs output through SciPy's independent cubic B-spline evaluator,
rather than using production's ordinary basis to validate another production
path. Coefficient QR normalization removes arbitrary scale and orientation from
the boundary-curvature check.

Known defects are identified by coefficient-space projectors, not individual SVD
columns. The large-range constraint signature is explicitly limited to the fixed
`1e6` fixture: cancellation leaves the limiting left interior ratio `[-4, 1]`
and right adjacent difference `[-1, 1]`. Its projector tolerance is `5e-6`,
covering the observed approximately `1.2e-6` cancellation residue. This is a
bounded signature of the fixture, not a general finite-difference oracle for
arbitrary large coordinate ranges.

For small ranges `1e-8` and `1e-7`, SciPy independently evaluates the erroneous
stencil-expanded knot interval. A dimensionless projector tolerance of `1e-10`
recognizes that specific wrong space. The actual large/small defect must still
violate the desired property before the custom sentinel is raised.

A successful test body must match the analytic two-boundary natural subspace.
Restoring constants or one boundary alone therefore cannot silently certify a
partial repair. A genuine analytic repair returns normally and becomes strict
XPASS under the retained narrow pytest markers. It does not get disguised as
another expected failure.

Both float32 and float64 input fixtures are covered. The current public routine
promotes these inputs to float64; this is input-representation coverage, not a
claim of native float32 spline computation.

## Committed regression coverage

`test_pr168_natural_spline_guards.py` constructs replacements independently with
SciPy and invokes the actual marked test bodies. Its 140 cases cover:

- zero, wrong-row/column shape, NaN/Inf and rank-deficient outputs;
- full-rank ordinary B-splines, unrelated finite subspaces and off-spline noise;
- unrelated runtime exceptions;
- known-defect spaces under signed permutations, orthogonal rotations, invertible
  mixing and very small column scaling;
- left-only/right-only partial repairs;
- independent analytic repairs under the same basis transformations;
- retained strict markers limited to the two designated sentinel exception types.

These replacement tests do not require the production numerical defect to remain
present. When the implementation is repaired, the marked desired-behavior tests
will XPASS and their obsolete expected-failure markers must be removed.

## Verification

- Focused final test pair: **153 passed / 14 strict xfailed / 0 failed**.
- Full survival/smoothing selection: **374 passed / 66 strict xfailed / 0 failed**.
- Each original large-zero and small-ordinary mutation produces exactly two
  ordinary assertion failures, with 13 passed and 12 other xfailed cases in the
  cycle-4 module. It is no longer accepted as the designated known defect.
- The independent analytic repair and an orthogonally rotated analytic repair
  each produce exactly six strict XPASS failures, with 13 passed and eight
  unrelated xfailed cases. No unrelated failure was observed in either replay.
- Both changed test modules pass standard Ruff and incremental whitespace checks.
  Maintained documentation contracts pass for 168 files; the link checker reports
  zero affected files. The prescribed high-signal Ruff set passes, and 471
  package/maintained-development Python files compile in memory.
- Environment: Python 3.12, NumPy 2.3.5, SciPy 1.17.0 and Torch 2.14.1+cpu,
  with one BLAS/OMP thread. Counts overlap and must not be summed as unique tests.

The full-repository CPU run and independent final review are recorded separately
in the PR description, tied to the frozen final source. These targeted checks
alone are not a claim that all repository tests or physical backends ran.

Final immutable head/tree identity, full candidate fingerprint, independent
review result and hosted check URLs are recorded in the PR description after
publication. Earlier-head hosted results are historical and are not substitutes
for the new-head check results.

## Preservation and evidence limits

All 286 tracked package paths, including all 278 production Python files, are
byte-identical to the repair-start head. Both native Torch model-X contract test
files, all unsupervised test files and the workflow configuration are unchanged.
The complete-PR production comparison remains 220 byte-identical Python files,
56 initial-docstring-only files, the existing Cox `__doc__` literal assignment,
and the single earlier intentional model-X Torch device/generator runtime change.

User-reported P100 evidence is source-bound to clean `dd719917`: 133 passed /
18 deselected / 0 skipped with `cuda1` excluded, including 18 actual `cuda0`
cases. Its extra 18-pass `cuda0` repetition is duplicate coverage. This follow-up
verifies unchanged relevant source/test blobs, but does not claim a new-head
physical-GPU run or independent inspection of those server logs.

No physical `cuda1`, CuPy, R, performance or statistical calibration run is
claimed. Skips and expected failures are not correctness passes. Issue #224,
UMAP issue #221, and the documented downstream Torch Lasso restrictions remain
separate implementation work. No merge, PR-state transition or issue closure is
part of this repair.
