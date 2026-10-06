# PR168 seventh documentation and runtime-contract follow-up

## Target and review scope

This is a fresh full-PR review using the repository's actual code-review skill,
review matrix, target-resolution rules and documentation-language policy. The
starting remote head is `d197a7f05291bf1474c86862c74fb0aa025f2411`, with comparison
base `3fba9af81db8624ab6e882cb153e7ede7ead7f63`. The initial full PR contains 196
changed paths. Earlier clean reviews are historical, not acceptance of this
candidate.

All 1,461 starting-head blobs and all 1,401 base blobs were materialized and
verified by Git blob hash. The exact unsigned head and signed base commit
objects were reconstructed from their public metadata and checked against their
known commit hashes, giving a clean writable checkout at the actual head.
The complete endpoint diff is available locally; oversized compare-API patch
omissions were not treated as unchanged content. Supporting base fixtures and
all execution caches are outside the publication candidate.

The active axes are documentation/public-contract reconciliation: API defaults,
methods and fitted results, objectives and intercepts, solver and inference
limits, preprocessing and resampling semantics, runtime help, runnable learner
examples, Chinese readability, regression quality and source-specific evidence.
No new numerical capability is being implemented. Numerical, dispatch and state
production changes remain out of scope.

## Documentation and test repairs

- Made the Ridge, SCAD and MCP learner examples self-contained and runnable on
  NumPy CPU. Explained the unpenalized intercept in the displayed objectives,
  stated appropriate nonconvex interpretation instead of blanket near-unbiased
  claims, and completed method, formula and weight reference guidance.
- Reconciled their installed help with the public API. Completed the linked
  RidgeCV and LassoCV references and corrected their result conventions:
  `best_score_` is negative mean squared error, `cv_results_` contains the fold
  `mse_path`, and the mean path is separately exposed as `mean_mse_`.
- Documented that a rejected KernelPCA refit can mix old projection arrays with
  new kernel-centering state, returning finite but wrong coordinates. The
  safe workflow discards the failed instance and fits a new one; legitimate
  zero-rank rejection is retained. Tests cover first-fit failure, `fit` and
  `fit_transform` refits, `transform`/`predict`, independent centered-Gram
  checks, a full-rollback repair control and unrelated-failure guards.
- Clarified that native generated Torch model-X construction currently ignores
  its requested seed and consumes the global Torch RNG. Covered the function,
  unified function and selector interfaces, the NumPy reproducible alternative,
  and supplied-knockoff/fixed-X boundaries. Clarified when `knockpy_sampler`
  is rejected versus ignored, including the earlier incorrect Chinese FAQ.
- Reconciled the shared CV guide and public-design page with the custom Ridge
  training-row limitation, and clarified that accepting a split does not
  establish shape, disjointness or statistical validity.
- Kept implementation/evidence status outside ordinary learner/API prose and
  synchronized English/Chinese explanations and changelogs.
- Rechecked the complete unsupervised family, survival/smoothing/distribution
  surface, shared helpers, feature-selection/GAM surface and connected guides.
  Existing numerical limitations remain documented; passing workaround tests
  does not establish that those implementations have been repaired.

## Underlying issues and deduplication

A fresh all-state inventory exhausted all 241 open/closed issue-and-PR entries
(126 issues), followed by live targeted searches immediately before publication.

### New issue: native Torch model-X seeding

[#242](https://github.com/TheHiddenObserver/statgpu/issues/242) records the
missing `generator=gen` on the native Torch model-X random draw. Independent
public-API replay on Torch 2.14.1+cpu returned maximum same-seed W difference
`7.365555601168252`; resetting the global RNG while changing the requested seed
returned identical W. A temporary in-memory repair restored same-seed identity
and distinct-seed draws. The production implementation is unchanged.

This reproduces with `method="corr_diff"`, so it is not the seeded Lasso cache
problem in #211, and it is not the fixed-X centering problem in #232.

### New issue: RidgeCV custom training subsets

[#243](https://github.com/TheHiddenObserver/statgpu/issues/243) records unweighted
RidgeCV complement-statistics reuse that checks validation coverage but does not
check whether each supplied training set really is the validation complement.
Valid deliberately reduced training sets can therefore be replaced silently.
Independent analytic normal equations and direct-estimator controls show alpha
1 selected instead of alpha 10, requested-fold MSE error 11.7630481 and agreement
with the unintended complement scores to 1.33e-15. Separate processes reproduce
identical output against both the starting head and the verified base.

An all-ones-weight diagnostic takes the explicit-index path and reproduces the
requested folds, but is not a general claim about every backend or CV setting.
The documented safe workflow uses an explicit external loop with the intended
training/validation rows. Strict guards identify the exact known complement-score
symptom, preserve independent analytic comparisons and expose unrelated failures
normally. This is separate from #241's direct Ridge centering cancellation and
#218's kernel-CV nonfinite candidate selection.

### Supplemental evidence on existing kernel issue

The finite-input KernelPCA failed-refit reproduction is attached to
[#236](https://github.com/TheHiddenObserver/statgpu/issues/236#issuecomment-6009698876),
whose acceptance already requires transactional/invalidation behavior and
includes the KernelPCA consumer. No duplicate issue was opened.

With a fitted linear kernel on `arange(6)` and query `[0.5, 2, 4]`, the valid
coordinates are `[2, 0.5, -1.5]`. Refitting constant finite data raises the
expected no-positive-eigenvalue error but retains `_fitted=True`, old `X_fit_`
and projection arrays while changing the kernel mean from 6.25 to 400. The
same query then gives `[-0.5, -2, -4]`. The defect also reproduces on the exact
comparison base and for both default `alpha=1` and `alpha=0`.

## Verification and preservation boundaries

The local complete CPU suite initially passed after installing the two missing
validation dependencies, statsmodels and rfc3339-validator, in an external
validation path. This resolves the earlier four local dependency failures;
repository dependency metadata and original assertions were not relaxed.
The final full-suite totals, content fingerprint, Git tree and hosted exact-head
checks are recorded in the PR description after final candidate verification.
Focused suites execute both language examples, inspect runtime signatures/help,
check analytic/reference behavior and distinguish narrow known-defect markers
from unrelated failures. Expected failures are tracked defects, not numerical
correctness passes; repaired behavior must require removing obsolete strict
markers. A broader exploratory whole-package high-signal Ruff scan reports
45 pre-existing legacy/type-annotation findings; the normalized findings are
identical on the verified base and candidate. They are outside this docs-only
repair. The repository-prescribed scoped static gate and standard Ruff on the
new tests are checked separately.

Every production Python file is compared with the independently blob-verified
base. Ordinary initial module/class/function docstrings are excluded for the
usual AST comparison. One existing exception remains a plain literal assigned
to `PenalizedCoxPHModel.__doc__`; its target, surrounding control flow and
assignment order are unchanged. Thus numerical/dispatch/state logic is
preserved, but strict AST identity after stripping only initial docstrings is
not claimed.

The final independent review is over the entire frozen candidate, not just this
cycle's incremental patch. Every published blob and the complete remote tree
are checked before the non-force branch update, then the remote head and its
hosted CI are checked through terminal state. Synthetic merge jobs must use the
same candidate tree. Previous-head CI is historical after a source change.

No physical CuPy/Torch CUDA, R, benchmark, empirical FDR or coverage-calibration
run is claimed. Optional Torch execution here is CPU evidence. No merge,
auto-merge or draft/ready-state change is part of this work.
