# PR168 fifth documentation/runtime-contract review

## Scope and source

This is a fresh full-PR documentation, installed-help, example and directly
connected runtime-contract review. Original remote head:
`bf9400063ee4f92e54091be50dc3a4e6dbea7c6e`; comparison base:
`3fba9af81db8624ab6e882cb153e7ede7ead7f63`; original tree:
`090eab2065ef1ee65a68145d7170e2c21e0783a4`.

All 158 originally changed paths are in scope, plus the bounded documentation,
docstring and test repairs described here. The starting materialized export
was verified against every present tracked Git blob in the exact remote tree;
it is not described as a complete Git checkout or a complete local test suite.
Prior clean verdicts and CI counts are historical. The final PR description
records the exact post-fix candidate fingerprint, publication tree, independent
review result and current-head hosted evidence.

Change classification is documentation/public-contract repair with regression
coverage. Active dimensions are API, formulas, solver/refit and inference
interpretation, backend boundaries, examples, failure semantics and evidence
provenance. No numerical implementation, executable dispatch, dependency,
release status, PR merge or draft/ready transition is included.

## Documentation corrections

- Repaired the README Quick Start: native Poisson sampling accepts a scalar
  mean, whereas the old example supplied one mean per observation and raised
  `TypeError`. A seeded NumPy sampler now generates the heterogeneous means;
  a modest explicit CPU CV grid keeps the standalone example runnable.
- Completed the method-inventory Poisson example, the GPU-cleanup example's
  inputs and the Torch dtype snippet's import. GPU examples are statically
  checked; no physical accelerator execution is inferred from that check.
- Reconciled the README benchmark attribution with its cited reports. Both
  identify Tesla P100 hardware, not the displayed RTX 4090; workloads and
  reference algorithms also differ. The unsupported table is retained in
  [the historical attribution record](../references/readme-historical-benchmarks.md)
  rather than relabeled as evidence for another device or current source.
- Expanded ordinary/generic/CV GLM installed help and reader/API guidance,
  including valid final-refit inference result access, failed-refit state,
  solver-specific behavior and estimation-only boundaries. The weighted
  ElasticNet diagnostic example now actually enables the inference whose
  training diagnostic it discusses.
- Completed SplineTransformer constructor/method/fitted-state help, held-out
  transform reuse, custom-knot boundaries and sample-weight limitations;
  filled bandwidth-selector/result-container help and exported Cox numerical
  error guidance.
- Added finite learned-array and prediction/feature checks for current kernel
  overflow behavior. Increasing regularization does not repair a nonfinite
  kernel matrix.
- Documented the UMAP mean-excess-distance graph bandwidth and fuzzy-union
  formulas, independently of its previously disclosed layout-force variant;
  neither graph construction nor optimization is claimed equivalent to
  reference umap-learn.
- Clarified that NMF/MiniBatchNMF inverse reconstruction accepts finite signed
  coordinates and computes their matrix product, although observation
  fit/encoding requires nonnegative data.
- Completed shared parameter-management help, corrected the installed
  Stouffer-weight overflow explanation to the shared raw normalization sum,
  and defined the noncircular moving-block bootstrap and its full-length
  zero-width-interval degeneracy.

EN/CN pages describe current user contracts and practical alternatives. Issue
tracking, audit chronology and source-specific validation stay in this record,
the PR description and the trackers rather than learner pages.

## Separately tracked implementation defects

### New #236: nonfinite kernel fits

[Issue #236](https://github.com/TheHiddenObserver/statgpu/issues/236) records
independent duplicate reproductions of finite polynomial-kernel overflow.
With `X=np.arange(8.)[:,None]` and `degree=200`, both KernelRidge and a seeded
two-landmark Nystroem fit can publish `_fitted=True` and NaN learned arrays and
outputs. Applicable nonfinite kernel controls can also escape validation.

This differs from #218's singular zero-alpha CV selection, #220's finite RBF
translation error and #231's device routing. Closed #81's finite-input guard
does not validate computed kernels or decompositions. Four strict expected
failures specify finite output or explicit rejection; they do not count the
existing behavior as correct.

### New #237: failed ordinary GLM refits

[Issue #237](https://github.com/TheHiddenObserver/statgpu/issues/237) records
auto/IRLS/FISTA refits that retain old coefficients while changing row-count or
formula/intercept state before an error. The weighted Poisson fixture changes
reported n from 30 to 5, likelihood from about -8.49815 to -1.41636, and AIC
from about 22.99631 to 8.83272 after rejecting a response-length mismatch.
A rejected no-intercept formula refit also changes predictions. The existing
explicit Newton/L-BFGS transaction wrapper preserves the earlier state.

This differs from GAM-specific #182/#189 and ElasticNet-initialization #183.
A repair may preserve the entire previous fit or invalidate it coherently;
the documentation recommends constructing a separate candidate and replacing
the live model only after successful fitting.

### New #238: large-penalty GAM nullspace shrinkage

[Issue #238](https://github.com/TheHiddenObserver/statgpu/issues/238) records
independent constant-response and diagonal-helper counterexamples: trace-scaled
jitter grows with lambda and adds a substantive ridge penalty to the nominally
unpenalized intercept/nullspace. A constant response of 5 is predicted as about
4.630 at lambda 1e10 and 0.556 at 1e12, despite an exact zero-objective constant
solution. The first value lies in the built-in GCV grid; this fixed-lambda
reproduction does not claim GCV selects it. User guidance now checks training
mean preservation and directs affected analyses to independently validated
nullspace-preserving solvers. Changing lambda changes the model, not the defect.
The original EDF helper help was also corrected to the tracked stabilized
operator, clipping and failure fallback.

### Independent review repairs

The first frozen full-PR audit additionally corrected ordinary Gaussian GLM's
normal/z inference distinction from nonrobust LinearRegression's Student-t
reference, completed CoxPHCV packed-target help, removed a contradictory
CuPy-only FAQ and refined Chinese R² terminology. A later full pass also
reconciled low-level kernel input help with its required backend arrays,
NumPy chi-squared conversion, callable-owned inputs/xp and the disclosed
cosine-denominator stabilization. Its exception-injection audit
found broad expected-failure markers that could hide unrelated setup/runtime
errors; those markers are narrowed to specific known symptoms while preserving
strict XPASS cleanup and independent failure detection.

### Existing summary architecture gap

PenalizedGLM_CV can complete supported final-refit inference and still raise
`AttributeError` from `summary()` because the selected generic estimator lacks
that method. This remains under [#154](https://github.com/TheHiddenObserver/statgpu/issues/154),
whose scope already includes CV delegation and result/summary contracts; no
duplicate issue was opened. The documented result-container workflow exposes
the actual final-refit inference without claiming a summary implementation.

All other previously registered numerical/API limitations remain outstanding.
No new runtime issue was established for signed NMF inverse reconstruction or
the explicitly documented UMAP graph-weight variant.

## Validation boundaries

Local runtime checks use Python 3.12.14, NumPy 2.3.5 and SciPy 1.17.0 on CPU.
Optional Torch structural checks use Torch CPU and are distinguished from
actual explicit-device execution. Every materialized production file is
compared with a Git-blob-verified base source after removing initial
module/class/function docstrings only; executable ASTs must remain identical.

The regressions include end-to-end README and bilingual examples, runtime
signature/default inventories, analytic kernel/graph/linear-reconstruction
references, old-versus-new refit-state checks and narrow strict expected
failures for independently reproduced unresolved defects. An unexpected
failure is not converted into a general expected failure. Newly passing
intended contracts require removal/review of their strict xfail markers.

Final aggregate counts, all-path hashes, frozen-candidate re-review and hosted
workflow results are recorded at their exact source identities in the PR
description. Focused results are not added together as unique full-suite
counts. CPU, skipped checks, mock dispatch and static GPU examples are not
physical CuPy/Torch CUDA, R, performance or statistical-calibration evidence.
