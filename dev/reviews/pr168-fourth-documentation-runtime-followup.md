# PR168 fourth documentation/runtime follow-up

## Scope and source

This review began from PR #168 head `fed51b4a5bfdf1e9ed65728c02b0ed50583493e4`, compared with base `3fba9af81db8624ab6e882cb153e7ede7ead7f63`. The complete original 131-path range was revisited, including all bilingual learner topics, runtime references, public help, examples and directly connected statistical consumers. The original full-range SHA-256 content fingerprint was `9eec4471d410ee9b8b919b055eed33e020aa2f4fdeb58d1a1f007e404239fd55`.

The change remains documentation, docstrings and regression tests. All production Python executable ASTs are compared with Git-blob-verified base sources after removing only initial module/class/function docstrings. Numerical algorithms, executable dispatch and dependency metadata are preserved. Historical review/CI records prove their own source, not this candidate; final published-head and validation details belong in the PR description.

## Documentation corrections

- Reconciled non-Gaussian SCAD/MCP oracle guidance with the actual refit reconstruction defect. Explicitly configured separate diagnostic refits do not correct data-dependent selection uncertainty.
- Completed runnable fixed-penalty Poisson inference reporting through `_inference_result`; the typed estimator has no `summary()` method. Clarified bootstrap tail probabilities, percentile intervals and centered-null interpretation.
- Added overflow-safe relative-weight preparation for Cauchy/Stouffer combination, retaining separate statistical calibration and probability-tail cautions.
- Distinguished raw weighted prediction scores from sparse Gaussian training diagnostics, including Lasso's directly affected weighted-use guidance. Completed Logistic precision-recall/all-one/all-zero boundaries and knockoff fast-profile consequences for tuning and selection.
- Documented valid Torch smoothing requests that currently fail on array-size checks. Two-dimensional query reshaping is a tested mitigation only for query cases; weights, absolute widths and bootstrap use the explicit NumPy alternative without silently changing the statistical request.
- Explained natural-cubic unit sensitivity and why rescaling mitigates but does not impose exact natural constraints. Improved spline introductory guidance, fitted-transform reuse and kernel constructor help.
- Added singular GMM and invalid MiniBatchKMeans initial-center precautions, plus NMF transform stopping semantics. Refined Chinese statistical and API explanations while preserving technical identifiers.

## Separately registered runtime defects

Every listed defect was reproduced independently of the documentation edits and deduplicated against all open and closed repository issues. Fresh live searches supplemented the complete inventory.

| Issue | Reproduced gap | Distinctness |
| --- | --- | --- |
| [#223](https://github.com/TheHiddenObserver/statgpu/issues/223) | NumPy-style `.size` checks reject valid Torch weights, widths, queries and KDE bootstrap | Argument-type failures, separate from #176/#177/#179/#203 |
| [#224](https://github.com/TheHiddenObserver/statgpu/issues/224) | Natural-cubic endpoint constraints depend on coordinate units and can exclude constants | Natural second-derivative constraints, distinct from cyclic #216 |
| [#225](https://github.com/TheHiddenObserver/statgpu/issues/225) | Singular diagonal/spherical GMM publishes fitted NaN results with finite inputs and `reg_covar=0` | Genuine singular covariance, distinct from translation #196 and nonfinite inputs |
| [#226](https://github.com/TheHiddenObserver/statgpu/issues/226) | Constructor-held MiniBatchKMeans initial centers bypass finite checks | Narrow residual gap after completed #81/#87; the umbrella was not reopened |
| [#227](https://github.com/TheHiddenObserver/statgpu/issues/227) | Oracle child reconstruction loses family, regularization, solver and device settings | Wrong refit model from raw code-object introspection, separate from method-resolution work |
| [#228](https://github.com/TheHiddenObserver/statgpu/issues/228) | Individually finite combination weights overflow their sum, changing p-values | Relative-weight normalization, distinct from probability-tail #174 and significance-level #210 |
| [#229](https://github.com/TheHiddenObserver/statgpu/issues/229) | Sparse Gaussian training R²/adjusted-R²/F diagnostics use re-centered working response totals | Correctly weighted evaluation score is unaffected; distinct from #187/#188/#208 |
| [#230](https://github.com/TheHiddenObserver/statgpu/issues/230) | ElasticNetCV with two/three rows reads missing result-schema keys | Small-sample schema, distinct from automatic-grid #207 |

Retaining a coherent last-successful knockoff selection after an explicitly failed refit was considered separately. No mixed state, false success or configuration-update retention was demonstrated; retention alone was not filed as a defect. The existing user warning remains accurate. Missing generic/typed `summary()` belongs to the existing architecture issue [#154](https://github.com/TheHiddenObserver/statgpu/issues/154); the broken example was corrected without creating a duplicate.

## Further independent whole-candidate review

A fresh complete-candidate audit at fingerprint `3542fd14a877c9020daaf5a1681f7fb8f1be87c291930c67a5e915e21c1140e9` identified additional cross-page, API, mathematical-scope, teaching and language findings. The subsequent local corrections cover:

- Lasso's default-intercept/weighted objective, complete runnable first workflow and direct API, inference-result layout without an intercept, simultaneous-inference host boundary and ineffective direct solver controls.
- Fixed-X Gaussian response assumptions and the distinction between raw and nuisance-projected knockoff geometry; a centered supplied-pair demonstration is deliberately not presented as an arbitrary-data repair.
- Current explicit-device exceptions in KDE, kernel regression and SplineTransformer, with actual array-placement checks and the intended strict device convention preserved.
- Callable-kernel `xp=None` handling, retained KernelPCA/Nystroem dimensions and scaled eigenvectors, plus training/held-out interpretation in a learner-oriented kernel workflow.
- Generic GLM provenance/oracle counterclaims, the shared combination-weight normalization explanation, public NNDescent inventory, approximate PCA whitening, Chinese reference headings and reciprocal language navigation.

Five further underlying gaps were independently reproduced and remain separate from documentation repair. They were registered after a fresh open/closed-issue deduplication check on 2026-10-06:

1. [#231](https://github.com/TheHiddenObserver/statgpu/issues/231): KDE/kernel-regression explicit Torch accelerator requests can run entirely on Torch CPU; SplineTransformer can let a Torch CPU input override an explicit accelerator request. Successful CPU reproduction does not constitute physical GPU validation.
2. [#232](https://github.com/TheHiddenObserver/statgpu/issues/232): Automatic fixed-X construction does not preserve the response-centering projection. A two-row, one-feature example has equal raw Grams but projected Grams 1 versus approximately zero, producing positive W for either nonconstant centered response direction. This is a null-sign-symmetry counterexample, not measured FDR exceedance; knockoff+ selects nothing in that one-feature example.
3. [#233](https://github.com/TheHiddenObserver/statgpu/issues/233): Direct sparse Gaussian `stopping="kkt"` is stored but not used to select the stopping rule. On an ill-scaled analytic fixture, Lasso/ElasticNet FISTA stops after two movement checks with KKT violations about 0.40/0.45. Specialized path/CV helpers must be distinguished from the selected final direct refit.
4. [#234](https://github.com/TheHiddenObserver/statgpu/issues/234): Public Lasso `admm_rho` is stored while unified ADMM receives an initial rho of 1.0. Adaptation depends on the solver route; the squared-error Cholesky route disables it. Recording actual dispatched arguments establishes the ignored constructor setting; merely comparing equal fitted optima would not.
5. [#235](https://github.com/TheHiddenObserver/statgpu/issues/235): Knockoff filters and selectors accept NaN q and return a successful empty selection with estimated FDR zero. A valid centered supplied pair isolates the target-validation issue from construction, threshold ties and cache defects. Finite-q workflows and narrow intended-rejection tests cover the public function/selector routes and both threshold rules.

No mixed state or false success was found in the separately investigated failed-knockoff-refit retention behavior, so no defect was registered for retention alone. The final complete candidate requires a new content-specific re-review; the before-state audit does not accept later edits.

## Final cross-page and regression reconciliation

The next complete-candidate pass also corrected ordinary GLM's solver-dependent C objective: default ordinary IRLS uses a slope penalty with gradient beta/(2C), whereas explicit ordinary Newton/L-BFGS/FISTA are unpenalized. The learner equations now distinguish unpenalized data-fit loss from both ordinary IRLS's C control and penalized GLM alpha, state intercept handling, and align selected defaults. CPU weighted/unweighted Poisson score equations independently verify those distinctions. README and both Torch guides link the current device exceptions without weakening correctly strict LinearRegression behavior.

The device-consumer inventory additionally includes KernelPCA/Nystroem. Their unavailable-CUDA checks genuinely raise on this CPU environment. In a separate fresh-process availability-only mock, the normal resolver reports a CUDA-configured Torch backend while fit/transform from NumPy or Torch-CPU inputs retains CPU tensors. That is source-traced/mocked routing evidence, not physical CUDA validation, and is distinct from intentionally host-resident fitted reporting arrays.

Zero-weight KDE guidance now distinguishes selecting a new bandwidth after filtering from preserving an already selected numerical factor. Refitting retained positive-weight observations with `bandwidth=original.bandwidth_factor_` preserves the tested covariance and density for nine selector/factor settings; rerunning R-style string selectors can change smoothing substantially. This does not establish that a factor influenced by zero-mass rows was scientifically appropriate.

Known-gap regressions are narrowed to the intended failure kinds. KKT tests no longer require historical early stopping, and missing-rejection tests cannot swallow unrelated programming/backend exceptions. Positive controls and future valid rejection/budget outcomes are tested separately.

A supplementary CPU fault injection found that cyclic-spline SVD error handling masks an injected `LinAlgError` with a TypeError from a nested exception tuple; a later singular-value variable is also inconsistent by source inspection. No naturally occurring SVD failure was observed. This falls under the explicit failure-behavior acceptance criterion of existing open [#216](https://github.com/TheHiddenObserver/statgpu/issues/216), so it is recorded here without a duplicate tracker or an executable-production change. It is not asserted repaired by the documentation.

## Reproduction and regression evidence

New executable coverage is in:

- `dev/tests/test_pr168_linear_feature_cycle4.py`
- `dev/tests/test_pr168_inference_cycle4.py`
- `dev/tests/test_pr168_survival_smoothing_cycle4.py`
- `dev/tests/test_pr168_unsupervised_cycle4.py`
- `dev/tests/test_pr168_linear_independent_cycle4.py`
- `dev/tests/test_pr168_smoothing_independent_cycle4.py`
- `dev/tests/test_pr168_cross_page_cycle4.py`

Ordinary assertions execute the actual bilingual workflows and verify safe alternatives. Strict expected failures assert the desired behavior for documented runtime defects; they must become unexpected passes when production repair succeeds. They are not successful numerical validation of the broken paths. Existing strict expected failures remain tied to their prior issues.

Representative independently replayed observations include oracle Logistic coefficient error about `0.2500` versus an explicit unpenalized fit; Cauchy p-value `0.5` instead of `0.0182223` under common large weight scaling; weighted ElasticNet training R² `-2.2616` versus raw weighted `0.17984`; natural-basis constant-fit error about `0.30876` on a range of length `1e6`; and fitted NaN GMM/MiniBatchKMeans outputs in the specified invalid configurations. These values identify deterministic fixtures, not universal numerical tolerances.

Executed environment: Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0 and Torch 2.14.1+cpu. Torch helper reproductions use CPU tensors. The GAM vector-query consumer and oracle child hardware-default risk include source-traced implications, not physical CUDA observations. No new physical CuPy/Torch CUDA, R or performance run was performed. CPU/doc results do not establish that the underlying numerical issues are fixed.
