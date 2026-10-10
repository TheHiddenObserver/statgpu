# PR 168 second complete documentation/runtime-contract review

## Target and scope

Original immutable head: `9086e4db6413864686a5e8d2319540f8d7ff1de7`.
Comparison base: `3fba9af81db8624ab6e882cb153e7ede7ead7f63`.

A fresh materialization was checked against every corresponding remote Git blob, and the complete remote tree was reconciled with the preceding published snapshot before editing. This is a full-PR documentation/public-contract review, including all nine learner topics, references, twelve unsupervised model destinations, runtime help and executable examples. The previous clean verdict was not reused.

The change remains documentation, docstrings and tests only. All 278 production Python executable ASTs are unchanged from the comparison base after removing only module/class/function docstrings. No runtime algorithm, dependency, merge status or Ready/Draft state is changed.

## Documentation corrections

- Fixed CoxPH's public penalty help: the implemented objective uses `penalty * ||beta||²`, without a factor of one half. An independent likelihood-score calculation checks the factor of two in its gradient.
- Completed kernel estimator and density-interval help and corrected the one-shot density helper's Gaussian-only wording.
- Made automatic-GAM examples reject a nonfinite selected GCV score before prediction and explained the all-invalid-grid failure.
- Explained zero-weight KDE log-density instability and tested filtering zero-weight rows before fitting against an analytic log-sum-exp reference.
- Scoped generic whole-cluster bootstrap to equal-size clusters; unequal-size truncation is not a valid whole-cluster resampler. Documented the mean fast path's one-dimensional limit and added a row-preserving matrix-mean example without that hint.
- Completed free-function resampling signatures and help, including Torch, actual vectorization coverage, callback probing, results and hint equivalence responsibilities.
- Explained the Torch small-shape incomplete-beta fallback's ordinary-probability and central-quantile failures, the tested explicit NumPy workaround, and the separate stable low-df Student-t two-sided helpers. Removed unqualified helper speed/accuracy claims.
- Documented ElasticNetCV's zero/near-zero L1-ratio automatic-grid limitation, Stepwise nonfinite-criterion behavior and manual evaluation-weight validation; added safe public-API workflows and regressions.
- Corrected unsupervised fitted-data ownership/help and incremental behavior descriptions; added random-initialized UMAP and training-derived centering guidance for spectral and distance failures.

## Separately registered implementation defects

Every new issue contains a deterministic executed reproduction, expected versus observed behavior, immutable source evidence, a bounded repair and explicit CPU/GPU evidence limits. Open and closed issues were searched; parent-level reproduction confirmed every candidate before filing.

- [#199](https://github.com/TheHiddenObserver/statgpu/issues/199): unequal-size generic cluster bootstrap truncates whole groups.
- [#200](https://github.com/TheHiddenObserver/statgpu/issues/200): the bootstrap mean hint mishandles matrix inputs.
- [#201](https://github.com/TheHiddenObserver/statgpu/issues/201): Torch incomplete-beta quadrature corrupts small-shape probabilities and central quantiles.
- [#202](https://github.com/TheHiddenObserver/statgpu/issues/202): automatic GAM publishes a fit when every GCV candidate is invalid.
- [#203](https://github.com/TheHiddenObserver/statgpu/issues/203): zero-weight observations corrupt Gaussian KDE log-sum-exp stabilization.
- [#204](https://github.com/TheHiddenObserver/statgpu/issues/204): UMAP sparse spectral initialization retains a null eigenvector and has unseeded eigensolver state.
- [#205](https://github.com/TheHiddenObserver/statgpu/issues/205): shared TSNE/UMAP/GPU-agglomerative pairwise distances are translation-unstable; UMAP's prior float32 cast also loses separations.
- [#206](https://github.com/TheHiddenObserver/statgpu/issues/206): Stepwise nonfinite criteria can block finite improvement or violate the feature cap.
- [#207](https://github.com/TheHiddenObserver/statgpu/issues/207): ElasticNetCV's automatic grid is unsuitable at the Ridge endpoint.
- [#208](https://github.com/TheHiddenObserver/statgpu/issues/208): shared squared-error scoring accepts negative evaluation weights.
- [#209](https://github.com/TheHiddenObserver/statgpu/issues/209): `force_vectorized=True` only enforces the first callback batch and can silently fall back later.

These are not runtime repairs. In particular #201 is distinct from #174's tail subtraction/LUT saturation, #203 from #179's distance cancellation, #205 from #197's separate KMeans distance helper and #194's TSNE bandwidth search, and #202 from #189's failed-refit state mutation. Preserving an entire old fit after early validation rejection was not classified as a new defect.

## Final independent documentation corrections

The complete post-fix review also aligned inherited-helper whole-cluster help, explained that fixed-tuning residual bootstrap refits can change the active variable set, corrected PCA inverse-transform finite-input validation, narrowed the R-dependent bandwidth-selector switch, distinguished GPU agglomerative single-linkage MST from other linkage updates, and removed the first-batch-only forced-vectorization guarantee. Focused tests cover these distinctions without repairing algorithms.

## Validation and evidence boundary

Ordinary regressions test correct formulas, supported API calls, safe workarounds and data/shape ownership. They do not require an incorrect runtime value to remain incorrect. Nine new strict expected-failure tests express intended correct Torch small-shape distribution results, in addition to the two inherited F-sampling expected failures. Skips identify unavailable capabilities; neither skips nor expected failures are numerical correctness passes.

The complete post-fix candidate is independently reviewed again. Final full-range fingerprints, fresh local counts and terminal hosted CI for the published SHA are recorded in the PR description after verification; this source record does not claim later evidence in advance. Physical CUDA/CuPy/Torch, R and performance checks were not performed, and old source-bound evidence does not certify this head.
