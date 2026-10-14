# PR 168 follow-up documentation and implementation audit

## Source and scope

This review started from PR #168 head `35e131edea5c27f1b5b6cd04564df6e41cd6b6b4`, compared with base `3fba9af81db8624ab6e882cb153e7ede7ead7f63`. The initial 72-path range was reviewed again, including the runtime APIs behind the documentation and the twelve unsupervised destination guides.

The follow-up changes documentation, public docstrings, executable documentation tests and this developer evidence. Numerical behavior is not repaired here. Constructor signatures and executable production ASTs are preserved. Earlier clean-review and CI records apply only to their original sources, not automatically to this follow-up.

## Documentation repairs

- Aligned shared helper backend selection, returned metadata, batching, cached-data use and output ownership with actual runtime behavior.
- Completed direct/CV model help and corrected formula precedence, weighted OLS diagnostics, multi-output inference boundaries and knockoff statistical assumptions.
- Corrected Cox CV fold scoring, near-tie preference, custom-grid ordering and split-iterator reuse; clarified GAM/KDE/spline parameters and failed-basis recovery.
- Removed unsupported distribution speed guarantees from public help and added a tested finite-observation density workflow.
- Replaced undefined-data examples in all twelve unsupervised guides with standalone CPU workflows, interpreted their outputs and removed benchmark/project-history inventories.
- Corrected unsupervised rank, regularization, dtype, host-transfer, automatic-neighbor and negative-sampling descriptions. Distinguished the standard UMAP reference objective from the implemented force update.
- Documented implementation limitations without presenting an unsupported option, a reference formula, a successful convergence flag or a GPU-shaped result as evidence of numerical correctness.

## Newly registered implementation defects

Each issue contains a bounded reproduction, observed versus expected behavior, immutable source links, impact, acceptance criteria and CPU/GPU evidence limits. Open and closed issues were searched before filing. Reproductions were executed independently of the documentation assertions.

- [#186](https://github.com/TheHiddenObserver/statgpu/issues/186): preserve missing observations in distribution PDF/PMF calculations
- [#187](https://github.com/TheHiddenObserver/statgpu/issues/187): linear_model: include precision-weight normalization in LinearRegression likelihood diagnostics
- [#188](https://github.com/TheHiddenObserver/statgpu/issues/188): linear_model: define safe multi-output F diagnostics for weighted LinearRegression
- [#189](https://github.com/TheHiddenObserver/statgpu/issues/189): semiparametric: make GAM failed refits transactional after basis mutation
- [#190](https://github.com/TheHiddenObserver/statgpu/issues/190): unsupervised: preserve disconnected isolated core components in DBSCAN CPU fallback
- [#191](https://github.com/TheHiddenObserver/statgpu/issues/191): unsupervised: use real backend identity for UMAP NumPy 2 CPU NNDescent and RNG dispatch
- [#192](https://github.com/TheHiddenObserver/statgpu/issues/192): backends: fix one-column NumPy scatter_add_2d used by one-dimensional UMAP
- [#193](https://github.com/TheHiddenObserver/statgpu/issues/193): unsupervised: align UMAP attraction and repulsion updates with the intended cross-entropy
- [#194](https://github.com/TheHiddenObserver/statgpu/issues/194): unsupervised: stabilize TSNE perplexity search and reject unnormalized joint affinities
- [#195](https://github.com/TheHiddenObserver/statgpu/issues/195): unsupervised: compute translation-stable PCA covariance for covariance and auto solvers
- [#196](https://github.com/TheHiddenObserver/statgpu/issues/196): unsupervised: stabilize GaussianMixture covariance updates and diagonal density under translation
- [#197](https://github.com/TheHiddenObserver/statgpu/issues/197): unsupervised: stabilize KMeans and MiniBatchKMeans distances and inertia under translation
- [#198](https://github.com/TheHiddenObserver/statgpu/issues/198): unsupervised: reject non-finite optimization and radius controls before publishing fitted results

Existing #173–#185 remain separately tracked. Closed #81 concerns observation-array finite validation; the new hyperparameter issue is narrower and distinct. Preserving an entire previous successful fit after an early rejected input and documented formula/data precedence were not misclassified as defects. No work-status changes or merges are part of this review.

## Validation and evidence boundary

Tests execute both languages' examples independently and check public signatures, numerical baselines, return schemas and supported workarounds. New ordinary regressions cover finite-input prechecks, positive-weight likelihood comparisons, separate-target diagnostics, fresh GAM recovery, full-SVD PCA, training-derived centering and rescaled t-SNE. They do not require an incorrect production value to remain incorrect. The only strict expected failures inherited in the documentation suite are the two existing F-sampling cases.

The exact final candidate fingerprint, complete local results and terminal hosted CI for the published head are recorded in the PR description after publication. This file does not claim those later checks in advance. Physical CuPy/Torch CUDA, R and performance runs were not performed in this audit; CPU and Torch CPU results are not GPU execution evidence.

