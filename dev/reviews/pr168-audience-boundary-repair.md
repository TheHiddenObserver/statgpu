# PR168 documentation audience-boundary repair

## Target and scope

- Baseline PR head: `880b7b8474ab4728ed44a4556b1b64faaa6e7b55`.
- Comparison base: `3fba9af81db8624ab6e882cb153e7ede7ead7f63`.
- Change type: documentation/evidence-only repair with focused documentation tests.
- Active review axes: DOC, ARTIFACT, bounded PERF claims, and preservation of the RidgeCV known-defect warning.
- The prepared source snapshot was verified against all 1,490 remote blobs and modes. Local Git metadata was reconstructed to that exact head before editing; an earlier snapshot's index was not treated as the source of truth.

The preceding complete audience audit read 162 active public pages and identified
28 cleanup candidates (22 substantive placement findings and six narrow editorial
findings). All 28 are repaired here. Of the other 134, 128 remain byte-identical;
the only six edits clarify the already-present RidgeCV warning across two language
versions of two CV guides and the API reference. Their substantive warning is
preserved rather than moved away from users. No unsupervised page is changed.

The [per-page inventory](pr168-audience-boundary-inventory.json) records all 162
baseline pages, their original and candidate Git blob IDs, and each disposition.
Thirteen developer schema/plan documents and eight historical release/changelog
pages were separately purpose-classified by the original audience audit; their
historical contents are not represented as newly audited numerical evidence.

## Repairs and destinations

- Nine English panel pages now explain observable numerical/backend behavior and
  aligned comparisons. Exact-source acceptance tables, runner catalogs, private
  thresholds/reduction mechanisms and old P100 claims live in the
  [panel provenance record](pr168-panel-validation-provenance.md). Existing CN
  pages already express the intended user contracts; their bytes are preserved.
  The migration preserves 55 displayed formulas, 21 fenced examples and six
  parameter sections byte-for-byte. Old validation-section anchors still resolve.
- Bilingual ANOVA and covariance pages retain hypotheses, assumptions, formulas,
  backend boundaries and failures. Their maintained-test catalogs are preserved,
  with original excerpts and source links, in [model provenance](pr168-model-validation-provenance.md).
  LogisticRegression comparisons now specify ordinary unweighted `C=0`, matched
  design/intercept, covariance and convergence settings; penalized comparisons
  require objective/penalty-scale alignment. No arbitrary-weight Logit claim is made.
- User benchmark guides now explain filters, metric scopes, missing values,
  comparison conditions and generated inventory/report/data links. Contributor
  Python/npm/Playwright, parser-registration and deployment workflows are in
  [frontend maintenance](../../frontend/docs/benchmark-dashboard-maintenance.md).
  Historical PR78 planning is in [dev/plans](../plans/statgpu_benchmark_dashboard_next_phase_plan.md),
  explicitly labelled historical, with its inbound reference repaired.
- Top-level benchmark pages no longer prescribe universal sample-count crossovers
  or expected speedups from April ElasticNet experiments. They retain qualified
  workload-specific observations, actual artifact links, missing-provenance limits
  and advanced reproduction runners. A row labelled Torch is not asserted to have
  executed Torch when the associated runner uses `device='cuda'`.
  Removed remote-phase identifiers remain in the [historical artifact index](../guides/historical-benchmark-artifacts.md).
- Narrow diagnostics/solver/float32/Cox wording repairs remove test-status or
  future-author instructions while preserving actionable numerical contracts.
  The displaced rationale is in [general-guide provenance](pr168-general-guide-provenance.md).

## Necessary warning retained

RidgeCV custom-training subsets remain a known implementation issue, tracked in
[#243](https://github.com/TheHiddenObserver/statgpu/issues/243), not an inherent
Ridge model restriction. Without `sample_weight`, validation partitions covering
every row exactly once can cause a smaller supplied training set to be replaced
by the validation set's full complement. This changes fold losses and may change
selected alpha. Ordinary complementary K-folds are unaffected by that substitution.
The guides and complete reference retain the external-CV workaround: fit `Ridge`
on exactly the specified training rows and evaluate the designated validation rows.
No numerical code for #243 or #245 is changed.

## Regression protection and evidence boundaries

The existing cross-page warning guard is strengthened rather than removed. A new
focused audience-boundary suite checks the known issue's conditions/consequence/
workaround, the user/developer destinations, historical identity, dashboard data
links and the relocated maintenance count example. That example is executed from
the actual fenced snippet, so a wrong JSON nesting level cannot pass a string-only
check. Existing model execution wrappers and numerical expected-failure guards
remain intact.

Independent review found and corrected stale archive matrix links, a wrong
`generation_id` lookup in the new count example, and missing relocated historical
summary links. Validation results and the final immutable publication identity
are recorded in the PR description after the final candidate is checked. Earlier
CPU/Torch/P100 results remain evidence only for their own recorded source; this
repair makes no new physical-GPU, R, benchmark-speed or calibration claim.

## Complete active-page disposition

| Baseline public page | Disposition | Rationale |
| --- | --- | --- |
| `README.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/cn/README.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/cn/benchmarks.md` | changed | Separated dashboard use from maintenance, derived coverage from generated data, and qualified historical speed evidence. |
| `docs/cn/getting-started/quickstart.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/cn/guides/benchmarks.md` | changed | Separated dashboard use from maintenance, derived coverage from generated data, and qualified historical speed evidence. |
| `docs/cn/guides/cox-cv-staged-safety.md` | changed | Moved future screening-author guidance to developer provenance; preserved current exhaustive evaluation and diagnostics. |
| `docs/cn/guides/cross-validation-design.md` | changed | Clarified RidgeCV known implementation defect; preserved conditions, consequence and external-CV workaround. |
| `docs/cn/guides/cross-validation.md` | changed | Clarified RidgeCV known implementation defect; preserved conditions, consequence and external-CV workaround. |
| `docs/cn/guides/device-and-memory.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/cn/guides/distribution-api.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/cn/guides/implemented-methods.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/cn/guides/inference-api.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/cn/guides/inference-modes.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/cn/guides/lbfgs-float32-precision-contract.md` | changed | Removed passive test/developer-change wording; retained practical statistical and numerical guidance. |
| `docs/cn/guides/loss-penalty-solver-framework.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/cn/guides/multiple-testing-combine-pvalues.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/cn/guides/nodewise-alpha-migration.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/cn/guides/penalized-glm-inference.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/cn/guides/penalized-solver-api-migration.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/cn/guides/pytorch-backend.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/cn/guides/regression-diagnostics.md` | changed | Removed passive test/developer-change wording; retained practical statistical and numerical guidance. |
| `docs/cn/guides/solver-algorithms.md` | changed | Removed passive test/developer-change wording; retained practical statistical and numerical guidance. |
| `docs/cn/guides/solver-penalty-matrix.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/cn/models/README.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/cn/models/adaptive-lasso.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/cn/models/anova.md` | changed | Moved maintained-test/evidence inventory to model provenance; retained hypotheses and added actionable comparison settings. |
| `docs/cn/models/covariance.md` | changed | Moved maintained-test/evidence inventory to model provenance; retained hypotheses and added actionable comparison settings. |
| `docs/cn/models/coxph.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/cn/models/elastic-net.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/cn/models/feature-selection.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/cn/models/generalized-linear-model.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/cn/models/kernel-methods.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/cn/models/knockoff.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/cn/models/lasso.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/cn/models/linear-regression.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/cn/models/logistic-regression.md` | changed | Replaced passive validation claim with aligned unweighted/unpenalized comparison guidance. |
| `docs/cn/models/losses.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/cn/models/mcp.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/cn/models/multiple-testing.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/cn/models/nonparametric.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/cn/models/ordered.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/cn/models/panel.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/cn/models/poisson-regression.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/cn/models/quantile.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/cn/models/ridge.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/cn/models/robust.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/cn/models/scad.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/cn/models/semiparametric.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/cn/models/splines.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/cn/models/unsupervised.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/cn/panel/architecture.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/cn/panel/between-ols.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/cn/panel/covariance.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/cn/panel/diagnostics.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/cn/panel/fama-macbeth.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/cn/panel/first-difference-ols.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/cn/panel/fit-statistics.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/cn/panel/panel-ols.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/cn/panel/pooled-ols.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/cn/panel/random-effects.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/cn/reference/coxph-diagnostics.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/cn/reference/estimator-api.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/cn/reference/feature-selection-api.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/cn/reference/linear-model-api.md` | changed | Clarified RidgeCV known implementation defect; preserved conditions, consequence and external-CV workaround. |
| `docs/cn/reference/survival-smoothing-api.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/cn/unsupervised/README.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/cn/unsupervised/agglomerative-clustering.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/cn/unsupervised/api-reference.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/cn/unsupervised/dbscan.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/cn/unsupervised/gaussian-mixture.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/cn/unsupervised/incremental-pca.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/cn/unsupervised/kmeans.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/cn/unsupervised/minibatch-kmeans.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/cn/unsupervised/minibatch-nmf.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/cn/unsupervised/nmf.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/cn/unsupervised/pca.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/cn/unsupervised/truncated-svd.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/cn/unsupervised/tsne.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/cn/unsupervised/umap.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/cn/usage.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/en/README.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/en/benchmarks.md` | changed | Separated dashboard use from maintenance, derived coverage from generated data, and qualified historical speed evidence. |
| `docs/en/getting-started/quickstart.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/en/guides/benchmarks.md` | changed | Separated dashboard use from maintenance, derived coverage from generated data, and qualified historical speed evidence. |
| `docs/en/guides/cox-cv-staged-safety.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/en/guides/cross-validation-design.md` | changed | Clarified RidgeCV known implementation defect; preserved conditions, consequence and external-CV workaround. |
| `docs/en/guides/cross-validation.md` | changed | Clarified RidgeCV known implementation defect; preserved conditions, consequence and external-CV workaround. |
| `docs/en/guides/device-and-memory.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/en/guides/distribution-api.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/en/guides/implemented-methods.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/en/guides/inference-api.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/en/guides/inference-modes.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/en/guides/lbfgs-float32-precision-contract.md` | changed | Removed passive test/developer-change wording; retained practical statistical and numerical guidance. |
| `docs/en/guides/loss-penalty-solver-framework.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/en/guides/multiple-testing-combine-pvalues.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/en/guides/nodewise-alpha-migration.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/en/guides/penalized-glm-inference.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/en/guides/penalized-solver-api-migration.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/en/guides/pytorch-backend.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/en/guides/regression-diagnostics.md` | changed | Removed passive test/developer-change wording; retained practical statistical and numerical guidance. |
| `docs/en/guides/solver-algorithms.md` | changed | Removed passive test/developer-change wording; retained practical statistical and numerical guidance. |
| `docs/en/guides/solver-penalty-matrix.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/en/guides/statgpu_benchmark_dashboard.md` | changed | Separated dashboard use from maintenance, derived coverage from generated data, and qualified historical speed evidence. |
| `docs/en/guides/statgpu_benchmark_dashboard_next_phase_plan.md` | changed | Relocated entire historical PR78 plan to dev/plans with history banner and repaired inbound references. |
| `docs/en/models/README.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/en/models/adaptive-lasso.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/en/models/anova.md` | changed | Moved maintained-test/evidence inventory to model provenance; retained hypotheses and added actionable comparison settings. |
| `docs/en/models/covariance.md` | changed | Moved maintained-test/evidence inventory to model provenance; retained hypotheses and added actionable comparison settings. |
| `docs/en/models/coxph.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/en/models/elastic-net.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/en/models/feature-selection.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/en/models/generalized-linear-model.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/en/models/kernel-methods.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/en/models/knockoff.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/en/models/lasso.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/en/models/linear-regression.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/en/models/logistic-regression.md` | changed | Replaced passive validation claim with aligned unweighted/unpenalized comparison guidance. |
| `docs/en/models/losses.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/en/models/mcp.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/en/models/multiple-testing.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/en/models/nonparametric.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/en/models/ordered.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/en/models/panel.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/en/models/poisson-regression.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/en/models/quantile.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/en/models/ridge.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/en/models/robust.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/en/models/scad.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/en/models/semiparametric.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/en/models/splines.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/en/models/unsupervised.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/en/panel/architecture.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/en/panel/between-ols.md` | changed | Moved exact-source validation/private implementation details to panel provenance; retained mathematical and observable numerical contracts. |
| `docs/en/panel/covariance.md` | changed | Moved exact-source validation/private implementation details to panel provenance; retained mathematical and observable numerical contracts. |
| `docs/en/panel/diagnostics.md` | changed | Moved exact-source validation/private implementation details to panel provenance; retained mathematical and observable numerical contracts. |
| `docs/en/panel/fama-macbeth.md` | changed | Moved exact-source validation/private implementation details to panel provenance; retained mathematical and observable numerical contracts. |
| `docs/en/panel/first-difference-ols.md` | changed | Moved exact-source validation/private implementation details to panel provenance; retained mathematical and observable numerical contracts. |
| `docs/en/panel/fit-statistics.md` | changed | Moved exact-source validation/private implementation details to panel provenance; retained mathematical and observable numerical contracts. |
| `docs/en/panel/panel-ols.md` | changed | Moved exact-source validation/private implementation details to panel provenance; retained mathematical and observable numerical contracts. |
| `docs/en/panel/pooled-ols.md` | changed | Moved exact-source validation/private implementation details to panel provenance; retained mathematical and observable numerical contracts. |
| `docs/en/panel/random-effects.md` | changed | Moved exact-source validation/private implementation details to panel provenance; retained mathematical and observable numerical contracts. |
| `docs/en/reference/coxph-diagnostics.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/en/reference/estimator-api.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/en/reference/feature-selection-api.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/en/reference/linear-model-api.md` | changed | Clarified RidgeCV known implementation defect; preserved conditions, consequence and external-CV workaround. |
| `docs/en/reference/survival-smoothing-api.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/en/unsupervised/README.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/en/unsupervised/agglomerative-clustering.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/en/unsupervised/api-reference.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/en/unsupervised/dbscan.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/en/unsupervised/gaussian-mixture.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/en/unsupervised/incremental-pca.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/en/unsupervised/kmeans.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/en/unsupervised/minibatch-kmeans.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/en/unsupervised/minibatch-nmf.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/en/unsupervised/nmf.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/en/unsupervised/pca.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/en/unsupervised/truncated-svd.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/en/unsupervised/tsne.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/en/unsupervised/umap.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/en/usage.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
| `docs/index.md` | preserved | Preserved byte-for-byte; no audience-placement finding. |
