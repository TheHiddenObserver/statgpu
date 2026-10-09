# PR #168 model-page structure audit

## Scope

This follow-up compares the verified source snapshot
`0691f8b33a51c98349ecd9219ad703c510e5d2b1` with the documentation and tests in this
commit. The PR comparison base was `3fba9af81db8624ab6e882cb153e7ede7ead7f63`.
All 1,486 original blobs were checked against the remote snapshot before editing.

The audit examined every file in `docs/en/models/` and `docs/cn/models/`: 27
corresponding page pairs, or 54 pages. It changed 24 pairs (48 pages); the three
navigation/reference pairs listed below remain unchanged. Dedicated guides
outside these two model directories were not rewritten by this follow-up.

The issue was the learning sequence, not a literal import-statement defect.
CoxPH's opening example combined imports, simulation, censoring, splitting,
fitting, diagnostics, summary, several predictions, and scoring; another section
then repeated data generation and a CPU fit. Similar overloaded examples and
undeclared input variables appeared elsewhere.

## Complete page inventory

Each row covers the entire English and Chinese page, not just its first fence.
“Changed” means both language versions were updated.

| Page pair | Result | Reader-facing repair or unchanged rationale |
|---|---|---|
| [Adaptive Lasso EN](../../docs/en/models/adaptive-lasso.md) / [CN](../../docs/cn/models/adaptive-lasso.md) | Changed | Replace undefined inputs with a staged train/test example; explain adaptive feature weights, intercept, tuning and inference limits. |
| [ANOVA EN](../../docs/en/models/anova.md) / [CN](../../docs/cn/models/anova.md) | Changed | Explain groups and the initial F test before Welch/Tukey alternatives; reuse inputs for optional GPU execution. |
| [Covariance EN](../../docs/en/models/covariance.md) / [CN](../../docs/cn/models/covariance.md) | Changed | Teach one LedoitWolf fit and outputs before robust/sparse/CV alternatives; clarify shared input preparation. |
| [CoxPH EN](../../docs/en/models/coxph.md) / [CN](../../docs/cn/models/coxph.md) | Changed | Separate short imports, censoring inputs, holdout, fit, risk interpretation and scoring; move survival and summary to later sections; reuse one dataset for GPU/CV. |
| [Elastic Net EN](../../docs/en/models/elastic-net.md) / [CN](../../docs/cn/models/elastic-net.md) | Changed | Separate training-only scaling, fit and prediction; introduce node-wise inference after its explanation; remove duplicated backend walkthroughs. |
| [Feature selection EN](../../docs/en/models/feature-selection.md) / [CN](../../docs/cn/models/feature-selection.md) | Changed | Explain inputs, selector fitting, selected-mask meaning and held-out transformation in order. |
| [GLM EN](../../docs/en/models/generalized-linear-model.md) / [CN](../../docs/cn/models/generalized-linear-model.md) | Changed | Stage count data, fitting, mean prediction and uncertainty; reuse declared data for tuning and explain formula setup separately. |
| [Kernel methods EN](../../docs/en/models/kernel-methods.md) / [CN](../../docs/cn/models/kernel-methods.md) | Changed | Start with kernel ridge prediction; introduce CV, KernelPCA and Nystroem in distinct explained sections sharing the data split. |
| [Knockoff EN](../../docs/en/models/knockoff.md) / [CN](../../docs/cn/models/knockoff.md) | Changed | Separate data construction, selection and the possible empty-result interpretation; retain sampling/FDR restrictions. |
| [Lasso EN](../../docs/en/models/lasso.md) / [CN](../../docs/cn/models/lasso.md) | Changed | Separate initial fitting/prediction; simultaneous inference explicitly reuses the preceding example's training data. |
| [Linear regression EN](../../docs/en/models/linear-regression.md) / [CN](../../docs/cn/models/linear-regression.md) | Changed | Separate data, fit, prediction and intervals; move covariance derivation beside inference choices; retain the distinct column-target hazard example. |
| [Logistic regression EN](../../docs/en/models/logistic-regression.md) / [CN](../../docs/cn/models/logistic-regression.md) | Changed | Explain binary labels, fitting, probability columns, classification and later uncertainty separately. |
| [Losses EN](../../docs/en/models/losses.md) / [CN](../../docs/cn/models/losses.md) | Changed | Preserve the reference role while supplying the previously undefined regression and survival inputs in short worked steps. |
| [MCP EN](../../docs/en/models/mcp.md) / [CN](../../docs/cn/models/mcp.md) | Changed | Separate imports, shaped input data, fitting and prediction before advanced optimization/inference content. |
| [Multiple testing EN](../../docs/en/models/multiple-testing.md) / [CN](../../docs/cn/models/multiple-testing.md) | Changed | Separate individual Holm decisions from the distinct Fisher global-null question and assumptions. |
| [Nonparametric EN](../../docs/en/models/nonparametric.md) / [CN](../../docs/cn/models/nonparametric.md) | Changed | Stage density, response regression and bootstrap-interval workflows; explain their different targets; reuse density data for GPU. |
| [Ordered models EN](../../docs/en/models/ordered.md) / [CN](../../docs/cn/models/ordered.md) | Changed | Add a seeded ordinal-data tutorial before reference material; separate probability interpretation, inference and Probit/GPU continuations. |
| [Poisson regression EN](../../docs/en/models/poisson-regression.md) / [CN](../../docs/cn/models/poisson-regression.md) | Changed | Separate counts, fit, expected-count prediction, coefficients and uncertainty while retaining formula and weighting examples. |
| [Quantile regression EN](../../docs/en/models/quantile.md) / [CN](../../docs/cn/models/quantile.md) | Changed | Supply a runnable response dataset; explain quantile predictions and pinball scoring before penalties, CV and inference. |
| [Ridge EN](../../docs/en/models/ridge.md) / [CN](../../docs/cn/models/ridge.md) | Changed | Put the objective before the example; separate analytic weights, fit, prediction and intervals; justify the distinct large-offset dataset. |
| [Robust regression EN](../../docs/en/models/robust.md) / [CN](../../docs/cn/models/robust.md) | Changed | Supply outlier-contaminated data and a clear fit/predict/MAE sequence before alternate losses, penalties, GPU use and inference limitations. |
| [SCAD EN](../../docs/en/models/scad.md) / [CN](../../docs/cn/models/scad.md) | Changed | Separate imports, shaped input data, fitting and prediction before advanced optimization/inference content. |
| [GAM EN](../../docs/en/models/semiparametric.md) / [CN](../../docs/cn/models/semiparametric.md) | Changed | Explain fitting, safety diagnostics and held-out prediction separately; introduce fixed-lambda refitting in smoothing selection. |
| [Splines EN](../../docs/en/models/splines.md) / [CN](../../docs/cn/models/splines.md) | Changed | Separate transformer training/query use; remove its duplicate dataset; retain distinct raw basis-family examples and limitations. |
| [Model index EN](../../docs/en/models/README.md) / [CN](../../docs/cn/models/README.md) | Unchanged | Appropriate navigation and support overview; no overloaded worked example or repeated setup. |
| [Panel index EN](../../docs/en/models/panel.md) / [CN](../../docs/cn/models/panel.md) | Unchanged | Appropriate model-choice/reference index linking dedicated panel guides; no analogous first-example defect. |
| [Unsupervised index EN](../../docs/en/models/unsupervised.md) / [CN](../../docs/cn/models/unsupervised.md) | Unchanged | Appropriate overview linking twelve dedicated model guides and API references; no overloaded tutorial. |

## Regression protection

`dev/tests/doc_examples.py` executes all Python fences within an explicitly
closed named example, in order and in one namespace. A continuation declares its
prerequisites with `example-requires`, accompanied by a reader-facing setup link.
The runner supplies no hidden arrays, imports or fitted models. Independent
execution of a later tutorial reconstructs the declared earlier tutorial once;
a reader proceeding through the page reuses that already completed context.

`dev/tests/test_doc_examples.py` checks missing/mismatched boundaries, unknown or
cyclic dependencies, ordered execution, and accidental fence truncation.
`dev/tests/test_progressive_model_examples.py` enumerates the required examples
and dependencies per page so deleting a later wrapper cannot silently reduce
execution coverage. Existing outcome-focused regressions continue to check
predictions, probability/score semantics, inference fields and CV results.

GPU snippets are compiled and their prerequisites/device requests inspected on
the CPU runner. They are not executed under a substituted backend. Unmarked
formula-interface sketches remain explicitly distinguished from runnable CPU
tutorials. Separate numerical, model-state and inference regression tests remain
in place.

## Scope and evidence limits

No production statistical implementation, dependency metadata or workflow is
changed by this follow-up. The full PR retains its earlier production changes;
this narrower documentation repair does not relabel the entire PR as docs-only.
No new physical-GPU, R, benchmark or calibration result is claimed. Existing
numerical limitations and their regression guards are retained. In particular,
this work does not implement the separately tracked issue #245.

Final exact-head test counts, independent-review outcome, remote-tree verification
and hosted checks are recorded in the PR description. Historical earlier-head
results are not evidence for a later changed candidate.
