# PR168 eighth documentation and runtime-contract follow-up

## Target, procedure and scope

This fresh whole-PR review starts at remote head
`6fe0252d93914e5856c00fe597eb15027a35d876`, against comparison/merge base
`3fba9af81db8624ab6e882cb153e7ede7ead7f63`. The starting PR has 207 changed
paths, and every one of the 1,466 starting-head blobs was independently hashed
against the freshly retrieved remote tree. The writable checkout was clean at
the actual head before edits. Prior review conclusions and CI are historical.

The actual repository code-review skill, target-resolution rules, review
matrix, `dev/AGENTS.md`, and documentation style policy govern this review.
The active axes are documentation/public-contract reconciliation: complete API
references and installed help, mathematical definitions, input/output and
failure semantics, independent examples, Chinese readability, and the quality
of regression evidence. Numerical, dispatch, state and dependency changes are
out of scope. No new statistical capability or backend is being implemented.

Domain reviews cover linear/feature/inference, survival/smoothing/GAM, and all
twelve unsupervised estimators. A separate reviewer independently checks the
whole PR and its final frozen candidate, rather than accepting either the
previous cycle's verdict or only this cycle's diff. Generated fixtures,
mutation probes, execution caches and evidence logs remain outside the
publication candidate.

## Documentation and API repairs

- Rebuilt the shared CV quick starts as independent seeded CPU examples for
  Ridge, Lasso and penalized Poisson, with a separate held-out test partition.
  Added a complete ordered/gapped-fold example, explanation of the full-data
  refit, and training-fold-only preprocessing guidance. Kept estimator-specific
  caveats visible, including custom Ridge training subsets and KernelRidgeCV's
  integer-only folds, lack of sample weights, and distinct MSE/R² reporting.
- Completed the Poisson learner journey, count/rate-multiplier interpretation,
  self-contained formula usage and inference boundaries. Completed CV wrapper
  installed help and typed-GLM constructor/reference coverage, including
  family-specific controls, actual defaults and supported public methods.
- Replaced unqualified portal-level SCAD/MCP oracle-property promises with
  their practical shrinkage interpretation. Improved Chinese explanatory
  vocabulary while retaining public identifiers and standard method names.
- Linked complete kernel, pairwise-function and spline APIs directly from the
  survival/smoothing reference, so its advertised API coverage is discoverable.
  Clarified that input NaN/Inf validation does not guarantee finite quantities
  computed later from valid finite inputs.
- Corrected UMAP's mathematical reference cross-entropy to sum over distinct
  unordered pairs, excluding undefined self-pair terms. Retained the caveat
  that the current optimizer's known forces are not a verified gradient of that
  reference objective. Explained host SciPy attraction-curve fitting alongside
  existing graph/spectral host work.
- Defined TSNE conditional Gaussian affinities, entropy/perplexity,
  symmetrization and normalization, with zero self probabilities. Distinguished
  the accepted runtime range from attainable entropy targets, including tied
  nearest-neighbor constraints. Corrected the order of randomized TruncatedSVD
  projection, small SVD and sign selection.

## Expected-failure quality repairs

Older tests sometimes raised their designated known-defect exception after any
numerical mismatch. Independent mutation experiments showed that unrelated
NaNs or arbitrary finite constants could therefore be mislabeled as expected
failures. A green suite containing those xfails did not exclude these unrelated
regressions.

The repaired guards identify the actual current numerical signature against
independent controls before raising their issue-specific exception. Examples
include chi-squared gamma omission, the Torch denominator floor, RBF distance
cancellation, ignored kernel weights, zero-alpha selection, GAM trace-scaled
jitter, small-shape distribution quadrature, p-value-weight overflow, weighted
training diagnostics and oracle reconstruction. Shape, finiteness, unexpected
runtime errors and incompatible finite corruption must remain ordinary test
failures. Known limitations remain desired-behavior strict xfails, not tests
that bless the defect as correct. Corrected outputs must become strict XPASS
so the marker cannot silently survive an implementation repair.

New regression coverage executes the independent learner examples, checks live
public signatures/defaults and methods, and exercises both known-defect and
unrelated-mutation branches. It does not alter production calculations to make
tests pass or relax the statistical reference tolerance.

## New formula prediction-row defect: #244

New issue [#244](https://github.com/TheHiddenObserver/statgpu/issues/244)
tracks silent prediction-row removal in formula-fitted `LinearRegression`,
`GeneralizedLinearModel`, `PoissonRegression`, `GammaRegression`,
`InverseGaussianRegression`, `NegativeBinomialRegression` and
`TweedieRegression`. A missing query predictor can cause Patsy to drop a row,
after which the ordinary prediction interfaces return a shorter unlabelled
array. The tested penalized typed wrappers already reject this case.
Ordinary LogisticRegression has no formula-fit interface and is not included.

Independent processes reproduce the same behavior at the starting head and
comparison base. For a line `y=1+0.2*x` and query `x=[0.2, NaN]`, ordinary
LinearRegression returns only `[1.04]`. Scoring against the valid flat response
`[1,3]` silently broadcasts that one retained prediction, reporting finite
R² `-0.9216` with no warning. The intended query has two observations, so this
is not a valid two-row model-quality calculation.

The ordinary GLM method and standalone LinearRegression use separate predict
implementations but both call `FormulaParser.transform` without rejecting
row removal. #178 already tracks general score shape/broadcast validation;
that score-only repair would not fix shortened predictions across the seven
consumers. #208's evaluation-weight validation and #237's failed-refit state
mixing are also distinct. The fresh all-state inventory and live targeted
searches found no existing formula-prediction-row tracker.

Bilingual model/reference/help now state this boundary and show explicit
complete-query checks with preserved row labels. Eight strict desired-behavior
cases cover the seven prediction consumers and the score consequence. Only
agreement with independently predicted retained rows or the exact broadcast
calculation triggers the known-defect sentinel; unrelated shapes, NaNs,
finite corruption and runtime failures remain failures. No production parser,
prediction or scoring implementation was changed.

## Independently reproduced existing TSNE defect

The new bounded TSNE reproduction is distinct evidence within existing #194,
not a new issue. The fresh all-state repository inventory contained 243
issue/PR records, including 128 issues. Targeted live searches also returned
#194, #198 and #205; #194 already owns impossible/unconverged perplexity search.
Its previously empty discussion received this evidence:

<https://github.com/TheHiddenObserver/statgpu/issues/194#issuecomment-6010406444>

```python
import numpy as np
from statgpu.unsupervised import TSNE

X = np.array([[0., 0.], [.4, 1.], [1.3, .2],
              [2.1, 1.9], [3.2, .7], [4., 2.5]])
for target in (3., 5.5):
    model = TSNE(perplexity=target, max_iter=250, init="random",
                 random_state=13, device="cpu").fit(X)
    P = model._joint_probabilities(model._get_backend(), X)
    off_diagonal = P[~np.eye(len(X), dtype=bool)]
    print(target, model._fitted, P.sum(),
          off_diagonal.min(), off_diagonal.max(), model.kl_divergence_)
```

Separate processes reproduce identical results on the starting head and base:

- Target 3: fitted, normalized affinities with off-diagonal range
  `[0.0010937900272026402, 0.0934036053674117]`, KL `0.2734009138614545`.
- Target 5.5: fitted, normalized but uniformly saturated affinities, each
  approximately `1/30`, KL `0.13062412414189917`; maximum deviation from
  uniform is `3.608224830031759e-16`.

There are only five other observations, so the entropy perplexity cannot
exceed five. The current validation permits 5.5 and the search accepts an
unattainable target without reporting failure. A valid target-3 control agrees
with an independent stable entropy root and joint symmetrization. The new
strict xfail recognizes only successful finite normalized uniform saturation;
unrelated runtime, shape, nonfinite or affinity errors remain failures.

This documentation candidate does not repair the search or close #194. The formula prediction-row defect is separately tracked in new #244; no
duplicate TSNE issue was filed.

## Preservation and evidence boundaries

All production Python files are compared with the verified base. Numerical,
dispatch and state logic must be identical after removing initial docstrings,
with the one pre-existing documentary exception already present in this PR:
the plain literal assigned to `PenalizedCoxPHModel.__doc__`. That assignment's
target, position, guard and every other executable AST node are unchanged.
This cycle adds no new non-initial-docstring exception. Strict all-file AST
identity after stripping only initial docstrings is not claimed.

Full frozen-candidate validation covers the complete local test tree, focused
mutation checks, documentation contracts, local links/fragments, Python fence
parsing, in-memory compilation, repository-prescribed static checks and
whitespace. The final PR description records the exact candidate fingerprint,
full/incremental path counts, final test totals, independent review identity,
published head/tree and exact-head hosted CI after terminal completion. Earlier
or intermediate test runs are not substituted for that final evidence.

Publication is a bounded blob upload and SHA-only tree update followed by a
non-force expected-head branch update. Every remote blob/mode and the full
remote path list are compared with the frozen local candidate. No merge,
auto-merge or Draft/Ready transition is requested or performed.

All new execution is NumPy CPU or explicitly scoped Torch CPU. No physical
CuPy/Torch CUDA, R, performance, empirical FDR or interval-coverage calibration
claim is made. Skips and strict expected failures are not correctness passes.
