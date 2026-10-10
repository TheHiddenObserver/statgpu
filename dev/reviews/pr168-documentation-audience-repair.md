# PR168 documentation audience repair

## Target and scope

This follow-up starts at `94758a78f3acc741ac0ba2a8c85076763117e7b0` and keeps
comparison base `3fba9af81db8624ab6e882cb153e7ede7ead7f63`. It repairs the audience
and teaching issues identified after the natural-spline guard follow-up. The
initial complete snapshot contains 1,475 blobs and 226 PR paths; every blob and
mode was checked against the exact remote tree before edits.

The procedure reads `CLAUDE.md`, `dev/AGENTS.md`, `dev/DOCUMENTATION_STYLE.md`,
and the actual `.claude/skills/code-review/SKILL.md`, `target-resolution.md` and
`review-matrix.md`. This is documentation/public-contract reconciliation with
executable documentation tests. No production numerical algorithm, public API,
dependency, workflow, Draft/Ready state, or underlying tracked defect is changed.

## Audience changes

- The bilingual Knockoff model guide now starts with a complete supplied-pair
  workflow, interprets selection and uncertainty, then introduces detailed
  limitations and advanced API material. The p=5, q=0.1 selector example explains
  that knockoff+ must select nothing: the smallest possible threshold ratio is
  1/p=0.2. It is not merely a potentially empty random example.
- Developer validation catalogs from LinearRegression, Ridge, Lasso, Elastic
  Net, Knockoff and nonparametric pages move to
  [model validation](../references/model-validation.md). Model pages keep the
  settings needed to compare objectives, intercepts, weights, covariance and
  reference distributions; each has an optional contributor link.
- GLM model pages keep supported solver/weight combinations, inverse-Gamma
  domain conditions and observable initialization errors. Armijo/private-loss
  and feasible-direction implementation details move to
  [weighted solver notes](../references/glm-weighted-solver-implementation.md).
- Cox model pages keep practical host-transfer and exhaustive-screening
  behavior. Public phase/call/origin diagnostics move to the bilingual
  [diagnostics reference](../../docs/en/reference/coxph-diagnostics.md);
  exact preparation/copy details belong in the developer Cox reference.
- UMAP separates actions and parameter meaning from advanced graph/force
  formulas. The graph-weight and standard-objective equations remain, together
  with the non-equivalence and numerical restrictions. GMM presents its complete
  example and interpretation before the detailed EM derivation. The spline
  section is titled for numerical accuracy and boundary conditions rather than
  implying a nonexistent strict/approx mode switch.

Legacy anchors are retained where sections move or names change. The changes do
not erase necessary mathematical explanations, current support limits, complete
API links, or statistical assumptions.

## Knockoff example and protection

The main controlled simulation fixes n=240, p=20, q=0.20 and `corr_diff` before
seeing results. QR reserves the intercept direction and produces centered,
orthogonal X/Xk with XᵀX=XkᵀXk=I and XᵀXk=0. The Gaussian linear response has
six signals and independent homoskedastic noise. This is a valid special design,
not a construction recipe for arbitrary observed X or a repair of the automatic
fixed-X centering defect.

The executed NumPy example selects `[0, 1, 2, 3, 4, 5, 15]`, with threshold
`1.019324242097392` and ratio `1/7`; the explanation identifies column 15 as a
false discovery. No tied absolute statistics occur. The reference selector
example returns `[]`, `(10, 0)` transformed columns, an all-false mask, infinite
threshold and the implementation's zero empty-result estimate. Neither result
is an empirical FDR calibration claim, and neither teaches post-hoc q tuning.

New tests independently check projected Gram conditions, the declared Gaussian
response generation, correlation-difference statistics, antisymmetry, complete
threshold counts, inevitable empty selection, retained warnings and anchors.
Only the old first-tutorial output assertion is migrated; its new expected
output and supplied-pair checks replace the obsolete automatically generated
design example. Other existing numerical and expected-failure guards are kept.

Placement tests check preserved formulas/limits, advanced diagnostics, links and
relocated catalogs. These complement contextual EN/CN audience review; passing
a phrase test alone is not a complete documentation review.

## Verification and evidence boundary

The final candidate is frozen by full path/mode/blob inventory, changed-path
SHA-256 and worktree fingerprint. Final full CPU, executable-example, independent
review, documentation/link/static outcomes and hosted check URLs are recorded
in the PR description against the published immutable head. Earlier-head CI is
historical and does not prove this candidate.

All `statgpu/` production blobs, workflows, model-X device/RNG tests, UMAP
known-defect guard and natural-spline guard tests remain byte-identical to the
follow-up start. The previous user-reported P100 result remains tied to clean
head `dd719917e336a9e8dc93386e13c35f861237abe4`: 133 passed, 18 deselected, zero
skipped, excluding cuda1. The relevant source/test blobs are unchanged, but this
does not claim a new-head physical GPU run or independent inspection of that
server log. No new CuPy, nondefault-GPU, R, benchmark, empirical-FDR or coverage
calibration evidence is asserted.
