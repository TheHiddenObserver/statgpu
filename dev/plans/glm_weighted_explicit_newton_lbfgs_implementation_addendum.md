# #150 implementation architecture addendum

Status: IMPLEMENTATION AMENDMENT / REVIEW REQUIRED
Parent plan: `dev/plans/glm_weighted_explicit_newton_lbfgs_plan.md`
Issue: #150

## Why this addendum exists

The review-clean parent plan preferred editing the canonical ordinary-GLM source in `statgpu/linear_model/_glm_base.py` to remove the stale weighted explicit smooth-solver guard and publish fit provenance.

During implementation through the GitHub connector, `_glm_base.py` is a large core file whose Contents-API mutation requires replacing the complete file. Reconstructing that file through a remote text write creates an avoidable truncation/copy risk unrelated to the numerical change. The repository already maintains a bounded runtime-contract installation chain under `statgpu/linear_model/__init__.py` for cross-cutting compatibility and inference contracts.

For this implementation, the ordinary-GLM public-boundary repair is therefore expressed as a narrowly scoped installer in:

`statgpu/linear_model/_glm_weighted_explicit_solver_contract.py`

This is an implementation-mechanism amendment. The statistical target, analytic-weight normalization, public solver authority, backend/device closure, consumer graph, and evidence requirements in the parent plan remain unchanged.

## Installer scope

The installer may patch only:

- `GeneralizedLinearModel.__init__` to initialize fit-provenance fields;
- `GeneralizedLinearModel._fit_smooth_solver` to pass already validated/backend-aligned `sample_weight` to explicit Newton/L-BFGS and, where needed, provide a family-valid initial point without changing the declared objective;
- `GeneralizedLinearModel.fit` to publish truthful solver/backend/device provenance after successful completion.

It must not patch or reinterpret:

- `OrderedGeneralizedLinearModel.fit`;
- standalone `LogisticRegression` / `LogisticRegressionCV`;
- `solver="auto"` resolution;
- IRLS/FISTA algorithms;
- explicit smooth-solver `C` semantics;
- covariance estimands or bootstrap contracts.

The shared weighted-L-BFGS numerical change itself remains ordinary source code in `statgpu/solvers/_lbfgs.py` and the loss capability marker remains at the `LossBase` / `GLMLoss` abstraction layers; those are not runtime installer shims.

## Weighted-L-BFGS sample-weight preparation amendment

The parent plan preferred extracting Newton's reviewed analytic-weight preparation into a single shared solver utility. The implementation instead keeps a dedicated `_prepare_lbfgs_sample_weight` in `_lbfgs.py` while reusing the canonical `_validated_sample_weight` primitive from `statgpu.solvers._utils`.

This narrower implementation is intentional: PR #142 already physically validated Newton's exact uniform/non-uniform routing, backend alignment, and line-search behavior. Refactoring Newton solely for code deduplication would move an already accepted numerical path and reopen a larger regression/evidence surface that #150 does not need.

The separate L-BFGS helper is acceptable only if its externally relevant contract remains locked to Newton's established analytic-weight semantics:

- identical shape/finite/non-negative/positive-total validation via `_validated_sample_weight`;
- the same historical floating-point `allclose` rule for treating effectively uniform weights as the unweighted path;
- exact equality for integer uniformity through the backend primitive;
- genuine non-uniform weights remain backend-native after alignment to the processed design;
- normalized objective semantics are `sum(w_i * contribution_i) / sum(w_i)`;
- positive global rescaling, uniform-weight identity, zero-weight-row equivalence, integer row replication, invalid-weight rejection, and backend parity are all regression-tested.

If final review finds semantic drift between Newton and L-BFGS preparation rather than mere implementation duplication, the helper split is not acceptable; extract a shared primitive before completion. No generic private helper is added solely to satisfy stylistic deduplication when behavior is already locked by tests.

## Inverse-link Gamma initialization amendment

Hosted family-matrix characterization exposed a pre-existing numerical weakness in the ordinary explicit smooth-solver path for `GammaRegression(link="inverse_power")`: an all-zero parameter start gives `eta = 0` for every row, immediately placing the inverse-link objective on its clipping boundary. Newton can then fail its first Armijo search before evaluating a meaningful interior step. This occurs at the initialization/domain boundary rather than in the new weighted objective algebra.

To keep the reviewed Gamma inverse-link support row numerically well-defined, the installer may provide a narrow family-aware initial point when an intercept is fitted:

- slopes start at zero;
- intercept starts at `1 / mean(y)` for unweighted fits;
- with analytic weights, intercept starts at `1 / weighted_mean(y)` using the same normalized active weights as the fitted objective;
- the initial vector is created on the selected NumPy/CuPy/Torch backend and concrete device;
- both Newton and L-BFGS receive the same family-valid start.

This changes only the optimization starting point. It does not change the Gamma loss, link, weight normalization, `C` semantics, convergence tolerance, solver identity, or reported parameterization.

Blocking characterization for this amendment:

- deterministic weighted inverse-link Gamma Newton and L-BFGS must run without `ConvergenceWarning` / L-BFGS line-search-failure warning on the maintained acceptance dataset;
- global positive weight-rescaling invariance must still hold;
- CPU/GPU parity remains subject to the frozen physical validator thresholds;
- other families keep their historical solver initialization unless separately reviewed.

## Physical-validator v2 amendment

The first static validator draft used a tighter per-solver tolerance and did not promote solver convergence/line-search warnings to hard acceptance failures. Hosted warning-as-error characterization showed that a physical artifact must not be able to publish `status="success"` while an accepted row reports solver stagnation.

No physical v1 artifact was executed. Before the first P100 run, the validator contract is therefore frozen as schema v2 with:

- `solver_tol = 1e-8` for the acceptance datasets, matching hosted warning-free characterization;
- unchanged CPU/GPU numerical thresholds: coefficient/intercept max absolute error `2e-5`, weight-rescaling drift `2e-6`;
- `ConvergenceWarning` and L-BFGS line-search-failure RuntimeWarning promoted to hard validator failure;
- weighted Negative-Binomial penalized and `PenalizedGLM_CV` consumers checked for coefficient/intercept parity, L-BFGS/backend/device provenance, and identical selected alpha relative to NumPy.

Because schema v1 never produced an accepted artifact, this is a pre-evidence contract correction rather than post-failure threshold tuning. Any later change after a physical v2 run must follow normal evidence-freshness/schema-review rules.

## Additional blocking gates introduced by this amendment

Because runtime installation changes the public method objects after import, final review must additionally prove:

1. **Idempotence** — repeated installer invocation leaves method identity unchanged.
2. **Introspection** — `inspect.signature` for `GeneralizedLinearModel.__init__`, `fit`, and `_fit_smooth_solver` remains the canonical public/source signature through `functools.wraps` / `__wrapped__`.
3. **Import-order safety** — importing `_glm_base` through Python's normal parent-package import semantics, or importing `statgpu.linear_model` directly, yields the same installed public contract and repeated package import does not wrap methods again.
4. **Subclass inheritance** — maintained thin ordinary-GLM wrappers inherit the installed contract without each requiring a separate monkey patch.
5. **Ordered isolation** — the ordered-model override continues to reject `sample_weight` and does not execute the ordinary smooth-solver installer path.
6. **No stale provenance** — provenance is published only after the wrapped fit succeeds; input-validation, weight-validation, and synthetic solver failures must not advertise attempted solver/backend work as completed execution.
7. **Installer ownership** — the module is installed once from `statgpu/linear_model/__init__.py` at a documented point after existing penalized/inference installers, and does not depend on their private state.
8. **Initialization isolation** — the inverse-Gamma warm start is activated only for the reviewed inverse-power Gamma row with an intercept; it must not alter other families or no-intercept solver behavior.
9. **Preservation routes** — ordinary `solver="auto"`, explicit IRLS, and explicit FISTA keep their prior numerical dispatch; the new provenance fields must describe those successful routes truthfully rather than changing them.

If fresh code review finds that this installer materially obscures ownership, breaks import/introspection semantics, conflicts with another runtime installer, or broadens the initialization change beyond the reviewed row, this amendment is rejected and the implementation must return to a canonical-source `_glm_base.py` patch before completion.

## Evidence impact

This amendment does not narrow the parent plan's completion gates. Final closure still requires:

- hosted family/solver/backend-preservation tests;
- warning-free characterization of the final claimed family/solver matrix on the maintained CPU acceptance datasets;
- independent statsmodels alignment for representative weighted ordinary Binomial/Poisson Newton and L-BFGS rows;
- NumPy/CuPy/Torch coverage for the new weighted GLM L-BFGS capability;
- ordinary Newton/L-BFGS public consumer closure;
- shared penalized/CV L-BFGS regression coverage;
- evergreen EN/CN docs and changelog reconciliation;
- frozen exact-source physical CUDA validator v2 and artifact;
- fresh exact-head code review with this addendum in scope.
