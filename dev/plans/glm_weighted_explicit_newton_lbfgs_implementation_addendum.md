# #150 implementation architecture addendum

Status: IMPLEMENTATION AMENDMENT / REVIEW REQUIRED
Parent plan: `dev/plans/glm_weighted_explicit_newton_lbfgs_plan.md`
Issue: #150

## Why this addendum exists

The review-clean parent plan preferred editing the canonical ordinary-GLM source in `statgpu/linear_model/_glm_base.py` to remove the stale weighted explicit smooth-solver guard and publish fit provenance.

During implementation through the GitHub connector, `_glm_base.py` is a large core file whose Contents-API mutation requires replacing the complete file. Reconstructing that file through a remote text write creates an avoidable truncation/copy risk unrelated to the numerical change. The repository already maintains a bounded runtime-contract installation chain under `statgpu/linear_model/__init__.py` for cross-cutting compatibility and inference contracts.

For this implementation, the ordinary-GLM public-boundary repair is therefore expressed as a narrowly scoped installer in:

`statgpu/linear_model/_glm_weighted_explicit_solver_contract.py`

This is an implementation-mechanism amendment. The statistical target, analytic-weight normalization, public solver authority, backend/device closure, consumer graph, and evidence requirements in the parent plan remain unchanged except for the explicit inverse-power-Gamma no-intercept support narrowing documented below.

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

The parent plan preferred extracting Newton's reviewed analytic-weight preparation into a single shared solver utility. The implementation instead keeps a dedicated `_prepare_lbfgs_sample_weight` in `_lbfgs.py` while reusing the canonical `_validate_sample_weight` / `_as_backend_vector` primitives from `statgpu.solvers._utils`.

This narrower implementation is intentional: PR #142 already physically validated Newton's exact uniform/non-uniform routing, backend alignment, and line-search behavior. Refactoring Newton solely for code deduplication would move an already accepted numerical path and reopen a larger regression/evidence surface that #150 does not need.

The separate L-BFGS helper is acceptable only if its externally relevant contract remains locked to Newton's established analytic-weight semantics. Fresh implementation review found that the first draft checked uniformity on the weight input's native backend/dtype before alignment, while Newton checks after alignment to the actual executed design. That ordering difference has now been removed. Both paths perform:

1. shape/finite/non-negative/positive-total validation;
2. alignment to the executed design backend/device/dtype;
3. the same historical floating-point `allclose` uniformity rule on that aligned vector;
4. normalization of uniform/effectively-uniform weights to the historical unweighted path;
5. backend-native retention of genuine non-uniform weights.

The remaining shared contract is:

- normalized objective semantics are `sum(w_i * contribution_i) / sum(w_i)`;
- positive global rescaling, uniform-weight identity, zero-weight-row equivalence, integer row replication, invalid-weight rejection, backend parity, and Newton/L-BFGS classification parity are regression-tested;
- no generic private Newton helper is refactored solely for code deduplication.

A dedicated dtype-alignment regression freezes the classification order: a weight vector that is non-uniform in its source dtype but becomes effectively uniform in the executed design dtype must be classified identically by Newton and L-BFGS.

## Inverse-link Gamma initialization amendment

Hosted family-matrix characterization exposed a pre-existing numerical weakness in the ordinary explicit smooth-solver path for `GammaRegression(link="inverse_power")`: an all-zero parameter start gives `eta = 0` for every row, immediately placing the inverse-link objective on its clipping boundary. Newton can then fail its first Armijo search before evaluating a meaningful interior step. This occurs at the initialization/domain boundary rather than in the new weighted objective algebra.

To keep the reviewed Gamma inverse-link support row numerically well-defined when an intercept is fitted, the installer provides a narrow family-aware initial point:

- slopes start at zero;
- intercept starts at `1 / mean(y)` for unweighted fits;
- with genuine analytic weights, intercept starts at `1 / weighted_mean(y)` using the same normalized active weights as the fitted objective;
- the initial vector is created on the selected NumPy/CuPy/Torch backend and concrete device;
- both Newton and L-BFGS receive the same family-valid start.

This changes only the optimization starting point. It does not change the Gamma loss, link, weight normalization, `C` semantics, convergence tolerance, solver identity, or reported parameterization.

Blocking characterization for this amendment:

- deterministic genuine-nonuniform weighted inverse-link Gamma Newton and L-BFGS with an intercept must run without `ConvergenceWarning` / L-BFGS line-search-failure warning on the maintained acceptance dataset;
- global positive weight-rescaling invariance must still hold;
- CPU/GPU parity remains subject to the frozen physical validator thresholds;
- other families keep their historical solver initialization unless separately reviewed.

### Reviewed support-matrix narrowing: no-intercept inverse-power Gamma

The parent plan initially asked to prove both `fit_intercept=True` and `False` across the ordinary family/link matrix. Characterization shows that this cannot be claimed generically for **genuine non-uniform weighted** inverse-power Gamma: without an intercept there is no maintained public initialization control and no generic guarantee that an arbitrary design admits a coefficient vector with strictly positive `X @ beta` for every row. A heuristic start could therefore turn an explicit capability claim into data-dependent solver luck.

Under the parent plan's rule allowing genuine family/solver limitations to be removed by reviewed amendment, #150 narrows only this newly opened genuine-nonuniform row:

- `GammaRegression(link="inverse_power", fit_intercept=True)` + genuine non-uniform weights + explicit Newton/L-BFGS: target supported on NumPy/CuPy/Torch;
- the same **genuine non-uniform** weighted explicit smooth-solver request with `fit_intercept=False`: fail closed before numerical work with a precise `fit_intercept=True` capability error;
- uniform and historically almost-uniform weight vectors retain the established solver rule and execute the historical unweighted no-intercept path;
- historical omitted-weight no-intercept behavior is unchanged by #150 and is not redefined as part of this repair.

Fresh implementation review caught and fixed an initial guard that rejected every non-`None` weight vector, including all-one/uniform weights. Newton/L-BFGS regressions now prove exact equality with the unweighted historical path for uniform and effectively-uniform no-intercept inverse-Gamma inputs.

All other final ordinary family/link rows retain the parent plan's `fit_intercept=True/False` review requirement where the model itself is well-defined. Hosted tests freeze the precise genuine-nonuniform weighted inverse-Gamma no-intercept rejection, and the physical matrix validates the supported inverse-Gamma row with its normal intercept-bearing public construction.

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
8. **Initialization isolation** — the inverse-Gamma warm start is activated only for the reviewed inverse-power Gamma row with an intercept; only genuine non-uniform weighted no-intercept requests fail precisely, while uniform/effectively-uniform and omitted-weight historical behavior remains unchanged.
9. **Preservation routes** — ordinary `solver="auto"`, explicit IRLS, and explicit FISTA keep their prior numerical dispatch; the new provenance fields must describe those successful routes truthfully rather than changing them.

If fresh code review finds that this installer materially obscures ownership, breaks import/introspection semantics, conflicts with another runtime installer, or broadens the initialization change beyond the reviewed row, this amendment is rejected and the implementation must return to a canonical-source `_glm_base.py` patch before completion.

## Evidence impact

This amendment does not narrow any other parent-plan completion gate. Final closure still requires:

- hosted family/solver/backend-preservation tests;
- warning-free characterization of the final claimed family/solver matrix on the maintained CPU acceptance datasets;
- independent statsmodels alignment for representative weighted ordinary Binomial/Poisson Newton and L-BFGS rows;
- NumPy/CuPy/Torch coverage for the new weighted GLM L-BFGS capability;
- ordinary Newton/L-BFGS public consumer closure;
- shared penalized/CV L-BFGS regression coverage;
- evergreen EN/CN docs and changelog reconciliation;
- frozen exact-source physical CUDA validator v2 and artifact;
- fresh exact-head code review with this addendum in scope.
