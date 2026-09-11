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

## Additional blocking gates introduced by this amendment

Because runtime installation changes the public method objects after import, final review must additionally prove:

1. **Idempotence** — repeated installer invocation leaves method identity unchanged.
2. **Introspection** — `inspect.signature` for `GeneralizedLinearModel.__init__`, `fit`, and `_fit_smooth_solver` remains the canonical public/source signature through `functools.wraps` / `__wrapped__`.
3. **Import-order safety** — importing `_glm_base` before `statgpu.linear_model`, or importing the package normally, yields the same installed public contract once `statgpu.linear_model` is imported.
4. **Subclass inheritance** — maintained thin ordinary-GLM wrappers inherit the installed contract without each requiring a separate monkey patch.
5. **Ordered isolation** — the ordered-model override continues to reject `sample_weight` and does not execute the ordinary smooth-solver installer path.
6. **No stale provenance** — provenance is published only after the wrapped fit succeeds; a failed refit follows the pre-existing ordinary-GLM state behavior and does not advertise an attempted solver/backend as completed work.
7. **Installer ownership** — the module is installed once from `statgpu/linear_model/__init__.py` at a documented point after existing penalized/inference installers, and does not depend on their private state.
8. **Initialization isolation** — the inverse-Gamma warm start is activated only for the reviewed inverse-power Gamma row with an intercept; it must not alter other families or no-intercept solver behavior.

If fresh code review finds that this installer materially obscures ownership, breaks import/introspection semantics, conflicts with another runtime installer, or broadens the initialization change beyond the reviewed row, this amendment is rejected and the implementation must return to a canonical-source `_glm_base.py` patch before completion.

## Evidence impact

This amendment does not narrow the parent plan's completion gates. Final closure still requires:

- hosted family/solver/backend-preservation tests;
- warning-free characterization of the final claimed family/solver matrix on the maintained CPU acceptance datasets;
- NumPy/CuPy/Torch coverage for the new weighted GLM L-BFGS capability;
- ordinary Newton/L-BFGS public consumer closure;
- shared penalized/CV L-BFGS regression coverage;
- evergreen EN/CN docs and changelog reconciliation;
- frozen exact-source physical CUDA validator and artifact;
- fresh exact-head code review with this addendum in scope.
