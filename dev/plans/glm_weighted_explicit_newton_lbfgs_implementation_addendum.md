# #150 implementation architecture addendum

Status: IMPLEMENTATION AMENDMENT / REVIEW REQUIRED
Parent plan: `dev/plans/glm_weighted_explicit_newton_lbfgs_plan.md`
Issue: #150

## Why this addendum exists

The review-clean parent plan preferred editing the canonical ordinary-GLM source in `statgpu/linear_model/_glm_base.py` to remove the stale weighted explicit smooth-solver guard and publish fit provenance.

During implementation through the GitHub connector, `_glm_base.py` is a large core file whose Contents-API mutation requires replacing the complete file. Reconstructing that file through a remote text write creates an avoidable truncation/copy risk unrelated to the numerical change. The repository already maintains a bounded runtime-contract installation chain under `statgpu/linear_model/__init__.py` for cross-cutting compatibility and inference contracts.

For this implementation, the ordinary-GLM public-boundary repair is therefore expressed as a narrowly scoped installer in:

`statgpu/linear_model/_glm_weighted_explicit_solver_contract.py`

This is an implementation-mechanism amendment only. The statistical, solver, backend, consumer, test, documentation, and evidence contracts in the parent plan remain unchanged.

## Installer scope

The installer may patch only:

- `GeneralizedLinearModel.__init__` to initialize fit-provenance fields;
- `GeneralizedLinearModel._fit_smooth_solver` to pass already validated/backend-aligned `sample_weight` to explicit Newton/L-BFGS;
- `GeneralizedLinearModel.fit` to publish truthful solver/backend/device provenance after successful completion.

It must not patch or reinterpret:

- `OrderedGeneralizedLinearModel.fit`;
- standalone `LogisticRegression` / `LogisticRegressionCV`;
- `solver="auto"` resolution;
- IRLS/FISTA algorithms;
- explicit smooth-solver `C` semantics;
- covariance estimands or bootstrap contracts.

The shared weighted-L-BFGS numerical change itself remains ordinary source code in `statgpu/solvers/_lbfgs.py` and the loss capability marker remains at the `LossBase` / `GLMLoss` abstraction layers; those are not runtime installer shims.

## Additional blocking gates introduced by this amendment

Because runtime installation changes the public method objects after import, final review must additionally prove:

1. **Idempotence** — repeated installer invocation leaves method identity unchanged.
2. **Introspection** — `inspect.signature` for `GeneralizedLinearModel.__init__`, `fit`, and `_fit_smooth_solver` remains the canonical public/source signature through `functools.wraps` / `__wrapped__`.
3. **Import-order safety** — importing `_glm_base` before `statgpu.linear_model`, or importing the package normally, yields the same installed public contract once `statgpu.linear_model` is imported.
4. **Subclass inheritance** — maintained thin ordinary-GLM wrappers inherit the installed contract without each requiring a separate monkey patch.
5. **Ordered isolation** — the ordered-model override continues to reject `sample_weight` and does not execute the ordinary smooth-solver installer path.
6. **No stale provenance** — provenance is published only after the wrapped fit succeeds; a failed refit follows the pre-existing ordinary-GLM state behavior and does not advertise an attempted solver/backend as completed work.
7. **Installer ownership** — the module is installed once from `statgpu/linear_model/__init__.py` at a documented point after existing penalized/inference installers, and does not depend on their private state.

If fresh code review finds that this installer materially obscures ownership, breaks import/introspection semantics, or conflicts with another runtime installer, this amendment is rejected and the implementation must return to a canonical-source `_glm_base.py` patch before completion.

## Evidence impact

This amendment does not narrow the parent plan's completion gates. Final closure still requires:

- hosted family/solver/backend-preservation tests;
- NumPy/CuPy/Torch coverage for the new weighted GLM L-BFGS capability;
- ordinary Newton/L-BFGS public consumer closure;
- shared penalized/CV L-BFGS regression coverage;
- evergreen EN/CN docs and changelog reconciliation;
- frozen exact-source physical CUDA validator and artifact;
- fresh exact-head code review with this addendum in scope.
