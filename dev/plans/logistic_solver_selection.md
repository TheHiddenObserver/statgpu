# Logistic classifier solver selection (#131)

## Baseline and scope

Baseline: master `3fba9af81db8624ab6e882cb153e7ede7ead7f63`.
This additive feature reconciles standalone logistic wrappers with an existing
backend-native L-BFGS implementation; it does not delegate to unpenalized GLM
fitting or change the shared solver. The existing docs-only PRs are independent.
Package metadata stays at 0.2.5; this feature is unreleased.

## Reconnaissance and design decisions

| Consumer | Baseline | Intended behavior |
| --- | --- | --- |
| LogisticRegression | dedicated IRLS, summed weighted likelihood + L2 | auto/irls unchanged; explicit lbfgs with same objective |
| LogisticRegressionCV CPU | direct IRLS candidates and refit | forward requested solver to every candidate and refit |
| LogisticRegressionCV GPU | batched IRLS candidates, direct IRLS refit | keep batched auto/irls; sequential backend-native lbfgs candidates/refit |
| Generic GLM and penalized GLM | separate normalized losses/solver policies | unchanged |
| Formula construction | standalone fit accepts arrays, not formula/data keywords | parsed design arrays keep ordinary intercept and solver behavior |
| BaseEstimator clone/set_params | captures raw solver controls generically | append parameter, validate constructor and each fit |

Active axes: API, backend/dtype/device/fallback, weighted objective/penalty,
solver/convergence, CV/cache/refit, existing covariance inference, documentation.
No performance claim or new inference procedure is introduced.

The design audit identified shared L-BFGS weight normalization and unpenalized
ordinary GLM delegation as incompatible with the standalone classifier objective.
A private adapter stores the validated original weights and includes the ridge
term in both value and gradient. The solver receives no separate sample weights
or penalty. It scales the *entire* objective by n or total weight to stabilize
line search; scaling only the likelihood would silently change C.

The intercept is excluded from ridge, C=0 retains the legacy unpenalized path,
and multiplying weights by a equals multiplying positive C by a. No newton or
fista selector is accepted by these wrappers. Explicit devices never fall back
to CPU. GPU L-BFGS CV computes in float64 and does not use the IRLS-only mixed
precision switch.

Existing inference runs at the selected penalized estimate and retains its
ridge curvature, covariance modes, likelihood diagnostics, and reporting
transfers. It does not correct shrinkage or CV-selection uncertainty.

Shared L-BFGS has gradient/accepted-step stopping and no returned status. The
adapter checks finite final value/gradient, preserves line-search warnings,
and publishes one estimator convergence warning. An exhausted budget requires
final gradient convergence; a failed line search is never marked converged.

## Validation and completion boundary

- Preserve default/auto/irls results exactly on deterministic data.
- Compare L-BFGS to an independently specified summed objective and derivatives.
- Cover intercepts, C=0/positive, uniform/nonuniform/zero weights and scaling.
- Cover constructor/get_params/set_params/clone, invalid controls, failure state,
  convergence/line-search metadata, and all existing covariance modes.
- Trace CV solver/device requests through candidate scoring and final refit;
  isolate cache entries by solver and ensure failures do not publish state.
- Exercise NumPy and Torch-CPU backend harness; keep conditional physical CuPy
  and Torch CUDA tests. CPU harness evidence is not physical GPU validation.
- Include final EN/CN docs in the independent immutable base/head review.
- Check hosted CI at final head after review fixes; do not merge.

Physical GPU validation must be run with `STATGPU_REQUIRE_PHYSICAL_GPU=1` on an
authorized GPU environment using `dev/tests/test_logistic_solver_contract.py`
and `dev/tests/test_logistic_cv_torch_dtype.py`. Until that evidence exists,
report `PARTIAL_REMOTE_PENDING`, not complete three-backend validation.
