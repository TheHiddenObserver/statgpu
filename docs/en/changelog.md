# Changelog

> Language: English<br>
> Last updated: 2026-08-05<br>
> This page: Changelog<br>
> Switch: [Chinese](../cn/changelog.md)

## Unreleased — PyTorch, validation, and sklearn compatibility

### Runtime safety

- Internal iterative Torch kernels now use a centralized compile policy.
  The default avoids `reduce-overhead` CUDA Graph capture, while
  `STATGPU_TORCH_COMPILE_MODE` permits explicit `default`,
  `reduce-overhead`, or eager-only operation.  Known CUDA Graph output
  lifecycle failures fall back to eager execution once; unrelated runtime
  errors remain visible.
- Maintained public numerical entry points are checked for NaN/Inf using
  NumPy, CuPy, or Torch reductions on the selected device. The matrix includes
  fit/predict/transform, inverse-transform, scoring, initialization arrays,
  and panel identifiers while preserving formula-owned missing-row semantics.
- Formula sample weights are aligned only after Patsy selects retained rows,
  then checked for shape, finite values, non-negativity, and positive total
  weight. Torch and CuPy alignment and inference weights remain device-native.
- Gaussian GLM FISTA now profiles the intercept with weighted feature and
  response means, matching the declared weighted squared-loss objective and
  closed-form weighted least squares when the penalty is zero.
- GLM sample weights now follow one analytic-weight convention across IRLS
  ridge scaling, line search, normalized pseudo-loglikelihood, AIC/BIC,
  dispersion, and sandwich inference. Globally rescaling weights leaves fitted
  parameters and reported diagnostics unchanged.
- Every supported GLM family, including penalized and CV estimators, now
  enforces its response domain before any solver or fold dispatch, using NumPy,
  Torch, or CuPy reductions on the selected backend. Scalar GLM responses
  accept one-dimensional or single-column input and reject multicolumn or
  length-mismatched data before solver/fold dispatch.
  Active IRLS/FISTA helper compilation uses the centralized compile policy, and
  unrelated linear-algebra/device failures are no longer masked as fallback.

### Estimator and test contracts

- Exact constructor arguments are retained separately from normalized
  runtime attributes so `sklearn.base.clone` works under legacy
  scikit-learn identity checks.
- Maintained pytest modules can no longer be hidden by broad `.gitignore`
  rules; manual GPU diagnostics have an explicit directory and ownership
  policy.

Related: Issue #45, Issue #81, Issue #82, Issue #83.
## 0.2.3 — 2026-08-04

### Survival analysis

- Completed CoxPH Phase 1 with Breslow, Efron, and Exact ties; delayed-entry
  and `(start, stop]` counting-process data; shared-coefficient stratification;
  subject identifiers; and `Surv(start, stop, event)` formula input.
- Added shared NumPy, CuPy, and Torch-CUDA risk-set primitives for objectives,
  gradients, information matrices, and baseline estimation. Exact tied-event
  partitions use backend-native dynamic programming.
- Extended `CoxPHCV` held-out partial likelihood to all supported tie methods,
  delayed entry, start-stop rows, strata, and subject-grouped folds.
- Hardened Cox inference, centered risk-set numerics, log-domain baseline
  prediction, formula NA alignment, singular-information handling, CV cache
  identity, fold eligibility, selected-penalty refitting, and failed-fit state
  resets.
- Hardened L1, L2, Elastic Net, SCAD, and MCP penalized Cox estimation; removed
  the unidentified intercept; corrected Cox-specific warm starts; and made the
  Torch Efron value, gradient, and Hessian paths native.

### Cross-validation and grouped penalties

- Requested CoxPHCV two-stage and successive-halving controls now execute one
  explicit exhaustive full-precision candidate pass, preserving deterministic
  selection while avoiding repeated complete-grid fitting.
- One-shot `CoxPHCV.cv_splits` iterators are reusable across repeated fit,
  scikit-learn clone, parameter reconstruction, and pickle.
- Public Group Lasso and Adaptive Group Lasso use the generic loss-gradient and
  exact group-proximal path consistently across supported backends.

### Validation and packaging

- Hosted workflow #960 passed on final reviewed head
  `f05a44ad363b46612e956e137e2f00d040765acb`: documentation, static, full CPU,
  and Python 3.9–3.12 regression jobs all passed; the complete CPU suite reported
  1881 passed and 662 skipped.
- The final exact-head physical-GPU promotion artifact is published as
  [schema-3 evidence](https://gist.github.com/TheHiddenObserver/afdcad86a243e68a918d852b92e984a4).
  It records 134/134 passing checks, zero child and nested return codes, empty
  gate-failure arrays, clean source state before and after execution, and SHA-256
  `bd4058450def691dd29e9d78853534016c6da70c33192a97dc312d95cbe5d76d`.
- The package version is now `0.2.3`. Release-package validation checks version
  consistency, builds the pure-Python wheel and sdist, runs `twine check`,
  validates artifact contents, and smoke-installs both distributions in clean
  environments.

## Earlier history

Detailed entries through 2026-08-03 are retained in
[the archived changelog](changelog-history-through-2026-08-03.markdown).
