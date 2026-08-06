# Changelog

- Decoupled direct LogisticRegression training confusion metrics from ROC/PR evaluation, so accuracy, precision, recall, and F1 remain available for one-class targets while ranking metrics keep their explicit support requirements; summary renders unavailable ranking metrics as NaN.

- Kept direct LogisticRegression analytic weights device-native on CuPy/Torch fits instead of copying the full vector to NumPy solely for the CPU inference cache.

- Closed follow-up review gaps in direct LogisticRegression and penalized CV: failed fits clear partial state, single-class confusion/table metrics remain available, and custom validation losses retain analytic weights.

- Aligned direct LogisticRegression prediction contracts across NumPy, CuPy, and Torch: hard labels are integer-valued, single-column responses score without broadcasting, and non-finite decision thresholds are rejected.

- Kept fitted likelihood diagnostics independent of covariance inference, so enabling inference cannot change AIC, BIC, or pseudo-R².

- Unified CPU, CuPy, and Torch fitted log-likelihood diagnostics with the registered numerically stable LogisticLoss objective.

- Completed the code-review fix cycle for scalar GLM runtime contracts: strict binary labels and controls, transactional refits, visible convergence, and backend-consistent analytic-weight diagnostics.

- Corrected arbitrary-link Binomial IRLS, backend-native warm starts, quadratic-penalty validation, and explicit penalized-CV fallback semantics.

- Removed universal ElasticNet backend thresholds, coefficient tolerances, and fixed speedup claims that were not established for the current exact-head environment; the model guide now requires workload-specific benchmarking and dtype/solver-specific validation.

- Corrected ElasticNet/Ridge scaling documentation and added a regression test confirming that `ElasticNet(alpha, l1_ratio=0)` matches `Ridge(alpha)` under the shared average-loss convention.

- Reconciled the ElasticNet API documentation with the implementation by correcting constructor defaults, removing nonexistent parameters, and replacing stale strict/approx guidance with the actual FISTA and post-fit inference semantics.

- Completed the public ElasticNet inference contract: the standalone wrapper now exposes and forwards inference options, and ElasticNetCV honors `compute_inference=True` on its final full-data refit with NumPy/CuPy/Torch matrix tests.

- Integrated transactional CV reset with the shared public finite-input guard, so NaN/Inf refit attempts invalidate stale RidgeCV, ElasticNetCV, LogisticRegressionCV, and unified penalized-CV state before validation raises.

- Made dedicated RidgeCV, ElasticNetCV, and LogisticRegressionCV refits failure-safe: every fit attempt clears stale fitted state, and CV selections are published only after the final model refit succeeds.

- Pinned AUTO-mode RidgeCV, ElasticNetCV, and LogisticRegressionCV final refits to the backend selected during CV, preventing silent Torch/CuPy backend drift after parameter selection.

- Preserved `device='auto'` through public RidgeCV, ElasticNetCV, and LogisticRegressionCV dispatch so GPU-resident inputs retain their owning backend; LogisticRegressionCV now validates 0/1 responses without a full GPU-to-CPU copy.

- Logistic and ElasticNet CV default regularization grids now use analytic weights and satisfy integer-weight row-replication equivalence; CV GPU-array device inspection no longer masks runtime failures.

- Dedicated Ridge, ElasticNet, and Logistic CV routines now preserve explicit Torch versus CuPy backend requests, normalize Device enum values consistently, and validate analytic weights before grid generation or degenerate returns.

- Corrected analytic-weight LogisticRegression IRLS across NumPy, CuPy, and Torch: weights now enter WLS curvature rather than the working-response denominator, and weighted likelihood/inference use the same objective. Narrowed penalized-CV alpha-grid and exact CuPy Ridge fallbacks so programming, CUDA OOM, and device errors propagate.

- Completed penalized-CV fallback hardening: optional Lipschitz recovery now recognizes NumPy/CuPy/Torch rank failures consistently, while alpha-grid estimation no longer hides memory or GPU infrastructure failures.

- Preserved the declared validation objective in penalized CV: non-Gaussian losses no longer silently fall back to MSE, weighted squared-error fallback retains validation weights, and GPU infrastructure failures propagate through layered CV fallbacks.

- Narrowed GPU linear-algebra fallbacks so only genuine rank/definiteness failures use least-squares, pseudo-inverse, ridge, or zero-block recovery; CUDA OOM, device, index, and programming errors now propagate.

> Language: English<br>
> Last updated: 2026-08-06<br>
> This page: Changelog<br>
> Switch: [Chinese](../cn/changelog.md)

## Unreleased — PyTorch, validation, and sklearn compatibility

### Runtime safety

- Armijo backtracking no longer treats generic `out of range` errors as recoverable numerical trials, preserving index/device programming errors.
- Proximal-Newton now backtracks on recognized numeric-domain ValueError trials while preserving unrelated contract and runtime failures.
- Shared backend linear solves now use least-squares fallback only for recognized rank failures and preserve CUDA OOM/device RuntimeErrors.
- Shared NumPy constructors now follow floating reference dtypes like the CuPy/Torch implementations, while integer references retain float64 numerical defaults.
- FISTA-family warm starts now follow the preprocessed design, and smooth proximal-Newton weights are normalized to the active backend/device/dtype before loss evaluation.
- Newton-family Armijo backtracking now suppresses only recognized numeric-domain trial failures and propagates CUDA OOM/device/runtime infrastructure errors.
- Solver sample-weight validation now propagates backend RuntimeError failures such as CUDA OOM/device errors instead of masking them as invalid-input ValueError exceptions.
- The executable solver matrix now treats Elastic Net as non-smooth and validates its precision through FISTA rather than a smooth-only solver.
- Newton, L-BFGS, and L-BFGS-B now fail explicitly for Elastic Net and other non-smooth penalties rather than optimizing only their smooth part.
- Newton-family, L-BFGS-family, and ADMM warm starts now follow the preprocessed design backend, device, and dtype rather than retaining the caller's original array placement.
- Removed the wrong Euclidean-prox Newton shortcut that duplicated smooth penalties. Smooth objectives retain Newton; non-smooth objectives explicitly use FISTA until a Hessian-metric proximal solver exists.
- Completed ADMM's Cholesky fallback initialization and hardened L-BFGS-B feasible directions and NaN-bound validation.
- Adjacent Newton, proximal-Newton, ADMM, FISTA-BB, L-BFGS, and L-BFGS-B paths now validate weights before curvature work, narrow singular-system fallbacks, preserve dtype/device for proximal Newton and CuPy bounds, and use the correct squared-gradient Armijo slope.
- Direct solver and penalized-CV sample-weight checks now remain on the selected backend, run before weighted Lipschitz operations, reject overflowing totals, and preserve HC1 analytic-weight scale invariance.
- Internal iterative Torch kernels now use a centralized, opt-in compile policy.
  Compilation remains eager when `STATGPU_TORCH_COMPILE_MODE` is unset,
  `auto`, or `disable`. Users can explicitly select `default` or
  `reduce-overhead`; known CUDA Graph output lifecycle failures then fall
  back to eager execution once, while unrelated runtime errors remain visible.
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
  accept non-empty real one-dimensional or single-column input and reject
  non-real, multicolumn, or length-mismatched data before solver/fold dispatch.
  Design matrices and analytic sample weights now use the same backend-native
  real/finite/shape/length contract in model, formula, CV, and direct IRLS paths.
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
