# Changelog

- Dedicated Ridge, ElasticNet, and Logistic CV routines now preserve explicit Torch versus CuPy backend requests, normalize Device enum values consistently, and validate analytic weights before grid generation or degenerate returns.

- Corrected analytic-weight LogisticRegression IRLS across NumPy, CuPy, and Torch: weights now enter WLS curvature rather than the working-response denominator, and weighted likelihood/inference use the same objective. Narrowed penalized-CV alpha-grid and exact CuPy Ridge fallbacks so programming, CUDA OOM, and device errors propagate.

- Completed penalized-CV fallback hardening: optional Lipschitz recovery now recognizes NumPy/CuPy/Torch rank failures consistently, while alpha-grid estimation no longer hides memory or GPU infrastructure failures.

- Preserved the declared validation objective in penalized CV: non-Gaussian losses no longer silently fall back to MSE, weighted squared-error fallback retains validation weights, and GPU infrastructure failures propagate through layered CV fallbacks.

- Narrowed GPU linear-algebra fallbacks so only genuine rank/definiteness failures use least-squares, pseudo-inverse, ridge, or zero-block recovery; CUDA OOM, device, index, and programming errors now propagate.

All notable changes to statgpu are documented here, organized by release and date.

## Unreleased — maintenance hardening

- Removed the over-broad Armijo `out of range` numerical marker so index and device programming errors propagate instead of being mistaken for recoverable trial-point domain failures.
- Made proximal-Newton Armijo backtracking treat recognized numeric-domain ValueError trials consistently with Newton while still propagating input-contract and infrastructure failures.
- Narrowed the shared backend linear-system fallback to genuine rank failures; CUDA OOM, device, and unrelated RuntimeError failures now propagate instead of being silently retried with least squares.
- Made shared NumPy zero/conversion helpers honor a floating reference array dtype, matching the existing CuPy/Torch backend contract while retaining float64 defaults for integer references.
- Normalized FISTA/FISTA-BB warm starts to the preprocessed design and converted smooth proximal-Newton sample weights to the active backend, device, and dtype before loss evaluation.
- Narrowed Newton-family Armijo trial exception handling to expected numeric-domain failures so CUDA OOM, device, and infrastructure errors remain visible to callers.
- Preserved backend RuntimeError failures (including CUDA OOM/device errors) during solver sample-weight validation instead of rewriting them as ordinary invalid-input ValueError exceptions.
- Aligned the executable loss/penalty/solver matrix with the maintained compatibility contract: Elastic Net precision is tested through FISTA, while smooth solvers are tested to reject it explicitly.
- Smooth Newton/L-BFGS solvers now reject Elastic Net and other non-smooth penalties before preprocessing instead of silently omitting their non-smooth objective component.
- Normalized Newton, proximal-Newton, L-BFGS, L-BFGS-B, and ADMM warm starts onto the preprocessed design backend, device, and dtype; added physical Torch/CuPy regression entry points.
- Removed the incorrect Euclidean-prox Newton shortcut that duplicated smooth penalty terms and solved the wrong non-smooth objective. Smooth L2/no-penalty requests retain Newton updates; non-smooth requests now explicitly use FISTA, and FISTA-LLA requires a future metric-prox capability.
- Completed ADMM's legitimate Cholesky-to-iterative fallback and kept L-BFGS-B directions/bounds feasible and backend-native.
- Hardened adjacent Newton, proximal-Newton, ADMM, FISTA-BB, L-BFGS, and L-BFGS-B contracts: validate weights before curvature work, only downgrade true singular systems, preserve dtype/device for proximal Newton and CuPy bounds, and use the correct squared-gradient Armijo slope.
- Kept direct solver and penalized-CV sample-weight checks backend-native, validated weights before weighted Lipschitz operations, rejected overflowing weight totals, and made HC1 analytic-weight inference invariant to global weight rescaling.
- Fixed Issue #45 by routing statgpu-owned Torch compilation through a
  centralized policy that avoids CUDA Graph lifecycle hazards for iterative
  solvers; compile decisions are observable, and only the known lifecycle
  failure falls back to eager execution. Performance comparison with
  `reduce-overhead` remains explicitly deferred.
- Addressed Issue #81 with backend-native finite-value validation at public
  estimator boundaries without full GPU-array transfers.
- Aligned formula sample weights after missing-row filtering across linear,
  GLM, and penalized estimators; retained Torch/CuPy weights on device; and
  corrected Gaussian GLM FISTA to use weighted centering and the intended
  weighted squared-loss intercept.
- Unified analytic-weight GLM semantics across IRLS ridge scaling, line search,
  pseudo-loglikelihood, AIC/BIC, dispersion, and sandwich inference; centralized
  active GLM Torch compilation; narrowed singular-system fallbacks; and added
  backend-native response-domain validation for every supported GLM family,
  including penalized estimators and cross-validation entrypoints; scalar
  GLMs now normalize single-column responses and reject empty, non-real,
  multicolumn, or length-mismatched responses before solver/fold dispatch;
  GLM design matrices and analytic weights now share backend-native real,
  finite, shape, length, and non-empty validation across model, CV, formula,
  and direct IRLS entrypoints.
- Addressed Issue #82 by preserving exact raw constructor arguments for
  legacy scikit-learn clone identity while retaining normalized runtime
  attributes and `set_params` bookkeeping.
- Addressed Issue #83 by making maintained `test_*.py` files visible to git,
  documenting the manual GPU diagnostic boundary, and adding maintained
  regression coverage.
## 0.2.3 — 2026-08-04

### Added
- Completed CoxPH Phase 1 with Breslow, Efron, and Exact ties; delayed-entry and `(start, stop]` counting-process data; shared-coefficient stratification; subject identifiers; and `Surv(start, stop, event)` formula input.
- Added shared NumPy, CuPy, and Torch-CUDA risk-set primitives for Cox objectives, gradients, information matrices, and baseline estimation, including backend-native dynamic programming for Exact ties.
- Extended `CoxPHCV` held-out partial likelihood to all supported tie methods, delayed entry, start-stop rows, strata, and subject-grouped folds.

### Changed
- Hardened Cox inference, numerical stability, formula NA alignment, singular-information handling, CV cache identity, fold eligibility, selected-penalty refitting, and failed-fit state resets.
- Hardened L1, L2, Elastic Net, SCAD, and MCP penalized Cox estimation; removed the unidentified intercept; corrected Cox-specific warm starts; and made Torch Efron value, gradient, and Hessian paths native.
- Standardized public Group Lasso and Adaptive Group Lasso behavior through the generic loss-gradient and exact group-proximal path across supported backends.
- Made requested CoxPHCV two-stage and successive-halving controls execute one explicit exhaustive full-precision candidate pass, avoiding repeated complete-grid fitting while preserving deterministic selection semantics.
- Made one-shot `CoxPHCV.cv_splits` iterators reusable across repeated fit, scikit-learn clone, parameter reconstruction, and pickle.

### Validation
- Hosted workflow #960 passed on the final reviewed head `f05a44ad363b46612e956e137e2f00d040765acb`: documentation, static, full CPU, and Python 3.9–3.12 regression jobs all passed; the complete CPU suite reported 1881 passed and 662 skipped.
- The final exact-head physical-GPU promotion artifact is published at https://gist.github.com/TheHiddenObserver/afdcad86a243e68a918d852b92e984a4. It records schema 3, 134/134 passing checks, zero child and nested return codes, empty gate-failure arrays, clean source state before and after execution, and SHA-256 `bd4058450def691dd29e9d78853534016c6da70c33192a97dc312d95cbe5d76d`.
- Added release-package validation that checks version consistency, builds the pure-Python wheel and sdist, runs `twine check`, validates artifact contents, clean-installs the sdist on Ubuntu, and clean-installs the same wheel on Ubuntu, Windows, and macOS.

### Packaging and release publication
- Bumped the package version to `0.2.3` in `pyproject.toml` and `statgpu/__init__.py`.
- The official wheel remains a universal `py3-none-any` artifact built with `STATGPU_NO_EXT=1`; optional Cython sources remain available in the sdist.
- Added the authoritative GitHub Release document at `.github/releases/v0.2.3.md`, a release-note completeness gate, and tag automation that publishes that file as the GitHub Release body after the PyPI job succeeds.

## Earlier history

Entries through 2026-07-27 are retained in
[`CHANGELOG-history-through-2026-07-27.md`](CHANGELOG-history-through-2026-07-27.md).
