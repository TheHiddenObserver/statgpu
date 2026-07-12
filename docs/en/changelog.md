# Changelog

> Language: English  
> Last updated: 2026-07-12  
> This page: Changelog  
> Switch: [Chinese](../changelog.md)

Language switch: [Chinese](../changelog.md)

## 2026-07

### Fixed and hardened (2026-07-12) — PR #79 Ridge follow-up

- Preserved statgpu's package-wide Ridge objective: average squared-error loss plus
  `(alpha/2) * ||beta||^2`, yielding `n*alpha` or `sum(w)*alpha` in exact normal
  equations. scikit-learn comparisons use the explicit mapped alpha instead of
  redefining statgpu's objective.
- Unified weighted Ridge behavior across the optimized wrapper, generic exact/FISTA/IRLS
  paths, formula fitting, CuPy/Torch exact routes, Gaussian inference, `RidgeCV`, and
  `PenalizedGLM_CV(loss="squared_error", penalty="l2")`.
- Corrected weighted centering, weighted inference design/residuals, weighted default
  alpha grids, Patsy missing-row side-array alignment, and global weight-rescaling
  invariance.
- GPU weight checks now synchronize scalar reductions only; default GPU Newton Ridge CV
  no longer constructs an unused host-side Gram cache.
- Added focused regression tests and updated Ridge documentation, validation scripts,
  and benchmarks to use the actual internal objective and explicit sklearn mapping.
- Latest validation remains `PARTIAL_REMOTE_PENDING`: all CPU/static gates pass, while
  physical CuPy/Torch CUDA numerical, memory, and performance checks remain required.

### Fixed and hardened (2026-07-11) — PR #79

- Completed an iterative full-repository review covering correctness, backend routing,
  API/statistical contracts, readability, maintainability, extensibility, performance
  risks, tests, and compliance with `dev/AGENTS.md`.
- Fixed backend/device validation, nested estimator parameters, Torch inference routing,
  UMAP fuzzy-union and RNG semantics, NNDescent neighbor validity, CV/KMeans input
  contracts, and Cox Efron observed-information orientation.
- Hardened optional GPU tests and full pytest collection; moved the remote GPU runner out
  of `dev/tests`; added Python 3.9–3.12 regression gates, a full CPU suite, package
  compilation, static-contract checks, and review-specific regression suites.
- Added `dev/reviews/pr79_full_repository_review.md`. Validation status is
  `PARTIAL_REMOTE_PENDING` until physical CuPy/Torch CUDA numerical, memory, and
  performance checks are completed.

### Added (2026-07-07)

- **Unified Inference Framework — Loss × Penalty Sandwich Engine**:
  - New module `statgpu/inference/_sandwich.py`: `compute_bread_avg`, `compute_meat_avg`,
    `assemble_cov_avg`, `m_estimation_inference` — average-scale M-estimation sandwich
    supporting `nonrobust` (model-based φ·H⁻¹/n) and `hc0`/`hc1` (robust sandwich
    H⁻¹·J·H⁻¹/n) for all Hessian-equipped losses
  - New module `statgpu/inference/_dispersion.py`: `glm_pearson_dispersion` for
    non-canonical GLMs (Gamma, IG, Tweedie), `robust_scale_dispersion` for M-estimators
  - **Expected Fisher interface**: `loss.fisher_information(X, coef, sample_weight)`
    added to LossBase (default `NotImplementedError`); implemented on `GammaLoss`
    (log-link W=1, inverse_power W=1/η²) and `TweedieLoss` (log-link W=μ^(2-p))
  - **Penalty curvature API**: `Penalty.curvature_diag(coef)` — returns P'' diagonal,
    default zeros; `L2Penalty` overrides to `α·ones`. SCAD/MCP raises `NotImplementedError`
  - **Penalized inference routing** (`penalized/_inference_mixin.py` +256 lines):
    - Sandwich for Hessian-equipped losses + L2/ElasticNet penalties
    - Oracle active-set refit for SCAD/MCP (Fan & Li 2001, conditions on selected model)
    - Bootstrapped inference entry point (phased rollout)
  - **GLM inference pipeline** (`_glm_base.py` +300 lines):
    - `compute_inference`, `cov_type` parameters on `GeneralizedLinearModel`
    - `_compute_inference()` reads fit-time metadata (penalty, solver, objective scale)
    - Aligned design matrix: intercept-first layout matching statsmodels `sm.add_constant(X, prepend=True)`
    - `summary()`, `aic`, `bic`, `loglikelihood` properties on GLM base
  - **Loss primitives** (`losses/_base.py` +79 lines):
    - `per_sample_score(X, y, coef)` — (n, p) per-observation scores for HC2/HC3/HAC
    - `score_outer(X, y, coef, sample_weight=None)` — memory-efficient score outer product
      with w_i² analytic-weight scaling for sandwich meat
  - **GLM wrapper exposure**: `compute_inference`, `cov_type` exposed on `PoissonRegression`,
    `GammaRegression`, `InverseGaussianRegression`, `NegativeBinomialRegression`,
    `TweedieRegression`
  - **QuantileRegression** (`wrappers/_quantile.py` +329 lines): standalone class with
    kernel-based inference (Powell 1991, Epanechnikov kernel + Hall-Sheather bandwidth)
    and bootstrap inference
  - Files: `statgpu/inference/_sandwich.py`, `_dispersion.py`; `statgpu/losses/_base.py`;
    `statgpu/penalties/_base.py`, `_l2.py`; `statgpu/glm_core/_gamma.py`, `_tweedie.py`;
    `statgpu/linear_model/_glm_base.py`, `penalized/_base.py`, `penalized/_inference_mixin.py`;
    `wrappers/_poisson.py`, `_gamma.py`, `_inverse_gaussian.py`, `_negative_binomial.py`,
    `_tweedie.py`, `_quantile.py`; `_ordered_logit.py`, `_ordered_probit.py`

- **Loss × Penalty × Solver Framework Guide** (EN+CN):
  - New docs: `docs/en/guides/loss-penalty-solver-framework.md` (+206),
    `docs/cn/guides/loss-penalty-solver-framework.md` (+205)
  - Complete dispatch logic: 12 losses × 10 penalties × 10 solvers
  - Auto-solver selection rules, penalty constraints, solver-penalty matrix

- **Ordered Logit/Probit — Newton-Raphson + Analytical Hessian + Inference**:
  - Replaced L-BFGS with Newton-Raphson + trust-region across all 3 backends
    - NumPy: vectorized analytical Hessian + `numpy.linalg.solve`
    - CuPy: native GPU Newton-Raphson, zero CPU round-trips for logit
    - Torch: native `torch.linalg.solve` with proper device/dtype handling
  - Convergence in 5–23 iterations for typical problems; trust-region inner loop
    (up to 20 ridge attempts per iteration) guarantees NLL decrease
  - Standardization: X internally standardized; coefficients and thresholds
    converted back to raw scale after convergence (`β_raw = β_fit / X_std`,
    `θ_raw = θ_fit + X_mean @ β_raw`)
  - Files modified: `statgpu/linear_model/_glm_base.py` (major rewrite)
    - Removed methods: `_ordered_nll_grad_fn`, `_ordered_gradient_vec` (dead code)
    - New methods: `_ordered_hessian_analytical`, `_compute_ordered_inference`,
      `_ordered_F_and_f`, `_ordered_gradient_torch`
    - Rewritten methods: `_fit_scipy_ordered`, `_fit_cupy_ordered`,
      `_fit_torch_ordered`, `_ordered_category_probs`, `predict_proba`
  - Files modified: `statgpu/glm_core/_gamma.py`, `statgpu/inference/_sandwich.py`
    (device-aware tensor creation fixes)

- **Ordered Model Inference** (`compute_inference=True`):
  - Analytical observed Hessian at MLE with proper block structure
    (β-β, β-θ, θ-θ) matching R `MASS::polr` and `ordinal::clm`
  - Standard errors via `sqrt(diag(H^{-1}))`; Wald z-statistics, two-sided p-values,
    95% confidence intervals via standard normal
  - Flat arrays `_bse`/`_zvalues`/`_pvalues`/`_conf_int`; use `_bse[:p]` for
    coefficients and `_bse[p:]` for thresholds
  - `loglikelihood`, `aic`, `bic` properties for ordered models
  - `summary()` method via `ParameterInferenceResult`
  - GPU inference: explicit `device='cuda'` or `device='torch'` now uses
    backend-native analytical Hessian inference (NumPy/CuPy/Torch); unsupported
    covariance types (`hc0`/`hc1`/`hac`) still raise `NotImplementedError`
  - Current limitations: `cov_type='nonrobust'` only, no `sample_weight`

- **Bug Fixes** (9 total, from code review):
  - **probit gradient**: `_ordered_gradient_torch` hardcoded `torch.sigmoid`;
    fixed to use `_ordered_link_derivative(family)` for correct probit dispatch
  - **probit f'(z)**: `_compute_ordered_inference` discarded correct probit f'(z)
    and recomputed using logit formula; fixed to use return value directly
  - **GPU silent fallback**: `_to_numpy()` silently converted GPU arrays to CPU;
    added `_resolve_backend` guard raising `NotImplementedError`
  - **pinv degradation**: `np.linalg.pinv` fallback silently degraded inference
    on singular Hessian; replaced with `LinAlgError` raise
  - **predict_proba double-division**: `X_scaled @ coef` where both are scaled
    (coef already divided by `X_std`); fixed to `X @ coef` on raw scale
  - **y.dtype guard**: `_fit_torch_ordered` didn't handle non-int64 torch tensors;
    added `elif y.dtype != torch.int64` check
  - **dead code**: Removed `_ordered_nll_grad_fn` and `_ordered_gradient_vec`
    (~70 lines, zero callers)
  - **loglikelihood**: Ordered model `loglikelihood` returned `nan` because
    `_loss`/`_X_design` not set; added `_final_nll` storage + property override

### Improved (2026-07-07)

- **Ordered model documentation** (EN + CN): complete rewrite with Newton-Raphson
  algorithm, analytical Hessian, inference API, parameter tables, CPU+GPU examples,
  strict vs approximate, external validation, and current limitations
- **Documentation files**: `docs/en/models/ordered.md`, `docs/cn/models/ordered.md`

### Validation (2026-07-07)

- Three-backend ordered logit benchmark: NumPy vs CuPy vs Torch single-step Hessian
  diff at machine precision (~1e-14); 24-iteration cumulative BSE diff ~4.5e-04
  due to math library divergence (`libm` vs NVIDIA `libdevice`)
- R `ordinal::clm` comparison: NLL agreement, same analytical Hessian structure
- All existing ordered model tests pass (4/4 CPU, 6 GPU skipped)

## 2026-06

### Added (2026-06-28) — PR #73

- **Loss Architecture — LossBase Extraction**:
  - Extracted `LossBase` from `GLMLoss` for quantile/robust/survival losses
  - `LossBase`: abstract base with `per_sample_value()`, `per_sample_gradient()` as single source of truth; derives `value()`, `gradient()`, `fused_value_and_gradient()` automatically
  - `GLMLoss` inherits from `LossBase` for GLM-specific features (canonical link, IRLS)
  - New loss classes: `QuantileLoss`, `HuberLoss`, `BisquareLoss`, `CoxPartialLikelihoodLoss`
  - New modules: `PenalizedQuantileRegression`, `PenalizedRobustRegression`, `PenalizedCoxPHModel`

- **Proximal IRLS-CD Solver**: New solver for quantile + SCAD/MCP
  - Algorithm: IRLS quadratic majorization + LLA nonconvex penalty + parallel diagonal majorization
  - CPU (numpy): ~3x faster than FISTA-LLA (60-120 iterations vs 1800+)
  - GPU (torch-CUDA): ~36x faster than CPU numpy for large problems (n=10K, p=500)
  - Three-backend: numpy, cupy, torch — core array operations backend-native; scalar convergence checks synchronize to host

