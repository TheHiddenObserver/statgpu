# GeneralizedLinearModel and Penalized GLM

> Language: English  
> Last updated: 2026-09-12
> This page: Model documentation  
> Switch: [Chinese](../../cn/models/generalized-linear-model.md)

Language switch: [Chinese](../../cn/models/generalized-linear-model.md)

## Overview

`GeneralizedLinearModel` provides the common GLM entry point for Gaussian, binomial, and Poisson families. `PenalizedGeneralizedLinearModel` and its typed wrappers add L1, L2, ElasticNet, group, and adaptive-penalty hooks while keeping the public API explicit.

Use typed penalized estimators for regularized models:

```python
from statgpu.linear_model import (
    GeneralizedLinearModel,
    PoissonRegression,
    PenalizedLinearRegression,
    PenalizedLogisticRegression,
    PenalizedPoissonRegression,
)
```

`Ridge`, `Lasso`, and `ElasticNet` are sklearn-style thin wrappers over penalized Gaussian regression.

For penalized coefficient inference, see [Penalized GLM inference](../guides/penalized-glm-inference.md).

## Path

- `statgpu.linear_model.GeneralizedLinearModel`
- `statgpu.linear_model.PoissonRegression`
- `statgpu.linear_model.PenalizedGeneralizedLinearModel`
- `statgpu.linear_model.PenalizedLinearRegression`
- `statgpu.linear_model.PenalizedLogisticRegression`
- `statgpu.linear_model.PenalizedPoissonRegression`
- `statgpu.linear_model.Ridge`
- `statgpu.linear_model.Lasso`
- `statgpu.linear_model.ElasticNet`
- Internal GLM core: `statgpu.glm_core`

## Objective Function

Ordinary GLM fits minimize the average negative log-likelihood for the selected family:

$$
\min_\beta \frac{1}{n}\sum_{i=1}^n \ell(y_i, x_i^\top\beta).
$$

With analytic `sample_weight`, supported weighted GLM paths instead minimize the normalized weighted average

$$
\min_\beta
\frac{\sum_i w_i\,\ell(y_i, x_i^\top\beta)}{\sum_i w_i}.
$$

Therefore multiplying every weight by the same positive constant does not change the fitted optimum. A row with weight zero contributes nothing to the objective. Weights must be finite and non-negative, and their total must be positive.

Penalized GLM adds a penalty term:

$$
\min_\beta \frac{1}{n}\sum_{i=1}^n \ell(y_i, x_i^\top\beta) + \alpha P(\beta),
$$

or the same normalized weighted loss plus `alpha * P(beta)` when analytic weights are active.

The intercept is not penalized. `statgpu.glm_core` is intentionally GLM-specific; Cox partial likelihood, panel objectives, time-series likelihoods, and zero-inflated composite likelihoods should use their own maintained objective layers rather than being forced into `glm_core`.

## Estimating Equation

Smooth GLMs solve score equations through maintained second-order/first-order paths when available. Non-smooth penalized objectives use proximal/KKT-style optimization through FISTA/FISTA-BB.

Current direct-fit `solver="auto"` behavior includes:

| Setting | `solver="auto"` behavior |
|---|---|
| `PenalizedLinearRegression(penalty="l2")` on NumPy/CPU | exact closed-form L2 path |
| squared error + L1/ElasticNet | FISTA/FISTA-BB sparse path |
| logistic/Poisson/Gamma/Inverse-Gaussian/Tweedie/Negative-Binomial + L2 | Newton under the maintained direct-fit dispatch |
| non-convex SCAD/MCP | FISTA + LLA continuation |
| quantile | FISTA / quantile-specific path |

Important device rule: explicit `device="cuda"` stays on CuPy, explicit `device="torch"` stays on Torch CUDA, and unsupported explicit solver/backend combinations fail visibly rather than silently falling back to CPU. Formula parsing may run on CPU, but fit/predict numerical work follows the selected backend.

### Explicit Newton and L-BFGS with analytic weights

For ordinary GLMs, explicit `solver="newton"` and `solver="lbfgs"` accept genuine non-uniform analytic weights on supported GLM families. Both solvers use the same normalized weighted objective shown above; the weight vector is part of every objective/gradient evaluation, and Newton also uses it in the Hessian. L-BFGS line-search trial points are evaluated under the same weights as the search direction.

An explicit solver request remains authoritative. Supplying `sample_weight` does not silently replace Newton or L-BFGS with IRLS/FISTA, and explicit CUDA/Torch requests do not fall back to CPU. Uniform weights preserve the ordinary unweighted numerical path.

This weighted L-BFGS statement is a **GLM loss contract**, not a blanket rule for every low-level `LossBase` consumer. Direct non-GLM losses such as robust, quantile, and Cox objectives keep their own solver/weight support boundaries. Ordered GLMs also retain their separate weight support policy.

Weighted penalized smooth GLMs use the same analytic-weight convention. Their existing solver-dispatch table remains authoritative; applicable L2 rows may use Newton or L-BFGS according to the maintained direct/CV policy rather than because weights are present.

## Covariance/Inference

Generic and typed penalized GLM estimators use `inference_method="auto"` as the recommended public request. Successful inference-enabled fits expose requested/resolved/reported method provenance, the inferential target, and tuning/selection conditioning.

For supported smooth non-Gaussian L2/no-penalty models, `auto` resolves to fixed-penalty `m_estimation`. Positive L2 fits target the penalized estimating equation; no-penalty aliases are canonicalized to zero-strength L2 and target the unpenalized population parameter. Current covariance support is `nonrobust`, `hc0`, and `hc1`; HC2/HC3/HAC fail closed on this penalized non-Gaussian path.

Analytic weights are supported, and numerical inference follows the backend/concrete device that actually executed the fit. Ordinary GLM inference uses the same fitted analytic-weight convention as estimation. Non-Gaussian L1/ElasticNet coefficient inference is not productized and fails closed. SCAD/MCP oracle inference is explicit rather than selected silently by `auto`; group penalties and penalized Cox remain estimation-only.

For supported Gaussian sparse penalties, `inference_method="bootstrap"` selects an unweighted residual bootstrap with `cov_type="nonrobust"`. The design matrix and fitted tuning configuration remain fixed: each draw resamples residuals, constructs a new Gaussian response around the fitted values, and refits the same penalized model. `n_bootstrap` controls the number of refits and `bootstrap_random_state` controls reproducibility.

Bootstrap execution follows the successful fit's backend and concrete device. CPU fits use NumPy; CuPy and Torch CUDA fits keep the bootstrap refits on the same GPU device. Final inference arrays use the standard NumPy reporting boundary. Weighted residual bootstrap, robust/HC or HAC/block bootstrap, non-Gaussian bootstrap, and Cox bootstrap are not supported and fail closed.

See [Penalized GLM inference](../guides/penalized-glm-inference.md) and [Inference Modes](../guides/inference-modes.md) for the complete support matrix, resampling boundaries, and statistical interpretation.

## Parameters

| Parameter | Default | Description |
|---|---:|---|
| `family` | model-specific | GLM family, for example `"gaussian"`, `"binomial"`, or `"poisson"` |
| `penalty` | `"l2"` or model-specific | `none`, `l1`, `l2`, `elasticnet`, and reserved structured penalties |
| `alpha` | `1.0` or model-specific | Penalty strength in statgpu objective scale |
| `l1_ratio` | `None` | ElasticNet mixing parameter |
| `fit_intercept` | `True` | Whether to fit an intercept |
| `solver` | `"auto"` | Solver dispatch; see estimating-equation section |
| `device` | `"auto"` | `cpu`, `cuda`, `torch`, or `auto` depending on estimator support |
| `max_iter` | model-specific | Maximum optimizer iterations |
| `tol` | model-specific | Convergence tolerance |
| `compute_inference` | `False` on generic/typed penalized GLM | Whether to compute coefficient inference |
| `inference_method` | `"auto"` on generic/typed penalized GLM | Public inference request; specialized sparse-Gaussian wrappers retain their established defaults |
| `cov_type` | `"nonrobust"` | Covariance convention; non-Gaussian penalized M-estimation currently supports nonrobust/HC0/HC1 |
| `hac_maxlags` | `None` | Retained control; does not imply HAC support for non-Gaussian penalized M-estimation |
| `formula` | `None` | Optional patsy-style formula used with `data` |
| `data` | `None` | DataFrame used with `formula` |

Alpha scaling is explicit. Do not compare same-named parameters across frameworks without conversion:

- Ridge: `sklearn_alpha = n_samples * statgpu_alpha`
- Logistic L2: `sklearn_C = 1 / (n_samples * statgpu_alpha)`
- Poisson L2: align against sklearn `PoissonRegressor(alpha=...)`
- Poisson L1/ElasticNet: align against statsmodels `fit_regularized`

## CPU+GPU Examples

```python
from statgpu.linear_model import GeneralizedLinearModel, PenalizedLogisticRegression

# Ordinary weighted Poisson GLM with an explicit smooth solver.
weighted_pois = GeneralizedLinearModel(
    family="poisson",
    solver="lbfgs",       # or "newton"
    device="cuda",        # CuPy CUDA; use "torch" for Torch CUDA
)
weighted_pois.fit(X, y_count, sample_weight=weights)

# CPU L2 logistic path: auto selects the maintained smooth solver.
logit_cpu = PenalizedLogisticRegression(
    penalty="l2",
    alpha=0.01,
    solver="auto",
    device="cpu",
)
logit_cpu.fit(X, y_binary, sample_weight=weights)

# GPU L2 logistic path follows the same weighted objective.
logit_gpu = PenalizedLogisticRegression(
    penalty="l2",
    alpha=0.01,
    solver="auto",
    device="cuda",
)
logit_gpu.fit(X, y_binary, sample_weight=weights)
```

Formula support is optional:

```bash
pip install statgpu[formula]
```

```python
from statgpu.linear_model import LinearRegression, PenalizedPoissonRegression

lm = LinearRegression()
lm.fit(formula="y ~ x1 + x2 + C(group)", data=df)
pred = lm.predict(df_new)

pois = PenalizedPoissonRegression(penalty="l2", alpha=0.01)
pois.fit(formula="count ~ exposure + x1", data=df)
```

Formula parsing runs on CPU and is intended as a convenience layer. When `sample_weight` is supplied with a formula, weights are aligned to the rows retained by formula/missing-data processing before numerical fitting. For very large data, pass explicit `X, y` arrays.

## strict/approx difference

Penalized GLM inference is fail-closed: supported non-Gaussian L2/no-penalty rows expose fixed-penalty M-estimation with nonrobust/HC0/HC1 covariance, while unsupported loss × penalty × method rows raise instead of substituting another inferential procedure. Residual bootstrap and SCAD/MCP oracle remain deliberately narrow explicit paths.

`solver="auto"` follows the maintained direct-fit dispatch. Analytic weights do not rewrite the public solver request: explicit smooth solvers remain explicit, while `auto` continues to use the same dispatch table as the corresponding unweighted model.

`PenalizedGLM_CV` defaults to `cv_strategy="strict"`. In strict mode every fold/alpha is evaluated with the requested `max_iter` and `tol`, and GPU optimizations are limited to caching, fused kernels, and batched validation-score transfers. The optional `cv_strategy="two_stage"` mode first screens the alpha grid with relaxed CV solves, then strictly refines the candidate alphas and performs a strict final refit. Because the screening step can change alpha ranking on close CV curves, two-stage mode emits `ApproximateCVWarning` unless `acknowledge_approx=True` is passed.

```python
from statgpu.linear_model import PenalizedGLM_CV

# Default: strict CV.
strict_cv = PenalizedGLM_CV(
    loss="poisson",
    penalty="elasticnet",
    cv_strategy="strict",
    device="cuda",
)

# Opt-in approximate screening, strict candidate refinement and final refit.
fast_cv = PenalizedGLM_CV(
    loss="poisson",
    penalty="elasticnet",
    cv_strategy="two_stage",
    acknowledge_approx=True,
    refine_top_k=3,
    device="cuda",
)
```

### Survival-aware penalized Cox CV

`PenalizedGLM_CV(loss="cox_ph")` uses a separate survival path rather than the
scalar-response GLM scorer. Pass `y` as an `(n_samples, 2)` array with columns
`[time, event]`. L1, L2, ElasticNet, SCAD, and MCP are supported on NumPy,
CuPy CUDA, and Torch CUDA. The path:

- preserves the two-column target and never fits an intercept;
- scores each held-out fold with unpenalized negative Cox partial likelihood
  per row;
- selects an alpha only when every evaluable fold supplies finite evidence;
- hard-fails without publishing fitted state when no alpha is supported; and
- refits `PenalizedCoxPHModel` with `compute_inference=False`.

```python
survival_y = np.column_stack([time, event])
cox_cv = PenalizedGLM_CV(
    loss="cox_ph",
    penalty="scad",              # l1, l2, elasticnet, scad, or mcp
    alpha_grid=[0.1, 0.03, 0.01],
    cv=5,
    cv_strategy="strict",
    loss_kwargs={"ties": "efron"},
    device="cpu",                # or "cuda" / "torch"
).fit(X, survival_y)
```

`cv_strategy="two_stage"`, `sample_weight`, dictionary targets, and
post-selection coefficient inference are not supported for this Cox branch.
`cv_results_` records per-fold losses, valid-evidence counts, event counts,
failure reasons, the tie method, and the final-refit class.

## Outputs

Common fitted attributes and methods include:

- `coef_`
- `intercept_`
- `n_iter_` when exposed by the selected solver
- `fit`
- `predict`
- `predict_proba` for logistic models
- `score` where implemented
- `cv_results_` for `PenalizedGLM_CV`, including `cv_strategy_`, `cv_selected_device_`, `refined_mask`, and stage-1 scores when two-stage screening is enabled

Future unified result objects are reserved for later work and are not part of this page's public contract.

## See Also

- [Solver × Penalty Compatibility Matrix](../guides/solver-penalty-matrix.md) — full dispatch table for loss × penalty × solver combinations, CV fast paths, and inference support status.

## FAQ

- Why is `statgpu.losses` not kept as a compatibility namespace? The uncommitted `losses` layer was GLM-specific, so it was renamed to `glm_core` to avoid implying a project-wide objective system.
- Does `device="cuda"` force GPU for every GLM solver? Yes for supported GLM solver paths: CuPy is used for the core computation, or a clear error is raised. There is no silent CPU fallback for explicit CUDA/Torch requests.
- What do `sample_weight` values mean here? On supported ordinary and penalized GLM paths they are analytic objective weights: the loss is normalized by `sum(weights)`. They are not survey-bootstrap weights or a request for weighted residual bootstrap.
- Should I use formula on large GPU workloads? Usually no. Formula parsing is CPU-side convenience; use explicit arrays for large-scale GPU jobs.
- Are `Ridge`, `Lasso`, and `ElasticNet` aliases? No. They are thin wrappers so sklearn-style constructor behavior can remain clear.

## External Validation

Local and hosted checks cover imports, solver/objective invariants, CPU references, and regression matrices. GPU numerical parity and concrete-device behavior are validated separately with maintained physical-CUDA validators before release claims are promoted.

Validation coverage includes:

- CPU/CuPy/Torch coefficient and intercept differences.
- Analytic-weight rescaling, uniform-weight, and zero-weight-row identities.
- Objective gap and KKT residual checks for penalized paths.
- Gaussian penalized comparison against sklearn Ridge/Lasso/ElasticNet.
- Logistic comparison against sklearn/statsmodels where objectives align.
- Poisson L2 comparison against sklearn.
- Poisson L1/ElasticNet comparison against statsmodels `fit_regularized`.
- Runtime benchmarks with warm-up and GPU synchronization when performance is measured.

Developer validation assets are separate from the user-facing inference API described above. Remote credentials must be supplied through environment variables and must not be committed.

## References

- McCullagh, P., & Nelder, J. A. (1989). *Generalized Linear Models* (2nd ed.). Chapman & Hall/CRC.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.
- Friedman, J., Hastie, T., & Tibshirani, R. (2010). Regularization paths for generalized linear models via coordinate descent. *Journal of Statistical Software*, 33(1), 1-22. [https://doi.org/10.18637/jss.v033.i01](https://doi.org/10.18637/jss.v033.i01)
- scikit-learn linear models documentation: [https://scikit-learn.org/stable/modules/linear_model.html](https://scikit-learn.org/stable/modules/linear_model.html)
- statsmodels GLM documentation: [https://www.statsmodels.org/stable/glm.html](https://www.statsmodels.org/stable/glm.html)
