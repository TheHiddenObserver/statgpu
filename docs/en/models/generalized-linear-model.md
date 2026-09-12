# GeneralizedLinearModel and Penalized GLM

> Language: English  
> Last updated: 2026-09-12  
> This page: Model documentation  
> Switch: [Chinese](../../cn/models/generalized-linear-model.md)

## Overview

`GeneralizedLinearModel` is the common entry point for core ordinary GLMs such as Gaussian, binomial, and Poisson models. Typed ordinary estimators such as `GammaRegression`, `InverseGaussianRegression`, `NegativeBinomialRegression`, and `TweedieRegression` use the same maintained GLM machinery for their family-specific behavior.

For regularized models, use `PenalizedGeneralizedLinearModel` or a typed wrapper such as `PenalizedLinearRegression`, `PenalizedLogisticRegression`, or `PenalizedPoissonRegression`. `Ridge`, `Lasso`, and `ElasticNet` are sklearn-style thin wrappers over penalized Gaussian regression.

If you are here for weighted GLMs, the main rules are:

- `sample_weight` is an **analytic objective weight** on supported GLM paths;
- the weighted loss is normalized by `sum(sample_weight)`, so multiplying all weights by one positive constant does not change the fitted optimum;
- explicit `solver="newton"` and `solver="lbfgs"` remain explicit—weights do not silently switch the solver;
- explicit `device="cuda"` and `device="torch"` remain on the requested GPU backend when that route is supported;
- unsupported combinations fail clearly instead of silently falling back to a different solver or CPU.

For penalized coefficient inference, see [Penalized GLM inference](../guides/penalized-glm-inference.md). For the complete solver table, see [Solver × Penalty Compatibility Matrix](../guides/solver-penalty-matrix.md).

## Public paths

- `statgpu.linear_model.GeneralizedLinearModel`
- `statgpu.linear_model.PoissonRegression`
- `statgpu.linear_model.GammaRegression`
- `statgpu.linear_model.InverseGaussianRegression`
- `statgpu.linear_model.NegativeBinomialRegression`
- `statgpu.linear_model.TweedieRegression`
- `statgpu.linear_model.PenalizedGeneralizedLinearModel`
- `statgpu.linear_model.PenalizedLinearRegression`
- `statgpu.linear_model.PenalizedLogisticRegression`
- `statgpu.linear_model.PenalizedPoissonRegression`
- `statgpu.linear_model.Ridge`
- `statgpu.linear_model.Lasso`
- `statgpu.linear_model.ElasticNet`

The internal GLM objective layer is `statgpu.glm_core`.

## Objective function and `sample_weight`

An ordinary unweighted GLM minimizes the average negative log-likelihood for its family:

$$
\min_\beta \frac{1}{n}\sum_{i=1}^n \ell(y_i, x_i^\top\beta).
$$

On a supported weighted path, `sample_weight=w` changes the data-fit term to the normalized weighted average

$$
\min_\beta
\frac{\sum_i w_i\,\ell(y_i, x_i^\top\beta)}{\sum_i w_i}.
$$

This convention has three useful consequences:

1. multiplying every weight by the same positive constant leaves the optimum unchanged;
2. a row with weight zero contributes nothing to the data-fit objective;
3. weights must be finite and non-negative, with a strictly positive total weight.

A penalized GLM adds the declared penalty:

$$
\min_\beta \frac{1}{n}\sum_{i=1}^n \ell(y_i, x_i^\top\beta) + \alpha P(\beta),
$$

or, when analytic weights are active,

$$
\min_\beta
\frac{\sum_i w_i\,\ell(y_i, x_i^\top\beta)}{\sum_i w_i}
+ \alpha P(\beta).
$$

The intercept is not penalized. `statgpu.glm_core` is intentionally GLM-specific; Cox partial likelihood, robust objectives, quantile objectives, and other non-GLM losses keep their own statistical contracts.

## Solver selection

Smooth GLMs use maintained first- or second-order solvers when available. Non-smooth penalized objectives use proximal/KKT-style paths such as FISTA and FISTA-BB.

Representative direct-fit `solver="auto"` behavior is:

| Setting | `solver="auto"` behavior |
|---|---|
| `PenalizedLinearRegression(penalty="l2")` on NumPy/CPU | exact closed-form L2 path |
| squared error + L1/ElasticNet | FISTA/FISTA-BB sparse path |
| logistic/Poisson/Gamma/Inverse-Gaussian/Tweedie/Negative-Binomial + L2 | Newton under the maintained direct-fit dispatch |
| non-convex SCAD/MCP | FISTA + LLA continuation |
| quantile | FISTA / quantile-specific path |

`solver="auto"` is a request for the maintained dispatch policy. An explicit solver name is different: statgpu either runs that supported solver or raises an error; it does not silently substitute another solver.

The same principle applies to devices. Explicit `device="cuda"` uses CuPy CUDA and explicit `device="torch"` uses Torch CUDA on supported routes. Formula parsing may run on CPU, but fit/predict numerical work follows the selected numerical backend.

### Explicit Newton and L-BFGS with analytic weights

For maintained ordinary GLM routes, explicit `solver="newton"` and `solver="lbfgs"` accept non-uniform analytic weights when that family/link combination supports the solver.

Both solvers optimize the normalized weighted objective shown above. The same weight vector is used consistently throughout the optimization:

- Newton uses it in objective, gradient, Hessian, and Armijo line-search evaluations;
- L-BFGS uses it in the initial gradient, current objective, every line-search candidate, and the accepted-point gradient.

This consistency is important: a weighted search direction is never paired with an unweighted line-search objective.

Adding `sample_weight` does **not** replace an explicit Newton/L-BFGS request with IRLS or FISTA. Likewise, an explicit CUDA/Torch request does not fall back to CPU. Uniform and historically effectively-uniform weights retain the historical unweighted numerical path.

#### Inverse-power Gamma exception

`GammaRegression(link="inverse_power")` has one intentional boundary. A genuinely non-uniform weighted explicit Newton/L-BFGS fit requires `fit_intercept=True`, because the maintained initialization needs a strictly positive family-valid starting predictor.

Therefore:

- non-uniform weights + explicit Newton/L-BFGS + `fit_intercept=True`: supported;
- non-uniform weights + explicit Newton/L-BFGS + `fit_intercept=False`: rejected before fitting;
- omitted, uniform, or effectively-uniform weights with `fit_intercept=False`: keep the historical unweighted behavior.

This is a narrow initialization boundary, not a general restriction on Gamma regression.

#### Scope of weighted L-BFGS support

Weighted L-BFGS is a **GLM loss capability**, not a blanket promise for every low-level `LossBase` implementation. Direct robust, quantile, and Cox losses keep their own weight semantics and may reject non-uniform weights in direct L-BFGS. Ordered GLMs also retain their separate weight policy.

Weighted penalized smooth GLMs use the same analytic-weight convention. Their existing direct-fit/CV dispatch remains authoritative: a supported L2 row may use Newton or L-BFGS according to that policy, not merely because weights are present.

## Covariance and inference

Generic and typed penalized GLM estimators use `inference_method="auto"` as the recommended public request. Successful inference-enabled fits expose requested/resolved/reported method provenance, the inferential target, and tuning/selection conditioning.

For supported smooth non-Gaussian L2/no-penalty models, `auto` resolves to fixed-penalty `m_estimation`. Positive L2 fits target the penalized estimating equation; no-penalty aliases are canonicalized to zero-strength L2 and target the unpenalized population parameter. Current covariance support is `nonrobust`, `hc0`, and `hc1`; HC2/HC3/HAC are not available on this penalized non-Gaussian path and raise instead of being substituted silently.

Analytic weights are supported on the maintained inference routes, and numerical inference follows the backend/concrete device that actually executed the fit. Ordinary GLM inference uses the same fitted analytic-weight convention as estimation. Non-Gaussian L1/ElasticNet coefficient inference is not productized. SCAD/MCP oracle inference must be requested explicitly; group penalties and penalized Cox remain estimation-only.

For supported Gaussian sparse penalties, `inference_method="bootstrap"` selects an unweighted residual bootstrap with `cov_type="nonrobust"`. The design matrix and fitted tuning configuration remain fixed: each draw resamples residuals, constructs a new Gaussian response around the fitted values, and refits the same penalized model. `n_bootstrap` controls the number of refits and `bootstrap_random_state` controls reproducibility.

Bootstrap execution follows the successful fit's backend and concrete device. CPU fits use NumPy; CuPy and Torch CUDA fits keep bootstrap refits on the same GPU device. Final inference arrays use the standard NumPy reporting boundary. Weighted residual bootstrap, robust/HC or HAC/block bootstrap, non-Gaussian bootstrap, and Cox bootstrap are not supported.

See [Penalized GLM inference](../guides/penalized-glm-inference.md) and [Inference Modes](../guides/inference-modes.md) for the complete support matrix, resampling boundaries, and statistical interpretation.

## Parameters

| Parameter | Default | Description |
|---|---:|---|
| `family` | model-specific | GLM family, for example `"gaussian"`, `"binomial"`, or `"poisson"` |
| `penalty` | `"l2"` or model-specific | `none`, `l1`, `l2`, `elasticnet`, and reserved structured penalties |
| `alpha` | `1.0` or model-specific | Penalty strength in statgpu objective scale |
| `l1_ratio` | `None` | ElasticNet mixing parameter |
| `fit_intercept` | `True` | Whether to fit an intercept |
| `solver` | `"auto"` | Solver dispatch; see the solver-selection section above |
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

## CPU + GPU examples

```python
from statgpu.linear_model import GeneralizedLinearModel, PenalizedLogisticRegression

# Ordinary weighted Poisson GLM with an explicit smooth solver.
weighted_pois = GeneralizedLinearModel(
    family="poisson",
    solver="lbfgs",       # or "newton"
    device="cuda",        # CuPy CUDA; use "torch" for Torch CUDA
)
weighted_pois.fit(X, y_count, sample_weight=weights)

# CPU L2 logistic path: auto chooses the maintained direct-fit solver.
logit_cpu = PenalizedLogisticRegression(
    penalty="l2",
    alpha=0.01,
    solver="auto",
    device="cpu",
)
logit_cpu.fit(X, y_binary, sample_weight=weights)

# GPU L2 logistic path uses the same weighted objective.
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

Formula parsing runs on CPU and is intended as a convenience layer. When `sample_weight` is supplied with a formula, weights are aligned to the rows retained after formula/missing-data processing before numerical fitting. For very large data, pass explicit `X, y` arrays.

## Strict and approximate CV

Penalized GLM inference is fail-closed: supported non-Gaussian L2/no-penalty rows expose fixed-penalty M-estimation with nonrobust/HC0/HC1 covariance, while unsupported loss × penalty × method combinations raise instead of substituting another inferential procedure. Residual bootstrap and SCAD/MCP oracle remain deliberately narrow explicit paths.

`solver="auto"` follows the maintained direct-fit dispatch. Analytic weights do not rewrite a public solver request: explicit smooth solvers remain explicit, while `auto` continues to use the maintained dispatch for the corresponding fit.

`PenalizedGLM_CV` defaults to `cv_strategy="strict"`. In strict mode every fold/alpha is evaluated with the requested `max_iter` and `tol`, and GPU optimizations are limited to caching, fused kernels, and batched validation-score transfers.

The optional `cv_strategy="two_stage"` mode first screens the alpha grid with relaxed CV solves, then strictly refines candidate alphas and performs a strict final refit. Because screening can change the ranking when CV curves are close, two-stage mode emits `ApproximateCVWarning` unless `acknowledge_approx=True` is passed.

```python
from statgpu.linear_model import PenalizedGLM_CV

# Default: strict CV.
strict_cv = PenalizedGLM_CV(
    loss="poisson",
    penalty="elasticnet",
    cv_strategy="strict",
    device="cuda",
)

# Opt-in approximate screening; candidate refinement and final refit stay strict.
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

`PenalizedGLM_CV(loss="cox_ph")` uses a separate survival path rather than the scalar-response GLM scorer. Pass `y` as an `(n_samples, 2)` array with columns `[time, event]`. L1, L2, ElasticNet, SCAD, and MCP are supported on NumPy, CuPy CUDA, and Torch CUDA. The path:

- preserves the two-column target and never fits an intercept;
- scores each held-out fold with unpenalized negative Cox partial likelihood per row;
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

`cv_strategy="two_stage"`, `sample_weight`, dictionary targets, and post-selection coefficient inference are not supported for this Cox branch. `cv_results_` records per-fold losses, valid-evidence counts, event counts, failure reasons, the tie method, and the final-refit class.

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

## See also

- [Solver × Penalty Compatibility Matrix](../guides/solver-penalty-matrix.md) — full dispatch table for loss × penalty × solver combinations, CV paths, and inference support.
- [Solver Algorithms](../guides/solver-algorithms.md) — algorithm-level description of Newton, L-BFGS, FISTA, and other solvers.
- [Loss Functions](losses.md) — low-level loss interfaces and their capability boundaries.

## FAQ

- **What does `sample_weight` mean here?** On supported ordinary and penalized GLM paths it is an analytic objective weight, normalized by `sum(weights)`. It is not a survey-bootstrap weight and does not request weighted residual bootstrap.
- **Does adding `sample_weight` change my solver?** No. Explicit solver requests stay explicit; `solver="auto"` continues to use the maintained dispatch policy.
- **Does `device="cuda"` force every GLM solver onto GPU?** On a supported route, yes: CuPy performs the numerical work. If the route is unsupported or the device is unavailable, statgpu raises instead of silently falling back to CPU.
- **Should I use formulas for very large GPU workloads?** Usually not. Formula parsing is a CPU-side convenience; explicit arrays are preferable for large GPU jobs.
- **Are `Ridge`, `Lasso`, and `ElasticNet` aliases?** No. They are thin wrappers with sklearn-style constructors.

## External validation

Local and hosted checks cover imports, solver/objective invariants, CPU references, and regression matrices. GPU numerical parity and concrete-device behavior are validated separately with maintained physical-CUDA validators before release claims are promoted.

Validation coverage includes:

- CPU/CuPy/Torch coefficient and intercept parity;
- analytic-weight global-rescaling, uniform-weight, and zero-weight-row identities;
- objective-gap and KKT-residual checks for penalized paths;
- comparisons with sklearn and statsmodels where the objective definitions align;
- runtime benchmarks with warm-up and GPU synchronization when performance is measured.

Developer validation assets are separate from the user-facing API described above. Remote credentials must be supplied through environment variables and must not be committed.

## References

- McCullagh, P., & Nelder, J. A. (1989). *Generalized Linear Models* (2nd ed.). Chapman & Hall/CRC.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.
- Friedman, J., Hastie, T., & Tibshirani, R. (2010). Regularization paths for generalized linear models via coordinate descent. *Journal of Statistical Software*, 33(1), 1-22. [https://doi.org/10.18637/jss.v033.i01](https://doi.org/10.18637/jss.v033.i01)
- scikit-learn linear models documentation: [https://scikit-learn.org/stable/modules/linear_model.html](https://scikit-learn.org/stable/modules/linear_model.html)
- statsmodels GLM documentation: [https://www.statsmodels.org/stable/glm.html](https://www.statsmodels.org/stable/glm.html)
