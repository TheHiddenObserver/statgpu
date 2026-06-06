# GeneralizedLinearModel and Penalized GLM

> Language: English  
> Last updated: 2026-05-20  
> This page: Model documentation  
> Switch: [Chinese](../../models/generalized-linear-model.md)

Language switch: [Chinese](../../models/generalized-linear-model.md)

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

\[
\min_\beta \frac{1}{n}\sum_{i=1}^n \ell(y_i, x_i^\top\beta)
\]

Penalized GLM adds a penalty term:

\[
\min_\beta \frac{1}{n}\sum_{i=1}^n \ell(y_i, x_i^\top\beta) + \alpha P(\beta)
\]

The intercept is not penalized. `statgpu.glm_core` is intentionally GLM-specific; Cox partial likelihood, panel objectives, time-series likelihoods, and zero-inflated composite likelihoods should use future objective layers rather than being forced into `glm_core`.

## Estimating Equation

Smooth GLMs solve score equations through IRLS/Newton/L-BFGS-style updates when available. Non-smooth penalized objectives use proximal/KKT-style optimization through FISTA.

Current solver behavior:

| Setting | `solver="auto"` behavior |
|---|---|
| `PenalizedLinearRegression(penalty="l2")` | `solver="exact"` closed-form L2 path |
| `PenalizedLinearRegression(penalty="l1"|"elasticnet")` | FISTA |
| `PenalizedLogisticRegression(penalty="l2")` on NumPy/CPU | IRLS |
| `PenalizedPoissonRegression(penalty="l2")` on NumPy/CPU | IRLS |
| `PenalizedLogisticRegression(penalty="l2")` on CuPy/Torch GPU | FISTA |
| `PenalizedPoissonRegression(penalty="l2")` on CuPy/Torch GPU | FISTA |
| explicit `solver="irls"` | backend-native IRLS on NumPy/CuPy/Torch |
| explicit `solver="newton"` | backend-native Newton on smooth objectives |
| explicit `solver="lbfgs"` | backend-native L-BFGS on smooth objectives (all GLM families + L2/ElasticNet) |
| non-smooth penalty with `solver="newton"` or `solver="lbfgs"` | raises `ValueError` |

Important device rule: explicit `device="cuda"` stays on CuPy, explicit `device="torch"` stays on Torch CUDA, and explicit solver choices do not silently fall back to CPU. Formula parsing may run on CPU, but the core fit/predict path is converted to the selected backend.

## Covariance/Inference

This page covers estimation-first GLM and penalized GLM APIs. Full strict inference parity is not yet exposed for the new penalized GLM layer. Existing inference-rich estimators remain documented separately:

- `LinearRegression`: classical, HC0-HC3, HAC.
- `Ridge`: classical, HC0-HC3, HAC.
- `LogisticRegression`: classical, HC0-HC3, HAC.
- `Lasso`: OLS-style and bootstrap inference paths.

Future GLM inference work should align with the project-wide strict inference gate before release.

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

# Ordinary Poisson GLM on GPU when the selected path supports it.
glm = GeneralizedLinearModel(family="poisson", device="cuda")
glm.fit(X, y_count)

# CPU L2 logistic path: auto selects IRLS.
logit_cpu = PenalizedLogisticRegression(
    penalty="l2",
    alpha=0.01,
    solver="auto",
    device="cpu",
)
logit_cpu.fit(X, y_binary)

# GPU L2 logistic path: auto selects GPU-capable FISTA.
logit_gpu = PenalizedLogisticRegression(
    penalty="l2",
    alpha=0.01,
    solver="auto",
    device="cuda",
)
logit_gpu.fit(X, y_binary)
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

Formula parsing runs on CPU and is intended as a convenience layer. For very large data, pass explicit `X, y` arrays.

## strict/approx difference

For the current GLM refactor, strict numerical validation is performed through remote CPU/GPU accuracy and external-framework comparison scripts. The new penalized GLM layer does not yet expose strict inference outputs such as robust standard errors and confidence intervals.

`solver="auto"` is device-aware for penalized GLMs. It picks exact Ridge for Gaussian L2, IRLS for smooth CPU logistic/poisson L2, and FISTA for CuPy/Torch GPU logistic/poisson L2. Explicit `irls`, `newton`, and `lbfgs` run on the selected backend when mathematically valid.

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

## FAQ

- Why is `statgpu.losses` not kept as a compatibility namespace? The uncommitted `losses` layer was GLM-specific, so it was renamed to `glm_core` to avoid implying a project-wide objective system.
- Does `device="cuda"` force GPU for every GLM solver? Yes for supported GLM solver paths: CuPy is used for the core computation, or a clear error is raised. There is no silent CPU fallback for explicit CUDA/Torch requests.
- Should I use formula on large GPU workloads? Usually no. Formula parsing is CPU-side convenience; use explicit arrays for large-scale GPU jobs.
- Are `Ridge`, `Lasso`, and `ElasticNet` aliases? No. They are thin wrappers so sklearn-style constructor behavior can remain clear.

## External Validation

Local checks cover imports and smoke tests only. Accuracy, runtime, GPU behavior, and external-framework comparisons run on the remote `myconda` environment.

**v23c full matrix benchmark (2026-05-20):** 1043/1043 ALL PASS across 7 families x 10 penalties x 3 scales x 3 backends, validated against sklearn and statsmodels. See `dev/tests/_bench_v23c_report.md` and `dev/tests/_bench_full_matrix.py`.

Validation coverage includes:

- CPU/CuPy/Torch coefficient and intercept differences.
- Objective gap and KKT residual checks for penalized paths.
- Gaussian penalized comparison against sklearn Ridge/Lasso/ElasticNet.
- Logistic comparison against sklearn.
- Poisson L2 comparison against sklearn.
- Poisson L1/ElasticNet comparison against statsmodels `fit_regularized`.
- Runtime benchmarks with warm-up and GPU synchronization.

Remote credentials must be supplied through environment variables and must not be committed.

## References

- McCullagh, P., & Nelder, J. A. (1989). *Generalized Linear Models* (2nd ed.). Chapman & Hall/CRC.
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.
- Friedman, J., Hastie, T., & Tibshirani, R. (2010). Regularization paths for generalized linear models via coordinate descent. *Journal of Statistical Software*, 33(1), 1-22. [https://doi.org/10.18637/jss.v033.i01](https://doi.org/10.18637/jss.v033.i01)
- scikit-learn linear models documentation: [https://scikit-learn.org/stable/modules/linear_model.html](https://scikit-learn.org/stable/modules/linear_model.html)
- statsmodels GLM documentation: [https://www.statsmodels.org/stable/glm.html](https://www.statsmodels.org/stable/glm.html)
