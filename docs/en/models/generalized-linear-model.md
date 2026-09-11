# GeneralizedLinearModel and Penalized GLM

> Language: English  
> Last updated: 2026-09-11  
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

For penalized coefficient inference, see the dedicated [Penalized GLM inference guide](../guides/penalized-glm-inference.md).

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
\min_\beta \frac{1}{n}\sum_{i=1}^n \ell(y_i, x_i^\top\beta)
$$

Penalized GLM adds a penalty term:

$$
\min_\beta \frac{1}{n}\sum_{i=1}^n \ell(y_i, x_i^\top\beta) + \alpha P(\beta)
$$

The intercept is not penalized. `statgpu.glm_core` is intentionally GLM-specific; Cox partial likelihood, panel objectives, time-series likelihoods, and zero-inflated composite likelihoods should use their own objective layers rather than being forced into `glm_core`.

## Estimating Equation and Solver Dispatch

Smooth GLMs use second-order or first-order optimization when available. Non-smooth penalized objectives use proximal/KKT-style optimization through FISTA/FISTA-BB.

Current direct-fit `solver="auto"` behavior is table-driven. In particular, smooth non-Gaussian L2 models currently resolve to Newton, not IRLS:

| Setting | `solver="auto"` behavior |
|---|---|
| `PenalizedLinearRegression(penalty="l2")` on NumPy/CPU | exact closed-form L2 path |
| squared error + L1/ElasticNet | FISTA/FISTA-BB sparse path |
| logistic/Poisson/Gamma/Inverse-Gaussian/Tweedie/Negative-Binomial + L2 | Newton under the maintained direct-fit dispatch |
| non-convex SCAD/MCP | FISTA + LLA continuation |
| quantile | FISTA / quantile-specific path |

Important device rule: explicit `device="cuda"` stays on CuPy, explicit `device="torch"` stays on Torch, and an unsupported explicit solver/backend combination raises instead of silently switching to CPU.

One targeted inference exception is documented rather than hidden: an inference-enabled, weighted, non-Gaussian L2/no-penalty fit with `solver="auto"` uses the existing backend-native FISTA path for that fit because Newton currently rejects non-uniform analytic weights. The public request remains `solver="auto"`; estimation-only benchmark dispatch is unchanged.

## Covariance / Inference

Penalized GLM coefficient inference is available for a deliberately bounded support matrix. The generic and typed penalized GLM surfaces use `inference_method="auto"` as the recommended default.

### Non-Gaussian L2 / no penalty

For a smooth non-Gaussian L2/no-penalty model, `auto` resolves to fixed-penalty M-estimation. Positive-penalty fits report the target as the penalized estimating equation rather than pretending to be an unpenalized/debiased coefficient.

Supported covariance choices are:

- `cov_type="nonrobust"` — model-based penalized-information covariance;
- `cov_type="hc0"`;
- `cov_type="hc1"`.

HC2, HC3, and HAC are not implemented for this penalized non-Gaussian path and fail visibly.

Analytic weights are supported. Numerical covariance/statistic/p-value/CI work follows the backend and concrete device that actually executed the fit. Reporting arrays may move to NumPy only after numerical inference is complete.

### Sparse and non-convex rows

- Gaussian L1/ElasticNet keep their established debiased and `post_selection_ols` contracts.
- Non-Gaussian L1/ElasticNet inference is not implemented and fails closed.
- SCAD/MCP oracle inference requires explicit `inference_method="oracle"`; `auto` does not silently select an active-set-conditional procedure.
- Group penalties and penalized Cox remain estimation-only.
- `inference_method="bootstrap"` in this repair is an **unweighted CPU Gaussian residual bootstrap**, not a universal GLM bootstrap. It preserves the fitted penalty family/alpha/intercept (and ElasticNet mixing where applicable), requires `cov_type="nonrobust"`, and does not rerun CV inside bootstrap samples.

### Requested, resolved, and reported method

Successful inference-enabled fits publish:

- `inference_requested_method_`;
- `inference_resolved_method_`;
- `inference_method_`;
- `inference_target_`;
- `penalty_conditioning_`;
- `penalty_selection_adjusted_`.

For L2/no-penalty models, explicitly requesting the historical `"debiased"` spelling is temporarily accepted with a `FutureWarning`, but the fitted result reports the actual method. L2 inference was never debiased-Lasso inference.

See [Penalized GLM inference](../guides/penalized-glm-inference.md) for the full support matrix and statistical interpretation.

## Parameters

| Parameter | Default | Description |
|---|---:|---|
| `family` / `loss` | model-specific | GLM family/loss, for example `"logistic"` or `"poisson"` |
| `penalty` | model-specific | `none`, `l1`, `l2`, `elasticnet`, and supported structured penalties |
| `alpha` | `1.0` or model-specific | Penalty strength in statgpu average-loss objective scale |
| `l1_ratio` | model-specific | ElasticNet mixing parameter |
| `fit_intercept` | `True` | Whether to fit an intercept |
| `solver` | `"auto"` | Backend-neutral solver request |
| `device` | `"auto"` | `cpu`, `cuda`, `torch`, or `auto` depending on estimator support |
| `max_iter` | model-specific | Maximum optimizer iterations |
| `tol` | model-specific | Convergence tolerance |
| `compute_inference` | `False` on generic/typed penalized GLM | Whether to compute coefficient inference |
| `inference_method` | `"auto"` on generic/typed penalized GLM | Resolve a supported method from loss/penalty/covariance; sparse Gaussian wrappers retain their established explicit defaults |
| `cov_type` | `"nonrobust"` | Covariance convention; non-Gaussian penalized M-estimation currently supports nonrobust/HC0/HC1 |
| `hac_maxlags` | `None` | Retained API control; does not imply HAC support for penalized non-Gaussian M-estimation |
| `formula` | `None` | Optional patsy-style formula used with `data` |
| `data` | `None` | DataFrame used with `formula` |

Alpha scaling is explicit. Do not compare same-named parameters across frameworks without conversion:

- Ridge: `sklearn_alpha = n_samples * statgpu_alpha`
- Logistic L2: align the objective before comparing to sklearn's `C` parameterization
- Poisson L2: align against sklearn `PoissonRegressor(alpha=...)`
- Poisson L1/ElasticNet: align against statsmodels `fit_regularized`

## CPU + GPU Examples

```python
from statgpu.linear_model import PenalizedLogisticRegression, PenalizedPoissonRegression

# Estimation-only L2 logistic fit.
logit = PenalizedLogisticRegression(
    penalty="l2",
    alpha=0.01,
    solver="auto",
    device="cpu",
)
logit.fit(X, y_binary)

# Fixed-penalty Poisson M-estimation inference.
pois = PenalizedPoissonRegression(
    penalty="l2",
    alpha=0.03,
    solver="auto",
    compute_inference=True,
    inference_method="auto",
    cov_type="hc0",
    device="cpu",            # also supported on maintained CuPy/Torch paths
)
pois.fit(X, y_count, sample_weight=w)

print(pois.inference_requested_method_)  # auto
print(pois.inference_resolved_method_)   # m_estimation
print(pois.inference_method_)            # m_estimation
print(pois._bse)
```

Formula support is optional:

```bash
pip install statgpu[formula]
```

```python
from statgpu.linear_model import PenalizedPoissonRegression

pois = PenalizedPoissonRegression(
    penalty="l2",
    alpha=0.01,
    compute_inference=True,
    cov_type="hc0",
)
pois.fit(formula="count ~ exposure + x1", data=df)
```

Formula parsing runs on CPU as a model-matrix convenience layer; the numerical fit/inference then follows the selected execution backend. For very large data, pass explicit arrays.

## Strict / approximate boundaries

Inference requests are fail-closed: an unsupported loss × penalty × method × covariance row raises rather than silently substituting another statistical procedure.

Residual bootstrap and SCAD/MCP oracle are intentionally narrow/conditional paths. Their limitations are public, and `auto` does not use them as universal fallbacks.

`PenalizedGLM_CV` defaults to `cv_strategy="strict"`. Fold/path/grid fits do **not** run coefficient inference. With `compute_inference=True`, inference runs once on the selected full-data final refit. The reported uncertainty is conditional on the selected alpha and explicitly records:

```text
penalty_conditioning_ = "cv_selected_penalty"
penalty_selection_adjusted_ = False
```

Therefore it does not claim to adjust for CV tuning-selection uncertainty.

The optional `cv_strategy="two_stage"` mode still uses approximate screening before strict candidate refinement/final refit and emits `ApproximateCVWarning` unless acknowledged.

### Survival-aware penalized Cox CV

`PenalizedGLM_CV(loss="cox_ph")` uses a separate survival path rather than the scalar-response GLM scorer. Pass `y` as an `(n_samples, 2)` array with columns `[time, event]`. L1, L2, ElasticNet, SCAD, and MCP are supported for estimation on maintained NumPy/CuPy/Torch paths.

The path:

- preserves the two-column target and never fits an intercept;
- scores held-out folds with unpenalized negative Cox partial likelihood per row;
- selects an alpha only when every evaluable fold supplies finite evidence;
- hard-fails without publishing fitted state when no alpha is supported; and
- refits `PenalizedCoxPHModel` with coefficient inference disabled.

`compute_inference=True` for the Cox branch is rejected: penalized Cox remains estimation-only.

## Outputs

Common fitted attributes and methods include:

- `coef_`;
- `intercept_`;
- `n_iter_` where exposed by the selected solver;
- `fit`, `predict`, and family-specific prediction helpers;
- `cv_results_` for `PenalizedGLM_CV`;
- when coefficient inference succeeds: `_bse`, `_pvalues`, `_conf_int`, `_inference_result`, and the requested/resolved/target/conditioning provenance fields listed above.

## See Also

- [Penalized GLM inference](../guides/penalized-glm-inference.md) — statistical targets, method resolver, resampling limits, backend provenance, and CV final-refit semantics.
- [Solver × Penalty Compatibility Matrix](../guides/solver-penalty-matrix.md) — solver dispatch and capability matrix.
- [Cross-Validation](../guides/cross-validation.md) — CV architecture and final-refit behavior.

## FAQ

- **Does `inference_method="auto"` mean a silent fallback?** No. It resolves only supported rows; unsupported rows fail closed.
- **Does non-Gaussian L2 M-estimation remove shrinkage bias?** No. It targets the fixed-penalty estimating equation and reports that target explicitly.
- **Does CV inference adjust for selecting alpha?** No. It conditions on the selected penalty and publishes `penalty_selection_adjusted_=False`.
- **Can non-Gaussian L1/ElasticNet use `bootstrap` as a fallback?** No. The maintained bootstrap in this repair is Gaussian residual bootstrap only.
- **Does explicit CUDA/Torch inference silently use CPU?** No. Supported numerical inference follows the executed backend/device or fails visibly.

## External Validation

The maintained model suite includes CPU regression tests, formula parity, independent covariance algebra checks, clone/API compatibility, and backend dispatch contracts. Physical CUDA acceptance remains a separate evidence tier and must be reported only when run on actual CuPy/Torch CUDA devices.

**v23c full matrix benchmark (2026-05-20):** 1043/1043 estimation cases passed across the historical family/penalty/backend matrix. That estimation evidence does not by itself certify the newer coefficient-inference contract; inference acceptance uses its own targeted tests and validators.

## References

- McCullagh, P., & Nelder, J. A. (1989). *Generalized Linear Models* (2nd ed.). Chapman & Hall/CRC.
- White, H. (1980). A heteroskedasticity-consistent covariance matrix estimator and a direct test for heteroskedasticity. *Econometrica*.
- MacKinnon, J. G., & White, H. (1985). Some heteroskedasticity-consistent covariance matrix estimators with improved finite sample properties. *Journal of Econometrics*.
- Friedman, J., Hastie, T., & Tibshirani, R. (2010). Regularization paths for generalized linear models via coordinate descent. *Journal of Statistical Software*, 33(1), 1-22.
