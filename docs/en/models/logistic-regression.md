# LogisticRegression

> Language: English  
> Last updated: 2026-10-03
> This page: Model documentation  
> Switch: [Chinese](../../cn/models/logistic-regression.md)

Language switch: [Chinese](../../cn/models/logistic-regression.md)

## Overview

`LogisticRegression` implements binary logit with IRLS or L-BFGS fitting on CPU/GPU, robust covariance options, and integrated classification metrics. `LogisticRegressionCV` selects the L2 regularization parameter `C` by cross-validation and refits on all training data. This API supports binary classification with optional L2 regularization; multiclass, L1, and elastic-net are not supported.

## Path

- `statgpu.linear_model.LogisticRegression`
- `statgpu.linear_model.LogisticRegressionCV`

## Objective Function

For positive `C`, both solvers minimize the same summed, optionally weighted objective:

$$
\min_{b,\beta}\; -\sum_i w_i\left[y_i\log p_i+(1-y_i)\log(1-p_i)\right]
+\frac{\|\beta\|_2^2}{2C},
\qquad p_i=\sigma(b+x_i^\top\beta).
$$

Here `sample_weight` supplies \(w_i\), or \(w_i=1\) when omitted. The intercept \(b\) is unpenalized and is omitted when `fit_intercept=False`. Larger positive `C` means weaker regularization. The legacy special value `C=0` disables the penalty; it does **not** mean infinitely strong regularization. Negative or non-finite `C` values are invalid.

L-BFGS divides the **whole objective, including the penalty**, by the sample count or total sample weight during optimization. This leaves the minimizer and the meaning of `C` unchanged. It does affect the scale of its gradient stopping criterion. When comparing other libraries, align the summed/averaged loss convention, penalty factor, weights, intercept treatment, and convergence settings.

## Solver Selection and Convergence

- `solver="auto"` resolves to `"irls"`, preserving the default fitting behavior.
- `solver="irls"` uses iteratively reweighted least squares. `tol` applies to the parameter-step norm.
- `solver="lbfgs"` uses limited-memory BFGS. `tol` applies to the gradient norm of the scaled objective or the norm of an accepted parameter step.
- `max_iter` limits iterations for the selected solver. Check `converged_` and `n_iter_`; a fit that fails to meet its stopping rule emits `ConvergenceWarning`.
- `get_params()["solver"]` retains the requested value, while `solver_` records the resolved solver after fitting. `summary()` shows both values and the convergence status; it requires `compute_inference=True`.

For positive `C`, the coefficient score equation is \(\sum_i w_i x_i(y_i-p_i)-\beta/C=0\); the intercept score has no penalty term. IRLS and L-BFGS target the same solution, but their iteration counts and the accuracy reached at the same `tol` need not match. Scaling poorly conditioned features can help either solver. L-BFGS avoids a Hessian solve at every optimization step, although requesting inference still requires covariance calculations.

### Backend and CV compatibility

All cells below support binary responses, an optional intercept, sample weights, and L2 regularization. Direct fits also support unpenalized `C=0`; the CV search uses positive `C` candidates.

| Requested solver | Resolved solver | NumPy CPU fit / CV | CuPy CUDA fit / CV | Torch CUDA fit / CV |
|---|---|---|---|---|
| `"auto"` | `"irls"` | IRLS / sequential IRLS fits | IRLS / batched IRLS | IRLS / batched IRLS |
| `"irls"` | `"irls"` | IRLS / sequential IRLS fits | IRLS / batched IRLS | IRLS / batched IRLS |
| `"lbfgs"` | `"lbfgs"` | L-BFGS / sequential L-BFGS fits | L-BFGS / sequential GPU L-BFGS fits | L-BFGS / sequential GPU L-BFGS fits |

`device="cpu"`, `"cuda"`, and `"torch"` select NumPy, CuPy CUDA, and Torch CUDA respectively. Explicit GPU requests raise an error if that backend is unavailable or fitting fails; they do not silently fall back to CPU. Only `device="auto"` allows automatic device selection. See also [solver and penalty compatibility](../guides/solver-penalty-matrix.md).

In CV, `auto`/`irls` retain the batched GPU path. L-BFGS fits candidates sequentially on the selected backend and does not use the batched IRLS solver. `gpu_cv_mixed_precision` affects only batched GPU IRLS; L-BFGS uses float64. The selected `C_` is refitted on all data with the same requested solver, and `solver_` reports the resolved solver. Unsupported solver names raise `ValueError`.

## Covariance/Inference

- `cov_type="nonrobust"`: information-matrix covariance.
- `cov_type="hc0"|"hc1"|"hc2"|"hc3"`: robust sandwich covariance variants.
- `cov_type="hac"`: Newey-West (Bartlett) covariance with optional `hac_maxlags`.
- Inference outputs use z-statistic conventions: `_bse`, `_zvalues`, `_pvalues`, `_conf_int`.
- `compute_inference=True` is required for inference fields.
- Likelihood, AIC, BIC, pseudo-R², and `converged_` remain available when `compute_inference=False`; covariance-based fields do not.

Both solvers use the same covariance calculations at the fitted estimate. With positive `C`, the information matrix (or sandwich bread) includes ridge curvature. These intervals and tests do not automatically correct for shrinkage bias or for the uncertainty from selecting `C` by CV. Changing the solver does not change that interpretation. CV inference is available through the final `estimator_`.

## Parameters

Complete `LogisticRegression` constructor parameters:

| Parameter | Default | Description |
|---|---:|---|
| `fit_intercept` | `True` | Whether to fit an unpenalized intercept |
| `C` | `1.0` | Inverse L2 strength for positive values; `0` disables regularization |
| `max_iter` | `100` | Maximum iterations for the selected solver |
| `tol` | `1e-4` | Positive stopping tolerance; interpretation depends on solver as above |
| `device` | `"auto"` | `cpu` / `cuda` / `torch` / `auto` |
| `n_jobs` | `None` | CPU job-count setting; does not parallelize logistic solver or CV candidate fits |
| `compute_inference` | `True` | Whether to compute inference statistics |
| `cov_type` | `"nonrobust"` | `nonrobust` / `hc0` / `hc1` / `hc2` / `hc3` / `hac` |
| `gpu_memory_cleanup` | `False` | Best-effort CuPy pool cleanup after each fit |
| `hac_maxlags` | `None` | Maximum lag for HAC; default follows a Newey-West-style heuristic |
| `solver` | `"auto"` | `auto` / `irls` / `lbfgs`; `auto` resolves to `irls` |

Complete `LogisticRegressionCV` constructor parameters:

| Parameter | Default | Description |
|---|---:|---|
| `Cs` | `None` | Positive candidate `C` values; `None` generates a data-dependent grid |
| `n_Cs` | `100` | Number of automatically generated candidates |
| `C_min_ratio` | `1e-3` | Smallest/largest `C` ratio for the automatic grid |
| `cv` | `5` | Number of folds when `cv_splits` is omitted |
| `cv_splits` | `None` | Explicit `(train_indices, validation_indices)` pairs |
| `fit_intercept` | `True` | Whether to fit an unpenalized intercept |
| `max_iter` | `100` | Iteration limit for candidate fits and final refit |
| `tol` | `1e-4` | Solver-specific stopping tolerance |
| `device` | `"auto"` | `cpu` / `cuda` / `torch` / `auto` |
| `n_jobs` | `None` | CPU job-count setting; does not parallelize candidate fits |
| `compute_inference` | `True` | Whether to compute inference for the final refit |
| `cov_type` | `"nonrobust"` | Covariance type for the final refit, as above |
| `gpu_memory_cleanup` | `False` | Best-effort CuPy pool cleanup |
| `random_state` | `None` | Random seed for generated CV splits |
| `gpu_cv_mixed_precision` | `True` | Mixed precision for batched GPU IRLS; unused by L-BFGS |
| `solver` | `"auto"` | Solver for candidate fits and final refit |

Both estimators accept `fit(X, y, sample_weight=None)`, where `y` contains binary 0/1 responses and weights, when supplied, are finite, nonnegative, and have positive total weight. `LogisticRegressionCV` does not expose `hac_maxlags`; HAC in its final refit uses the default lag rule.

## Fit and CV Example

```python
import numpy as np
from statgpu.linear_model import LogisticRegression, LogisticRegressionCV

rng = np.random.default_rng(42)
X = rng.normal(size=(400, 4))
probability = 1 / (1 + np.exp(-(0.3 + X @ [0.8, -0.5, 0.2, 0.0])))
y = rng.binomial(1, probability)
weights = rng.uniform(0.5, 1.5, size=len(y))

model = LogisticRegression(
    solver="lbfgs", device="cpu", C=1.0,
    max_iter=300, tol=1e-7, cov_type="hc1",
).fit(X, y, sample_weight=weights)
print(model.get_params()["solver"], model.solver_, model.converged_)
print(model.predict_proba(X[:3]))

cv_model = LogisticRegressionCV(
    Cs=[0.1, 1.0, 10.0], cv=3, random_state=42,
    solver="lbfgs", device="cpu", max_iter=300, tol=1e-7,
    compute_inference=False,
).fit(X, y, sample_weight=weights)
print(cv_model.C_, cv_model.solver_, cv_model.estimator_.converged_)
print(cv_model.mean_loss_)
```

Use `solver="auto"` or `"irls"` to compare IRLS on the same data. For a supported GPU installation, change `device` to `"cuda"` or `"torch"`; the same example and solver choices apply. CV minimizes mean held-out log loss. `mean_loss_` contains one value per candidate, `cv_results_["loss_path"]` contains fold-level losses, and `best_score_` is the negative of the selected mean loss.

## strict/approx difference

No separate approx inference mode is exposed in this API. Robust covariance choice (`hc*`/`hac`) is the main practical trade-off between assumptions and computational cost. Neither solver adds a shrinkage-bias or post-selection inference correction.

## Outputs

- Coefficients and fitting: `intercept_`, `coef_`, `n_iter_`, `solver_`, `converged_`
- Inference: `_bse`, `_zvalues`, `_pvalues`, `_conf_int`; printed by `summary()`
- Fit/metrics: `loglikelihood`, `loglikelihood_null`, `aic`, `bic`, `pseudo_rsquared`, `accuracy`, `precision`, `recall`, `f1`, `auc`, `average_precision`
- Prediction methods: `predict_proba`, `predict`, `predict_with_threshold`
- Evaluation methods: `confusion_matrix`, `classification_table`, `roc_curve`, `roc_auc_score`, `precision_recall_curve`, `average_precision_score`, `evaluate_classification`
- Plot helpers: `plot_roc_curve`, `plot_precision_recall_curve` (`matplotlib` optional dependency)
- CV: `C_`, `Cs_`, `mean_loss_`, `cv_results_`, `best_score_`, `cv_selected_device_`, `solver_`, and final `estimator_`; final convergence is `estimator_.converged_`

## FAQ

- How do I fit unregularized MLE? Use `C=0`. A large positive `C` (for example, `1e10`) approximates it. With complete separation, a finite unregularized MLE may not exist.
- Why are inference statistics z-values instead of t-values? Logistic regression inference follows large-sample normal approximation.
- Are evaluation methods GPU-native? Core prediction and metrics stay on the selected supported backend; plotting converts to NumPy for rendering.

## External Validation

- `dev/tests/test_external_consistency.py`
  - `test_logistic_robust_covariance_matches_statsmodels`
  - `test_logistic_robust_covariance_gpu_matches_statsmodels`
- Cross-backend artifact for `hc2/hc3/hac`:
  - `results/remote_covariance_full_compare_2026-04-10.json`

## References

- McCullagh, P., & Nelder, J. A. (1989). *Generalized Linear Models* (2nd ed.). Chapman & Hall/CRC.
- Hosmer, D. W., Lemeshow, S., & Sturdivant, R. X. (2013). *Applied Logistic Regression* (3rd ed.). Wiley.
