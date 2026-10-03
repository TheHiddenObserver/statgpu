# LogisticRegression

> Language: English  
> Last updated: 2026-10-03
> This page: Model documentation  
> Switch: [Chinese](../../cn/models/logistic-regression.md)

Language switch: [Chinese](../../cn/models/logistic-regression.md)

## When to use this model

`LogisticRegression` models the probability of a binary outcome, such as whether an event occurs. It combines a linear predictor with the logistic function, so predicted probabilities stay between 0 and 1. Use it for binary prediction or for interpreting associations on the log-odds scale. Association alone does not establish causality.

This standalone estimator uses IRLS and optional L2 regularization on CPU/GPU. It does not implement multiclass, L1, or Elastic Net fitting. For broader penalty choices, see [penalized GLM](generalized-linear-model.md); do not assume those estimators share this class's `C` or solver interface.

## A complete CPU example

The example generates nonseparable 0/1 data, fits on 600 rows, and evaluates the remaining 200. `C=0` requests this estimator's unregularized fit; the default `C=1.0` instead applies L2 regularization.

```python
import numpy as np
from statgpu.linear_model import LogisticRegression

rng = np.random.default_rng(42)
X = rng.normal(size=(800, 3))
true_coef = np.array([0.8, -0.6, 0.3])
p = 1.0 / (1.0 + np.exp(-(-0.2 + X @ true_coef)))
y = rng.binomial(1, p)

model = LogisticRegression(
    C=0, device="cpu", cov_type="hc1",
    compute_inference=True, max_iter=200, tol=1e-8,
).fit(X[:600], y[:600])

print("converged:", model.converged_)
print("coef:", np.round(model.coef_, 3))
print("odds ratios:", np.round(np.exp(model.coef_), 3))
print("P(y=1):", np.round(model.predict_proba(X[600:603])[:, 1], 3))
print("held-out accuracy:", round(float(model.score(X[600:], y[600:])), 3))
print("95% coefficient intervals:", np.round(model._conf_int, 3))
```

For this seed, rounded output starts as follows (small numerical differences across environments are expected):

```text
converged: True
coef: [ 0.849 -0.501  0.138]
odds ratios: [2.338 0.606 1.148]
P(y=1): [0.388 0.306 0.563]
held-out accuracy: 0.65
```

### Reading the results

- `coef_[j]` is a change in log odds for a one-unit increase in feature j, holding other features fixed. `exp(coef_[j])` is an odds ratio, not a probability ratio or a direct probability change.
- `predict_proba(X)` has shape `(n_samples, 2)`: columns are probabilities for classes 0 and 1. `predict(X)` uses a 0.5 threshold; `predict_with_threshold(X, threshold=...)` allows a threshold chosen for the application.
- `score(X, y)` is classification accuracy. Evaluate it on held-out data; for imbalanced outcomes also examine precision/recall or the precision–recall curve.
- `_conf_int` contains 95% coefficient intervals. With an intercept, its first row and the first entries of `_bse`, `_zvalues`, `_pvalues` refer to the intercept; subsequent rows follow the input feature order.
- Check `converged_` before interpreting coefficients or inference. `n_iter_` records iteration count. A fitted object alone is not proof of convergence.

## Inputs and model definition

Import with `from statgpu.linear_model import LogisticRegression`.
`fit(X, y, sample_weight=None)` returns the fitted estimator. Supply finite numeric `X` of shape `(n_samples, n_features)` and `y` of shape `(n_samples,)`, encoded as 0 or 1. Prediction data must use the same feature columns in the same order. Encode categorical features and handle missing values before array fitting.

Optional `sample_weight` is a length-n vector of finite nonnegative analytic weights with positive total weight. With no weights, set each mathematical weight below to 1. This standalone estimator minimizes a **summed** weighted negative log-likelihood plus a slope penalty:

$$
\eta_i=b+x_i^\top\beta,\qquad p_i=\frac{1}{1+\exp(-\eta_i)},
$$

$$
Q(b,\beta)=-\sum_i w_i\{y_i\log p_i+(1-y_i)\log(1-p_i)\}
+\frac{\alpha_C}{2}\|\beta\|_2^2,
\qquad
\alpha_C=\begin{cases}1/C,&C>0,\\0,&C=0.\end{cases}
$$

The intercept is not penalized. For positive `C`, larger values weaken regularization; very large `C` only approximates an unpenalized fit. `C=0` is a special legacy convention that removes the penalty exactly, not infinitely strong regularization. Unlike an average-loss objective, multiplying all weights by a positive constant changes the effective penalty when `C>0`; do not transfer regularization values from another API without matching the objective scale.

## Fitting and estimating equations

IRLS solves the weighted estimating equations:

$$
\sum_i w_i x_i(y_i-p_i)-\alpha_C\beta=0,
\qquad \sum_i w_i(y_i-p_i)=0\quad\text{(with intercept)}.
$$

`max_iter` and `tol` control the iteration. The slope equation includes the penalty term for the default `C=1.0`; the unpenalized score equation applies only when `C=0` (or as an approximation at very large `C`). This estimator does not expose a `solver` selection parameter.

## Covariance/Inference

- `cov_type="nonrobust"`: inverse information matrix at `C=0`; inverse penalized curvature when `C>0`.
- `cov_type="hc0"|"hc1"|"hc2"|"hc3"`: robust sandwich covariance variants.
- `cov_type="hac"`: Newey-West (Bartlett) covariance with optional `hac_maxlags`.
- Inference outputs use z-statistic conventions: `_bse`, `_zvalues`, `_pvalues`, `_conf_int`.
- `compute_inference=True` is required for inference fields.
- Likelihood, AIC, BIC, pseudo-R², and `converged_` remain available when `compute_inference=False`; covariance-based fields do not.

At positive `C`, inference is computed around the penalized fit; it is not debiasing for the unpenalized population coefficient. Robust choices use the penalized inverse curvature as the sandwich bread. These quantities do not correct shrinkage bias or uncertainty from selecting `C`. For ordinary unpenalized Logit inference use `C=0`, check identification and convergence, and choose covariance assumptions appropriate to the data. `summary()` requires `compute_inference=True`.

## Parameters

| Parameter | Default | Description |
|---|---:|---|
| `fit_intercept` | `True` | Whether to fit an intercept |
| `C` | `1.0` | Positive: inverse L2 strength; `0`: unpenalized legacy convention |
| `max_iter` | `100` | Max IRLS iterations |
| `tol` | `1e-4` | Convergence tolerance |
| `device` | `"auto"` | `cpu` / `cuda` / `torch` / `auto` |
| `n_jobs` | `None` | CPU parallelism control inherited from the base estimator; not an IRLS solver selector |
| `compute_inference` | `True` | Whether to compute inference stats |
| `cov_type` | `"nonrobust"` | `nonrobust` / `hc0` / `hc1` / `hc2` / `hc3` / `hac` |
| `hac_maxlags` | `None` | Max lag for `cov_type="hac"`; default follows Newey-West style heuristic |
| `gpu_memory_cleanup` | `False` | Best-effort CuPy pool cleanup after each fit |

## GPU use

After running the data-generation part of the CPU example, the equivalent CuPy fit is:

```python
import cupy as cp
from statgpu.linear_model import LogisticRegression

X_gpu = cp.asarray(X[:600])
y_gpu = cp.asarray(y[:600])
model_gpu = LogisticRegression(C=0, device="cuda", cov_type="hc1").fit(X_gpu, y_gpu)
p_gpu = model_gpu.predict_proba(cp.asarray(X[600:603]))[:, 1]
```

`device="torch"` selects Torch CUDA. Explicit `cuda`/`torch` requests require the corresponding installed package and usable CUDA device; they do not silently fall back to CPU. Prediction arrays use the selected supported backend; plot helpers convert results to NumPy for rendering. See [device and memory](../guides/device-and-memory.md) for installation/routing details.

## strict/approx difference

No separate approx inference mode is exposed in this API. Robust covariance choice (`hc*`/`hac`) is the main practical trade-off between assumptions and computational cost.

## Outputs

- Coefficients/convergence: `intercept_`, `coef_`, `n_iter_`, `converged_`
- Inference: `_bse`, `_zvalues`, `_pvalues`, `_conf_int`
- Fit/metrics: `loglikelihood`, `loglikelihood_null`, `aic`, `bic`, `pseudo_rsquared`, `accuracy`, `precision`, `recall`, `f1`, `auc`, `average_precision`
- Prediction methods: `predict_proba`, `predict`, `predict_with_threshold`
- Evaluation methods: `confusion_matrix`, `classification_table`, `roc_curve`, `roc_auc_score`, `precision_recall_curve`, `average_precision_score`, `evaluate_classification`
- Plot helpers: `plot_roc_curve`, `plot_precision_recall_curve` (`matplotlib` optional dependency)

## Parameter choices and common problems

- Scale continuous features when their units should not determine the L2 penalty. Learn preprocessing from training data only and apply the same transformation at prediction time.
- For predictive use, choose positive `C` and the decision threshold using validation data; the smallest training error is not a reliable tuning rule. Keep the final test set separate.
- `C=0` is useful for ordinary unpenalized inference. Perfect or near separation can make unpenalized coefficients unstable or nonfinite in the limit; a larger iteration budget alone does not fix separation or collinearity.
- If `converged_` is false, inspect scaling, separation and rank before increasing `max_iter` or loosening `tol`. Do not report p-values as reliable simply because attributes exist.
- Use HC covariance for an appropriate heteroskedastic/score-robust analysis; HAC additionally depends on observation order and lag selection. Shuffling time-ordered observations changes the meaning of HAC.
- If `summary()` reports inference unavailable, refit with `compute_inference=True`; prediction and likelihood diagnostics can still be used with inference disabled.
- Binary labels must be 0/1; encode other class names first. Missing/nonfinite inputs and mismatched shapes must be corrected before fitting.

## API reference and validation

The parameter table above covers the constructor. The fitting interface is `fit(X, y, sample_weight=None)`; inherited `get_params` / `set_params` support estimator configuration. `score(X, y)` returns accuracy. `summary()` displays the inference report. Training-data metric properties include `accuracy`, `precision`, `recall`, `f1`, `auc`, and `average_precision`; use the evaluation methods on separate data for generalization assessment.

Complete method signatures, returns, and estimator docstrings are available in the [public class API source](../../../statgpu/linear_model/wrappers/_logistic.py); `help(LogisticRegression)` also exposes that API in the installed version. The standalone distribution is binary; inference uses large-sample z statistics. This estimator has no dedicated formula argument in `fit`.

CPU/GPU estimates and covariance calculations are compared with statistical reference implementations under aligned settings, including near-unregularized comparisons with `statsmodels.Logit`. Such comparisons are scoped to their data and covariance assumptions, not universal accuracy guarantees.

## References

- McCullagh, P., & Nelder, J. A. (1989). *Generalized Linear Models* (2nd ed.). Chapman & Hall/CRC.
- Hosmer, D. W., Lemeshow, S., & Sturdivant, R. X. (2013). *Applied Logistic Regression* (3rd ed.). Wiley.

