# LogisticRegression

> Language: English  
> Last updated: 2026-10-14
> This page: Model documentation  
> Switch: [Chinese](../../../cn/models/regression/logistic-regression.md)

## When to use this model

`LogisticRegression` models the probability of a binary outcome, such as whether an event occurs. It combines a linear predictor with the logistic function, so predicted probabilities stay between 0 and 1. Use it for binary prediction or for interpreting associations on the log-odds scale: the logarithm of $p/(1-p)$, where $p$ is the event probability. Association alone does not establish causality.

This standalone estimator offers IRLS or L-BFGS fitting with optional L2 regularization on CPU/GPU. The default `solver="auto"` selects IRLS. `LogisticRegressionCV` chooses a positive `C` by cross-validation and refits on all supplied training rows. Neither class implements multiclass, L1, or Elastic Net fitting. For broader penalty choices, see [penalized GLM](generalized-linear-model.md); do not assume those estimators share this class's `C` or solver interface.

<a id="cpu-example"></a>

## A complete CPU example

Run these steps in order. Start with fitting and prediction; the last step optionally adds coefficient intervals. Only a CPU is needed.

### 1. Import

<!-- learner-example: logistic-unpenalized -->
```python
import numpy as np
from statgpu.linear_model import LogisticRegression
```

### 2. Prepare binary data

`X` has shape `(800, 3)`: one observation per row and one numeric feature per column. `y` is a length-800 vector of 0/1 labels. The simulated probabilities generate the labels; fitting real data does not require knowing those probabilities. Use the first 600 rows for training and the remaining 200 for evaluation, retaining the same feature order.

```python
rng = np.random.default_rng(42)
X = rng.normal(size=(800, 3))
true_coef = np.array([0.8, -0.6, 0.3])
p = 1.0 / (1.0 + np.exp(-(-0.2 + X @ true_coef)))
y = rng.binomial(1, p)
```

### 3. Fit and check convergence

`C=0` is this estimator's special value for removing the penalty exactly; the default `C=1.0` adds L2 shrinkage. Disable inference initially to focus on the fitted model.

```python
model = LogisticRegression(
    C=0, device="cpu", compute_inference=False, max_iter=200, tol=1e-8,
).fit(X[:600], y[:600])
print("converged:", model.converged_)
print("coef:", np.round(model.coef_, 3))
print("odds ratios:", np.round(np.exp(model.coef_), 3))
```

For this seed, coefficients round to `[0.849, -0.501, 0.138]` and odds ratios to `[2.338, 0.606, 1.148]`. Odds mean the probability of occurrence divided by the probability of non-occurrence, $p/(1-p)$. Holding other features fixed, a unit increase in the first feature multiplies this ratio by about 2.338. For example, a starting probability of 0.5 gives odds of 1; multiplying the odds by 2.338 gives probability $2.338/(1+2.338)\approx0.70$, rather than multiplying 0.5 by 2.338. Different starting probabilities give different probability changes. Check that `converged_` is true before interpreting the estimates.

### 4. Predict on held-out rows

```python
probability = model.predict_proba(X[600:])
print("P(y=1):", np.round(probability[:3, 1], 3))
print("held-out accuracy:", round(float(model.score(X[600:], y[600:])), 3))
```

`predict_proba` generally returns shape `(n_samples, 2)`, here a `(200, 2)` array with class-0 and class-1 probabilities. The first three class-1 probabilities round to `[0.388, 0.306, 0.563]`, and accuracy to `0.65`. `predict` uses a 0.5 threshold; `predict_with_threshold` allows another threshold. For imbalanced outcomes, also examine precision/recall or the precision–recall curve.

### 5. Optional: coefficient uncertainty

Reuse the training data above and refit with inference enabled. Here HC1 requests a score-robust covariance estimate; its assumptions and the interpretation at positive C are discussed under Covariance/Inference below. Changing the covariance convention does not change these coefficients.

```python
model.set_params(compute_inference=True, cov_type="hc1")
model.fit(X[:600], y[:600])
print("95% coefficient intervals:", np.round(model._conf_int, 3))
```

`_conf_int` has shape `(4, 2)`: intercept first, then the three input columns. `_bse`, `_zvalues`, and `_pvalues` use that order too. These are marginal coefficient intervals, not intervals for an individual event probability.
<!-- example-end: logistic-unpenalized -->

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

The intercept is not penalized. For positive `C`, larger values weaken regularization; very large `C` only approximates an unpenalized fit. `C=0` is a special legacy convention that removes the penalty exactly, not infinitely strong regularization. Negative or nonfinite `C` is invalid. Unlike an average-loss objective, multiplying all weights by a positive constant changes the effective penalty when `C>0`; do not transfer regularization values from another API without matching the objective scale.

## Fitting and estimating equations

IRLS and L-BFGS target the same weighted estimating equations:

$$
\sum_i w_i x_i(y_i-p_i)-\alpha_C\beta=0,
\qquad \sum_i w_i(y_i-p_i)=0\quad\text{(with intercept)}.
$$

The slope equation includes the penalty term for the default `C=1.0`; the unpenalized score equation applies only when `C=0` (or as an approximation at very large `C`). Changing between the two solvers does not change this objective or the meaning of `C`.

### Solver choice and convergence

- `solver="auto"` resolves to `"irls"`, preserving the default fitting algorithm.
- `solver="irls"` uses iteratively reweighted least squares. `tol` tests the Euclidean norm of the parameter step.
- `solver="lbfgs"` uses limited-memory BFGS. It divides the **entire objective, including the penalty**, by n without weights or by `sum(sample_weight)` with weights, and tests the Euclidean norm of that scaled objective's gradient against `tol`. A small parameter step alone does not establish convergence.
- `max_iter` caps iterations of the selected solver. Inspect `converged_` and `n_iter_`; failure to meet the stopping rule emits `ConvergenceWarning`. A line-search failure is not reported as convergence.
- `get_params()["solver"]` retains the requested value; fitted `solver_` records `"irls"` or `"lbfgs"`. With inference enabled, `summary()` shows the requested/resolved solver and convergence status. Unsupported solver names raise `ValueError`.

Whole-objective scaling leaves the optimum unchanged but changes the numerical gradient scale. The same `tol` therefore does not imply the same precision or iteration count across IRLS and L-BFGS. Feature scaling can help either algorithm. L-BFGS avoids a Hessian linear-system solve at each optimization iteration; enabled covariance inference still requires curvature calculations.

### CPU example: L-BFGS and C selection

This separate example is self-contained. It fits a weighted model and tunes `C` on the first 300 rows, leaving 100 rows for prediction. The input features already have comparable scales; learned preprocessing would need to be fitted within each CV training fold.

<!-- api-example: logistic-lbfgs -->
```python
import numpy as np
from statgpu.linear_model import LogisticRegression, LogisticRegressionCV

rng_lbfgs = np.random.default_rng(42)
X_lbfgs = rng_lbfgs.normal(size=(400, 4))
p_lbfgs = 1 / (1 + np.exp(-(0.3 + X_lbfgs @ [0.8, -0.5, 0.2, 0.0])))
y_lbfgs = rng_lbfgs.binomial(1, p_lbfgs)
weights_lbfgs = rng_lbfgs.uniform(0.5, 1.5, size=300)

model_lbfgs = LogisticRegression(
    solver="lbfgs", device="cpu", C=1.0,
    max_iter=300, tol=1e-7, compute_inference=False,
).fit(X_lbfgs[:300], y_lbfgs[:300], sample_weight=weights_lbfgs)
print(model_lbfgs.get_params()["solver"], model_lbfgs.solver_, model_lbfgs.converged_)
probabilities_lbfgs = model_lbfgs.predict_proba(X_lbfgs[300:303])
print(probabilities_lbfgs)
assert model_lbfgs.solver_ == "lbfgs" and model_lbfgs.converged_
assert probabilities_lbfgs.shape == (3, 2)
assert np.isfinite(probabilities_lbfgs).all()
assert np.allclose(probabilities_lbfgs.sum(axis=1), 1.0)

cv_lbfgs = LogisticRegressionCV(
    Cs=[0.1, 1.0, 10.0], cv=3, random_state=42,
    solver="lbfgs", device="cpu", max_iter=300, tol=1e-7,
    compute_inference=False,
).fit(X_lbfgs[:300], y_lbfgs[:300], sample_weight=weights_lbfgs)
print(cv_lbfgs.C_, cv_lbfgs.solver_, cv_lbfgs.estimator_.converged_)
print(cv_lbfgs.mean_loss_)
assert cv_lbfgs.solver_ == cv_lbfgs.estimator_.solver_ == "lbfgs"
assert cv_lbfgs.estimator_.converged_
assert cv_lbfgs.mean_loss_.shape == (3,)
assert cv_lbfgs.cv_results_["loss_path"].shape == (3, 3)
assert np.isfinite(cv_lbfgs.cv_results_["loss_path"]).all()
assert np.isclose(cv_lbfgs.best_score_, -np.min(cv_lbfgs.mean_loss_))
```
<!-- example-end: logistic-lbfgs -->

Both fitted solver values are `"lbfgs"`. CV selects the smallest mean validation log-loss; `mean_loss_` stores one value per `Cs_` candidate, `cv_results_["loss_path"]` stores candidate-by-fold losses, and `best_score_` is the negative selected mean loss. `estimator_` is the full-training-data refit; its `converged_` describes that final fit. To compare IRLS, change `solver` to `"auto"` or `"irls"`.

### Backend and CV behavior

The combinations below support binary responses, optional intercepts, sample weights, and L2 penalties. Direct fitting also accepts unpenalized `C=0`; CV searches positive `C` candidates only.

| Requested solver | Resolved solver | NumPy CPU fit / CV | CuPy CUDA fit / CV | Torch CUDA fit / CV |
|---|---|---|---|---|
| `"auto"` | `"irls"` | IRLS / sequential IRLS | IRLS / batched IRLS | IRLS / batched IRLS |
| `"irls"` | `"irls"` | IRLS / sequential IRLS | IRLS / batched IRLS | IRLS / batched IRLS |
| `"lbfgs"` | `"lbfgs"` | L-BFGS / sequential L-BFGS | L-BFGS / sequential GPU L-BFGS | L-BFGS / sequential GPU L-BFGS |

`LogisticRegressionCV` applies the same requested solver to every candidate and the final refit, and exposes its resolved value in `solver_`. `gpu_cv_mixed_precision` affects GPU batched IRLS only; L-BFGS candidate fits use float64 on the selected backend. It does not switch to CPU to run L-BFGS. See the [solver–penalty matrix](../../guides/solver-penalty-matrix.md) and [CV guide](../../guides/cross-validation.md) for related interfaces and fold choices.

## Covariance/Inference

- `cov_type="nonrobust"`: inverse information matrix at `C=0`; inverse penalized curvature when `C>0`.
- `cov_type="hc0"|"hc1"|"hc2"|"hc3"`: robust sandwich covariance variants.
- `cov_type="hac"`: Newey-West (Bartlett) covariance with optional `hac_maxlags`.
- Inference outputs use z-statistic conventions: `_bse`, `_zvalues`, `_pvalues`, `_conf_int`.
- `compute_inference=True` is required for inference fields.
- Likelihood, AIC, BIC, pseudo-R², and `converged_` remain available when `compute_inference=False`; covariance-based fields do not.

Both solvers use the same covariance calculation at the fitted estimate. At positive `C`, inference is computed around the penalized fit; it is not debiasing for the unpenalized population coefficient. Robust choices use the penalized inverse curvature as the sandwich bread. These quantities do not correct shrinkage bias or uncertainty from selecting `C`. For ordinary unpenalized Logit inference use `C=0`, check identification and convergence, and choose covariance assumptions appropriate to the data. `summary()` requires `compute_inference=True`. For CV, inspect inference on the final `estimator_`.

For the summed-loss convention above, let $d_i=(1,x_i^\top)^\top$, $D$ stack the rows $d_i^\top$, and $P=\operatorname{diag}(0,1,\ldots,1)$ (remove the intercept entry when absent). The penalized information and HC0 sandwich are

$$
H=D^\top\operatorname{diag}\{w_i p_i(1-p_i)\}D+\alpha_C P,
\qquad
\widehat V_{\mathrm{HC0}}=H^{-1}\left(\sum_i s_i s_i^\top\right)H^{-1},
\quad s_i=w_i d_i(y_i-p_i).
$$

Nonrobust covariance is $H^{-1}$; HC1–HC3 alter the score-outer-product corrections and HAC adds lagged score products. For coefficient j, $z_j=\hat\theta_j/\sqrt{\widehat V_{jj}}$ and the 95% marginal interval uses the normal 0.975 quantile. This formula does not turn positive-C inference into unpenalized inference.

## Parameters

| Parameter | Default | Description |
|---|---:|---|
| `fit_intercept` | `True` | Whether to fit an intercept |
| `C` | `1.0` | Positive: inverse L2 strength; `0`: unpenalized legacy convention |
| `max_iter` | `100` | Maximum iterations of the selected solver |
| `tol` | `1e-4` | Finite positive stopping tolerance; step norm for IRLS, scaled-objective gradient norm for L-BFGS |
| `device` | `"auto"` | `cpu` / `cuda` / `torch` / `auto` |
| `n_jobs` | `None` | Shared estimator configuration; this class does not use it to control parallel fitting or select a solver |
| `compute_inference` | `True` | Whether to compute inference stats |
| `cov_type` | `"nonrobust"` | `nonrobust` / `hc0` / `hc1` / `hc2` / `hc3` / `hac` |
| `hac_maxlags` | `None` | Max lag for `cov_type="hac"`; default follows Newey-West style heuristic |
| `gpu_memory_cleanup` | `False` | Best-effort CuPy pool cleanup after each fit |
| `solver` | `"auto"` | `auto` / `irls` / `lbfgs`; `auto` resolves to `irls` |

This table covers `LogisticRegression`. The [complete CV constructor reference](../../reference/linear-model-api.md#logisticregressioncv) includes grid/fold controls and `gpu_cv_mixed_precision`. CV does not accept `hac_maxlags`; final-refit HAC uses the automatic lag rule.

## GPU use

Reuse the imports and `X`, `y` from the completed [CPU example](#cpu-example). To request the same unpenalized prediction fit on CuPy CUDA:

```python
import cupy as cp

X_gpu = cp.asarray(X[:600])
y_gpu = cp.asarray(y[:600])
model_gpu = LogisticRegression(
    C=0, device="cuda", compute_inference=False, max_iter=200, tol=1e-8,
).fit(X_gpu, y_gpu)
p_gpu = model_gpu.predict_proba(cp.asarray(X[600:603]))[:, 1]
```

`device="torch"` selects Torch CUDA. Explicit `cuda`/`torch` requests require the corresponding installed package and usable CUDA device; they do not silently fall back to CPU. Prediction arrays use the selected supported backend; plot helpers convert results to NumPy for rendering. See [device and memory](../../guides/device-and-memory.md) for installation/routing details.

## strict/approx difference

No separate approx inference mode is exposed in this API. Robust covariance choice (`hc*`/`hac`) is the main practical trade-off between assumptions and computational cost.

## Outputs

- Coefficients/convergence: `intercept_`, `coef_`, `n_iter_`, `solver_`, `converged_`
- Inference: `_bse`, `_zvalues`, `_pvalues`, `_conf_int`
- Fit/metrics: `loglikelihood`, `loglikelihood_null`, `aic`, `bic`, `pseudo_rsquared`, `accuracy`, `precision`, `recall`, `f1`, `auc`, `average_precision`
- Prediction methods: `predict_proba`, `predict`, `predict_with_threshold`
- Evaluation methods: `confusion_matrix`, `classification_table`, `roc_curve`, `roc_auc_score`, `precision_recall_curve`, `average_precision_score`, `evaluate_classification`
- Plot helpers: `plot_roc_curve`, `plot_precision_recall_curve` (`matplotlib` optional dependency)
- CV: `C_`, `Cs_`, `mean_loss_`, `cv_results_`, `best_score_`, `cv_selected_device_`, `solver_`, and final `estimator_`; final stopping status is `estimator_.converged_`

## Parameter choices and common problems

- Scale continuous features when their units should not determine the L2 penalty. Learn preprocessing from training data only and apply the same transformation at prediction time.
- For predictive use, choose positive `C` and the decision threshold using validation data; the smallest training error is not a reliable tuning rule. Keep the final test set separate.
- `C=0` is useful for ordinary unpenalized inference. Perfect or near separation can make unpenalized coefficients unstable or nonfinite in the limit; a larger iteration budget alone does not fix separation or collinearity.
- If `converged_` is false, inspect scaling, separation and rank before increasing `max_iter` or loosening `tol`. Do not report p-values as reliable simply because attributes exist.
- Use HC covariance for an appropriate heteroskedastic/score-robust analysis; HAC additionally depends on observation order and lag selection. Shuffling time-ordered observations changes the meaning of HAC.
- If `summary()` reports inference unavailable, refit with `compute_inference=True`; prediction and likelihood diagnostics can still be used with inference disabled.
- `roc_curve`, `roc_auc_score` and `evaluate_classification` require both classes in the evaluation labels. `include_curves=False` only omits curve arrays; the combined evaluator still computes ROC AUC and raises for a one-class subset. Use `classification_table` or `confusion_matrix` for threshold metrics on such a subset.
- `precision_recall_curve` and `average_precision_score` need at least one positive label; they raise `ValueError` on an all-zero evaluation subset. All-one subsets are accepted, but their perfect precision/AP does not measure the ability to distinguish the two classes. The corresponding plot and training `average_precision` property follow the same restriction.
- Binary labels must be 0/1; encode other class names first. Missing/nonfinite inputs and mismatched shapes must be corrected before fitting.

<a id="api-reference-and-validation"></a>

## API reference and comparisons

Use the [complete LogisticRegression method reference](../../reference/linear-model-api.md#logisticregression) for threshold limits, metric dictionaries, curve shapes and plotting returns; [LogisticRegressionCV](../../reference/linear-model-api.md#logisticregressioncv) has its own constructor and [selection/result contract](../../reference/linear-model-api.md#cv-methods-and-results). Shared inherited methods are in the [estimator API](../../reference/estimator-api.md).

The parameter table above covers the constructor. The fitting interface is `fit(X, y, sample_weight=None)`; inherited `get_params` / `set_params` support estimator configuration. `score(X, y)` returns accuracy. `summary()` displays the inference report. Training-data metric properties include `accuracy`, `precision`, `recall`, `f1`, `auc`, and `average_precision`; use the evaluation methods on separate data for generalization assessment.

Complete method signatures, returns, and estimator docstrings are available in the [public class API source](../../../../statgpu/linear_model/wrappers/_logistic.py); `help(LogisticRegression)` also exposes that API in the installed version. The standalone distribution is binary; inference uses large-sample z statistics. This estimator has no dedicated formula argument in `fit`.

For an ordinary unweighted, unpenalized comparison with `statsmodels.Logit`, use `C=0` and match the response rows, encoded design/intercept, covariance convention (including any finite-sample correction), and convergence settings. Check that both fits converge. For positive `C`, first establish an equivalent penalized reference: align the summed-loss normalization, slope-only L2 penalty, and penalty scale before comparing coefficients or covariance. Matching numerical settings does not make different covariance assumptions equivalent.

## References

- McCullagh, P., & Nelder, J. A. (1989). *Generalized Linear Models* (2nd ed.). Chapman & Hall/CRC.
- Hosmer, D. W., Lemeshow, S., & Sturdivant, R. X. (2013). *Applied Logistic Regression* (3rd ed.). Wiley.

