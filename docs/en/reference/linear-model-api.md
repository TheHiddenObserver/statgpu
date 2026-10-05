# Linear-model API reference

> Language: English  
> Last updated: 2026-10-05  
> Switch: [Chinese](../../cn/reference/linear-model-api.md)

All classes here import from `statgpu` or `statgpu.linear_model`. `X` is finite numeric `(n,p)` data; prediction uses the fitted feature order. Analytic weights are a finite nonnegative `(n,)` vector with positive sum. For weights after formula row filtering, see [formula inputs](#formula-inputs). Explicit GPU use requires the corresponding installed CUDA backend; see [device and memory](../guides/device-and-memory.md).

Every class here inherits [get_params/set_params and four inference helpers](estimator-api.md). The generic helpers accept supplied data/p-values; their presence does not guarantee that a CV wrapper publishes coefficient inference arrays on itself. For CV, read inference from `estimator_`.

The public `coef_` arrays and successful coefficient-inference arrays of these direct estimators are stored as NumPy arrays, including after GPU fitting; scalar intercepts are Python numbers. This storage convention differs from backend-native prediction/evaluation arrays. There is no constructor `dtype` control or blanket promise to preserve the input floating dtype; use the actual returned array dtype when integrating with other libraries.

## LinearRegression

```text
LinearRegression(fit_intercept=True, device='auto', n_jobs=None, compute_inference=True, gpu_memory_cleanup=False, cov_type='nonrobust', hac_maxlags=None)
```
| Parameter | Default | Meaning and restrictions |
|---|---|---|
| `fit_intercept` | `True` | Fit an intercept; formula syntax overrides the array-fit choice. |
| `device` | `"auto"` | `cpu`/`cuda` (CuPy)/`torch` (Torch CUDA)/`auto`; explicit GPU requests require a usable backend. |
| `n_jobs` | `None` | Shared CPU worker setting; it does not select a solver or promise parallel fitting in these wrappers. |
| `compute_inference` | `True` | Enable supported post-fit uncertainty; `summary()` needs successful inference. |
| `gpu_memory_cleanup` | `False` | Best-effort backend memory-pool cleanup after fitting. |
| `cov_type` | `"nonrobust"` | `nonrobust`, `hc0`, `hc1`, `hc2`, `hc3`, `hac`; see each model’s inference contract. |
| `hac_maxlags` | `None` | Nonnegative integer HAC lag; automatic `floor(4*(n/100)**(2/9))`, capped at n−1, for Linear/Logistic. |

`fit(X=None, y=None, sample_weight=None, formula=None, data=None)` returns `self`. Use arrays or `formula` plus `data`; do not supply both. Currently, a formula fit silently takes its X/y from `data` even if array X/y are also supplied; it does not reject the conflict. A single column response is flattened by fit; genuine multi-output `y` has shape `(n,t)`.

| Method/result | Contract |
|---|---|
| `predict(X)` | `(m,)` for one target or `(m,t)` for multiple targets; NumPy on CPU, native supported GPU array on GPU. Formula-fitted models also accept a prediction DataFrame. The device is resolved again at prediction; use an explicit device if later global-device changes must not change output placement. |
| `score(X,y)` | Unweighted R²; averages individual target R² in multi-output. No `sample_weight` argument. Pass one-dimensional single-target `y`: a column-shaped response currently broadcasts incorrectly. Constant targets receive 0 rather than a meaningful explained-variance interpretation. |
| `summary()` | Prints a single-output table; returns `None`. Requires successful inference and positive residual degrees of freedom. |
| `coef_`, `intercept_`, `rank_` | Shapes `(p,)`/scalar, or `(t,p)`/`(t,)`; `rank_` is fitted design rank. |
| `rsquared`, `rsquared_adj` | Training R² (using fitted weights; pooled for multi-output) and residual-DoF adjustment. These are distinct from held-out `score`. |
| `fvalue`, `f_pvalue` | Single-target residual-based F diagnostic, not an HC/HAC joint Wald test. Weighted multi-output access currently raises `TypeError`; unweighted pooled output is not a joint multivariate test. |
| `llf`, `aic`, `bic` | Unweighted single-target Gaussian likelihood/information criteria. Weighted values omit the WLS log-weight normalization and depend on weight scale. Multi-output AIC/BIC are `None`, and pooled `llf` is not a joint multivariate likelihood. See [diagnostic limits](../models/linear-regression.md#diagnostic-limitations-with-weights-or-multiple-targets). |
| `_bse`, `_tvalues`, `_pvalues`, `_conf_int` | Intercept first when fitted. For k parameters: `(k,)` and CI `(k,2)`; CPU multi-output uses `(k,t)` and `(k,t,2)`. Inference-disabled/unavailable arrays are `None`. |

Classical covariance uses t-reference inference; HC/HAC uses normal-reference inference despite the `_tvalues` name. CPU multi-output inference is supported; GPU multi-output fitting requires `compute_inference=False`. See the [learner page](../models/linear-regression.md) for interpretation and assumptions.

## LogisticRegression

```text
LogisticRegression(fit_intercept=True, C=1.0, max_iter=100, tol=0.0001, device='auto', n_jobs=None, compute_inference=True, cov_type='nonrobust', gpu_memory_cleanup=False, hac_maxlags=None)
```
| Parameter | Default | Meaning and restrictions |
|---|---|---|
| `fit_intercept` | `True` | Fit an unpenalized intercept; this class has no formula interface. |
| `device` | `"auto"` | `cpu`/`cuda` (CuPy)/`torch` (Torch CUDA)/`auto`; explicit GPU requests require a usable backend. |
| `n_jobs` | `None` | Shared CPU worker setting; it does not select a solver or promise parallel fitting in these wrappers. |
| `compute_inference` | `True` | Enable supported post-fit uncertainty; `summary()` needs successful inference. |
| `gpu_memory_cleanup` | `False` | Best-effort backend memory-pool cleanup after fitting. |
| `cov_type` | `"nonrobust"` | `nonrobust`, `hc0`, `hc1`, `hc2`, `hc3`, `hac`; see each model’s inference contract. |
| `hac_maxlags` | `None` | Nonnegative integer HAC lag; automatic `floor(4*(n/100)**(2/9))`, capped at n−1, for Linear/Logistic. |
| `C` | `1.0` | Finite nonnegative real. Positive means inverse L2 strength; zero removes the penalty exactly. |
| `max_iter` | `100` | Positive integer IRLS iteration budget. |
| `tol` | `1e-4` | Finite positive convergence tolerance. |

`fit(X,y,sample_weight=None)` returns `self`; binary `y` must be 0/1 with n entries. There is no formula argument, multiclass mode, solver selector, L1 or Elastic Net penalty in this standalone class. `coef_` is `(p,)`; `intercept_` is scalar. `converged_` and `n_iter_` describe optimization, not statistical adequacy.

| Method | Arguments, defaults and return |
|---|---|
| `predict_proba(X)` | `(m,2)` probabilities for classes 0 and 1. |
| `predict(X)` | `(m,)` integer labels; class 1 when probability ≥0.5. |
| `predict_with_threshold(X,threshold=0.5)` | Same labels with a finite real threshold in `[0,1]`; boolean thresholds are rejected. |
| `score(X,y)` | Python float unweighted accuracy; no threshold or weights argument. |
| `confusion_matrix(X,y,threshold=0.5)` | `(2,2)` array `[[tn,fp],[fn,tp]]`: rows true labels, columns predictions. |
| `classification_table(X,y,threshold=0.5)` | Dictionary: `tn`, `fp`, `fn`, `tp`, `accuracy`, `precision`, `recall`, `specificity`, `f1`, `support_negative`, `support_positive`. Undefined ratio denominators are represented by 0. |
| `roc_curve(X,y)` | Tuple `(fpr,tpr,thresholds)`; equal-length arrays, decreasing thresholds beginning at infinity. Needs both classes for an informative ROC curve. |
| `roc_auc_score(X,y)` | Scalar trapezoidal ROC area. |
| `precision_recall_curve(X,y)` | Tuple `(precision,recall,thresholds)`; equal-length arrays here, decreasing thresholds beginning at infinity with precision=1, recall=0. Do not assume another library's array-length convention. |
| `average_precision_score(X,y)` | Scalar precision integrated over recall increments. |
| `evaluate_classification(X,y,threshold=0.5,include_curves=True)` | One probability evaluation; dictionary described below. |
| `plot_roc_curve(X,y,ax=None,label=None)` | Requires matplotlib; creates or draws on supplied Axes, returns Axes. `label=None` includes computed AUC. |
| `plot_precision_recall_curve(X,y,ax=None,label=None)` | Same plot contract, with average precision in the default label. |
| `summary()` | Prints the inference report and returns `None`; requires successful enabled inference. |

Evaluation arrays/scalars use NumPy on CPU and the supported CuPy/Torch backend on GPU; `score` is a Python float. Plotting transfers small results to NumPy. None of these evaluation methods accepts `sample_weight`; weighting fit does not make evaluation weighted.

`evaluate_classification` always returns `threshold`, `confusion_matrix`, `classification_table`, `roc_auc`, `average_precision`. With `include_curves=True`, it adds `roc_curve={fpr,tpr,thresholds}` and `precision_recall_curve={precision,recall,thresholds}`. For external probabilities, use `statgpu.metrics.evaluate_binary_classification(y_true,y_score,threshold=0.5,include_curves=True,backend="auto")` or the top-level alias `statgpu.evaluate_binary_classification`; supply one-dimensional class-1 scores.

Training properties are `loglikelihood`, `loglikelihood_null`, `aic`, `bic`, `pseudo_rsquared`, `accuracy`, `precision`, `recall`, `f1`, `auc`, `average_precision`. `pseudo_rsquared` is McFadden's `1-loglikelihood/loglikelihood_null`, not R² or accuracy. Inference arrays `_bse`, `_zvalues`, `_pvalues` have `(k,)` and `_conf_int` has `(k,2)`, intercept first. They use normal-reference inference around the fitted penalized coefficients when C>0 and remain unavailable when inference is disabled. Training metrics do not measure generalization.

## ElasticNet

```text
ElasticNet(alpha=1.0, l1_ratio=0.5, fit_intercept=True, max_iter=1000, tol=0.0001, stopping='coef_delta', device='auto', n_jobs=None, solver='fista', cpu_solver='fista', lipschitz_L=None, gpu_memory_cleanup=False, compute_inference=False, inference_method='debiased', cov_type='nonrobust', hac_maxlags=None, *, nodewise_alpha=None)
```
| Parameter | Default | Meaning and restrictions |
|---|---|---|
| `alpha` | `1.0` | Nonnegative total regularization strength under average squared loss. |
| `l1_ratio` | `0.5` | Mixing proportion in `[0,1]`; 0 gives the Ridge objective, 1 the Lasso objective. |
| `fit_intercept` | `True` | Unpenalized intercept; formula syntax controls formula fits. |
| `max_iter` | `1000` | Positive solver iteration budget. |
| `tol` | `1e-4` | Positive convergence tolerance. |
| `stopping` | `"coef_delta"` | `coef_delta` or `kkt`; coefficient movement or KKT-based stopping. |
| `device` | `"auto"` | `cpu`/`cuda` (CuPy)/`torch` (Torch CUDA)/`auto`; explicit GPU requests require a usable backend. |
| `n_jobs` | `None` | Shared CPU worker setting; it does not select a solver or promise parallel fitting in these wrappers. |
| `solver` | `"fista"` | Backend-neutral solver. Other values are combination-specific; see the solver–penalty matrix. |
| `cpu_solver` | `"fista"` | Deprecated compatibility argument; use `solver`. Explicit legacy requests can warn/reject; see the migration guide. |
| `lipschitz_L` | `None` | Optional positive upper bound for the smooth-loss gradient Lipschitz constant; normally let the solver estimate it. |
| `gpu_memory_cleanup` | `False` | Best-effort backend memory-pool cleanup after fitting. |
| `compute_inference` | `False` | Default is prediction/estimation only. |
| `inference_method` | `"debiased"` | `debiased`, `post_selection_ols`, `bootstrap`; `auto` resolves to `debiased`. Deprecated `cpu_ols`/`gpu_ols` mean `post_selection_ols`. |
| `nodewise_alpha` | `None` | Keyword-only finite positive precision-tuning value or automatic design-side rule; never changes prediction coefficients. |
| `cov_type` | `"nonrobust"` | Method-specific behavior below; HC/HAC is not implemented by the debiased path. |
| `hac_maxlags` | `None` | Nonnegative HAC lag for `post_selection_ols`; does not turn debiased inference into HAC inference. |

| Method/result | Contract |
|---|---|
| `fit(X=None,y=None,sample_weight=None,initial_coef=None,**kwargs)` | Returns `self`; `y` is one-dimensional. `kwargs` accepts `formula=None,data=None`. `initial_coef` is a length-p starting vector. Omission on a later fit currently retains a previously supplied vector; use a new estimator for default initialization, especially after changing p. |
| `predict(X,return_cpu=True)` | `(m,)` predictions. Default returns NumPy even after GPU fitting; `False` preserves the fitted NumPy/CuPy/Torch backend. |
| `score(X,y,sample_weight=None)` | Python float R²; optional evaluation weights define weighted mean and residual sums. They are not automatically inherited from training. Check weights are finite, nonnegative, length m and have positive sum: this score path currently accepts negative weights and can return invalid R² above 1. Use flat y and NumPy/host evaluation y and weights; GPU-native response conversion is not supplied by this squared-error score path. |
| `summary()` | Prints coefficient/inference reporting and returns `None`; requires successful enabled inference. |
| `coef_`, `intercept_`, `n_iter_` | Penalized prediction slopes `(p,)`, scalar intercept, iteration count. Reaching the iteration limit does not establish convergence. |
| `_params`, `_bse`, `_tvalues`, `_zvalues`, `_pvalues`, `_conf_int` | Reporting parameters/uncertainty, not necessarily prediction coefficients. Usually `(k,)` and `(k,2)` for CI, with intercept first; availability depends on the inference method. |
| `_inference_result`, `nodewise_alpha_` | Structured result (`params`, `bse`, `statistic`, `pvalues`, `conf_int`, `method`, `distribution`, `metadata`) and resolved multi-feature node-wise tuning. |
| `inference_requested_method_`, `inference_resolved_method_`, `inference_method_`, `inference_target_`, `penalty_conditioning_`, `penalty_selection_adjusted_` | Successful inference records the requested/resolved procedure and conditional target. These are reporting attributes, not extra constructor parameters. |
| `rsquared`, `rsquared_adj`, `fvalue`, `f_pvalue`, `llf`, `aic`, `bic` | Available fit diagnostics. AIC/BIC/F are compatibility plug-in summaries, not general penalty-aware effective-DoF or selective-inference criteria; missing state can yield `None`/NaN. |

### Covariance and inference behavior

| `inference_method` | `cov_type`/weights | Meaning |
|---|---|---|
| `debiased` (also `auto`) | Uses a residual-variance/node-wise model-based formula. Current HC/HAC settings are accepted but **do not change this covariance**; do not interpret them as robust inference. | Normal-reference marginal intervals for corrected coefficients; assumptions include design/sparsity/model conditions. Public prediction coefficients stay penalized. |
| `post_selection_ols` | `nonrobust`, HC0–HC3 or HAC, with optional analytic weights | Refit OLS/WLS on active slopes; t reference for nonrobust, normal for HC/HAC. No general selection adjustment. |
| `bootstrap` | Only `cov_type="nonrobust"`, no sample weights | Residual bootstrap of the same penalized model at fixed tuning; percentile intervals. |

For reproducible residual bootstrap, set `model.bootstrap_random_state = seed` and `model.n_bootstrap = B` **before fit**, with B≥2 (default 200). These are supplementary attributes, not accepted constructor/get_params/set_params arguments. `random_state` is not an ElasticNet constructor parameter. The returned method is `residual_bootstrap`, with a `bootstrap_percentile` distribution. Bootstrap does not repeat CV or retune the penalty. Each resample refits the full penalized design, so its nonzero coefficient set can change. These are not generally selection-adjusted intervals. Generic `bootstrap_statistic` is a different operation.

## ElasticNetCV

```text
ElasticNetCV(l1_ratio=0.5, *, alphas=None, n_alphas=100, alpha_min_ratio=0.001, cv=5, cv_splits=None, fit_intercept=True, device='auto', n_jobs=None, compute_inference=False, max_iter=1000, tol=0.0001, random_state=None, nodewise_alpha=None)
```
| Parameter | Default | Meaning and restrictions |
|---|---|---|
| `l1_ratio` | `0.5` | Scalar or candidate sequence in `[0,1]`; a scalar tunes alpha only. Use explicit alphas for zero or near-zero ratios; see grid limitations below. |
| `alphas` | `None` | Positive finite candidate sequence; omitted: construct a separate automatic grid for each ratio. The automatic rule is L1-based, not a Ridge-specific search. |
| `n_alphas` | `100` | Positive integer grid size when `alphas=None`. |
| `alpha_min_ratio` | `1e-3` | Positive minimum-to-maximum alpha ratio for the automatic log grid; normally in `(0,1]`. |
| `cv` | `5` | Integer fold count ≥2; use enough observations in every training/validation split. |
| `cv_splits` | `None` | Explicit iterable of `(train_indices, validation_indices)` overrides generated folds. |
| `fit_intercept` | `True` | Intercept in fold fits and final refit. |
| `device` | `"auto"` | `cpu`/`cuda` (CuPy)/`torch` (Torch CUDA)/`auto`; explicit GPU requests require a usable backend. |
| `n_jobs` | `None` | Accepted configuration; candidate parallelism is not implemented by this class. |
| `compute_inference` | `False` | Debiased inference on final full-data refit only; no inference-method selector. |
| `max_iter` | `1000` | Per-fit iteration budget. |
| `tol` | `1e-4` | Per-fit convergence tolerance. |
| `random_state` | `None` | Integer seed for generated shuffled K-fold splits. |
| `nodewise_alpha` | `None` | Final-refit precision tuning only; does not change CV grid or scoring. |


## LogisticRegressionCV

```text
LogisticRegressionCV(Cs=None, n_Cs=100, C_min_ratio=0.001, cv=5, cv_splits=None, fit_intercept=True, max_iter=100, tol=0.0001, device='auto', n_jobs=None, compute_inference=True, cov_type='nonrobust', gpu_memory_cleanup=False, random_state=None, gpu_cv_mixed_precision=True)
```
| Parameter | Default | Meaning and restrictions |
|---|---|---|
| `Cs` | `None` | Positive finite C candidates; omitted: automatic data-dependent log grid. `C=0` is not a CV candidate. |
| `n_Cs` | `100` | Positive integer automatic grid size. |
| `C_min_ratio` | `1e-3` | Positive minimum-to-maximum C ratio; normally `(0,1]`. |
| `cv` | `5` | Integer fold count ≥2; use enough observations in every training/validation split. |
| `cv_splits` | `None` | Explicit iterable of `(train_indices, validation_indices)` overrides generated folds. |
| `fit_intercept` | `True` | Intercept in fold fits and final refit. |
| `max_iter` | `100` | Per-fit IRLS iteration budget. |
| `tol` | `1e-4` | Per-fit convergence tolerance. |
| `device` | `"auto"` | `cpu`/`cuda` (CuPy)/`torch` (Torch CUDA)/`auto`; explicit GPU requests require a usable backend. |
| `n_jobs` | `None` | Shared CPU worker setting; it does not select a solver or promise parallel fitting in these wrappers. |
| `compute_inference` | `True` | Inference on final refit only, conditional on selected C. |
| `cov_type` | `"nonrobust"` | `nonrobust`, `hc0`, `hc1`, `hc2`, `hc3`, `hac`; see each model’s inference contract. |
| `gpu_memory_cleanup` | `False` | Best-effort backend memory-pool cleanup after fitting. |
| `random_state` | `None` | Integer seed for generated shuffled K-fold splits. |
| `gpu_cv_mixed_precision` | `True` | GPU CV mixed-precision option; does not choose a different statistical objective. |


## CV methods and results
Both classes use `fit(X,y,sample_weight=None) -> self`, `predict(X)`, `score(X,y)`, and inherited `summary()`. Fit requires array/design-matrix input, not `formula`/`data`. `score` is unweighted held-out R² for ElasticNetCV and accuracy for LogisticRegressionCV, **not** the CV selection loss. ElasticNetCV.predict returns NumPy through the final estimator's default; LogisticRegressionCV also exposes `predict_proba(X) -> (m,2)` on the final model's backend. Other classification methods are on `estimator_`.

Generated folds are shuffled K-fold, not automatically stratified/grouped/time-aware. Supply explicit reusable index-pair lists for those designs. Fold weights affect both training and validation loss; every relevant split needs positive weight mass. Fit preprocessing only within each training fold. These wrappers do not accept a preprocessing pipeline or a scoring callable. For fold-local learned scaling, use an external CV loop/pipeline rather than scaling all rows before internal CV.

Candidates are refitted on all provided training rows after selecting minimum mean validation loss. Inference runs on this final `estimator_` only and conditions on the selected tuning; it does not adjust for tuning uncertainty. Predictions and outer `coef_`/`intercept_` come from that final fit. `summary()` delegates to it.

### Grid and edge behavior

Provide positive finite grid entries deliberately. Current CV code filters invalid/nonpositive alpha/C entries and replaces an empty surviving grid with an automatic grid. ElasticNetCV also filters l1_ratio values outside `[0,1]`, using 0.5 if none survive. Do not rely on these fallbacks to validate a scientific search. Direct LogisticRegression's special `C=0` is excluded from CV. Automatic ElasticNet alpha grids can differ by ratio; inspect the returned mapping.

For ElasticNetCV, the automatic maximum is the largest absolute weighted-centered
X/y cross-product divided by `sum(sample_weight) * max(l1_ratio, 1e-6)` (unit
weights when omitted), with a lower floor of `1e-6`. At `l1_ratio=0`, this can
produce only huge penalties and nearly constant predictions even when a useful
Ridge fit exists. Supply explicit `alphas` for zero/near-zero ratios. The rule
centers X/y regardless of `fit_intercept`; use an explicit grid for no-intercept
fits too. A finite CV score only identifies the best candidate in the supplied
range, not an adequate tuning range.

Use at least four rows and enough rows per fold. The current ElasticNetCV path below four rows does not provide a usable result; increase the data or fit a direct estimator. LogisticRegressionCV with fewer than four rows or only one C candidate performs a final fit without an informative CV comparison: loss arrays and `best_score_` are NaN.

### CV results

Let r be the number of l1 ratios, a the maximum candidate alpha count, c the C count, and f the fold count.

| Class/result | Shape and meaning |
|---|---|
| ElasticNetCV `alpha_`, `l1_ratio_` | Selected scalar tuning values. |
| ElasticNetCV `cv_results_` | `mse_path` `(r,a,f)`, `mean_mse`/`std_mse` `(r,a)`, `alphas` dictionary `{ratio_index: alpha_array}`, `l1_ratios` `(r,)`, scalar `best_alpha`, `best_l1_ratio`. Alpha axis follows each stored grid. |
| ElasticNetCV `best_score_` | **Negative** selected mean MSE; larger is better. Not the positive loss or final-model R². |
| ElasticNetCV `nodewise_alpha_` | Final estimator's resolved precision tuning, or `None`. |
| LogisticRegressionCV `C_`, `Cs_` | Selected scalar and actual candidate array `(c,)`. |
| LogisticRegressionCV `cv_results_` | Only `loss_path` `(c,f)`; no `mean_loss` key. |
| LogisticRegressionCV `mean_loss_`, `best_score_` | Mean log-loss `(c,)` and its selected **negative** value; not final-model accuracy. |
| Both: `coef_`, `intercept_`, `n_iter_`, `estimator_`, `cv_selected_device_` | Final prediction slopes `(p,)`, scalar intercept, final iteration count, fitted direct estimator, and final device selection. |

ElasticNetCV has no direct-fit `solver`, `stopping`, `inference_method`, or `cov_type` arguments. LogisticRegressionCV has no `hac_maxlags` argument; HAC therefore uses the final estimator's automatic lag rule. Do not copy an entire direct-estimator constructor into CV.

### A reproducible CPU tuning example

<!-- api-example: linear-cv -->
```python
import numpy as np
from statgpu.linear_model import ElasticNetCV, LogisticRegressionCV

rng = np.random.default_rng(19)
X = rng.normal(size=(160, 3))
y = 1 + X @ np.array([1.0, -0.5, 0.0]) + rng.normal(scale=0.4, size=160)
enet = ElasticNetCV(
    l1_ratio=[0.3, 0.7], alphas=[0.03, 0.1], cv=3,
    device="cpu", random_state=7,
).fit(X[:120], y[:120])
print(enet.alpha_, enet.l1_ratio_, enet.score(X[120:], y[120:]))
assert np.isclose(enet.best_score_, -np.nanmin(enet.cv_results_["mean_mse"]))

probability = 1 / (1 + np.exp(-X[:, 0]))
y_binary = rng.binomial(1, probability)
logit = LogisticRegressionCV(
    Cs=[0.1, 1.0], cv=3, device="cpu", random_state=7,
    compute_inference=False,
).fit(X[:120], y_binary[:120])
report = logit.estimator_.evaluate_classification(X[120:], y_binary[120:], include_curves=False)
print(logit.C_, report["classification_table"]["accuracy"])
assert set(logit.cv_results_) == {"loss_path"}
assert np.isclose(logit.best_score_, -np.nanmin(logit.mean_loss_))
```

The printed score is evaluated on the untouched final 40 rows. The assertions explain the sign/schema rather than promising a particular selected model for every dataset. These simulated features already share a common scale; no full-data learned preprocessing is applied.

## Formula inputs
Only LinearRegression and direct ElasticNet here support formulas. For both, supply only `formula`/`data`, not simultaneous array X/y; current formula parsing replaces those arrays without a conflict error. Install optional pandas/patsy. `formula="y ~ x + C(group)"` supplies numeric/categorical terms; `~ 0 + ...` removes the intercept regardless of the constructor. Interactions/transforms follow Patsy syntax. Formula fitting can drop rows with missing terms. Weights may describe all original data rows or exactly the retained rows; matching is positional, not by arbitrary Series labels.

Prediction DataFrames rebuild the stored design and categorical levels. Unknown levels or missing values that would drop prediction rows raise; array prediction must supply the already encoded non-intercept columns in training order. Keep transformations and level definitions consistent. LogisticRegression and both CV wrappers have no formula argument; construct a suitable design first, avoiding preprocessing leakage.

<!-- api-example: formula-models -->
```python
import numpy as np
import pandas as pd
from statgpu import LinearRegression, ElasticNet

df = pd.DataFrame({"x": np.arange(12.0), "group": ["a", "b"] * 6})
df["y"] = 1.0 + 2.0 * df["x"] + (df["group"] == "b").astype(float)
for cls in (LinearRegression, ElasticNet):
    model = cls(device="cpu", compute_inference=False)
    model.fit(formula="y ~ x + C(group)", data=df)
    assert model.predict(df.iloc[:3]).shape == (3,)
```
