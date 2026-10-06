# Linear-model API reference

> Language: English  
> Last updated: 2026-10-06  
> Switch: [Chinese](../../cn/reference/linear-model-api.md)

Classes here import from `statgpu` or `statgpu.linear_model`, except `SCADRegression` and `MCPRegression`, which are exported only by `statgpu.linear_model`. `X` is finite numeric `(n,p)` data; prediction uses the fitted feature order. Analytic weights are a finite nonnegative `(n,)` vector with positive sum. For weights after formula row filtering, see [formula inputs](#formula-inputs). Explicit GPU use requires the corresponding installed CUDA backend; see [device and memory](../guides/device-and-memory.md).

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
| `roc_curve(X,y)` | Tuple `(fpr,tpr,thresholds)`; equal-length arrays, decreasing thresholds beginning at infinity. Raises `ValueError` if the evaluation labels do not contain both classes. |
| `roc_auc_score(X,y)` | Scalar trapezoidal ROC area. |
| `precision_recall_curve(X,y)` | Tuple `(precision,recall,thresholds)`; equal-length arrays here, decreasing thresholds beginning at infinity with precision=1, recall=0. Do not assume another library's array-length convention. Requires at least one positive label; all-zero evaluation y raises `ValueError`, while all-one y is accepted. |
| `average_precision_score(X,y)` | Scalar precision integrated over recall increments; same positive-label requirement as the precision–recall curve. |
| `evaluate_classification(X,y,threshold=0.5,include_curves=True)` | One probability evaluation; dictionary described below. |
| `plot_roc_curve(X,y,ax=None,label=None)` | Requires matplotlib; creates or draws on supplied Axes, returns Axes. `label=None` includes computed AUC. |
| `plot_precision_recall_curve(X,y,ax=None,label=None)` | Same plot contract, with average precision in the default label. |
| `summary()` | Prints the inference report and returns `None`; requires successful enabled inference. |

Evaluation arrays/scalars use NumPy on CPU and the supported CuPy/Torch backend on GPU; `score` is a Python float. Plotting transfers small results to NumPy. None of these evaluation methods accepts `sample_weight`; weighting fit does not make evaluation weighted.

`evaluate_classification` always returns `threshold`, `confusion_matrix`, `classification_table`, `roc_auc`, `average_precision`. It requires both classes in evaluation y even with `include_curves=False`, because scalar ROC AUC is still computed; otherwise it raises `ValueError`. For a one-class subset use `classification_table` or `confusion_matrix` for threshold metrics. With `include_curves=True`, it adds `roc_curve={fpr,tpr,thresholds}` and `precision_recall_curve={precision,recall,thresholds}`. For external probabilities, use `statgpu.metrics.evaluate_binary_classification(y_true,y_score,threshold=0.5,include_curves=True,backend="auto")` or the top-level alias `statgpu.evaluate_binary_classification`; supply one-dimensional class-1 scores.

Training properties are `loglikelihood`, `loglikelihood_null`, `aic`, `bic`, `pseudo_rsquared`, `accuracy`, `precision`, `recall`, `f1`, `auc`, `average_precision`. `pseudo_rsquared` is McFadden's `1-loglikelihood/loglikelihood_null`, not R² or accuracy. Inference arrays `_bse`, `_zvalues`, `_pvalues` have `(k,)` and `_conf_int` has `(k,2)`, intercept first. They use normal-reference inference around the fitted penalized coefficients when C>0 and remain unavailable when inference is disabled. Training metrics do not measure generalization.

## Lasso

```text
Lasso(alpha=1.0, fit_intercept=True, max_iter=1000, tol=0.0001, stopping='coef_delta', inference_method='debiased', n_bootstrap=200, bootstrap_random_state=None, enable_simultaneous_inference=False, simultaneous_method='maxz_bootstrap', simultaneous_alpha=0.05, simultaneous_n_bootstrap=1000, simultaneous_random_state=None, simultaneous_include_intercept=False, device='auto', n_jobs=None, compute_inference=True, solver='fista', cpu_solver='coordinate_descent', lipschitz_L=None, admm_rho=1.0, gpu_memory_cleanup=False, *, nodewise_alpha=None)
```

| Parameter | Default | Description |
|---|---:|---|
| `alpha` | `1.0` | Nonnegative L1 regularization strength under average squared loss. |
| `fit_intercept` | `True` | Whether to fit an intercept. |
| `max_iter` | `1000` | Maximum optimization iterations. |
| `tol` | `1e-4` | Convergence tolerance. |
| `stopping` | `"coef_delta"` | Stored `coef_delta` / `kkt` request; currently ignored by direct-fit stopping checks. See the direct-control note below. |
| `inference_method` | `"debiased"` | `post_selection_ols` / `debiased` / `bootstrap`; ordinary `auto` resolves to `debiased`. With `enable_simultaneous_inference=True`, explicitly use `debiased`: the constructor currently rejects `auto`. Deprecated `cpu_ols` and `gpu_ols` aliases remain temporarily accepted. |
| `nodewise_alpha` | `None` | Node-wise Lasso penalty for `debiased` inference. Explicit positive values override the standardized design-side automatic rule. |
| `n_bootstrap` | `200` | Residual-bootstrap refit count; use at least 2 draws. |
| `bootstrap_random_state` | `None` | RNG seed for residual-bootstrap inference. |
| `enable_simultaneous_inference` | `False` | Enable simultaneous inference (debiased only). |
| `simultaneous_method` | `"maxz_bootstrap"` | Simultaneous-inference method; currently `maxz_bootstrap`. |
| `simultaneous_alpha` | `0.05` | Simultaneous family-wise error level; must be strictly in `(0, 1)` when simultaneous inference is enabled. |
| `simultaneous_n_bootstrap` | `1000` | Positive integer multiplier-bootstrap draw count for max-\|Z\| calibration when simultaneous inference is enabled. |
| `simultaneous_random_state` | `None` | RNG seed for simultaneous bootstrap. |
| `simultaneous_include_intercept` | `False` | Whether the debiased intercept is included in both the simultaneous target set and max-\|Z\| calibration family. |
| `device` | `"auto"` | Execution device: `auto`, `cpu`, `cuda` (CuPy), or `torch` (Torch CUDA). |
| `n_jobs` | `None` | Shared CPU worker setting; not a solver selector or parallel-fit guarantee. |
| `compute_inference` | `True` | Whether to compute post-fit inference. |
| `solver` | `"fista"` | Backend-neutral direct-fit solver; use `coordinate_descent` for the CPU CD path or another supported solver as appropriate. |
| `cpu_solver` | `"coordinate_descent"` | **Deprecated compatibility parameter.** It does not select the current direct-fit algorithm; use `solver` instead. |
| `lipschitz_L` | `None` | Optional user-supplied Lipschitz constant for compatible iterative solvers. |
| `admm_rho` | `1.0` | Stored request; unified ADMM currently ignores it and starts with rho=1.0. Adaptation depends on the solver path; the direct Cholesky solve keeps rho fixed. |
| `gpu_memory_cleanup` | `False` | Best-effort GPU memory cleanup after fit where supported. |

`nodewise_alpha` is keyword-only. This is the L1-only wrapper, not an alias
for the full ElasticNet constructor: `l1_ratio`, `cov_type`, `hac_maxlags`, and
`initial_coef` are not Lasso constructor/fit controls. It uses the default
nonrobust inference configuration. See the [learner page](../models/lasso.md)
for the objective, tuning guidance and statistical interpretation.

### Lasso methods and shapes

| Method | Input and return contract |
|---|---|
| `fit(X=None,y=None,sample_weight=None,formula=None,data=None)` | Returns `self`. Array inputs: finite numeric X `(n,p)` and one-dimensional y `(n,)`; optional analytic weights `(n,)`, finite, nonnegative and with positive sum. Alternatively use formula/data, as described under [formula inputs](#formula-inputs). Do not supply both routes: formula parsing currently replaces array inputs rather than rejecting the conflict. |
| `predict(X,return_cpu=True)` | X `(m,p)` in the fitted feature order, or a prediction DataFrame after formula fitting. Returns predictions `(m,)`; default is NumPy, including after GPU fitting. `return_cpu=False` preserves the fitted NumPy/CuPy/Torch numerical backend. |
| `score(X,y,sample_weight=None)` | Python float R² on evaluation data. Use flat `(m,)` y and NumPy/host response and weights. Evaluation weights are separate from training weights. Validate finite, nonnegative length-m weights of positive sum yourself: the current score method can accept negative weights and return invalid R². |
| `summary()` | Prints a coefficient/inference table and returns `None`; needs successful inference, not merely a prediction fit. |
| `get_params(deep=True)`, `set_params(**params)` | Constructor configuration dictionary and updates returning self; nonempty valid updates reset fitted state. Refit before predicting. See [parameter management](estimator-api.md#parameter-management). |
| `adjust_pvalues`, `combine_pvalues`, `bootstrap_statistic`, `permutation_test` | Inherited helpers for supplied/cached p-values or supplied data. Complete signatures, arguments, returns and restrictions are in the [shared estimator reference](estimator-api.md#inference-helpers). They do not repeat Lasso tuning or correct selection uncertainty automatically. |

### Lasso fitted results

Let k=p+1 when an intercept is fitted and k=p otherwise. Formula intercept
syntax takes precedence over the constructor. `coef_` and all coefficient
reporting arrays below are NumPy arrays, even after GPU fitting.

| Result | Shape and meaning |
|---|---|
| `coef_`, `intercept_`, `n_iter_` | Penalized prediction slopes `(p,)`, scalar intercept (zero without an intercept), and iteration count. None of these certifies KKT optimality. |
| `_params`, `_bse`, `_tvalues` / `_zvalues`, `_pvalues`, `_conf_int` | Successful inference: vectors `(k,)`, CI `(k,2)`. With an intercept, row 0 is its reporting estimate and remaining rows are slopes. Without an intercept, every row is a slope in feature order. Fields can remain `None` without inference. `_params` may be debiased or active-set-refit parameters, not the penalized prediction fit. |
| `_inference_result` | Structured `params`, `bse`, `statistic`, `pvalues`, `conf_int`, `method`, `distribution`, `metadata`. Use these fields to identify the actual procedure and target. |
| `nodewise_alpha_` | Resolved node-wise penalty after successful multi-feature debiased inference; `None` for one-feature or other inference paths. |
| `_conf_int_simultaneous` | Joint intervals in the `(k,2)` reporting layout after enabled debiased simultaneous inference. With an intercept excluded from the family, its row remains the marginal interval; inclusion makes it part of max-\|Z\| calibration. Ordinary `_conf_int` always remains marginal. |
| `inference_requested_method_`, `inference_resolved_method_`, `inference_method_`, `inference_target_`, `penalty_conditioning_`, `penalty_selection_adjusted_` | Procedure/target attributes for supported reporting paths. Some may remain `None` for successful `post_selection_ols`; inspect `_inference_result.method` and metadata instead. |
| `rsquared`, `rsquared_adj`, `fvalue`, `f_pvalue`, `llf`, `aic`, `bic` | Training diagnostics when available. Weighted debiased inference can misstate R²/F totals by re-centering a working response; prefer `score` on original data with validated weights. Likelihood/AIC/BIC/F summaries are not generally penalty-effective-DoF or selective-inference criteria; missing state can yield `None`/NaN. |

Prediction coefficients remain penalized under all three inference methods.
`post_selection_ols` refits the selected columns for diagnostic inference;
`debiased` corrects coefficients for marginal normal-reference inference;
`bootstrap` resamples empirical residuals and refits the full penalized design
at fixed alpha. None automatically adjusts for tuning/selection uncertainty.
Bootstrap accepts only unweighted nonrobust Gaussian-model inference and has
public constructor controls `n_bootstrap` and `bootstrap_random_state`.

For `fit_intercept=True`, simultaneous debiased inference reuses the fitted
NumPy/CuPy/Torch backend. Result metadata can include
`simultaneous_numerical_backend`, `simultaneous_numerical_device`,
`simultaneous_reporting_backend` and `simultaneous_reporting_boundary`.
`fit_intercept=False` instead uses the NumPy host helper for simultaneous
calculations, even when marginal inference ran on GPU.

The direct `stopping` setting is currently ineffective: Gaussian CPU FISTA/CD
and GPU FISTA check coefficient movement; ADMM checks primal/dual residuals.
The separate Lasso CV/path helper does not establish direct/final-refit KKT
certification. `admm_rho` is stored but ignored by unified ADMM, which starts at
rho=1.0. Adaptation depends on the solver path; the direct squared-error Cholesky
solve keeps rho fixed. See [control limitations](../models/lasso.md#solvers-and-current-control-limitations).

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
| `stopping` | `"coef_delta"` | Stored `coef_delta` / `kkt` request; currently ignored by direct Gaussian stopping checks. FISTA/CD use coefficient movement; ADMM uses primal/dual residuals. |
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
| `inference_requested_method_`, `inference_resolved_method_`, `inference_method_`, `inference_target_`, `penalty_conditioning_`, `penalty_selection_adjusted_` | Debiased/bootstrap inference records public procedure/target fields. The current `post_selection_ols` path can leave `inference_method_` and `inference_target_` as `None` despite successful inference; read `_inference_result.method` and `_inference_result.metadata` for that path. These are reporting attributes, not extra constructor parameters. |
| `rsquared`, `rsquared_adj`, `fvalue`, `f_pvalue`, `llf`, `aic`, `bic` | Available fit diagnostics. After weighted debiased inference, `rsquared`/`rsquared_adj` use a re-centered working response and need not equal raw weighted R²; use validated `score(X,y,sample_weight=weights)` instead, as shown in [weighted diagnostics](../models/elastic-net.md#weighted-training-diagnostics). AIC/BIC/F are compatibility plug-in summaries, not general penalty-aware effective-DoF or selective-inference criteria; missing state can yield `None`/NaN. |

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
`ElasticNetCV` and `LogisticRegressionCV` use `fit(X,y,sample_weight=None) -> self`, `predict(X)`, `score(X,y)`, and inherited `summary()`. Fit requires array/design-matrix input, not `formula`/`data`. `score` is unweighted held-out R² for ElasticNetCV and accuracy for LogisticRegressionCV, **not** the CV selection loss. ElasticNetCV.predict returns NumPy through the final estimator's default; LogisticRegressionCV also exposes `predict_proba(X) -> (m,2)` on the final model's backend. Other classification methods are on `estimator_`.

Generated folds are shuffled K-fold, not automatically stratified/grouped/time-aware. Supply explicit reusable index-pair lists for those designs. Validate each pair as one-dimensional integer indices with nonempty, disjoint training/validation sets and no repeated rows. The current shared splitter casts to integers, flattens arrays and skips empty pairs; it does not reject overlap or duplicates. Invalid splits can leak validation rows into training. Fold weights affect both training and validation loss; every relevant split needs positive weight mass. Fit preprocessing only within each training fold. These wrappers do not accept a preprocessing pipeline or a scoring callable. For fold-local learned scaling, use an external CV loop/pipeline rather than scaling all rows before internal CV.

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
LinearRegression, direct Lasso, ElasticNet, Ridge, SCADRegression and MCPRegression here support formulas. For these classes, supply only `formula`/`data`, not simultaneous array X/y; current formula parsing replaces those arrays without a conflict error. Install optional pandas/patsy. `formula="y ~ x + C(group)"` supplies numeric/categorical terms; `~ 0 + ...` removes the intercept regardless of the constructor. Interactions/transforms follow Patsy syntax. Formula fitting can drop rows with missing terms. Weights may describe all original data rows or exactly the retained rows; matching is positional, not by arbitrary Series labels.

Prediction DataFrames rebuild the stored design and categorical levels. Unknown levels or missing values that would drop prediction rows raise; array prediction must supply the already encoded non-intercept columns in training order. Keep transformations and level definitions consistent. LogisticRegression, ElasticNetCV, LogisticRegressionCV, RidgeCV and LassoCV have no formula argument; construct a suitable design first, avoiding preprocessing leakage.

<!-- api-example: formula-models -->
```python
import numpy as np
import pandas as pd
from statgpu import LinearRegression, ElasticNet, Lasso

df = pd.DataFrame({"x": np.arange(12.0), "group": ["a", "b"] * 6})
df["y"] = 1.0 + 2.0 * df["x"] + (df["group"] == "b").astype(float)
for cls in (LinearRegression, ElasticNet, Lasso):
    model = cls(device="cpu", compute_inference=False)
    model.fit(formula="y ~ x + C(group)", data=df)
    assert model.predict(df.iloc[:3]).shape == (3,)
```

## GeneralizedLinearModel

```text
GeneralizedLinearModel(family='gaussian', fit_intercept=True, max_iter=100, tol=0.0001, C=1.0, device='auto', n_jobs=None, solver='auto', gpu_memory_cleanup=False, compute_inference=False, cov_type='nonrobust')
```

| Parameter | Default | Meaning and restrictions |
|---|---|---|
| `family` | `'gaussian'` | gaussian, binomial, poisson, gamma, inverse_gaussian, negative_binomial, or tweedie. Family-specific link/dispersion controls belong to typed wrappers. |
| `fit_intercept` | `True` | Fit an unpenalized intercept; formula syntax overrides this choice. |
| `max_iter` | `100` | Solver iteration budget. |
| `tol` | `0.0001` | Numerical convergence tolerance. |
| `C` | `1.0` | Ordinary IRLS adds ||beta||²/(4C) for positive C; C=0 removes it. Explicit newton/lbfgs/fista ignore C. |
| `device` | `'auto'` | cpu, cuda (CuPy), torch (Torch CUDA), or auto. Explicit GPU requests require the corresponding CUDA backend. |
| `n_jobs` | `None` | Shared CPU-worker configuration; no parallel-fit guarantee. |
| `solver` | `'auto'` | auto, irls, fista, newton, lbfgs; ordinary auto selects IRLS. Solver changes can change the C-penalized objective. |
| `gpu_memory_cleanup` | `False` | Best-effort GPU memory-pool cleanup. |
| `compute_inference` | `False` | Enable supported M-estimation coefficient inference. |
| `cov_type` | `'nonrobust'` | nonrobust, hc0, hc1 for ordinary GLM inference; no hac_maxlags constructor argument. |


`fit(X=None,y=None,sample_weight=None,formula=None,data=None)` returns self. Use finite numeric X `(n,p)` and scalar-response y `(n,)`; a single response column is flattened. Supply arrays or formula/data, not both: formula currently replaces simultaneous array arguments without rejecting the conflict. Default links are identity for Gaussian, logit for binomial, and log for the other listed families. Binomial requires 0/1 labels, Poisson/Negative Binomial nonnegative responses, and Gamma/Inverse Gaussian strictly positive responses. The default Tweedie power is 1.5 and permits zero. Family/link settings can impose further restrictions.

| Method/result | Contract |
|---|---|
| `predict(X)` | Response mean `(m,)` on the currently resolved device. Binomial returns probabilities, not labels or a two-column matrix. Formula DataFrames rebuild the fitted design. |
| `summary()` | Returns a string; does not print it. Use `print(model.summary())`. With inference disabled it still reports coefficients/diagnostics; before fit it returns a not-fitted string. |
| `family_to_loss()` | Returns the internal loss-name string, e.g. gaussian → squared_error and binomial → logistic. |
| `coef_`, `intercept_`, `n_iter_` | NumPy slopes `(p,)`, scalar intercept, iteration count. Iteration count alone does not certify convergence. |
| `_bse`, `_zvalues`, `_pvalues`, `_conf_int` | Successful inference arrays `(k,)`, `(k,2)` with the intercept first if fitted; k=p+1 or p. |
| `loglikelihood`, `llf`, `aic`, `bic` | Pseudo-likelihood diagnostics omit parameter-independent constants. Weighted loglikelihood is −n times the weighted-average per-row loss. Do not compare absolute values across packages or incompatible outcomes/rows/weights. |

Ordinary GLM coefficient inference uses a normal/z reference for all supported families, including Gaussian. It differs from nonrobust `LinearRegression` and the shared squared-error L2/Ridge Student-t path. Check `model._inference_result.distribution`; the `_zvalues` name reflects this normal-reference calculation.

There is no `score` or `predict_proba` method on this generic ordinary class. `get_params(deep=True)`, `set_params(**params)`, `adjust_pvalues`, `combine_pvalues`, `bootstrap_statistic`, and `permutation_test` follow the [shared API](estimator-api.md); those resampling helpers do not automatically refit a GLM.

### Failed ordinary-GLM refits

Current auto/IRLS/FISTA refit errors can leave an object marked fitted while mixing earlier coefficients with new row counts or formula/intercept settings. Predictions and likelihood/AIC/BIC can change even though the new fit raised. Do not reuse that object's outputs after a failed refit; construct a fresh estimator and complete a successful fit. The shared ordinary typed wrappers inherit this limitation. Explicit newton/lbfgs currently restore their earlier fitted state after a failed attempt; this preservation does not mean the new data were fitted.


## PenalizedGeneralizedLinearModel

```text
PenalizedGeneralizedLinearModel(loss='squared_error', penalty='l1', alpha=1.0, l1_ratio=0.5, penalty_kwargs=None, fit_intercept=True, max_iter=1000, tol=0.0001, device='auto', n_jobs=None, cpu_solver='fista', solver='auto', lipschitz_L=None, gpu_memory_cleanup=False, compute_inference=False, inference_method='auto', cov_type='nonrobust', hac_maxlags=None, stopping='coef_delta', lla=True, max_lla_iters=50, lla_tol=1e-06, loss_kwargs=None, *, nodewise_alpha=None)
```

| Parameter | Default | Meaning and restrictions |
|---|---|---|
| `loss` | `'squared_error'` | Loss names include squared_error, logistic, poisson, gamma, inverse_gaussian, negative_binomial, tweedie and quantile. Consult the [loss reference](../models/losses.md) for other families and their restrictions. |
| `penalty` | `'l1'` | none, l1, l2, elasticnet, scad, mcp, adaptive_l1, supported group penalties, or a Penalty object. |
| `alpha` | `1.0` | Penalty strength on the average-loss scale. A supplied Penalty object owns its own configuration. |
| `l1_ratio` | `0.5` | L1 fraction for elasticnet. |
| `penalty_kwargs` | `None` | Additional penalty constructor settings, e.g. groups or shape controls. |
| `fit_intercept` | `True` | Unpenalized intercept; formula syntax takes precedence. |
| `max_iter` | `1000` | Per-solve iteration budget. |
| `tol` | `0.0001` | Numerical tolerance. |
| `device` | `'auto'` | cpu, cuda, torch, auto; see the backend guide. |
| `n_jobs` | `None` | Shared CPU-worker setting where used. |
| `cpu_solver` | `'fista'` | Deprecated compatibility control; it no longer selects the direct-fit algorithm. Explicit non-None values, including the historical default, emit FutureWarning; omitted/default clone replay does not. Use `solver`; see the [migration guide](../guides/penalized-solver-api-migration.md). |
| `solver` | `'auto'` | Choices include auto, fista, fista_bb, admm, irls, newton, lbfgs and exact; support depends on the loss and penalty, and unsupported explicit combinations raise. Shared ADMM does not support Quantile; see the [compatibility matrix](../guides/solver-penalty-matrix.md). |
| `lipschitz_L` | `None` | Optional Lipschitz bound for compatible proximal paths. |
| `gpu_memory_cleanup` | `False` | Best-effort GPU memory-pool cleanup. |
| `compute_inference` | `False` | Run supported post-fit inference only when True. |
| `inference_method` | `'auto'` | Resolve a supported method from loss/penalty; consult the inference matrix. |
| `cov_type` | `'nonrobust'` | Method-specific covariance; non-Gaussian smooth inference supports nonrobust/hc0/hc1. |
| `hac_maxlags` | `None` | HAC lag control only on paths supporting HAC. |
| `stopping` | `'coef_delta'` | Stored convergence request; direct sparse Gaussian fits currently ignore the kkt choice. |
| `lla` | `True` | Enable local linear approximation for supported nonconvex penalties. |
| `max_lla_iters` | `50` | Maximum outer LLA iterations. |
| `lla_tol` | `1e-06` | Outer LLA convergence tolerance. |
| `loss_kwargs` | `None` | Loss-specific controls, e.g. link, Negative Binomial dispersion alpha, or Tweedie power. |
| `nodewise_alpha` | `None` | Keyword-only tuning for supported debiased precision estimation; not the fit penalty. |


`fit(X=None,y=None,sample_weight=None,formula=None,data=None)` returns self; scalar-response shapes and formula rules match the ordinary GLM above. There is no generic `initial_coef`, `n_bootstrap`, or `bootstrap_random_state` constructor parameter; wrapper-specific controls are not interchangeable.

`predict(X,return_cpu=True)` returns `(m,)`, with NumPy output by default and the fitted native backend when False. Squared error returns the linear fit; GLM families return response means, except logistic returns 0/1 labels (class 1 only above probability 0.5). This generic class has no `predict_proba` or `summary`; use a suitable typed wrapper for its additional methods. `score(X,y,sample_weight=None)` is response-scale R², including on logistic labels, not classification accuracy or deviance pseudo-R². Use one-dimensional y and validate finite nonnegative evaluation weights with positive sum; shared score validation is incomplete.

Fitted `coef_` `(p,)` and scalar `intercept_` describe prediction. Read supported inference through `_inference_result` and method-specific arrays, not by assuming prediction slopes equal corrected/refitted inferential parameters. `_inference_result.to_dict()` returns a reporting dictionary; `to_dataframe()` requires pandas. The [inference guide](../guides/penalized-glm-inference.md) defines methods, estimands and limitations. The six shared configuration/p-value/resampling methods listed above also apply.


## PenalizedGLM_CV

```text
PenalizedGLM_CV(loss='squared_error', penalty='l2', alpha_grid=None, n_alphas=100, l1_ratio=0.5, cv=5, cv_splits=None, random_state=0, device='auto', max_iter=1000, tol=0.0001, solver='auto', cv_strategy='strict', acknowledge_approx=False, refine_top_k=3, loss_kwargs=None, penalty_kwargs=None, *, compute_inference=False, inference_method='auto', cov_type='nonrobust', hac_maxlags=None)
```

| Parameter | Default | Meaning and restrictions |
|---|---|---|
| `loss` | `'squared_error'` | Scalar-response loss; cox_ph uses a separate survival contract. |
| `penalty` | `'l2'` | Tunable supported penalty; none is rejected as non-tunable. |
| `alpha_grid` | `None` | Explicit finite grid; ordinary tunable penalties require positive values. An omitted grid is generated for the selected loss/penalty. |
| `n_alphas` | `100` | Requested automatic grid size. |
| `l1_ratio` | `0.5` | Fixed Elastic Net mixture, not a sequence searched by this class. |
| `cv` | `5` | Generated shuffled K-fold count. |
| `cv_splits` | `None` | Explicit disjoint nonempty integer train/validation indices; one-shot iterators are materialized and reused. |
| `random_state` | `0` | Seed for generated folds. |
| `device` | `'auto'` | Explicit cpu/cuda/torch or size-aware automatic CV routing; inspect cv_selected_device_. |
| `max_iter` | `1000` | Strict fold/refit iteration budget. |
| `tol` | `0.0001` | Strict fold/refit tolerance. |
| `solver` | `'auto'` | Requested compatible solver; inference does not choose a different tuning grid. |
| `cv_strategy` | `'strict'` | strict or two_stage; two_stage screens approximately before strict refinement. |
| `acknowledge_approx` | `False` | Acknowledge two-stage approximation and suppress its warning. |
| `refine_top_k` | `3` | Number of promising candidates to refine, subject to refinement safeguards. |
| `loss_kwargs` | `None` | Family/link settings forwarded to fits and validation loss. |
| `penalty_kwargs` | `None` | Penalty-specific configuration. |
| `compute_inference` | `False` | Keyword-only; supported final-refit inference, not fold inference. |
| `inference_method` | `'auto'` | Keyword-only final-refit inference request. |
| `cov_type` | `'nonrobust'` | Keyword-only final-refit covariance. |
| `hac_maxlags` | `None` | Keyword-only; retained only where the final inference method supports HAC. |


Invalid/nonpositive scalar alpha entries are filtered with a warning; an empty surviving grid triggers automatic generation, also with a warning. Validate the intended search range yourself instead of relying on this fallback.

This generic CV class is distinct from ElasticNetCV and LogisticRegressionCV. `fit(X,y,sample_weight=None)` returns self and has no formula interface. Scalar-response CV always fits an intercept; no public `fit_intercept` option is exposed here. Use an external CV loop with a suitable direct estimator for no-intercept scalar fits. The `cox_ph` branch instead requires the [survival target and no-intercept contract](../models/coxph.md).

`predict(X)` and `score(X,y,sample_weight=None)` delegate to `estimator_`; scalar-response score is R², including for logistic labels. `alpha_` is selected by minimum mean held-out loss, and `best_score_` is its negative. It is not the held-out R² returned by score. Final inference conditions on selected alpha and does not adjust for tuning uncertainty.

For a candidate count a and fold count f, `alpha_grid_` and `cv_results_["alpha"]` have shape `(a,)`; `mean_score` has `(a,)`, and `all_scores` has `(f,a)`. These score arrays hold losses (smaller is better). Other keys are `device_sizing_fold_count`, `cv_strategy_`, `cv_selected_device_`, `mean_score_stage1`, `all_scores_stage1`, and `refined_mask`. Stage-one arrays are None in strict mode; the boolean refinement mask has `(a,)`. `coef_`, `intercept_`, and `estimator_` describe the final full-data fit. `cv_strategy_` and `cv_selected_device_` are also attributes.

`summary(*args,**kwargs)` currently raises AttributeError after successful generic final-refit inference because it delegates to a final estimator without a summary method. Without inference it raises RuntimeError instead. Read `estimator_._inference_result.to_dict()` (or `to_dataframe()` with pandas) for supported inference; the failure to render a summary does not itself change the fitted coefficients. See the runnable result example on the [GLM page](../models/generalized-linear-model.md#reading-cv-inference-results). Shared configuration/p-value/resampling methods apply; `get_params` materializes a one-shot custom splitter for reuse.


## Ridge

```text
Ridge(alpha=1.0, fit_intercept=True, device='auto', n_jobs=None, gpu_memory_cleanup=False, compute_inference=True, cov_type='nonrobust', hac_maxlags=None, max_iter=1000, tol=0.0001, solver='exact', cpu_solver='fista', lipschitz_L=None)
```

Every constructor parameter and its meaning is listed in the
[complete Ridge parameter table](../models/ridge.md#parameters).
This wrapper accepts only a scalar response. Default `solver="exact"` fits L2
regression; `compute_inference=True` enables the supported Gaussian covariance
path. The optimized CPU exact path has the documented
[large-offset limitation](../models/ridge.md#large-feature-offsets).

| Method | Arguments, defaults and return |
|---|---|
| `fit(X=None,y=None,sample_weight=None,formula=None,data=None)` | Returns self. Use finite X `(n,p)` and one-dimensional y `(n,)`, or formula/data. Optional analytic weights `(n,)` must be finite, nonnegative and have positive total. Formula syntax owns the intercept. Do not pass both arrays and formula/data: formula input currently replaces arrays without a conflict error. See [formula inputs](#formula-inputs) for row alignment and prediction DataFrames. |
| `predict(X,return_cpu=True)` | Prediction `(m,)` for X `(m,p)` or a formula prediction DataFrame. NumPy by default, including after GPU fits; False retains the fitted numerical backend. |
| `score(X,y,sample_weight=None)` | Python float R². Use one-dimensional response and host/NumPy response/weights. Evaluation weights are independent of training weights; validate finite nonnegative length-m weights with positive total yourself because current shared validation is incomplete. |
| `summary()` | Prints a coefficient/inference table, returns None; raises RuntimeError when inference is disabled or unavailable. |
| `get_params(deep=True)`, `set_params(**params)` | Configuration dictionary and updates returning self. A nonempty valid update clears fitted state; refit before using results. |
| `adjust_pvalues`, `combine_pvalues`, `bootstrap_statistic`, `permutation_test` | Full argument/default/return contracts are in the [shared estimator reference](estimator-api.md#inference-helpers). They do not automatically refit this model or repeat selection/tuning. |

`coef_` is a NumPy `(p,)` array, `intercept_` a scalar (zero without an intercept),
and `n_iter_` an iteration count, not a global-optimality certificate.
With successful inference, `_params`, `_bse`, `_tvalues`, `_pvalues` have `(k,)`
and `_conf_int` has `(k,2)`, where k=p+1 with an intercept and p otherwise.
The intercept comes first. `_inference_result` records the method/distribution
and reporting metadata. Nonrobust inference uses Student-t; HC/HAC uses a normal
reference. The diagnostic properties `rsquared`, `rsquared_adj`, `fvalue`,
`f_pvalue`, `llf`, `aic`, `bic` can be None without the necessary fitted inference
state. These plug-in diagnostics are not general effective-degrees-of-freedom
or tuning-adjusted criteria; robust covariance does not make F a robust Wald test.


## SCADRegression

```text
SCADRegression(alpha=1.0, a=3.7, fit_intercept=True, max_iter=1000, tol=0.0001, device='auto', compute_inference=False, solver='auto', gpu_memory_cleanup=False)
```

Every constructor parameter and its meaning is listed in the
[complete SCADRegression parameter table](../models/scad.md#parameters).
Import with `from statgpu.linear_model import SCADRegression`; the top-level
`statgpu` module does not export it. `alpha` must be finite and positive, and
`a` must be finite and greater than 2.
Keep `compute_inference=False`: the wrapper does not expose `inference_method`,
and enabling inference raises. Use the generic penalized-linear interface from
the [model page](../models/scad.md#covarianceinference) for explicit inference.

| Method | Arguments, defaults and return |
|---|---|
| `fit(X=None,y=None,sample_weight=None,formula=None,data=None)` | Returns self. Use finite X `(n,p)` and one-dimensional y `(n,)`, or formula/data. Optional analytic weights `(n,)` must be finite, nonnegative and have positive total. Formula syntax owns the intercept. Do not pass both arrays and formula/data: formula input currently replaces arrays without a conflict error. See [formula inputs](#formula-inputs) for row alignment and prediction DataFrames. |
| `predict(X,return_cpu=True)` | Prediction `(m,)` for X `(m,p)` or a formula prediction DataFrame. NumPy by default, including after GPU fits; False retains the fitted numerical backend. |
| `score(X,y,sample_weight=None)` | Python float R². Use one-dimensional response and host/NumPy response/weights. Evaluation weights are independent of training weights; validate finite nonnegative length-m weights with positive total yourself because current shared validation is incomplete. |
| `summary()` | Prints a coefficient/inference table, returns None; raises RuntimeError when inference is disabled or unavailable. |
| `get_params(deep=True)`, `set_params(**params)` | Configuration dictionary and updates returning self. A nonempty valid update clears fitted state; refit before using results. |
| `adjust_pvalues`, `combine_pvalues`, `bootstrap_statistic`, `permutation_test` | Full argument/default/return contracts are in the [shared estimator reference](estimator-api.md#inference-helpers). They do not automatically refit this model or repeat selection/tuning. |

`coef_` is a NumPy `(p,)` array, `intercept_` a scalar (zero without an intercept),
and `n_iter_` an iteration count, not a global-optimality certificate.
With inference disabled, `_bse`, `_tvalues`, `_pvalues`, `_conf_int` and the
inherited diagnostic properties `rsquared`, `rsquared_adj`, `fvalue`, `f_pvalue`,
`llf`, `aic`, `bic` are unavailable (normally None). Use `score` or explicit
held-out prediction loss for evaluation. Do not enable inference merely to make
`summary()` work: the specialized constructor cannot select a supported method.


## MCPRegression

```text
MCPRegression(alpha=1.0, gamma=3.0, fit_intercept=True, max_iter=1000, tol=0.0001, device='auto', compute_inference=False, solver='auto', gpu_memory_cleanup=False)
```

Every constructor parameter and its meaning is listed in the
[complete MCPRegression parameter table](../models/mcp.md#parameters).
Import with `from statgpu.linear_model import MCPRegression`; the top-level
`statgpu` module does not export it. `alpha` must be finite and positive, and
`gamma` must be finite and greater than 1.
Keep `compute_inference=False`: the wrapper does not expose `inference_method`,
and enabling inference raises. Use the generic penalized-linear interface from
the [model page](../models/mcp.md#covarianceinference) for explicit inference.

| Method | Arguments, defaults and return |
|---|---|
| `fit(X=None,y=None,sample_weight=None,formula=None,data=None)` | Returns self. Use finite X `(n,p)` and one-dimensional y `(n,)`, or formula/data. Optional analytic weights `(n,)` must be finite, nonnegative and have positive total. Formula syntax owns the intercept. Do not pass both arrays and formula/data: formula input currently replaces arrays without a conflict error. See [formula inputs](#formula-inputs) for row alignment and prediction DataFrames. |
| `predict(X,return_cpu=True)` | Prediction `(m,)` for X `(m,p)` or a formula prediction DataFrame. NumPy by default, including after GPU fits; False retains the fitted numerical backend. |
| `score(X,y,sample_weight=None)` | Python float R². Use one-dimensional response and host/NumPy response/weights. Evaluation weights are independent of training weights; validate finite nonnegative length-m weights with positive total yourself because current shared validation is incomplete. |
| `summary()` | Prints a coefficient/inference table, returns None; raises RuntimeError when inference is disabled or unavailable. |
| `get_params(deep=True)`, `set_params(**params)` | Configuration dictionary and updates returning self. A nonempty valid update clears fitted state; refit before using results. |
| `adjust_pvalues`, `combine_pvalues`, `bootstrap_statistic`, `permutation_test` | Full argument/default/return contracts are in the [shared estimator reference](estimator-api.md#inference-helpers). They do not automatically refit this model or repeat selection/tuning. |

`coef_` is a NumPy `(p,)` array, `intercept_` a scalar (zero without an intercept),
and `n_iter_` an iteration count, not a global-optimality certificate.
With inference disabled, `_bse`, `_tvalues`, `_pvalues`, `_conf_int` and the
inherited diagnostic properties `rsquared`, `rsquared_adj`, `fvalue`, `f_pvalue`,
`llf`, `aic`, `bic` are unavailable (normally None). Use `score` or explicit
held-out prediction loss for evaluation. Do not enable inference merely to make
`summary()` work: the specialized constructor cannot select a supported method.


## RidgeCV

```text
RidgeCV(alphas=None, n_alphas=100, alpha_min_ratio=0.001, cv=5, cv_splits=None, fit_intercept=True, device='auto', n_jobs=None, compute_inference=True, cov_type='nonrobust', gpu_memory_cleanup=False, random_state=None, gpu_cv_mixed_precision=True)
```

| Parameter | Default | Meaning |
|---|---|---|
| `alphas` | `None` | Explicit positive finite candidates; omitted: generate a data-dependent grid. Invalid/nonpositive entries are filtered; an empty surviving grid falls back to automatic generation. |
| `n_alphas` | `100` | Automatic grid size when alphas is omitted. |
| `alpha_min_ratio` | `0.001` | Minimum/maximum ratio for the automatic grid; choose a positive value normally no greater than 1. |
| `cv` | `5` | Generated shuffled K-fold count, at least 2. |
| `cv_splits` | `None` | Explicit reusable list of (train_indices, validation_indices); validate nonempty disjoint integer subsets yourself. RidgeCV has the custom-training-subset restriction below. |
| `fit_intercept` | `True` | Fit an intercept in CV and final refit. |
| `device` | `'auto'` | cpu, cuda (CuPy), torch (Torch CUDA), auto; explicit unavailable GPU requests raise. |
| `n_jobs` | `None` | Shared worker configuration; no candidate-parallelism guarantee. |
| `compute_inference` | `True` | Compute supported inference only on the final full-data refit, conditional on selected alpha. |
| `cov_type` | `'nonrobust'` | Final Ridge covariance: nonrobust, hc0, hc1, hc2, hc3, hac. There is no hac_maxlags constructor control; HAC uses the automatic rule. |
| `gpu_memory_cleanup` | `False` | Request best-effort GPU cache cleanup for final fitting. |
| `random_state` | `None` | Seed for generated folds; not residual-bootstrap randomness. |
| `gpu_cv_mixed_precision` | `True` | Enable mixed precision during GPU CV; final-refit inference has its own numerical path. |

`fit(X,y,sample_weight=None)` returns self, accepting finite X `(n,p)`,
one-dimensional y `(n,)`, and optional analytic weights `(n,)`. There is no
formula interface. `predict(X)` returns NumPy `(m,)` through the final estimator.
`score(X,y)` returns unweighted R² and has no weight argument; for separately
validated evaluation weights use `estimator_.score(X,y,sample_weight=...)`.
`summary()` prints the final estimator's report and returns None, requiring
successful final inference. Shared `get_params(deep=True)`, `set_params(**params)`,
`adjust_pvalues`, `combine_pvalues`, `bootstrap_statistic`, and `permutation_test`
follow the [estimator reference](estimator-api.md).

`alpha_` is selected by minimum mean validation MSE; `best_score_` is its
**negative**, not the positive loss or final-model R². For a alphas and f folds,
`alphas_`/`mean_mse_` are `(a,)`, and `cv_results_` contains only `mse_path` `(a,f)`.
There is no `cv_results_["mean_mse"]` key. `coef_` `(p,)`, scalar `intercept_`,
`n_iter_` and `estimator_` describe the full-data refit. Inference belongs to
`estimator_` and conditions on alpha; it does not account for tuning uncertainty.
`cv_selected_device_` records the final device. There is no public solver,
max_iter or tol control on RidgeCV; the final estimator uses Ridge's exact path.
The [large-offset caution](../models/ridge.md#large-feature-offsets) also matters
when interpreting this final Ridge fit.

Generated folds are shuffled K-fold, not grouped, stratified or time-aware.
Use reusable explicit index-pair lists for a chosen design and validate them
before fitting; caller-side validation is necessary, not implied by acceptance.
Each relevant weighted fold needs positive weight mass. Fit learned preprocessing
inside each training fold, using an external CV loop when needed. Use at least
four observations, two candidates and two folds for an actual tuning comparison;
small/single-candidate paths can simply refit and report NaN losses/best_score_.
A chosen grid can still miss the useful penalty range.

### Custom RidgeCV training subsets

With no sample weights, when validation sets partition every row once,
RidgeCV currently substitutes each validation set's full complement for the
supplied training indices. A deliberately smaller training subset is therefore
not honored, even if it is valid and disjoint. Validation losses and alpha
selection can change. Do not use this path for custom excluded/embargoed rows.
For those designs use an external loop that fits Ridge on each exact training
subset and evaluates its designated validation rows. Ordinary complete K-folds
already use complementary training sets and are not affected by this substitution.


## LassoCV

```text
LassoCV(alphas=None, n_alphas=12, alpha_min_ratio=0.001, cv=5, cv_splits=None, fit_intercept=True, device='auto', n_jobs=None, compute_inference=False, max_iter=3000, tol=0.0001, stopping='coef_delta', solver='fista', cpu_solver=None, method='standard', cd_kkt_check_every=None, inference_method='post_selection_ols', lipschitz_L=None, admm_rho=1.0, gpu_memory_cleanup=False, random_state=None, gpu_cv_mixed_precision=True, cv_solver='auto', *, nodewise_alpha=None)
```

| Parameter | Default | Meaning |
|---|---|---|
| `alphas` | `None` | Explicit positive finite candidates; omitted: generate a data-dependent grid. Invalid/nonpositive entries are filtered; an empty surviving grid falls back to automatic generation. |
| `n_alphas` | `12` | Automatic grid size when alphas is omitted. |
| `alpha_min_ratio` | `0.001` | Minimum/maximum ratio for the automatic grid; choose a positive value normally no greater than 1. |
| `cv` | `5` | Generated shuffled K-fold count, at least 2. |
| `cv_splits` | `None` | Explicit reusable list of (train_indices, validation_indices); validate nonempty disjoint integer subsets yourself. |
| `fit_intercept` | `True` | Fit an intercept in CV and final refit. |
| `device` | `'auto'` | cpu, cuda (CuPy), torch (Torch CUDA), auto; explicit unavailable GPU requests raise. |
| `n_jobs` | `None` | Shared worker configuration; no candidate-parallelism guarantee. |
| `compute_inference` | `False` | Compute supported inference only on the final full-data refit, conditional on selected alpha. |
| `max_iter` | `3000` | Iteration budget for CV solves and the final direct fit. |
| `tol` | `0.0001` | Convergence tolerance for CV solves and final fitting. |
| `stopping` | `'coef_delta'` | Final-refit request only; current direct Gaussian stopping does not honor the kkt setting. It does not select the CV path stopping check. |
| `solver` | `'fista'` | Final full-data Lasso solver; does not choose the CV solver. |
| `cpu_solver` | `None` | Deprecated CPU CV alias; use cv_solver. Conflicting explicit CPU requests raise. On GPU it warns but does not replace FISTA. |
| `method` | `'standard'` | standard or glmnet CV path mode. On CPU, glmnet requires coordinate descent; GPU CV remains FISTA. Not an inference method. |
| `cd_kkt_check_every` | `None` | Positive integer CPU CV coordinate-descent KKT-check interval. None resolves to 1 for standard or 4 for glmnet. Not a final-refit certificate. |
| `inference_method` | `'post_selection_ols'` | Final-refit post_selection_ols, debiased, bootstrap or supported auto request; see Lasso inference restrictions. Deprecated aliases normalize at the compatibility boundary. |
| `lipschitz_L` | `None` | Optional compatible final-refit Lipschitz bound; not a CV-path control. |
| `admm_rho` | `1.0` | Forwarded to final Lasso; currently ignored by unified ADMM, which starts at rho=1.0. |
| `gpu_memory_cleanup` | `False` | Request best-effort GPU cache cleanup for final fitting. |
| `random_state` | `None` | Seed for generated folds; not residual-bootstrap randomness. |
| `gpu_cv_mixed_precision` | `True` | Enable mixed precision during GPU CV; final-refit inference has its own numerical path. |
| `cv_solver` | `'auto'` | auto, coordinate_descent or fista for CV. auto selects CPU CD or GPU FISTA; explicit coordinate_descent is CPU-only. |
| `nodewise_alpha` | `None` | Keyword-only final-refit debiased precision tuning; does not change the grid, fold losses or selected alpha. |

`fit(X,y,sample_weight=None)` returns self, accepting finite X `(n,p)`,
one-dimensional y `(n,)`, and optional analytic weights `(n,)`. There is no
formula interface. `predict(X)` returns NumPy `(m,)` through the final estimator.
`score(X,y)` returns unweighted R² and has no weight argument; for separately
validated evaluation weights use `estimator_.score(X,y,sample_weight=...)`.
`summary()` prints the final estimator's report and returns None, requiring
successful final inference. Shared `get_params(deep=True)`, `set_params(**params)`,
`adjust_pvalues`, `combine_pvalues`, `bootstrap_statistic`, and `permutation_test`
follow the [estimator reference](estimator-api.md).

`alpha_` is selected by minimum mean validation MSE; `best_score_` is its
**negative**, not the positive loss or final-model R². For a alphas and f folds,
`alphas_`/`mean_mse_` are `(a,)`, and `cv_results_` contains only `mse_path` `(a,f)`.
There is no `cv_results_["mean_mse"]` key. `coef_` `(p,)`, scalar `intercept_`,
`n_iter_` and `estimator_` describe the full-data refit. Inference belongs to
`estimator_` and conditions on alpha; it does not account for tuning uncertainty.
`mse_path_` also exposes the `(a,f)` loss array. `cv_solver_` records the resolved
CV algorithm, while `solver` controls the final Lasso only. `nodewise_alpha_`
exposes final-refit precision tuning where applicable. Source-static signatures
can omit the installed keyword-only `nodewise_alpha`; the constructor above is
the runtime public API. LassoCV does not expose the direct Lasso simultaneous or
residual-bootstrap draw/seed constructor controls.

Generated folds are shuffled K-fold, not grouped, stratified or time-aware.
Use reusable explicit index-pair lists for a chosen design and validate them
before fitting; caller-side validation is necessary, not implied by acceptance.
Each relevant weighted fold needs positive weight mass. Fit learned preprocessing
inside each training fold, using an external CV loop when needed. Use at least
four observations, two candidates and two folds for an actual tuning comparison;
small/single-candidate paths can simply refit and report NaN losses/best_score_.
A chosen grid can still miss the useful penalty range.


## RidgeCV and LassoCV CPU example

<!-- api-example: ridge-lasso-cv -->
```python
import numpy as np
from statgpu import LassoCV, RidgeCV

rng = np.random.default_rng(64)
X = rng.normal(size=(160, 5))
y = 1.5 + 2 * X[:, 0] - X[:, 1] + rng.normal(scale=0.4, size=160)
models = {}
for cls in (LassoCV, RidgeCV):
    model = cls(
        alphas=[0.03, 0.1, 0.3], cv=3, random_state=7,
        device="cpu", compute_inference=False,
    ).fit(X[:120], y[:120])
    models[cls.__name__] = model
    assert set(model.cv_results_) == {"mse_path"}
    assert model.cv_results_["mse_path"].shape == (3, 3)
    assert np.isclose(model.best_score_, -np.min(model.mean_mse_))
    print(cls.__name__, model.alpha_, round(model.score(X[120:], y[120:]), 3))
```

Both select alpha 0.03 on these data; held-out R² rounds to 0.965 for LassoCV
and 0.966 for RidgeCV. No learned preprocessing uses the held-out rows.
