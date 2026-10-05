# Survival and smoothing API reference

> Language: English  
> Last updated: 2026-10-05  
> Switch: [Chinese](../../cn/reference/survival-smoothing-api.md)

Use this page to look up a call after the [CoxPH](../models/coxph.md), [GAM](../models/semiparametric.md), or [nonparametric](../models/nonparametric.md) walkthrough. Signatures below include public arguments and defaults; `*` starts keyword-only arguments. Shape notation: `n` training rows, `p` features, `q` query rows, `r` response columns. Fitted methods require a successful fit. Generic [parameter management](estimator-api.md#parameter-management) and [inference helpers](estimator-api.md#inference-helpers) are documented separately. Generic resampling helpers do not automatically supply model-valid survival inference or GAM confidence bands.

## CoxPH and CoxPHCV

Import both from `statgpu.survival`. Constructor meanings, allowed values, and inference restrictions are in the [CoxPH parameter tables](../models/coxph.md#parameters) and [CoxPHCV table](../models/coxph.md#coxphcv-parameters).

<!-- signature: CoxPH -->
```text
CoxPH(ties='breslow', tol=1e-09, max_iter=100, device='auto', n_jobs=None, compute_inference=True, compute_cindex=True, cov_type='nonrobust', gpu_memory_cleanup=False, penalty=0.0, inference_mode='strict')
```

<!-- signature: CoxPHCV -->
```text
CoxPHCV(penalties=None, n_penalties=100, penalty_min_ratio=0.001, cv=5, cv_splits=None, ties='breslow', tol=1e-09, max_iter=100, device='auto', n_jobs=None, compute_inference=True, cov_type='nonrobust', inference_mode='strict', gpu_memory_cleanup=False, random_state=None)
```

<!-- signature: CoxPH.fit -->
```text
CoxPH.fit(X=None, time=None, event=None, entry=None, cluster=None, init_coef=None, formula=None, data=None, *, start=None, strata=None, subject_id=None)
```

<!-- signature: CoxPHCV.fit -->
```text
CoxPHCV.fit(X, time, event=None, entry=None, cluster=None, *, start=None, strata=None, subject_id=None)
```


### Fitting and selection

- `X`: finite real `(n,p)` design, with no intercept or constant column. CoxPH also accepts `(n,)`; use a matrix for CoxPHCV.
- `time`: positive finite `(n,)` stop/follow-up times. `event`: binary `(n,)`; fitting needs observed events. If `event=None`, `time` is a packed `(n,2)` `[time,event]` or `(n,3)` `[start,stop,event]` target.
- `entry`/`start`: mutually exclusive `(n,)` aliases, `0 <= start < time`; omitted entry is zero. Do not combine either alias with a packed three-column target.
- `strata`: optional `(n,)` labels for separate risk sets/baselines and shared coefficients. `subject_id`: `(n,)` labels for repeated observations and subject-preserving CV. `cluster`: `(n,)` labels required by `cov_type="cluster"`; cluster alone does not group CV folds.
- `init_coef`: CoxPH-only finite `(p,)` initialization. `formula` and `data`: CoxPH-only alternative using a pandas DataFrame and optional pandas/Patsy dependencies; use `Surv(time,event)` or `Surv(start,stop,event)`. Formula missing-row removal is applied to aligned auxiliary labels. Prediction reuses the saved design transformation and rejects silently dropped rows.
- Both `fit` methods return `self`. Unsupported Exact-tie robust inference raises `NotImplementedError`; estimation-only Exact fitting is supported. Fit/input failures must not be interpreted as usable fitted results.
- CV `penalties` is a nonempty finite nonnegative vector. With `None`, `n_penalties` and `penalty_min_ratio` control the automatic grid; comparable feature scaling matters. `cv_splits` overrides generated folds and supplies nonempty disjoint integer train/validation indices. Subject overlap is rejected when `subject_id` is supplied. `random_state` controls generated splits.
- Selection maximizes mean unpenalized held-out partial log likelihood over the same evaluable folds; it does not optimize the C-index. A candidate must have finite scores and convergence on every effective fold. No qualifying candidate raises an error. The selected penalty is then refitted on all supplied training rows. Inference runs only in the final refit and does not correct tuning uncertainty.

### Prediction, scoring, and summaries

<!-- signature: CoxPH.predict -->
```text
CoxPH.predict(X)
```

<!-- signature: CoxPH.predict_risk_score -->
```text
CoxPH.predict_risk_score(X)
```

<!-- signature: CoxPH.predict_hazard_ratio -->
```text
CoxPH.predict_hazard_ratio(X)
```

<!-- signature: CoxPH.predict_survival -->
```text
CoxPH.predict_survival(X, times=None, strata=None)
```

<!-- signature: CoxPH.score -->
```text
CoxPH.score(X, time, event=None, start=None, strata=None, subject_id=None)
```

<!-- signature: CoxPH.summary -->
```text
CoxPH.summary()
```

<!-- signature: CoxPHCV.predict -->
```text
CoxPHCV.predict(X)
```

<!-- signature: CoxPHCV.predict_risk_score -->
```text
CoxPHCV.predict_risk_score(X)
```

<!-- signature: CoxPHCV.predict_hazard_ratio -->
```text
CoxPHCV.predict_hazard_ratio(X)
```

<!-- signature: CoxPHCV.predict_survival -->
```text
CoxPHCV.predict_survival(X, times=None, strata=None)
```

<!-- signature: CoxPHCV.score -->
```text
CoxPHCV.score(X, time, event=None, start=None, strata=None, subject_id=None)
```

<!-- signature: CoxPHCV.summary -->
```text
CoxPHCV.summary()
```


| Method | Inputs and return |
|---|---|
| `predict_risk_score` | Same feature order, `(q,p)` design; backend-native `(q,)` log-risk `X @ coef_`. |
| `predict`, `predict_hazard_ratio` | Backend-native `(q,)` relative hazards `exp(X @ coef_)`; not probabilities. Nonrepresentable exponentials raise rather than clip. |
| `predict_survival` | `times=None` uses the union of fitted event times; otherwise a finite scalar or 1D sequence in training time units. Returns `(curves,times)`, backend arrays `(q,n_times)` and `(n_times,)`. Explicit ordering is preserved. Before the first failure, survival is 1; beyond the last, the fitted step function stays flat. |
| `strata` for survival | One known label per query for every explicitly stratified fit, including a single fitted stratum. Missing/unseen labels raise. A no-failure stratum returns survival 1. Baseline availability requires `compute_inference=True`. |
| `score` | `X`, separate or packed targets, optional `start`, `strata`, `subject_id`; no `entry` alias. Returns a scalar Harrell-style C-index. Multi-stratum fits require labels; supplied labels must be known. Subject labels exclude within-subject pairs. No comparable pairs returns `0.5`, which is not evidence of adequate validation. |
| `summary` | Prints the fitted summary and returns `None`; use `model.summary()`, not `print(model.summary())`. CoxPHCV delegates to `estimator_`. |

### Coefficients, inference, and CV results

| Field | Shape/type and interpretation |
|---|---|
| `coef_`, `hazard_ratios_` | NumPy `(p,)`; coefficients and their exponentials, respectively. No intercept. |
| `_bse`, `_zvalues`, `_pvalues` | NumPy `(p,)` standard errors, coefficient z statistics, and two-sided normal p-values when inference is enabled; otherwise `None`. These established result fields have leading underscores. |
| `_conf_int` | NumPy `(p,2)`, fixed **95% marginal coefficient-scale** normal intervals. `np.exp(model._conf_int)` gives hazard-ratio intervals; `summary()` displays that exponentiated version. No confidence-level argument is exposed here. |
| `log_likelihood` | CoxPH property, scalar unpenalized partial log likelihood at the fitted coefficient; available after estimation-only fitting too. |
| `aic`, `bic` | CoxPH scalar properties for unpenalized fits. With `k=p` and event count `d`, AIC is `-2*log_likelihood+2*k`; BIC is `-2*log_likelihood+log(d)*k`. Access after positive-penalty fitting raises `RuntimeError`. |
| `concordance_index` | CoxPH training C-index property; `None` when `compute_cindex=False`. |
| `converged_`, `n_iter_` | Boolean and integer; also inspect `termination_reason_`, `optimization_stop_reason_`, `final_kkt_inf_`, and `final_kkt_normalized_`. |
| `penalties_`, `penalty_`, `best_score_` | CoxPHCV searched penalty vector, selected scalar, and best mean held-out partial log likelihood. |
| `estimator_` | CoxPHCV final fitted CoxPH. Access its CoxPH properties, e.g. `cv.estimator_.log_likelihood`; they are not all delegated as CV properties. |
| `cv_results_` | Dictionary: `pl_path`, `converged_path`, `failure_path` have penalty-by-fold structure; `mean_pl` and `effective_fold_counts` are per-penalty vectors; `fold_valid` identifies effective folds. Additional backend/cache diagnostics are described in the [Cox guide](../models/coxph.md#outputs). |

CoxPHCV copies coefficient and inference arrays from the final estimator. `inference_method_`, `inference_target_`, `penalty_conditioning_`, and `penalty_selection_adjusted_` explain the inferential target; `inference_backend_`, `inference_approximate_`, `inference_fallback_reason_`, and transfer fields describe execution. Joint-test availability is reported separately from marginal intervals. Positive penalties use fixed-penalty estimating-equation inference, not debiased or selection-adjusted inference; see [covariance and inference](../models/coxph.md#covariance-and-inference).

<!-- example: cox-output-reference-cpu -->
```python
import numpy as np
from scipy.stats import norm
from statgpu.survival import CoxPH

rng = np.random.default_rng(53)
X = rng.normal(size=(160, 2))
event_time = rng.exponential(np.exp(-X @ np.array([0.5, -0.3])))
censor_time = rng.exponential(2.0, size=len(X))
time = np.maximum(np.round(np.minimum(event_time, censor_time), 1), 0.1)
event = (event_time <= censor_time).astype(int)
model = CoxPH(ties="efron", device="cpu").fit(X, time, event)
coefficient_ci = np.column_stack([
    model.coef_ - norm.ppf(0.975) * model._bse,
    model.coef_ + norm.ppf(0.975) * model._bse,
])
hazard_ratio_ci = np.exp(coefficient_ci)
query = X[:2]
curves, grid = model.predict_survival(query, times=[0.0, 0.5, 1.0])
failure_times = np.unique(time[event == 1])
increments = np.array([
    np.sum((time == t) & (event == 1)) / np.exp(X[time >= t] @ model.coef_).sum()
    for t in failure_times
])
baseline = np.array([increments[failure_times <= t].sum() for t in grid])
survival_from_formula = np.exp(-np.exp(query @ model.coef_)[:, None] * baseline)
print(coefficient_ci.shape, hazard_ratio_ci.shape, curves.shape)
print(np.allclose(coefficient_ci, model._conf_int))
print(np.allclose(curves, survival_from_formula))
summary_result = model.summary()
assert summary_result is None
```


The first three printed lines are `(2, 2) (2, 2) (2, 3)`, `True`, and `True`, followed by the printed summary. The explicit sum reconstructs the implemented Breslow baseline even though coefficients use Efron ties.

## GAM

Import `GAM` from `statgpu.semiparametric`. All constructor meanings and admissible values are in the [GAM table](../models/semiparametric.md#complete-constructor-and-output-reference).

<!-- signature: GAM -->
```text
GAM(n_splines=20, degree=3, lam=None, penalty_order=2, knot_method='quantile', gamma=1.0, device='auto', n_jobs=None)
```

<!-- signature: GAM.fit -->
```text
GAM.fit(X, y=None, **fit_params)
```

<!-- signature: GAM.predict -->
```text
GAM.predict(X)
```

<!-- signature: GAM.summary -->
```text
GAM.summary()
```


- `fit`: finite numeric `(n,p)` or single-feature `(n,)` `X`; one finite response per row. A one-column `y` is flattened. `y=None` is not a usable missing-target fit. Returns `self`. Only `X` and `y` are used: additional `fit_params`, including `sample_weight`, are currently ignored and do not enable weighted fitting. Do not pass them.
- `predict`: `(q,p)`, or a vector interpreted according to fitted feature count; returns a NumPy `(q,)` even after GPU fitting. Reuses knots, boundaries, and centering. It does not offer reliable extrapolation.
- `summary`: prints and returns a dictionary with `n_features`, `n_splines_per_feature`, `spline_degree`, `penalty_order`, `smoothing_parameter`, `effective_df`, `intercept`, and, only after automatic selection, `gcv_score`.
- Fitted fields: backend-native `coef_` of length `1+sum(n_basis_j)` and per-feature `knots_`; scalar `intercept_`, `edf_`, `lam_`; `gcv_score_` is a float for automatic selection and `None` for fixed lambda; `n_features_` is an integer. Coefficients describe centered basis functions, not raw-feature slopes.
- `lam=None` searches the built-in 100-value grid; there is no GAM custom-grid/CV-estimator API. Choose other settings with external validation. No GAM-specific `score`, sample-weight objective, family/link, coefficient inference, or confidence-band method is provided. Generic inherited helpers do not create these capabilities.
- Always refit after `set_params`. Some GAM parameter updates currently retain the previous fitted arrays; continued prediction before refitting can use inconsistent state. A fresh GAM instance is the safest way to change the basis design.

## Kernel density estimation

Import all following names from `statgpu.nonparametric`. `KDE` is an alias subclass of `KernelDensityEstimator` with the same constructor and methods. Constructor arguments are keyword-only.

<!-- signature: KernelDensityEstimator -->
```text
KernelDensityEstimator(*, bandwidth='scott', weights=None, kernel='gaussian', backend='auto', device='auto', n_jobs=None, gpu_memory_cleanup=False)
```

<!-- signature: KernelDensityEstimator.fit -->
```text
KernelDensityEstimator.fit(X, y=None)
```

<!-- signature: KernelDensityEstimator.pdf -->
```text
KernelDensityEstimator.pdf(points, *, batch_size=1024)
```

<!-- signature: KernelDensityEstimator.logpdf -->
```text
KernelDensityEstimator.logpdf(points, *, batch_size=1024)
```

<!-- signature: KernelDensityEstimator.__call__ -->
```text
KernelDensityEstimator.__call__(points, *, batch_size=1024)
```

<!-- signature: KernelDensityEstimator.predict -->
```text
KernelDensityEstimator.predict(X)
```

<!-- signature: KernelDensityEstimator.score_samples -->
```text
KernelDensityEstimator.score_samples(X)
```

<!-- signature: KernelDensityEstimator.score -->
```text
KernelDensityEstimator.score(X, y=None)
```

<!-- signature: KernelDensityEstimator.to_numpy_metadata -->
```text
KernelDensityEstimator.to_numpy_metadata()
```

<!-- signature: fit_kde -->
```text
fit_kde(samples, *, bandwidth='scott', weights=None, kernel='gaussian', backend='auto')
```

<!-- signature: kde_pdf -->
```text
kde_pdf(samples, points, *, bandwidth='scott', weights=None, kernel='gaussian', backend='auto', return_log=False, batch_size=1024)
```


| Argument | Meaning and restrictions |
|---|---|
| `bandwidth` | Positive finite scalar factor, or `scott`, `silverman`, `nrd0`, `nrd`, `ucv`, `bcv`, `sj`, `sj-ste`, `sj-dpi`. Numeric values are covariance factors, not widths in input units. |
| `weights` | Constructor/functional-fit input, not a `fit` keyword: nonnegative finite length `n`, positive sum; normalized internally. Default `None` means equal weights. Covariance needs more than one effectively weighted observation. |
| `kernel` | Default `gaussian`; also `rectangular`, `triangular`, `epanechnikov`, `biweight`, `triweight`, `cosine`, `optcosine`; the last two are 1D-only. |
| `backend` | `auto`, `numpy`, `cupy`, `torch`; selects the array library. `auto` follows device/global configuration. |
| `device`, `n_jobs`, `gpu_memory_cleanup` | Estimator-only device control, shared job option, and best-effort GPU-cache cleanup. The kernel evaluators do not use `n_jobs` for a parallel query pool. Functional helpers do not expose these arguments. |
| `batch_size` | Positive query-batch size for `pdf`, `logpdf`, `__call__`, and `kde_pdf`; not a KDE constructor or `predict`/`score_samples`/`score` argument. Some NumPy fast paths evaluate a small workload together. |
| `return_log` | `kde_pdf` only; `False` returns density, `True` log density. |

`fit(X,y=None)` ignores `y` and returns `self`; `fit_kde` returns a fitted `KDE`. Samples are finite `(n,)` or `(n,p)`, with `n>=2`. Queries are finite `(q,p)`; a vector means multiple queries for one feature or one query for a multivariate fit. `pdf`, `predict`, and `__call__` return backend-native `(q,)` density; `logpdf` and `score_samples` return log density. `score` returns Python `float`, the unweighted mean query log density, and ignores `y`. Compact-support kernels can return density 0 and log density `-inf`.

For large-offset coordinates, subtract a training-derived offset from both samples and queries before fitting/evaluation. Current quadratic-distance calculations can lose precision without this centering, including KDE log-density and multivariate density/regression. This translation does not change the intended statistical estimator. Explicit `backend="torch"` alone does not guarantee CUDA placement; see [device guidance](../guides/device-and-memory.md).

Fitted attributes: `samples_` `(n,p)`, normalized `weights_` `(n,)`, scalar `bandwidth_factor_`, `bandwidth_info_` (selection result or `None` for numeric bandwidth), `covariance_` and `inv_covariance_` `(p,p)`, scalar `norm_const_` and `inv_norm_const_`, `kernel_`, `backend_`, `n_samples_`, and `n_features_`. `to_numpy_metadata()` returns a dictionary with `bandwidth_factor`, `bandwidth_selection`, `n_samples`, `n_features`, `backend`, `kernel`, `covariance`, `inv_covariance`, and `weights`; arrays are host NumPy arrays.

## Kernel regression

`KernelRegressionRegressor` is an alias subclass of `KernelRegression` with the same constructor and methods. Shared kernel, bandwidth, weight, backend, device, jobs, and cleanup controls have the KDE meanings above.

<!-- signature: KernelRegression -->
```text
KernelRegression(*, bandwidth='scott', weights=None, kernel='gaussian', regression='nw', kernel_metric='full', bandwidth_per_feature=None, backend='auto', device='auto', n_jobs=None, batch_size=1024, min_effective_weight=1e-12, gpu_memory_cleanup=False)
```

<!-- signature: KernelRegression.fit -->
```text
KernelRegression.fit(X, y)
```

<!-- signature: KernelRegression.predict -->
```text
KernelRegression.predict(points, *, batch_size=None, min_effective_weight=None)
```

<!-- signature: KernelRegression.__call__ -->
```text
KernelRegression.__call__(points, *, batch_size=None, min_effective_weight=None)
```

<!-- signature: KernelRegression.score -->
```text
KernelRegression.score(X, y)
```

<!-- signature: KernelRegression.to_numpy_metadata -->
```text
KernelRegression.to_numpy_metadata()
```

<!-- signature: fit_kernel_regression -->
```text
fit_kernel_regression(samples, targets, *, bandwidth='scott', weights=None, kernel='gaussian', regression='nw', kernel_metric='full', bandwidth_per_feature=None, backend='auto')
```

<!-- signature: kernel_regression_predict -->
```text
kernel_regression_predict(samples, targets, points, *, bandwidth='scott', weights=None, kernel='gaussian', regression='nw', kernel_metric='full', bandwidth_per_feature=None, backend='auto', batch_size=1024, min_effective_weight=1e-12)
```


- `regression="nw"` is Nadaraya–Watson; `"local_linear"` fits an intercept and local slopes per query. There is no global coefficient vector or coefficient inference.
- `kernel_metric="full"` uses the weighted covariance; `"diagonal"` removes off-diagonal terms. `bandwidth_per_feature=None` selects a scalar factor. Positive absolute per-feature widths (a scalar can broadcast) require the diagonal metric and bypass the `bandwidth` selector.
- `fit(X,y)` returns `self`; `fit_kernel_regression(samples,targets,...)` returns fitted `KernelRegression`. Samples follow the KDE contract; targets are finite `(n,)` or `(n,r)`. Predictions preserve target dimensionality: `(q,)` or `(q,r)`, including `(q,1)` for a one-column target.
- Constructor `batch_size=1024` and `min_effective_weight=1e-12` become prediction defaults. Method values `None` inherit them; explicit positive values override them for that call. When local total weight is too small, prediction returns the weighted training-target mean. Local-linear instability may use numerical stabilization or NW fallback.
- `kernel_regression_predict` fits and predicts once, with the listed evaluation overrides. Functional helpers expose `backend`, not estimator-only device/jobs/cleanup settings.
- `score` returns a host scalar R-squared after flattening all target columns together, not the average of per-target scores; constant targets return `0.0`. Use separate target metrics when scales differ.
- Regression-only bandwidth aliases `cv`, `cv_ls`, `cv-nw`, `cv-ll` search leave-one-out MSE. Selection uses full covariance, and multivariate local-linear selection uses NW predictions. Validate the actual intended diagonal/multivariate-local-linear model externally; these names are not KDE selectors.

In addition to the common fitted fields (excluding KDE normalization constants), regression exposes `targets_` `(n,r)`, `n_targets_`, `target_mean_` `(r,)`, `target_was_1d_`, `regression_`, `kernel_metric_`, and `bandwidth_per_feature_` (backend vector or `None`). Metadata has `bandwidth_factor`, `bandwidth_selection`, `bandwidth_per_feature`, `n_samples`, `n_features`, `n_targets`, `backend`, `kernel`, `kernel_metric`, `regression`, `covariance`, `inv_covariance`, `weights`, and `target_mean`.

## Density confidence intervals

<!-- signature: kde_confidence_interval -->
```text
kde_confidence_interval(samples, points, *, bandwidth='scott', weights=None, kernel='gaussian', backend='auto', n_resamples=200, confidence_level=0.95, random_state=None, method='normal', bootstrap_method='percentile', return_bootstrap_samples=False, batch_size=1024)
```

<!-- signature: kde_bootstrap_confidence_interval -->
```text
kde_bootstrap_confidence_interval(samples, points, *, bandwidth='scott', weights=None, kernel='gaussian', backend='auto', n_resamples=200, confidence_level=0.95, random_state=None, method='percentile', return_bootstrap_samples=False, batch_size=1024)
```

<!-- signature: KDEBootstrapResult.to_dict -->
```text
KDEBootstrapResult.to_dict()
```


Shared `samples`, `points`, `bandwidth`, `weights`, `kernel`, and `backend` have the KDE meanings. Use a finite `confidence_level` strictly between 0 and 1; do not pass NaN. `n_resamples` is a positive integer; it is validated even on the normal path. `random_state` seeds bootstrap sampling. `batch_size` controls density evaluation; a NumPy bootstrap fast path may evaluate all queries together.

- `kde_confidence_interval(method="normal")`: only 1D Gaussian KDE; the fitted covariance determines the absolute width. It returns an asymptotic, pointwise, zero-lower-clipped normal interval. The result has `n_resamples=0` and `bootstrap_samples=None`, even if `return_bootstrap_samples=True`.
- `kde_confidence_interval(method="bootstrap", bootstrap_method="percentile")`: percentile resampling; no other bootstrap method is supported.
- `kde_bootstrap_confidence_interval(method="percentile")`: convenience wrapper for that bootstrap path. Here `method` names the percentile procedure, whereas the general function's `method` chooses normal versus bootstrap.
- `return_bootstrap_samples=True` returns the bootstrap replicate matrix; otherwise it is omitted from the result. Intervals do not adjust smoothing bias or give simultaneous coverage. The [tutorial](../models/nonparametric.md#cpu-example-3-pointwise-density-intervals) explains selector refitting and why the illustrated interpretation is restricted to independent, equally weighted data. Nonuniform weights are used both for sampling and reweighting and need a study-specific justification.

`KDEBootstrapResult` is the result container for **both** methods. Arrays are NumPy: `points` `(q,)` for one feature or `(q,p)` otherwise; `estimate`, `lower`, `upper` `(q,)`; optional `bootstrap_samples` `(n_resamples,q)`. Other fields are `confidence_level`, `n_resamples`, `random_state`, `kernel`, `backend`, and `metadata`. The backend label records the fitted array library, not the output-array location. Metadata includes `method`, `bandwidth`, `batch_size`, `n_features`, plus `n_eff` for normal intervals or `bootstrap_method` for bootstrap. `to_dict()` converts result arrays to lists and omits `bootstrap_samples` when absent.

<!-- example: kde-normal-reference-cpu -->
```python
import numpy as np
from scipy.stats import norm
from statgpu.nonparametric import fit_kde, kde_confidence_interval

samples = np.linspace(-2.0, 2.0, 60)
points = np.array([-1.0, 0.0, 1.0])
model = fit_kde(samples, bandwidth=0.4, backend="numpy")
ci = kde_confidence_interval(samples, points, bandwidth=0.4, backend="numpy")
h = np.sqrt(model.covariance_[0, 0])
n_eff = 1.0 / np.sum(model.weights_ ** 2)
se = np.sqrt(ci.estimate / (2 * np.sqrt(np.pi) * n_eff * h))
normal_lower = np.maximum(ci.estimate - norm.ppf(0.975) * se, 0.0)
normal_upper = ci.estimate + norm.ppf(0.975) * se
print(ci.estimate.shape, ci.n_resamples, ci.bootstrap_samples)
assert np.allclose(ci.lower, normal_lower)
assert np.allclose(ci.upper, normal_upper)
```


The printed result is `(3,) 0 None`. This checks the formula, not actual coverage for every dataset.

## Lower-level bandwidth selectors

Most users should set estimator `bandwidth`. These public functions are for callers who already have validated backend arrays and covariance information.

<!-- signature: select_bandwidth -->
```text
select_bandwidth(bandwidth, *, n_eff, n_features, samples_2d, weights_1d, data_cov, xp, enable_r_selectors=True, weighted_r_selector_strategy='quantile_resample', multivariate_selector_strategy='projection_pca_1d', estimator='kde', targets=None, regression='nw', kernel='gaussian')
```

<!-- signature: select_bandwidth_factor -->
```text
select_bandwidth_factor(bandwidth, *, n_eff, n_features, samples_2d, weights_1d, data_cov, xp, enable_r_selectors=True, weighted_r_selector_strategy='quantile_resample', multivariate_selector_strategy='projection_pca_1d', estimator='kde', targets=None, regression='nw', kernel='gaussian')
```

<!-- signature: BandwidthSelectionResult.to_dict -->
```text
BandwidthSelectionResult.to_dict()
```


| Argument | Required input or behavior |
|---|---|
| `bandwidth` | Same selector names/scalar factors as the chosen estimator; regression CV names require `estimator="kernel_regression"`. |
| `n_eff`, `n_features` | Positive effective sample size and feature count `p`. For normalized weights, `n_eff=1/sum(weights_1d**2)`. |
| `samples_2d`, `weights_1d`, `data_cov`, `xp` | Backend arrays `(n,p)`, normalized `(n,)` weights, unscaled `(p,p)` weighted covariance, and the matching NumPy/CuPy/Torch array module. Use finite compatible inputs; passing a module does not by itself move arrays to a GPU. |
| `enable_r_selectors=True` | Enables R-style names such as `nrd0`, `ucv`, and `sj`; `False` rejects those names. These routines do not launch R. |
| `weighted_r_selector_strategy` | `"quantile_resample"` is the supported strategy for nonuniform weights in R-style selection. |
| `multivariate_selector_strategy` | `"projection_pca_1d"` is the supported multivariate extension for R-style selectors. It is not an exact multivariate R equivalent. |
| `estimator`, `targets`, `regression`, `kernel` | Default `"kde"`; choose `"kernel_regression"` with compatible targets for regression-CV selection. `regression` is `"nw"` or `"local_linear"`; `kernel` selects the regression kernel. Defaults are in the signatures. |

`select_bandwidth` returns `BandwidthSelectionResult`; `select_bandwidth_factor` returns its scalar `factor`. Result fields: `factor`, `method`, `n_features`, `n_eff`, `used_r_selector`, `weighted`, `weighted_strategy`, `multivariate_strategy`, `selector_dimension`, and `details`. `to_dict()` returns those same names. `details` is method-specific diagnostic information; do not assume all selectors have the same keys. `used_r_selector` identifies the R-style numerical selector, not an external R process. Some constant/sparse samples cause selectors to fail.

<!-- example: bandwidth-selector-reference-cpu -->
```python
import numpy as np
from statgpu.nonparametric import fit_kde, select_bandwidth, select_bandwidth_factor

samples = np.linspace(-2.0, 2.0, 40)
base = fit_kde(samples, bandwidth=1.0, backend="numpy")
selector_inputs = dict(
    n_eff=1.0 / np.sum(base.weights_ ** 2),
    n_features=base.n_features_, samples_2d=base.samples_,
    weights_1d=base.weights_, data_cov=base.covariance_, xp=np,
)
selection = select_bandwidth("scott", **selector_inputs)
factor = select_bandwidth_factor("scott", **selector_inputs)
print(selection.method, round(factor, 6))
assert np.isclose(factor, 40 ** (-1 / 5))
assert factor == selection.factor
```


The printed result is `scott 0.478176`. Reusing a bandwidth-one KDE supplies the covariance and normalized weights without duplicating their preparation.

## Implementation links

For implementation details rather than API lookup: [CoxPH](../../../statgpu/survival/_cox.py), [CoxPHCV](../../../statgpu/survival/_cox_cv.py), [GAM](../../../statgpu/semiparametric/_gam.py), [KDE and intervals](../../../statgpu/nonparametric/kernel_smoothing/_kde.py), [kernel regression](../../../statgpu/nonparametric/kernel_smoothing/_kernel_regression.py), [selectors](../../../statgpu/nonparametric/kernel_smoothing/_bandwidth_selection.py). Statistical references are in the linked learner pages.
