# CoxPH

> Language: English<br>
> Last updated: 2026-10-09<br>
> This page: Model documentation<br>
> Switch: [Chinese](../../cn/models/coxph.md)

## Overview

Use Cox regression when the question is **how predictors relate to the time until
an event**, such as a machine failure, and observation can end before the event
occurs. Regressing only observed failure times discards useful follow-up from
units that have not failed; classifying event/no-event ignores how long each unit
was observed.

For ordinary right-censored data, each row contains predictors `X`, a positive
observed follow-up `time`, and an `event` indicator:

- `event=1`: the event occurred at `time`;
- `event=0`: observation ended at `time` without observing the event. Its true
  event time is later than that follow-up, not zero or known never to occur.

Keep censored rows. The usual interpretation assumes censoring is independent of
the event process conditional on the modeled covariates and study design.

The model is

$$
h_s(t\mid x)=h_{0s}(t)\exp(x^\top\beta).
$$

Here $h_{0s}$ is the baseline hazard in stratum $s$, $x$ is a covariate vector,
and $\beta$ is the shared coefficient vector. The baseline describes how the
instantaneous event hazard changes over time, and predictors multiply it.
For fixed covariates within the same stratum, the **proportional-hazards (PH)
assumption** says that this hazard ratio stays constant over time. A one-unit
increase in feature `j`, holding the others fixed, multiplies the hazard by
`exp(coef_[j])`. A hazard ratio is not an event probability, a survival time, or
by itself a causal effect. Check whether PH and the study design are reasonable;
optimizer convergence alone does not validate these assumptions. Time-varying
covariates can change `x(t)`, but their fitted coefficient remains constant.

`CoxPH` implements proportional-hazards regression with Breslow, Efron, or Exact
tie handling on NumPy, CuPy CUDA, and Torch CUDA. It supports ordinary
right-censored observations, delayed entry, counting-process `(start, stop]`
rows, independent strata, time-varying covariates, robust/cluster covariance,
and L2-penalty selection through `CoxPHCV`.

<!-- example: coxph-cpu-walkthrough -->
## Imports

```python
import numpy as np
from statgpu.survival import CoxPH
```

## First CPU Walkthrough

Run the following steps in order with NumPy and StatGPU. Start with a
right-censored fit and held-out evaluation; the later survival, CV, and GPU
examples reuse this data rather than generating another dataset.

### Prepare the inputs

`X` has one row per subject and one column per feature, with shape
`(n_samples, n_features)`. Both `time` and `event` are one-dimensional vectors
with one entry per row. `time` is the event or censoring time; `event=1` means
an observed event and `event=0` means right censoring. Do not add an intercept
or constant column to `X`.

Simulate 400 independent subjects with three comparable feature scales.
For your own analysis, replace this block with your data-loading step.

```python
rng = np.random.default_rng(42)
X = rng.normal(size=(400, 3))
true_coef = np.array([0.8, -0.5, 0.3])
event_time = rng.exponential(scale=np.exp(-(X @ true_coef)))
censor_time = rng.exponential(scale=2.0, size=len(X))
time = np.minimum(event_time, censor_time)
event = (event_time <= censor_time).astype(np.int64)
```

### Hold out evaluation data

Reserve the last 100 subjects before fitting. For real data, choose a split
that respects subjects, groups, or time. Learn scaling and other preprocessing
from the training partition only.

```python
X_train, X_test = X[:300], X[300:]
time_train, time_test = time[:300], time[300:]
event_train, event_test = event[:300], event[300:]
```

### Fit and check convergence

Use the explicit CPU backend and Efron tie handling. Keep
`compute_inference=True` so the later examples can use coefficient uncertainty
and the baseline needed for survival curves. This does not test the PH assumption.

```python
model = CoxPH(
    ties="efron", device="cpu", compute_inference=True,
).fit(X_train, time_train, event_train)
if not model.converged_:
    raise RuntimeError(
        f"{model.optimization_stop_reason_}: "
        f"normalized KKT={model.final_kkt_normalized_}"
    )
```

After convergence, inspect the direction and hazard ratio of each feature:

```python
print("Coefficients:", model.coef_)
print("Per-feature hazard ratios:", model.hazard_ratios_)
print("Convergence:", model.termination_reason_, model.n_iter_)
```

For this seed, the coefficients are approximately `[0.852, -0.457, 0.312]`
and their hazard ratios are `[2.345, 0.633, 1.367]`. For example, increasing the
first feature by one unit multiplies the estimated instantaneous hazard by
about 2.35, holding the others fixed. The second feature is associated with a
lower hazard. These are associations in simulated data, not practical treatment
recommendations.

### Predict relative risk

First compare two held-out profiles without treating a hazard ratio as a probability.

```python
log_risk = model.predict_risk_score(X_test[:2])
relative_hazard = model.predict(X_test[:2])
print("Log-risk:", log_risk)
print("Relative hazard:", relative_hazard)
```

`predict_risk_score(X)` is `X @ coef_`; `predict(X)` and
`predict_hazard_ratio(X)` are `exp(X @ coef_)`, relative to the same-stratum
zero-covariate profile. To compare two profiles, exponentiate their log-risk
difference. `hazard_ratios_` instead has one value per **feature**.

### Evaluate held-out ranking

Use all 100 held-out subjects with their times and event indicators to compute the C-index.

```python
held_out_cindex = model.score(X_test, time_test, event_test)
print("Held-out C-index:", held_out_cindex)
```
<!-- example-end: coxph-cpu-walkthrough -->

The held-out C-index is about `0.766`: larger predicted hazard tends to rank
subjects with earlier observed events ahead of comparable longer-surviving
subjects. This is discrimination, not probability calibration or R-squared.
A value near `0.5` is neutral ranking, `1.0` is perfect ranking on permissible
pairs, and a value below `0.5` suggests reversed ranking. No permissible pairs
also return `0.5`; that is insufficient evaluation evidence, not proof of a
random-quality model. Do not use training concordance as a held-out estimate.

## Predict survival curves

Run the [CPU walkthrough](#first-cpu-walkthrough) first and reuse its `model`
and `X_test`. Estimating the probability of remaining event-free at a given
time requires the fitted baseline as well as relative risk.

<!-- example-requires: coxph-cpu-walkthrough -->
<!-- example: coxph-survival-prediction -->
```python
requested_times = np.array([0.0, 0.5, 1.0, 2.0])
curves, curve_times = model.predict_survival(
    X_test[:2], times=requested_times,
)
print("Survival shape and times:", curves.shape, curve_times)
```
<!-- example-end: coxph-survival-prediction -->

`predict_survival` returns the tuple **`(curves, times)`**, not just a matrix:
`curves` has shape `(n_new, n_times)` and `times` has shape `(n_times,)`.
Here these are `(2, 4)` and `(4,)`; each row estimates the probability of
remaining event-free beyond the corresponding time. Curves lie in `[0, 1]` and
are non-increasing on an increasing time grid. `times=None` uses the fitted
baseline's event-time grid (the union for strata); explicit times retain the
requested order. The baseline is stepwise and flat beyond its last event time,
so extending the grid does not establish reliable long-term extrapolation.
For counting-process data, a supplied prediction row describes a fixed
covariate profile, not automatic integration over a future covariate trajectory.

### From relative hazard to survival probability

For a fixed covariate profile in stratum $s$,

$$
H_s(t\mid x)=H_{0s}(t)\exp(x^\top\beta),\qquad
S_s(t\mid x)=\Pr(T>t\mid x,s)=\exp\{-H_s(t\mid x)\}.
$$

$H_{0s}(t)=\int_0^t h_{0s}(u)\,du$ is the cumulative baseline hazard. The
implementation estimates it with Breslow increments at distinct event times:

$$
\widehat H_{0s}(t)=\sum_{t_k\le t}
\frac{d_{sk}}{\sum_{j\in R_s(t_k)}\exp(x_j^\top\widehat\beta)}.
$$

Here $d_{sk}$ is the number of events in stratum $s$ at $t_k$, and $R_s(t_k)$
contains its rows satisfying `start < t_k <= stop`. These baseline increments
are used even when coefficients are fitted with `ties="efron"` or `"exact"`.
This explains why a relative hazard alone is insufficient for a survival
probability: the fitted baseline is also needed. The [API reference example](../reference/survival-smoothing-api.md#coefficients-inference-and-cv-results)
reconstructs the returned curves and explains coefficient versus hazard-ratio
confidence intervals.

## Input Shapes and Fit API

The matrix API is `CoxPH(...).fit(X, time, event, ...)` or
`CoxPHCV(...).fit(X, time, event, ...)`; both return the fitted estimator.

| Input | Shape / contract |
|---|---|
| `X` | Finite real matrix `(n_samples, n_features)`; no intercept/constant column. Preserve feature order at prediction. `CoxPH` also accepts a one-dimensional single feature; use a matrix for `CoxPHCV`. |
| `time` | Finite positive vector `(n_samples,)`, in one consistent time unit; event or censoring time, or interval stop time. |
| `event` | Vector `(n_samples,)` containing only `0` or `1`; at least one observed event is needed for fitting. |
| `entry` / `start` | Optional vector `(n_samples,)` with `0 <= start < time`; aliases, never pass both. Omitted means entry at zero. |
| `strata` | Optional labels `(n_samples,)` for independent risk sets and separate baseline hazards with shared coefficients. |
| `subject_id` | Optional labels `(n_samples,)` identifying repeated rows; use for subject-aware concordance, robust aggregation, and CV splitting. |
| `cluster` | Optional labels `(n_samples,)`; required by `cov_type="cluster"`. Cluster labels alone do not make CV folds group-preserving; use suitable `subject_id` or explicit `cv_splits`. |
| `init_coef` | Optional finite vector `(n_features,)` for a `CoxPH` initial estimate; not a `CoxPHCV.fit` argument. |
| `formula`, `data` | Alternative `CoxPH.fit` interface described below; not accepted by `CoxPHCV.fit`. |

Both estimators also accept `fit(X, y)` with `event` omitted: `y` is either
`(n_samples, 2)` with columns `[time, event]`, or `(n_samples, 3)` with columns
`[start, stop, event]`. Do not pass a separate `entry`/`start` with the
three-column form. The same packed targets work in `score`; its interval
argument is `start`, not `entry`. Prepare missing numeric values before matrix
fitting; non-finite arrays are rejected rather than silently dropped.

The [complete reader-facing API reference](../reference/survival-smoothing-api.md#coxph-and-coxphcv)
collects signatures, return shapes, method restrictions, and inference outputs.
Constructor options are also in [Parameters](#parameters) and
[CoxPHCV Parameters](#coxphcv-parameters). Canonical signatures and method
contracts are maintained in [`CoxPH`](../../../statgpu/survival/_cox.py) and
[`CoxPHCV`](../../../statgpu/survival/_cox_cv.py).

Important behavior:

- explicit `device="cuda"` and `device="torch"` requests never silently fall back to CPU;
- `entry=` and `start=` are aliases and are mutually exclusive;
- a row is in the risk set at time `t` exactly when `start < t <= stop` and its
  stratum matches the event stratum;
- `subject_id=` identifies repeated rows from one subject for concordance,
  sandwich aggregation, and subject-preserving CV folds;
- `compute_inference=False` performs estimation only and leaves inference and
  baseline-hazard fields unset.

## Parameters

| Parameter | Default | Description |
|---|---:|---|
| `ties` | `"breslow"` | `"breslow"`, `"efron"`, or `"exact"` |
| `tol` | `1e-9` | Newton/KKT convergence tolerance |
| `max_iter` | `100` | Maximum iterations |
| `device` | `"auto"` | `"cpu"`, `"cuda"`, `"torch"`, or `"auto"` |
| `n_jobs` | `None` | Accepted shared CPU-job setting; the current Cox fit/CV loop does not use it to parallelize folds. |
| `compute_inference` | `True` | Compute covariance, tests, and baseline hazards |
| `compute_cindex` | `True` | Compute training concordance |
| `cov_type` | `"nonrobust"` | `"nonrobust"`, `"hc0"`, `"hc1"`, or `"cluster"` |
| `penalty` | `0.0` | Non-negative L2 penalty |
| `inference_mode` | `"strict"` | `"approx"` is a compatibility alias; both settings use the same robust covariance calculation. See [Covariance and Inference](#covariance-and-inference). |
| `gpu_memory_cleanup` | `False` | Best-effort CuPy/Torch cache cleanup |

## Cross-Validation

`CoxPHCV` evaluates an L2 penalty grid using the same tie method, start/entry,
strata, and backend semantics, then refits a `CoxPH` estimator at the selected
penalty. When `subject_id` is supplied, every row from a subject remains wholly
inside one automatically generated fold. User-provided `cv_splits` are rejected
if they leak a subject between train and validation. `inference_mode` and
`compute_inference` are forwarded to the final refit.

Run this after the [CPU walkthrough](#first-cpu-walkthrough). Search only its
300 training subjects, then evaluate the selected/refitted model on the same
untouched 100-subject test partition. L2 shrinks coefficients; because it acts on
coefficient magnitudes, comparable feature scales matter. For real data, fit
preprocessing within each training fold rather than using validation/test data.

<!-- example-requires: coxph-cpu-walkthrough -->
<!-- example: coxph-cpu-cv -->
```python
from statgpu.survival import CoxPHCV

cv_model = CoxPHCV(
    penalties=[0.0, 0.1, 1.0, 10.0],
    cv=3,
    random_state=42,
    ties="efron",
    device="cpu",
    compute_inference=False,
).fit(X_train, time_train, event_train)
if not cv_model.converged_:
    raise RuntimeError(cv_model.optimization_stop_reason_)

```

Inspect candidate scores and valid-fold counts, then evaluate the final model on the held-out test set:

```python
cv_mean_pl = cv_model.cv_results_["mean_pl"]
cv_fold_counts = cv_model.cv_results_["effective_fold_counts"]
cv_test_cindex = cv_model.score(X_test, time_test, event_test)
print("Penalty grid:", cv_model.penalties_)
print("Mean held-out partial log-likelihood:", cv_mean_pl)
print("Effective fold counts:", cv_fold_counts)
print("Selected penalty:", cv_model.penalty_)
print("Test C-index:", cv_test_cindex)
```
<!-- /example: coxph-cpu-cv -->

This example selects `penalty_=1.0` and obtains a test C-index of approximately
`0.766`; CV is not guaranteed to improve that score. `best_score_` is the
selected **mean held-out partial log-likelihood**, not the C-index returned by
`score()`, and should only be compared for the same dataset/folds and scoring
convention. `cv_results_["pl_path"]` has shape `(n_penalties, n_folds)`;
`mean_pl` and `effective_fold_counts` each have shape `(n_penalties,)`.
Each fold contributes its summed partial log likelihood, without normalization
by rows or events. For custom grids, numerical near-ties prefer stronger penalties, so the selected
score can be slightly below the largest mean. Custom-grid results retain the
input penalty order even though evaluation proceeds from stronger to weaker penalties.
Inspect `converged_path`, `failure_path`, and `fold_valid` when candidates cannot
be evaluated. A selectable candidate needs finite scores and convergence on
all the same effective folds; a fold requires events in both partitions. If
none qualifies, fitting raises rather than publishing a selection.

`estimator_` is the final `CoxPH` refitted on all supplied training rows using
`penalty_`. With `compute_inference=False`, risk and hazard-ratio predictions
remain available but survival curves and inference do not. Enabling final-refit
inference does not correct for tuning uncertainty or shrinkage bias; see
[Penalty Scaling and Penalized Inference](#penalty-scaling-and-penalized-inference).

### CoxPHCV Parameters

| Parameter | Default | Description |
|---|---:|---|
| `penalties` | `None` | Non-empty finite non-negative one-dimensional L2 grid; `None` requests an automatic grid. |
| `n_penalties` | `100` | Number of values in the automatic grid. |
| `penalty_min_ratio` | `1e-3` | Minimum/maximum ratio for the automatic grid; in `(0, 1]`. |
| `cv` | `5` | Number of automatically generated folds, at least two. |
| `cv_splits` | `None` | Explicit `(train_indices, validation_indices)` pairs, overriding generated folds; non-empty disjoint one-dimensional integer indices in range. One-shot iterators are materialized once and reused, including during parameter inspection/cloning. |
| `ties` | `"breslow"` | `"breslow"`, `"efron"`, or `"exact"` for candidates and refit. |
| `tol` | `1e-9` | Candidate/refit convergence tolerance. |
| `max_iter` | `100` | Candidate/refit maximum iterations. |
| `device` | `"auto"` | `"cpu"`, `"cuda"`, `"torch"`, or `"auto"`. |
| `n_jobs` | `None` | Shared CPU-job option, forwarded to the refit; the current CV loop is not parallelized by this option. |
| `compute_inference` | `True` | Compute inference and baseline hazards only in the final refit. |
| `cov_type` | `"nonrobust"` | Covariance contract for the final refit. |
| `inference_mode` | `"strict"` | `"approx"` is a compatibility alias; both settings use the same robust covariance calculation. See [Covariance and Inference](#covariance-and-inference). |
| `gpu_memory_cleanup` | `False` | Best-effort GPU cache cleanup at public work boundaries. |
| `random_state` | `None` | Seed for automatically generated CV folds. |

Unlike `CoxPH`, `CoxPHCV` has no constructor `penalty` or `compute_cindex`;
use `penalties` and call `score` explicitly. The complete source reference is
[`CoxPHCV`](../../../statgpu/survival/_cox_cv.py).


### L1/L2/ElasticNet/SCAD/MCP model-family CV

`CoxPHCV` above is the canonical L2 Cox selector and may run the configured
final-refit inference. The public penalized-model family uses the separate
survival-aware branch of `PenalizedGLM_CV`. This additional example reuses `X_train`, `time_train`, and `event_train`
from the [CPU walkthrough](#first-cpu-walkthrough), keeping the test subjects
out of penalty selection:

<!-- example-requires: coxph-cpu-walkthrough -->
<!-- example: coxph-penalized-family-cv -->
```python
from statgpu.linear_model import PenalizedGLM_CV

survival_y = np.column_stack([time_train, event_train])
penalized_cv = PenalizedGLM_CV(
    loss="cox_ph",
    penalty="mcp",               # l1, l2, elasticnet, scad, or mcp
    alpha_grid=[0.1, 0.03, 0.01],
    cv=5,
    cv_strategy="strict",
    loss_kwargs={"ties": "efron"},
    device="cpu",                # or "cuda" / "torch"
).fit(X_train, survival_y)
```
<!-- /example: coxph-penalized-family-cv -->

This branch keeps the `(time, event)` target two-dimensional, forbids an
intercept, evaluates unpenalized held-out partial likelihood, and requires
finite evidence from every evaluable fold. If no alpha satisfies that contract,
fit raises and publishes no selected alpha or fitted estimator. The final refit
is `PenalizedCoxPHModel(compute_inference=False)`; post-selection coefficient
inference, `two_stage`, sample weights, and dictionary targets are unsupported.
No-penalty aliases are non-tunable and are rejected by this CV path; use a
direct model fit instead.

The two Cox APIs use different loss scales. `PenalizedCoxPHModel` minimizes

$$
-\ell(\beta)/n + P_\alpha(\beta),
\qquad P_\alpha(\beta)=\frac{\alpha}{2}\lVert\beta\rVert_2^2
\quad\text{for the built-in L2 penalty}.
$$

Here `n` is the number of rows actually fitted, including censored observations
and after any formula missing-row removal, not the number of events. With the
same rows, features, and tie method, built-in
`penalty="l2", alpha=a` therefore matches the objective of
`CoxPH(penalty=n*a/2)`; `alpha` and `penalty` are not interchangeable values.
The default family `alpha=1.0` is positive regularization, whereas `CoxPH`
defaults to `penalty=0.0`. A supplied penalty object uses its own strength.

The penalized-family CV branch minimizes the mean over folds of each fold's
`-ell_validation / n_validation`; its `best_score_` is the negative of the
selected mean, hence a mean row-normalized held-out log likelihood. Numerical
near-ties prefer the stronger alpha. This differs
from `CoxPHCV`'s mean of summed fold log likelihoods. In addition, each training
fold and the final refit have their own row count. A direct-fit conversion
using one `n` therefore does not make the two CV searches equivalent. Use the
appropriate API's selection and final-refit convention throughout an analysis.

Custom folds may be general non-empty disjoint train/validation splits, including
forward `TimeSeriesSplit` or repeated holdout; they need not be complementary or
cover each row exactly once. Fold indices are validated before any candidate fit
and must be one-dimensional exact integers in range. With an automatic grid,
ElasticNet uses the zero-model KKT boundary
`alpha_max = ||gradient L(0)||_inf / l1_ratio` when `l1_ratio > 0`; a penalty
object supplies its own ratio. Pure L2 (`l1_ratio=0`) has no finite all-zero KKT
threshold and uses the raw zero-score norm as a documented grid heuristic.

For large `device="auto"` searches, Torch and CuPy are selected only after their
CUDA backend reports an operational device. Generic fallback sizing uses only
evaluable folds whose training and validation partitions both contain events;
other normalized folds remain visible in `failure_path` but do not inflate GPU
work. An importable but unusable CuPy installation therefore falls back to
CPU; explicit `device="cuda"` remains strict and raises instead of falling
back.

## Prediction and Scoring

`predict`, `predict_risk_score`, `predict_hazard_ratio`, `predict_survival`, and
`score` execute on the fitted backend for array inputs. A model fitted with
`device="auto"` pins its actual `effective_device_`; later global device changes do
not migrate its prediction or scoring backend. Stratified survival
prediction requires one known stratum label per prediction row, including when
the fit contained only one explicit stratum. Missing or unseen labels raise
`ValueError`.

`score()` uses the same row-label encoder: supplied strata must have shape
`(n_samples,)`, labels must be known when the model was explicitly stratified,
and a multi-stratum fitted model requires scoring labels. Malformed scalar,
two-dimensional, wrong-length, or unseen labels consistently raise
`ValueError` before backend concordance work. Survival curves use log-domain
baseline accumulation for numerical stability. Formula-fitted models apply
their saved design transformation before prediction.

A fitted stratum with no observed failures has a valid empty baseline-hazard
state. Its cumulative baseline hazard is zero at every time, so
`predict_survival()` returns exactly one for that stratum. This applies to
explicit times, automatically selected times, mixed-stratum prediction rows,
and the delegated `CoxPHCV` path; a mismatched stored time/hazard shape remains
an invalid state.

`predict_risk_score()` returns the unexponentiated log-risk. Hazard-ratio
prediction APIs use one strict float64 exponential boundary across canonical,
CV, and penalized Cox models; canonical/CV fitted `hazard_ratios_` use the same
boundary. A value that would overflow to infinity or underflow to zero raises
`CoxFitNumericalError` during canonical/CV fit or `FloatingPointError` during
prediction; values are never silently clipped to an estimator-specific
threshold. `PenalizedCoxPHModel` also exposes
`predict_risk_score()` so extreme finite log-risk remains directly available.

## Outputs

- parameters: `coef_`, `hazard_ratios_`;
- inference: `_bse`, `_zvalues`, `_pvalues`, `_conf_int` when enabled;
- diagnostics: `log_likelihood`, `aic`, `bic`, `concordance_index` where defined;
- convergence: `converged_`, `termination_reason_`,
  `optimization_stop_reason_`, `n_iter_`,
  `final_kkt_inf_`, `final_kkt_normalized_`;
- provenance: `inference_method_`, `inference_backend_`,
  `inference_approximate_`, `inference_fallback_reason_`,
  `inference_target_`, `penalty_conditioning_`, `penalty_selection_adjusted_`,
  `full_host_transfer_performed_`.

`CoxPHCV` additionally exposes `cv_full_host_transfer_performed_`,
`final_refit_full_host_transfer_performed_`, and `orchestration_device_` so
data-movement audits do not confuse host CV selection with the final refit.
Its `cv_results_` separates selection-origin fields (`scoring_device`,
`selection_origin_device`,
`candidate_preparation_origin_device`, and total preparation counts) from
invocation fields (`selection_cache_hit`, `requested_fit_device`,
`effective_device`, and `*_this_call` counts). A cache hit reports zero fold preparation and target
transfer work for that invocation without rewriting the origin device.
If a finite-input candidate returns non-finite fitted coefficients or
likelihood, `CoxPH` raises `CoxFitNumericalError` (a
`FloatingPointError` subclass); `CoxPHCV` excludes only that candidate while
letting input, allocator, CUDA, and unexpected runtime errors propagate.

For coefficient inference, `_bse`, `_zvalues`, and `_pvalues` are `(p,)`
NumPy arrays. `_conf_int` is `(p, 2)` and contains fixed **95% marginal
coefficient-scale** intervals; `np.exp(model._conf_int)` gives hazard-ratio
intervals, as printed by `summary()`. Inference-disabled fits leave these
fields `None`. `summary()` prints and returns `None`.

`log_likelihood`, `aic`, `bic`, and `concordance_index` are CoxPH properties.
AIC/BIC access raises for positive penalties; unpenalized BIC uses the event
count, not the row count. For a CV model, access these properties on
`estimator_`. See the [output table and runnable verification](../reference/survival-smoothing-api.md#coefficients-inference-and-cv-results).

<a id="cpu-and-gpu-examples"></a>
## Optional GPU execution

Continue after the complete [CPU walkthrough](#first-cpu-walkthrough).
Transfer that same training partition and three held-out profiles to the chosen
GPU; no second dataset or repeated CPU fit is needed. Choose the CuPy or Torch
example independently; you do not need both backends installed.

### CuPy / CUDA

Convert the training and prediction inputs to CuPy arrays:

<!-- example-requires: coxph-cpu-walkthrough -->
<!-- example: coxph-cupy-fit -->
```python
import cupy as cp

X_cp = cp.asarray(X_train)
time_cp = cp.asarray(time_train)
event_cp = cp.asarray(event_train)
X_test_cp = cp.asarray(X_test[:3])
```

Fit and predict the three held-out profiles; the result remains a CuPy array.

```python
cupy_model = CoxPH(
    ties="efron", device="cuda", compute_inference=False,
).fit(X_cp, time_cp, event_cp)
cupy_log_risk = cupy_model.predict_risk_score(X_test_cp)
```
<!-- example-end: coxph-cupy-fit -->

To select an L2 penalty, reuse those CuPy training arrays:

<!-- example-requires: coxph-cupy-fit -->
<!-- example: coxph-cupy-cv -->
```python
from statgpu.survival import CoxPHCV

cupy_cv = CoxPHCV(
    penalties=[0.0, 0.1, 1.0, 10.0], cv=3, random_state=42,
    ties="efron", device="cuda", compute_inference=False,
).fit(X_cp, time_cp, event_cp)
```
<!-- example-end: coxph-cupy-cv -->

### Torch / CUDA

Alternatively, start from the CPU walkthrough’s NumPy arrays and run this Torch version independently:

<!-- example-requires: coxph-cpu-walkthrough -->
<!-- example: coxph-torch-fit -->
```python
import torch

X_t = torch.as_tensor(X_train, dtype=torch.float64, device="cuda")
time_t = torch.as_tensor(time_train, dtype=torch.float64, device="cuda")
event_t = torch.as_tensor(event_train, dtype=torch.float64, device="cuda")
X_test_t = torch.as_tensor(X_test[:3], dtype=torch.float64, device="cuda")
```

Fit with `device="torch"`; prediction results remain CUDA tensors.

```python
torch_model = CoxPH(
    ties="efron", device="torch", compute_inference=False,
).fit(X_t, time_t, event_t)
torch_log_risk = torch_model.predict_risk_score(X_test_t)
```
<!-- example-end: coxph-torch-fit -->

Torch cross-validation likewise reuses the training arrays above:

<!-- example-requires: coxph-torch-fit -->
<!-- example: coxph-torch-cv -->
```python
from statgpu.survival import CoxPHCV

torch_cv = CoxPHCV(
    penalties=[0.0, 0.1, 1.0, 10.0], cv=3, random_state=42,
    ties="efron", device="torch", compute_inference=False,
).fit(X_t, time_t, event_t)
```
<!-- example-end: coxph-torch-cv -->

Explicit CUDA requests raise an error when the package, CUDA runtime, or device
is unavailable; they never silently run on CPU. These examples disable inference.
Set `compute_inference=True` when covariance, tests, or survival curves are needed.

## Objective Function and Estimating Equation

For row `i`, start time `a_i`, stop time `b_i`, event indicator `delta_i`, and
stratum `s_i`, the risk set for an event at `t` is

$$
R_s(t)=\{i : a_i < t \le b_i,\ s_i=s\}.
$$

Without tied failures, the stratified Cox partial log likelihood is

$$
\ell(\beta)=\sum_s\sum_{i:\delta_i=1,\ s_i=s}
\left[x_i^\top\beta-
\log\left\{\sum_{j\in R_s(b_i)}\exp(x_j^\top\beta)\right\}\right].
$$

Breslow, Efron, and Exact ties replace the tied-event denominator according to
their respective definitions but retain the same `(start, stop]` risk sets.
With `penalty=lambda`, StatGPU maximizes the total, summed objective

$$
Q_\lambda(\beta)=\ell(\beta)-\lambda\lVert\beta\rVert_2^2.
$$

Writing `U(beta)` for the unpenalized partial-likelihood score, the fitted
coefficient solves

$$
U_\lambda(\beta)=U(\beta)-2\lambda\beta=0.
$$

If $J(\beta)=-\partial U(\beta)/\partial\beta$ is the unpenalized observed
information, the derivative used by penalized Newton steps is
$A(\beta)=J(\beta)+2\lambda I_p$.

## Risk Sets and Tie Methods

`ties="breslow"` and `ties="efron"` use their standard tied-event partial
likelihoods. `ties="exact"` evaluates the exact tied-event denominator with an
elementary-symmetric dynamic program. The same counting-process risk-set engine
is used for delayed entry, strata, Exact ties, L2-penalized fits, and GPU robust
inference, which keeps the `(start, stop]` convention consistent across backends.
`CoxPH` and `CoxPHCV` accept one-dimensional stratum, subject, and cluster
labels and encode them internally. Exact dynamic programming avoids enumerating
all event subsets, but its computation and memory needs still increase with data
size, tied-event group size, and feature count. Ordinary right-censored data can
reuse calculations across nested risk sets; delayed-entry data use a calculation
suited to their risk-set structure.

When a temporary workspace exceeds its configured limit, the selected backend
uses a lower-memory algorithm rather than silently switching to CPU.
`STATGPU_EXACT_NESTED_MAX_BYTES`, `STATGPU_EXACT_BATCH_MAX_BYTES`, and
`STATGPU_COX_GROUP_MAX_BYTES` each default to 512 MiB and limit their respective
workspaces, not the total memory used by a fit. See the
[developer reference](../../../dev/references/coxph-implementation-and-evidence.md#implementation-snapshot)
for configuration details, algorithms, and their applicability.

## Formula Interface

These are interface sketches, not standalone examples: provide a pandas
DataFrame `df` containing the named columns and install the optional pandas/Patsy
formula dependencies. Both survival response forms are accepted:

```python
CoxPH().fit(formula="Surv(time, event) ~ age + C(group)", data=df)
CoxPH().fit(
    formula="Surv(start, stop, event) ~ age + treatment",
    data=df,
    strata=df["clinic"],
    subject_id=df["patient_id"],
)
```

Formula row removal is applied consistently to `entry`/`start`, `cluster`,
`strata`, and `subject_id`. Supplying `entry=` or `start=` together with a
three-column `Surv(start, stop, event)` response raises an error.

## Optimization and Convergence

Newton iterations use line search and final-state KKT verification. A failed
line search does not update coefficients and cannot report convergence. Public
fitted-state fields include:

- `converged_`;
- `termination_reason_`;
- `optimization_stop_reason_`;
- `n_iter_`;
- `final_kkt_inf_`;
- `final_kkt_normalized_`.

The likelihood, gradient, Hessian, covariance, baseline hazard, and public
convergence state are evaluated from the final coefficient vector.
`termination_reason_` is the interpreted user-level category and is one of
`kkt_converged`, `line_search_failed`, or `stalled_with_large_kkt`.
`optimization_stop_reason_` preserves the raw solver exit, including
`max_iter`; warnings also report this raw reason. This makes it possible to
identify an iteration-limit exit; reaching the iteration limit does not itself
establish convergence.

## Penalty Scaling and Penalized Inference

`penalty` is the `lambda` in the total partial-likelihood objective above. It is
not divided by the sample size or event count, and CoxPH has no intercept to
penalize. The same numeric `penalty` does not guarantee the same effective
regularization strength at different data sizes. For comparisons across data
sets or sample sizes, tune `penalty` with `CoxPHCV` under the intended sampling scale;
when reproducing software that minimizes an average loss, explicitly convert
that package's penalty convention rather than assuming the numeric values are
identical.

For a positive L2 penalty, let `J` be the unpenalized observed Cox information
at the fitted coefficient and `A = J + 2 * penalty * I_p`. The fixed-penalty
frequentist plug-in covariance is

$$
\widehat{\operatorname{Var}}(\widehat\beta)=A^{-1}JA^{-1},
\qquad A=J+2\lambda I_p.
$$

rather than `A^-1`. The latter is a penalized curvature or Laplace-style
quantity and is not published as a frequentist sampling covariance. Robust
penalized fits use the same penalized bread and the unpenalized aggregated score
outer product as meat.

The resulting SE/z/p/CI and penalized Wald test are conditional on the supplied
penalty. They target the penalized estimating equation; they are not debiased
inference for the unpenalized coefficient and do not account for shrinkage bias
or for selecting the penalty by cross-validation. `CoxPHCV` copies this same
contract from its final refit and explicitly reports
`penalty_selection_adjusted_=False`. Following `PenalizedGLM` result naming,
`inference_method_` is the concise `"m_estimation"` for a positive-penalty fit;
bread, meat, covariance convention, target, and conditioning details remain
separately available in inference metadata.

Classical likelihood-ratio and score tests plus AIC/BIC are suppressed for a
penalized fit rather than reported as if the estimate were an unconstrained
maximum-likelihood estimate. This contract is separate from
`PenalizedCoxPHModel`, whose L1/elastic-net/SCAD/MCP interface remains
estimation-only.

## Covariance and Inference

The [CPU walkthrough](#first-cpu-walkthrough) already enabled inference.
Inspect its coefficient table without refitting. The standard errors, z tests,
and intervals use the default model-based covariance; their interpretation
and other covariance choices are explained below.

<!-- example-requires: coxph-cpu-walkthrough -->
<!-- example: coxph-summary -->
```python
model.summary()
```
<!-- example-end: coxph-summary -->

| `cov_type` | Meaning |
|---|---|
| `"nonrobust"` | Model-based covariance; inverse information when unpenalized, fixed-penalty sandwich otherwise |
| `"hc0"` | Score-sandwich covariance |
| `"hc1"` | Score-sandwich covariance with finite-unit correction |
| `"cluster"` | Cluster-robust covariance; pass `cluster=` to `fit` |

For an unpenalized fit, nonrobust covariance is the usual inverse observed
information. Positive-penalty covariance follows the dedicated contract above.

For Breslow and Efron ties, strict robust inference uses statgpu's internal exact
counting-process score residuals; it does not require statsmodels. Repeated rows
are summed by `subject_id` before forming HC0/HC1 meat, and cluster covariance is
summed by `cluster`.

Robust inference requires identifiable independent-unit variation. HC0 and
cluster covariance require at least two independent units after subject or
cluster aggregation. HC1 additionally requires `n_units > n_features`, because
its finite-unit multiplier is exactly `n_units / (n_units - n_features)`.
Violations raise `RuntimeError`; statgpu does not replace a non-positive degrees-
of-freedom denominator with an arbitrary finite value. Materially negative or
non-positive robust marginal variances also fail strict inference instead of
publishing zero standard errors and misleading significance statistics.

Positive marginal variances do not guarantee that the robust covariance is
valid over the complete parameter space. StatGPU classifies the symmetrized
covariance spectrum with a scale-aware tolerance. A positive-definite matrix
supports marginal and joint Wald inference. A positive-semidefinite but
rank-deficient matrix retains per-coefficient robust SE/z/p/CI while exposing
`wald_test_available_=False` and `wald_test_failure_reason_`; the summary prints
`Robust Wald test unavailable` rather than applying an unstable inverse or
printing a bare `nan`. A materially negative eigenvalue means the matrix is not
a valid covariance estimator, so strict inference raises `RuntimeError` and the
fit transaction clears public fitted state instead of publishing its diagonal.
Likelihood-ratio and score tests remain classical, model-based tests even when
coefficient and Wald inference use a robust covariance; the summary labels this
distinction explicitly.

`inference_mode="strict"` is the default. `inference_mode="approx"` remains
accepted for backward compatibility, but the unified public fit path treats it
as a compatibility-only alias and still computes the exact counting-process
score sandwich. Consequently successful public fits report
`inference_approximate_=False` and no approximation fallback reason.
Here, exact refers to computing the counting-process score residuals, not to
finite-sample exact tests. Coefficient z tests and confidence intervals still
use a large-sample normal approximation, including with `ties="exact"`.

Exact ties currently support model-based (`cov_type="nonrobust"`) inference only.
Requesting HC0, HC1, or cluster inference with `ties="exact"` raises
`NotImplementedError`. If `compute_inference=False`, a robust covariance label
is accepted but no covariance is computed.

Inference provenance is exposed through:

- `inference_method_`;
- `inference_backend_`;
- `inference_approximate_`;
- `inference_fallback_reason_`;
- `inference_target_`;
- `penalty_conditioning_`;
- `penalty_selection_adjusted_`;
- `wald_test_available_` and `wald_test_failure_reason_`;
- `full_host_transfer_performed_`.

## Device transfers and CV screening

GPU fitting still uses host memory for some preprocessing. Ordinary GPU
Breslow/Efron fits copy the complete sorted time and event vectors to the host
for event-group metadata, so `full_host_transfer_performed_=True` even when the
design matrix stays on the GPU. For `CoxPHCV`, this flag covers both CV and the
final refit. The [device and CV diagnostics reference](../reference/coxph-diagnostics.md)
explains the phase-specific transfer flags and how to distinguish this call's
device from the origin of reused selection results.

Requesting `STATGPU_COXPHCV_TWO_STAGE` or
`STATGPU_COXPHCV_SUCCESSIVE_HALVING` currently emits a `RuntimeWarning` and uses
one exhaustive full-precision pass over all candidates on NumPy, CuPy, and
Torch. These controls do not currently reduce the candidate set or enable
approximate screening. `cv_results_` reports
`staged_safety_strategy="single_pass_exhaustive"`; the
[screening-control guide](../guides/cox-cv-staged-safety.md) lists the requested
and effective modes and candidate masks.

## Support Matrix

| Capability | Breslow | Efron | Exact | NumPy | CuPy | Torch |
|---|---|---|---|---|---|---|
| ordinary right censoring | supported | supported | supported | supported | supported | supported |
| delayed entry / `(start, stop]` | supported | supported | supported | supported | supported | supported |
| independent `strata` | supported | supported | supported | supported | supported | supported |
| non-negative L2 `penalty` | supported | supported | supported | supported | supported | supported |
| nonrobust inference | supported | supported | supported | supported | supported | supported |
| HC0 / HC1 / cluster inference | supported | supported | not implemented | supported | supported | supported |
| backend-native prediction arrays | supported | supported | supported | NumPy | CuPy | Torch |

`predict_survival` requires fitted baseline hazards, so leave
`compute_inference=True` when survival curves are needed. Risk-score and hazard-
ratio prediction do not require a baseline.

## External Comparisons and Implementation Reference

Align the design matrix, tie method, penalty scale, covariance convention, and
convergence settings before comparing other software.
[External comparisons and GPU validation records](../../../dev/references/coxph-implementation-and-evidence.md#historical-external-validation)
identify the source and environment they tested. Historical results are not a
blanket guarantee for the current version or every dataset and device.

## FAQ and Common Failure Modes

| Symptom | Meaning and action |
|---|---|
| Explicit `device="cuda"` or `device="torch"` fails | The requested package, CUDA runtime, or device is unavailable. Install a compatible backend or use `device="cpu"`; StatGPU does not silently fall back. |
| `predict_survival()` says the baseline is unavailable | Refit with `compute_inference=True`; risk-score and hazard-ratio prediction do not need a baseline. |
| Stratified survival prediction rejects labels | An explicitly stratified fit always requires one known training stratum per prediction row with shape `(n_samples,)`, even when training used one stratum. |
| Stratified scoring rejects labels | A multi-stratum fit requires one known label per row. A single-stratum fit may omit labels; supplied labels must still have shape `(n_samples,)` and be known. |
| Survival is exactly one for a known stratum | That fitted stratum had no observed failures, so its cumulative baseline hazard is identically zero. This is a valid fitted state, not missing baseline data. |
| `HC1 covariance requires n_units > n_features` | Increase independent subjects/clusters, reduce the feature count, or use a covariance contract justified for the study design. |
| Robust covariance requires at least two units | A one-subject or one-cluster sandwich cannot estimate between-unit variation. Estimation-only fitting remains available with `compute_inference=False`. |
| Observed information is singular | Check collinearity, invariant columns, separation/saturation, and event support; reduce the design or use an explicitly justified L2 penalty. |
| Hazard-ratio prediction raises `FloatingPointError` | `exp(X @ coef_)` is outside finite float64 range. Inspect `predict_risk_score()`, rescale features, and check extrapolation. |
| `converged_` is false | Inspect `optimization_stop_reason_`, `final_kkt_inf_`, and `final_kkt_normalized_`; increasing `max_iter` alone does not repair a failed line search or ill-conditioned design. |
| Exact ties are slow or hit a workspace gate | Exact uses dynamic programming instead of enumerating all subsets, but computation and memory needs still increase with risk-set size, tied-event group size, and feature count. Assess the resource cost for your data; consider Breslow or Efron when scientifically appropriate. |
| `score()` returns `0.5` | This can reflect neutral/tied ranking, or no permissible concordance pairs; the latter also returns the documented neutral value `0.5`. Check whether the data contain evaluable pairs. |

## Limitations

- robust/cluster covariance for Exact ties is not implemented;
- Exact ties use dynamic programming; large risk sets or tied-event groups can still require substantial computation and memory, also depending on feature count.
- frailty/random-effect terms are not implemented;
- optional `torch.compile` acceleration requires compatible Triton-capable
  hardware and is not part of the portable correctness contract.

## References

- Cox, D. R. (1972). Regression models and life-tables. *JRSS B*, 34(2), 187–220.
- Breslow, N. (1974). Covariance analysis of censored survival data. *Biometrics*, 30(1), 89–99.
- Efron, B. (1977). The efficiency of Cox's likelihood function for censored data. *JASA*, 72(359), 557–565.
- Lin, D. Y., & Wei, L. J. (1989). The robust inference for the Cox proportional hazards model. *JASA*, 84(408), 1074–1078.
- R survival documentation: [`coxph`](https://stat.ethz.ch/R-manual/R-devel/library/survival/html/coxph.html).
