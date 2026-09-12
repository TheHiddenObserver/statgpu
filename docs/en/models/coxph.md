# CoxPH

> Language: English<br>
> Last updated: 2026-08-04<br>
> This page: Model documentation<br>
> Switch: [Chinese](../../cn/models/coxph.md)

## Overview

`CoxPH` implements proportional-hazards regression with Breslow, Efron, or Exact
tie handling on NumPy, CuPy CUDA, and Torch CUDA. It supports ordinary
right-censored observations, delayed entry, counting-process `(start, stop]`
rows, independent strata, time-varying covariates, robust/cluster covariance,
and L2-penalty selection through `CoxPHCV`.

Important behavior:

- explicit `device="cuda"` and `device="torch"` requests never silently fall back to CPU;
- `entry=` and `start=` are aliases and are mutually exclusive;
- a row is in the risk set at time `t` exactly when `start < t <= stop` and its
  stratum matches the event stratum;
- `subject_id=` identifies repeated rows from one subject for concordance,
  sandwich aggregation, and subject-preserving CV folds;
- `compute_inference=False` performs estimation only and leaves inference and
  baseline-hazard fields unset.

## Why the survival target is structured

For ordinary right-censored Cox data, one observation is defined jointly by

$$
(T_i,\delta_i),
$$

where $T_i$ is the observed time and $\delta_i\in\{0,1\}$ distinguishes an event from censoring. Two observations with the same time but different event indicators make different contributions to the risk-set likelihood.

`PenalizedCoxPHModel` and `PenalizedGLM_CV(loss="cox_ph")` therefore keep an `(n,2)` `[time,event]` target when they use the generic estimator/CV surface. This is a **structured survival response**, not ordinary two-output regression. The survival-aware CV path needs both columns to construct training risk sets and evaluate held-out partial likelihood.

The canonical `CoxPH` API exposes `time`, `event`, `start/entry`, and `strata` separately because that is a more natural surface for the full survival model. The penalized GLM-family Cox path is intentionally narrower and currently represents standard right-censored outcomes with `[time,event]`.

## Why Cox has no intercept

The Cox PH model is

$$
h(t\mid x)=h_0(t)\exp(x^\top\beta).
$$

Adding an intercept $c$ gives

$$
h(t\mid x)=h_0(t)e^c\exp(x^\top\beta)
=\widetilde h_0(t)\exp(x^\top\beta),
$$

so the constant is completely confounded with the unknown baseline hazard. The same fact is visible directly in the partial likelihood:

$$
\frac{e^{x_i^\top\beta+c}}
{\sum_{j\in R_i}e^{x_j^\top\beta+c}}
=
\frac{e^{x_i^\top\beta}}
{\sum_{j\in R_i}e^{x_j^\top\beta}}.
$$

The intercept is therefore **not identifiable** from the Cox partial likelihood. `PenalizedCoxPHModel` requiring `fit_intercept=False` is part of the model contract, not a missing optimization feature. A future API-cleanup discussion could consider removing a permanently-false parameter, but “support Cox intercept fitting” would be statistically incorrect.

## Import

```python
from statgpu.survival import CoxPH, CoxPHCV
```

## CPU and GPU Examples

The three backends use the same statistical inputs and return prediction arrays
on the fitted backend. Run this deterministic data setup once:

```python
import numpy as np

from statgpu.survival import CoxPH, CoxPHCV

rng = np.random.default_rng(20260730)
n = 256
X = rng.normal(size=(n, 3))
log_risk = X @ np.array([0.45, -0.30, 0.20])
event_time = rng.exponential(scale=np.exp(-log_risk))
censor_time = rng.exponential(scale=1.8, size=n)
time = np.minimum(event_time, censor_time)
event = (event_time <= censor_time).astype(np.float64)
```

NumPy / CPU:

```python
cpu_model = CoxPH(
    ties="efron",
    device="cpu",
    compute_inference=False,
).fit(X, time, event)
cpu_log_risk = cpu_model.predict_risk_score(X[:3])
```

CuPy / CUDA:

```python
import cupy as cp

X_cp = cp.asarray(X)
time_cp = cp.asarray(time)
event_cp = cp.asarray(event)
cupy_model = CoxPH(
    ties="efron",
    device="cuda",
    compute_inference=False,
).fit(X_cp, time_cp, event_cp)
cupy_log_risk = cupy_model.predict_risk_score(X_cp[:3])
```

Torch / CUDA:

```python
import torch

X_t = torch.as_tensor(X, dtype=torch.float64, device="cuda")
time_t = torch.as_tensor(time, dtype=torch.float64, device="cuda")
event_t = torch.as_tensor(event, dtype=torch.float64, device="cuda")
torch_model = CoxPH(
    ties="efron",
    device="torch",
    compute_inference=False,
).fit(X_t, time_t, event_t)
torch_log_risk = torch_model.predict_risk_score(X_t[:3])
```

When the requested package, CUDA runtime, or device is unavailable, explicit CUDA requests raise instead of silently moving to CPU. Set `compute_inference=True` when covariance, tests, or survival curves are required.

## Objective and estimating equations

For row `i` with start time `a_i`, stop time `b_i`, event indicator `delta_i`, and stratum `s_i`, the risk set at time `t` is

$$
R_s(t)=\{i : a_i < t \le b_i,\ s_i=s\}.
$$

With no tied failures, the stratified Cox partial log-likelihood is

$$
\ell(\beta)=\sum_s\sum_{i:\delta_i=1,\ s_i=s}
\left[x_i^\top\beta-
\log\left\{\sum_{j\in R_s(b_i)}\exp(x_j^\top\beta)\right\}\right].
$$

Breslow, Efron, and Exact replace the tied-event denominator according to their definitions while using the same `(start, stop]` risk set. With `penalty=lambda`, statgpu maximizes the sum-scale objective

$$
Q_\lambda(\beta)=\ell(\beta)-\lambda\lVert\beta\rVert_2^2.
$$

If `U(beta)` is the unpenalized partial-likelihood score, the fitted coefficient satisfies

$$
U_\lambda(\beta)=U(\beta)-2\lambda\beta=0.
$$

If $J(\beta)=-\partial U(\beta)/\partial\beta$ is the unpenalized observed information, penalized Newton uses $A(\beta)=J(\beta)+2\lambda I_p$.

## Risk sets and tie methods

`ties="breslow"` and `ties="efron"` use the corresponding tied-event partial likelihood; `ties="exact"` computes the Exact denominator with elementary-symmetric dynamic programming. Delayed entry, strata, Exact ties, L2 penalized fitting, and GPU robust inference share the same counting-process risk-set engine, so all three backends follow the same `(start, stop]` convention.

For ordinary right-censored Exact fitting, risk sets are nested within each stratum. statgpu sorts by stratum and descending stop time and reuses one segmented elementary-symmetric prefix dynamic program across all failure groups on NumPy, CuPy, and Torch. This avoids Python loops over strata and repeated risk-set scans. Failure numerators use backend-native grouped reductions instead of a dense `failure-groups × samples` mask. The prefix workspace is capped at 512 MiB by `STATGPU_EXACT_NESTED_MAX_BYTES` before allocation.

On Torch CUDA, PyTorch 2.0 multi-dimensional `cumsum(dim=0)` over long axes can dominate this linear prefix DP. When rows are at least 2,048 and tail-moment channels are at most 64, statgpu can lay out channels contiguously, run efficient one-dimensional CUDA scans, then reconstruct the original shape on device. `STATGPU_TORCH_EXACT_SCAN_MIN_ROWS` and `STATGPU_TORCH_EXACT_SCAN_MAX_CHANNELS` control these conservative gates. CPU, small samples, and wide tensors keep native Torch scanning. `STATGPU_TORCH_EXACT_SCAN_STRATEGY` can be `auto`, `native`, or `channelwise`; `auto` enables channelwise scanning only on the measured Torch 2.0 + Pascal/P100 combination and uses native scanning elsewhere. Extra transpose/output workspace is included in the nested-workspace cap; if the base DP fits but the channelwise extras do not, the nested algorithm continues with native Torch scanning rather than falling back to the more expensive generic Exact path.

Delayed entry breaks nested-prefix eligibility. When there are at least 8 strata, GPU backends first handle eligible failure groups in one backend-native global batch; fewer GPU strata and NumPy use per-stratum batches to avoid computing empty cross-stratum masks. `STATGPU_EXACT_BATCH_MAX_BYTES` independently caps this workspace at 512 MiB. A global batch that exceeds the cap retries by stratum and then uses the bounded per-group path; score-residual construction and conservative numerical-range gates also retain their normalized implementations. These are algorithmic fallbacks, not CPU fallbacks.

For Breslow/Efron delayed-entry objectives, `STATGPU_COX_GROUP_MAX_BYTES` caps dense failure-group workspace at 512 MiB by default. If even one failure group exceeds the cap, the selected GPU backend switches to a numerically stable multi-pass row-streaming moment calculation so batch size 1 cannot allocate an unbounded `O(n)` mask.

The inference stage of a completed fit also builds the Breslow baseline hazard. For ordinary right-censored rows, statgpu sorts within stratum by descending stop time and obtains all risk denominators from one log-risk prefix: NumPy uses `logaddexp.accumulate`, Torch uses `logcumsumexp`, and CuPy uses a shifted exponential cumulative sum behind a conservative predictor-range gate. Extreme CuPy predictors and delayed-entry rows keep the numerically stable backend-native per-failure-group implementation. This removes the previous `failure-groups × samples` risk-mask scan from the common ordinary-right-censored path.

## Formula interface

Two survival responses are supported:

```python
CoxPH().fit(formula="Surv(time, event) ~ age + C(group)", data=df)
CoxPH().fit(
    formula="Surv(start, stop, event) ~ age + treatment",
    data=df,
    strata=df["clinic"],
    subject_id=df["patient_id"],
)
```

When Formula handling drops missing rows, `entry`/`start`, `cluster`, `strata`, and `subject_id` are aligned to the same retained rows. A three-column `Surv(start, stop, event)` already defines the start time and cannot be combined with `entry=` or `start=`.

## Optimization and convergence

Newton iterations use line search and perform a final KKT check at the final parameters. A failed line search does not update coefficients and cannot report convergence. Public fit state includes:

- `converged_`;
- `termination_reason_`;
- `optimization_stop_reason_`;
- `n_iter_`;
- `final_kkt_inf_`;
- `final_kkt_normalized_`.

Likelihood, gradient, Hessian, covariance, baseline hazard, and public convergence state are recomputed from the final coefficient vector. `termination_reason_` is the interpreted user-facing classification and is one of `kkt_converged`, `line_search_failed`, or `stalled_with_large_kkt`. `optimization_stop_reason_` preserves the raw solver stop condition (including `max_iter`) and warnings report that raw value as well, so budget exhaustion remains auditable without being misrepresented as a separate convergence certificate.

## Penalty scale and penalized inference

`penalty` is the `lambda` in the sum-scale partial-likelihood objective above; it is not divided by the number of rows or events. CoxPH has no intercept to penalize. Replicating every observation therefore doubles likelihood and score contributions without automatically doubling the user-supplied penalty, changing the effective regularization strength. Across datasets or sample sizes, tune with `CoxPHCV` on the target sampling scale. When reproducing external software that uses an average-loss convention, convert its penalty scale explicitly rather than assuming the same numeric value.

For a positive L2 penalty, if `J` is the unpenalized Cox observed information at the fitted coefficients and `A = J + 2 * penalty * I_p`, the fixed-penalty frequentist plug-in covariance is

```text
A^-1 J A^-1
```

not `A^-1`; the latter is closer to penalized curvature or a Laplace-style quantity and should not be published as a frequentist sampling covariance. Penalized robust inference likewise uses the penalized bread while the meat remains the unpenalized aggregated score outer product.

SE/z/p/CI and the penalized Wald test are therefore conditional on the supplied penalty and target the penalized estimating equation; they are not debiased inference for the unpenalized coefficient and do not correct shrinkage bias or CV penalty-selection uncertainty. `CoxPHCV` copies the same contract from its final refit and reports `penalty_selection_adjusted_=False`. To stay aligned with `PenalizedGLM` result naming, positive-penalty fits use the concise `"m_estimation"` value for `inference_method_`; the bread/meat/covariance convention, inference target, and conditioning remain separately recorded in inference metadata.

Penalized fitting disables classical likelihood-ratio and score tests plus AIC/BIC rather than reporting a penalized estimate as an unrestricted MLE. This contract is separate from `PenalizedCoxPHModel`; its L1/elastic-net/SCAD/MCP interface remains estimation-only.

## Covariance and inference

| `cov_type` | Meaning |
|---|---|
| `"nonrobust"` | Model covariance; information inverse when unpenalized, fixed-penalty sandwich when penalized |
| `"hc0"` | Score-sandwich covariance |
| `"hc1"` | HC1 correction applied to the independent-unit score meat |
| `"cluster"` | Sandwich after summing subject-level scores by `cluster` |

Model-based inference uses the backend-native information inverse and a Normal reference for coefficient statistics. HC0/HC1 and cluster covariance use counting-process score residuals to form the meat and do not depend on statsmodels. Repeated rows from one subject are aggregated by `subject_id` before forming the HC0/HC1 meat; cluster covariance aggregates by `cluster`.

Robust inference requires identifiable independent-unit variation. After subject or cluster aggregation, HC0 and cluster covariance require at least two independent units; HC1 additionally requires `n_units > n_features` because its finite-unit correction is strictly `n_units / (n_units - n_features)`. Violating these conditions raises `RuntimeError` rather than substituting an arbitrary finite denominator. A materially negative covariance diagonal or non-positive robust marginal variance also fails strict inference rather than publishing zero standard errors and misleading significance.

Positive marginal variances do not by themselves make a robust covariance valid in the full parameter space. statgpu classifies the spectrum of the symmetrized covariance using a scale-aware tolerance before any inversion. Positive-definite covariance supports both marginal inference and the joint Wald test. Rank-deficient PSD covariance still retains coefficient-wise robust SE/z/p/CI but sets `wald_test_available_=False`, records `wald_test_failure_reason_`, and shows `Robust Wald test unavailable` in the summary instead of using an unstable inverse or printing a bare `nan`. A materially negative eigenvalue means the matrix is not a valid covariance estimator; strict inference raises `RuntimeError` and clears the current fit state rather than publishing marginal inference just because the diagonal entries happen to be positive. Likelihood-ratio and score tests remain classical model-based tests even when coefficient-wise and Wald inference use robust covariance; the summary labels that distinction explicitly.

`inference_mode="strict"` is the default. For backward compatibility, `inference_mode="approx"` remains an accepted compatibility-only alias; the unified fit path still computes the exact counting-process score sandwich. Successful fits therefore report `inference_approximate_=False` and no approximation-fallback reason.

Exact ties currently support model covariance only (`cov_type="nonrobust"`). Requesting HC0, HC1, or cluster inference under `ties="exact"` raises `NotImplementedError`. With `compute_inference=False`, a robust `cov_type` label may be retained without computing covariance.

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

For `CoxPHCV`, `full_host_transfer_performed_` summarizes the whole fit, including fold construction and penalty selection orchestrated on the host. `cv_full_host_transfer_performed_` and `final_refit_full_host_transfer_performed_` separately mark whether at least one full device-resident training component crossed to the host during CV and final refit. This includes sorted targets and any retained entry/strata/subject vector even when the design matrix itself stays on GPU. `orchestration_device_` records the CV orchestration device. Ordinary GPU Breslow/Efron preprocessing sorts on the selected backend, then copies the full sorted time and event vectors to the host to build failure-group metadata, so `full_host_transfer_performed_=True` is reported.

When `STATGPU_COXPHCV_TWO_STAGE` or `STATGPU_COXPHCV_SUCCESSIVE_HALVING` is requested, NumPy, CuPy, and Torch currently disable experimental screening for correctness. CoxPHCV emits `RuntimeWarning` and evaluates all candidates in one ordinary exhaustive full-precision pass. Public diagnostics record `staged_safety_strategy="single_pass_exhaustive"`, requested/effective staged states, an all-true `full_precision_candidate_mask`, and an all-false `screened_out_candidate_mask`. Each effective fold is prepared exactly once in that pass; no staged retained cache is enabled and no fold is prepared again across stages. Preparation counts and target-transfer counts remain available in `cv_results_`. `selection_cache_hit`, `requested_fit_device`, `fold_backend_preparation_count_this_call`, and `candidate_target_host_transfer_count_this_call` are call-local; `selection_origin_device`, `candidate_preparation_origin_device`, and `scoring_device` preserve selection provenance; `effective_device` records the current request/final-refit device. One target preparation represents the complete time/event metadata preparation; vector-transfer counts record the two actual vector copies.

## Parameters

| Parameter | Default | Description |
|---|---:|---|
| `ties` | `"breslow"` | `"breslow"`, `"efron"`, or `"exact"` |
| `tol` | `1e-9` | Newton/KKT convergence threshold |
| `max_iter` | `100` | Maximum iterations |
| `device` | `"auto"` | `"cpu"`, `"cuda"`, `"torch"`, or `"auto"` |
| `compute_inference` | `True` | Compute covariance, tests, and baseline hazard |
| `compute_cindex` | `True` | Compute training concordance |
| `cov_type` | `"nonrobust"` | `"nonrobust"`, `"hc0"`, `"hc1"`, or `"cluster"` |
| `penalty` | `0.0` | Non-negative L2 penalty |
| `inference_mode` | `"strict"` | `"strict"` or compatibility alias `"approx"`; both run exact inference |
| `gpu_memory_cleanup` | `False` | Try to release CuPy/Torch caches |

## Support matrix

| Capability | Breslow | Efron | Exact | NumPy | CuPy | Torch |
|---|---|---|---|---|---|---|
| ordinary right-censoring | supported | supported | supported | supported | supported | supported |
| delayed entry / `(start, stop]` | supported | supported | supported | supported | supported | supported |
| independent `strata` | supported | supported | supported | supported | supported | supported |
| non-negative L2 `penalty` | supported | supported | supported | supported | supported | supported |
| nonrobust inference | supported | supported | supported | supported | supported | supported |
| HC0 / HC1 / cluster inference | supported | supported | not implemented | supported | supported | supported |
| backend-native prediction arrays | supported | supported | supported | NumPy | CuPy | Torch |

`predict_survival` needs a fitted baseline hazard, so keep `compute_inference=True` when survival curves are required. Risk-score and hazard-ratio prediction do not depend on the baseline.

## Cross-validation

`CoxPHCV` evaluates an L2 penalty grid under the same tie/start/entry/strata/backend semantics and refits `CoxPH` with the selected penalty. With `subject_id`, all rows from one subject stay in the same automatically generated fold; user-supplied `cv_splits` that leak one subject across train and validation are rejected. `inference_mode` and `compute_inference` are forwarded to the final refit.

```python
cpu_cv = CoxPHCV(
    penalties=[0.0, 0.01, 0.1],
    cv=5,
    device="cpu",
    compute_inference=False,
).fit(X, time, event)
```

The same penalty search accepts the CuPy or Torch CUDA arrays shown above:

```python
cupy_cv = CoxPHCV(
    penalties=[0.0, 0.01, 0.1], cv=5, device="cuda",
    compute_inference=False,
).fit(X_cp, time_cp, event_cp)

torch_cv = CoxPHCV(
    penalties=[0.0, 0.01, 0.1], cv=5, device="torch",
    compute_inference=False,
).fit(X_t, time_t, event_t)
```

### L1/L2/ElasticNet/SCAD/MCP family CV

The `CoxPHCV` path above remains the canonical L2 Cox selector and can run final-refit inference when configured. The public penalized model family uses the separate survival-aware `PenalizedGLM_CV` route:

```python
from statgpu.linear_model import PenalizedGLM_CV

survival_y = np.column_stack([time, event])
penalized_cv = PenalizedGLM_CV(
    loss="cox_ph",
    penalty="mcp",               # l1, l2, elasticnet, scad, or mcp
    alpha_grid=[0.1, 0.03, 0.01],
    cv=5,
    cv_strategy="strict",
    loss_kwargs={"ties": "efron"},
    device="cpu",                # also "cuda" / "torch"
).fit(X, survival_y)
```

That route always preserves the two-column `(time, event)` target, forbids an intercept, scores by the unpenalized held-out partial likelihood, and requires every evaluable fold to provide finite evidence. If no alpha satisfies the contract, fitting raises and publishes neither a selected alpha nor a fitted estimator. Final refit uses `PenalizedCoxPHModel(compute_inference=False)`; post-selection coefficient inference, `two_stage`, sample weights, and dictionary targets are unsupported. The no-penalty alias is not a tunable model for this CV route and is rejected in favor of direct model fitting.

Custom folds may be ordinary non-empty, disjoint train/validation splits, including forward `TimeSeriesSplit` or repeated holdout; they do not need to be complements and validation rows need not form a partition. Indices are validated before any candidate fit and must be one-dimensional exact integers within range. For automatic grids, ElasticNet with `l1_ratio > 0` uses the zero-model KKT boundary `alpha_max = ||gradient L(0)||_inf / l1_ratio`, including the ratio carried by a penalty object. Pure L2 (`l1_ratio=0`) has no finite all-zero KKT threshold, so the raw infinity norm of the zero-model score is used as the documented grid heuristic.

Large `device="auto"` searches select GPU only after the Torch or CuPy CUDA backend reports a genuinely usable device. General fallback sizing counts only evaluable folds whose train and validation targets both contain events; other normalized folds remain in `failure_path` but do not inflate GPU-work estimates. An importable CuPy runtime with no usable device falls back to CPU; explicit `device="cuda"` remains strict and raises instead of silently falling back.

## Prediction and scoring

For array inputs, `predict`, `predict_risk_score`, `predict_hazard_ratio`, `predict_survival`, and `score` run on the fitted backend. Once a `device="auto"` fit resolves to an `effective_device_`, changing global device state later does not migrate that fitted estimator. Stratified survival prediction requires one training-known stratum label for every prediction row; even a model fitted with a single explicit stratum cannot omit the labels, and missing/unknown labels raise `ValueError`. Survival curves accumulate the baseline in the log domain for numerical stability. Formula-fitted models apply the stored design transformation before prediction.

`score()` reuses the same row-label encoder: supplied strata must have shape `(n_samples,)`; an explicitly stratified model accepts only training-known labels, and multi-stratum fits require labels at scoring time. Scalar, two-dimensional, wrong-length, and unknown-label inputs raise `ValueError` before backend concordance evaluation.

A fitted stratum with no observed failures has a valid empty baseline-hazard state. Its cumulative baseline hazard is zero at every time, so `predict_survival()` returns exactly 1. Explicit times, automatic times, mixed-strata prediction rows, and the `CoxPHCV` delegation path all obey this contract; mismatched stored time/hazard shapes still raise as invalid state.

`predict_risk_score()` returns the unexponentiated log risk. Canonical, CV, and penalized Cox hazard-ratio prediction APIs use one strict float64 exponential representability boundary; canonical/CV fitted `hazard_ratios_` uses the same boundary. Values that would overflow to infinity or underflow to zero raise `CoxFitNumericalError` during canonical/CV fitting and `FloatingPointError` during prediction instead of being silently clipped to estimator-specific thresholds. `PenalizedCoxPHModel` also exposes `predict_risk_score()`, so extreme but finite log risks remain directly accessible.

## Outputs

- parameters: `coef_`, `hazard_ratios_`, `n_iter_`, `converged_`,
  `termination_reason_`, `optimization_stop_reason_`, `final_kkt_inf_`,
  `final_kkt_normalized_`;
- inference: `covariance_`, `standard_errors_`, `z_values_`, `p_values_`,
  `confidence_intervals_`, `inference_method_`, `inference_target_`,
  `penalty_conditioning_`, `penalty_selection_adjusted_`;
- joint tests: `wald_test_available_`, `wald_test_failure_reason_`, `wald_test_`,
  `likelihood_ratio_test_`, `score_test_`;
- model comparison: `log_likelihood_`, `aic_`, `bic_`;
- baseline: `baseline_hazard_`, `baseline_cumulative_hazard_`, `baseline_survival_`;
- performance/provenance: `effective_device_`, `inference_backend_`,
  `full_host_transfer_performed_`, `orchestration_device_`,
  `cv_full_host_transfer_performed_`, `final_refit_full_host_transfer_performed_`.

## External validation

CPU precision is aligned against R `survival` as the definition reference for ordinary right-censoring, tied events, delayed entry, start-stop, strata, robust, cluster, and subject-level cases. `lifelines` is used as an additional directional check but is not authoritative when its tie or penalty convention differs from R.

## GPU validation

Physical CUDA promotion is not replaced by hosted CI. Any change affecting Cox numerical paths requires CuPy and Torch CUDA validation at the exact clean source SHA; an artifact only certifies the source SHA it records.

## References

- Cox, D. R. (1972). Regression Models and Life-Tables. *JRSS B*, 34(2), 187-220.
- Therneau, T. M. & Grambsch, P. M. (2000). *Modeling Survival Data: Extending the Cox Model*. Springer.