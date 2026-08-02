# CoxPH

> Language: English<br>
> Last updated: 2026-08-02<br>
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

Explicit CUDA requests raise an error when that backend or a CUDA device is not
available; they never silently run the model on CPU. Set
`compute_inference=True` when covariance, tests, or survival curves are needed.

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
The public `CoxPH` and `CoxPHCV` estimators factorize one-dimensional labels:
host strings/objects and finite numeric CuPy/Torch labels are encoded internally
as consecutive int64 codes. Low-level counting-process primitives do not
factorize labels and therefore require finite integer-valued numeric codes
representable as signed int64.

For ordinary right-censored Exact fits, the risk sets are nested within each
stratum. StatGPU sorts rows by stratum and decreasing stop time, then reuses one
segmented elementary-symmetric prefix dynamic program across every failure group
on NumPy, CuPy, and Torch without a Python loop over strata.
This removes the repeated risk-set scan that made work grow with both sample
count and failure-group count. Failure numerators use backend-native grouped
reductions instead of a dense failure-group-by-sample mask. The prefix workspace
defaults to a 512 MiB ceiling
controlled by `STATGPU_EXACT_NESTED_MAX_BYTES` and is checked before allocation.

On Torch CUDA, long multidimensional `cumsum(dim=0)` calls in PyTorch 2.0 can
dominate this otherwise linear prefix DP. For at least 2,048 rows and at most 64
trailing moment channels, StatGPU therefore lays out each channel contiguously,
runs the efficient one-dimensional CUDA scan per channel, and stacks the results
back on device. `STATGPU_TORCH_EXACT_SCAN_MIN_ROWS` and
`STATGPU_TORCH_EXACT_SCAN_MAX_CHANNELS` control these conservative gates, and
`STATGPU_TORCH_EXACT_SCAN_STRATEGY` accepts `auto`, `native`, or `channelwise`.
`auto` enables the split scan only for the benchmarked Torch 2.0 + Pascal/P100
combination; unbenchmarked Torch/GPU combinations use the native scan. CPU,
small, or wide inputs also keep Torch's native multidimensional scan. The additional
transpose/output workspace is included in the existing nested-workspace check:
if the base DP fits but the channel-scan workspace does not, the nested
algorithm stays active and uses the native Torch scan rather than falling back
to the more expensive general Exact path.

Delayed entry prevents the nested-prefix shortcut. With at least eight strata,
GPU backends first try all eligible failure groups in one backend-native batch;
smaller GPU cases and NumPy use per-stratum batches to avoid empty cross-stratum
mask work. The separate 512 MiB ceiling is controlled by
`STATGPU_EXACT_BATCH_MAX_BYTES`. An oversized global batch is retried per
stratum before the memory-bounded per-group path; score-residual requests and
conservative numerical-range gates also retain the normalized implementation.
These are explicit algorithmic fallbacks, never implicit CPU fallbacks.

For Breslow/Efron delayed-entry objectives,
`STATGPU_COX_GROUP_MAX_BYTES` controls the dense failure-group workspace
(512 MiB by default). If even one failure group exceeds the ceiling,
the selected GPU backend uses a stable multi-pass row-streaming moment
calculation. This keeps an extreme single stratum/risk set bounded instead of
letting the minimum batch size allocate an unbounded `O(n)` mask.

Full-fit inference also constructs a Breslow baseline hazard. For ordinary
right-censored rows, StatGPU now sorts each stratum by decreasing stop time and
computes every risk denominator from one log-risk prefix. NumPy uses
`logaddexp.accumulate`, Torch uses `logcumsumexp`, and CuPy uses a shifted
exponential cumulative sum within a conservative predictor-range gate. Extreme
CuPy predictors and delayed-entry rows retain the stable backend-native
per-failure-group calculation. This removes the former
failure-group-by-sample risk-mask scan from the common right-censored path.

## Formula Interface

Both survival response forms are accepted:

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
`max_iter`; warnings also report this raw reason. Thus budget exhaustion remains
auditable without presenting it as a separate convergence certificate.

## Penalty Scaling and Penalized Inference

`penalty` is the `lambda` in the total partial-likelihood objective above. It is
not divided by the sample size or event count, and CoxPH has no intercept to
penalize. Consequently, duplicating the observations doubles the likelihood
and score contributions without doubling the supplied penalty, so it changes
the effective regularization strength. For comparisons across data sets or
sample sizes, tune `penalty` with `CoxPHCV` under the intended sampling scale;
when reproducing software that minimizes an average loss, explicitly convert
that package's penalty convention rather than assuming the numeric values are
identical.

For a positive L2 penalty, let `J` be the unpenalized observed Cox information
at the fitted coefficient and `A = J + 2 * penalty * I_p`. The fixed-penalty
frequentist plug-in covariance is

```text
A^-1 J A^-1
```

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

For `CoxPHCV`, `full_host_transfer_performed_` describes the complete fit,
including host-orchestrated fold construction and selection. The more specific
`cv_full_host_transfer_performed_` and
`final_refit_full_host_transfer_performed_` attributes identify which phase
moved at least one complete device-resident training component to the host;
this includes sorted targets and retained entry, strata, or subject vectors,
even when the design matrix remains on the GPU. `orchestration_device_` records
where CV orchestration ran. Ordinary GPU Breslow/Efron preprocessing sorts on
the selected backend, then copies the complete sorted time and event vectors to
the host to build failure-group metadata, so it reports
`full_host_transfer_performed_=True`. Ordinary `CoxPHCV` prepares that metadata
once per fold and reuses it across every staged penalty pass in the complete
selector invocation when the estimated retained workspace fits
`STATGPU_COXPHCV_FOLD_CACHE_MAX_BYTES` (512 MiB by default). Above that gate,
stages repeat fold preparation so retained GPU memory stays bounded.
`fold_state_cache_enabled` and the estimate/limit fields make that routing
auditable. Preparation and target-transfer counts are exposed in `cv_results_`.
The invocation fields
`selection_cache_hit`, `requested_fit_device`,
`fold_backend_preparation_count_this_call`, and
`candidate_target_host_transfer_count_this_call` remain separate from the
selection-origin fields such as `selection_origin_device`,
`candidate_preparation_origin_device`, and `scoring_device`.
`effective_device` records the current requested/final-refit device. A target
preparation count represents one complete time/event metadata preparation;
the vector-transfer count records its two actual vector copies.

## Parameters

| Parameter | Default | Description |
|---|---:|---|
| `ties` | `"breslow"` | `"breslow"`, `"efron"`, or `"exact"` |
| `tol` | `1e-9` | Newton/KKT convergence tolerance |
| `max_iter` | `100` | Maximum iterations |
| `device` | `"auto"` | `"cpu"`, `"cuda"`, `"torch"`, or `"auto"` |
| `compute_inference` | `True` | Compute covariance, tests, and baseline hazards |
| `compute_cindex` | `True` | Compute training concordance |
| `cov_type` | `"nonrobust"` | `"nonrobust"`, `"hc0"`, `"hc1"`, or `"cluster"` |
| `penalty` | `0.0` | Non-negative L2 penalty |
| `inference_mode` | `"strict"` | `"strict"` or compatibility alias `"approx"`; both are exact |
| `gpu_memory_cleanup` | `False` | Best-effort CuPy/Torch cache cleanup |

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

## Cross-Validation

`CoxPHCV` evaluates an L2 penalty grid using the same tie method, start/entry,
strata, and backend semantics, then refits a `CoxPH` estimator at the selected
penalty. When `subject_id` is supplied, every row from a subject remains wholly
inside one automatically generated fold. User-provided `cv_splits` are rejected
if they leak a subject between train and validation. `inference_mode` and
`compute_inference` are forwarded to the final refit.

```python
cpu_cv = CoxPHCV(
    penalties=[0.0, 0.01, 0.1],
    cv=5,
    device="cpu",
    compute_inference=False,
).fit(X, time, event)
```

The same penalty search runs on CuPy or Torch CUDA arrays prepared above:

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

### L1/L2/ElasticNet/SCAD/MCP model-family CV

`CoxPHCV` above is the canonical L2 Cox selector and may run the configured
final-refit inference. The public penalized-model family uses the separate
survival-aware branch of `PenalizedGLM_CV`:

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
    device="cpu",                # or "cuda" / "torch"
).fit(X, survival_y)
```

This branch keeps the `(time, event)` target two-dimensional, forbids an
intercept, evaluates unpenalized held-out partial likelihood, and requires
finite evidence from every evaluable fold. If no alpha satisfies that contract,
fit raises and publishes no selected alpha or fitted estimator. The final refit
is `PenalizedCoxPHModel(compute_inference=False)`; post-selection coefficient
inference, `two_stage`, sample weights, and dictionary targets are unsupported.

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

## External Validation and Reproducibility

The maintained R baseline uses R 4.4.1 with `survival` 3.8.9 and aligns ties,
Newton `max_iter=80`, and `tol=1e-8`. At `n=3000`, `p=10`, the Breslow and
Efron comparisons use 3,000 independent HC1 units and 120 cluster units. The
maximum StatGPU-versus-R coefficient/SE/p-value differences were
`5.55e-16`/`1.39e-16`/`8.00e-19` for HC1 and
`5.55e-16`/`1.32e-16`/`2.22e-16` for cluster covariance. Unsupported
statsmodels covariance modes are recorded as unsupported rather than relabeled
as external evidence.

Machine-readable R comparison artifacts:

- `results/benchmark_frontend_sources/coxph_robust_inference_breslow_pr80_20260729_schema11.json`;
- `results/benchmark_frontend_sources/coxph_robust_inference_efron_pr80_20260729_schema11.json`.

These are fixed-source, shape-specific comparisons, not a universal accuracy or
performance guarantee. Exact-ties and performance conclusions remain bound to
their dedicated artifacts listed in `dev/reviews/pr80_review_fix.md`.

### Exact-Source Physical-GPU Evidence

Physical-GPU evidence is pinned to an exact source commit so that later code or
documentation changes cannot silently inherit a broader validation claim.

| Field | Current audited evidence |
|---|---|
| Source commit | `d688f760d8a0678c3c52c657a50178dad1b5ab3d` |
| Artifact | `results/benchmark_frontend_sources/coxph_completion_contract_pr80_20260802_schema16.json` |
| Artifact SHA-256 | `f0b47df704d2a0895cd1d66019c8676ff8a525d0f85e827d90ba816ad02b4837` |
| Schema / tier | `16` / `remote-full` |
| Hardware | Tesla P100-SXM2-16GB |
| Software | Python 3.9.16, NumPy 1.24.2, CuPy 13.6.0, Torch 2.0.0+cu117 |
| Structured GPU cases | CuPy 14/14; Torch 14/14 |
| Targeted tests | 516 passed, 7 expected warnings |
| Source audit | `source_clean=true`; 43/43 recorded Git-blob hashes matched |
| Gate failures | `[]` |

The schema-16 scope retains the schema-15 prediction/scoring, CV preparation,
prepared-state, packed-target, numerical-boundary, workspace, concordance,
robust-inference, fixed-penalty inference, shared strata-scoring, eventless-
stratum, and canonical-validator gates. It additionally exercises the
survival-aware penalized-Cox CV path for L1, L2, ElasticNet, SCAD, and MCP on
both GPU backends, including complete finite fold evidence, selected-alpha and
direct final-refit coefficient parity, no-intercept behavior, and fitted-backend
pinning after the global device changes. The targeted matrix also runs the
sklearn <=1.2 `CompositePenalty` constructor-identity regression.

This is not a new performance-crossover benchmark or a new R external-alignment
run; those claims remain tied to their dedicated artifacts and detailed history
in `dev/reviews/pr80_review_fix.md`. Runtime or maintained-test changes after
the source commit above require their own exact-source refresh before they can
claim the same physical-GPU evidence.

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
| Exact ties are slow or hit a workspace gate | Exact likelihood is combinatorial in tied-event group size. Prefer Breslow/Efron when scientifically acceptable, or reduce the largest exact tie block. |
| `score()` returns `0.5` | There were no permissible concordance pairs; `0.5` is the documented neutral value. |

## Limitations

- robust/cluster covariance for Exact ties is not implemented;
- Exact ties use combinatorial dynamic programming and are intended for modest
  tied-event groups rather than unrestricted large tie blocks;
- frailty/random-effect terms are not implemented;
- optional `torch.compile` acceleration requires compatible Triton-capable
  hardware and is not part of the portable correctness contract.

## References

- Cox, D. R. (1972). Regression models and life-tables. *JRSS B*, 34(2), 187–220.
- Breslow, N. (1974). Covariance analysis of censored survival data. *Biometrics*, 30(1), 89–99.
- Efron, B. (1977). The efficiency of Cox's likelihood function for censored data. *JASA*, 72(359), 557–565.
- Lin, D. Y., & Wei, L. J. (1989). The robust inference for the Cox proportional hazards model. *JASA*, 84(408), 1074–1078.
- R survival documentation: [`coxph`](https://stat.ethz.ch/R-manual/R-devel/library/survival/html/coxph.html).
