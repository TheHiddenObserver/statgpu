# CoxPH

> Language: English<br>
> Last updated: 2026-07-28<br>
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

## Risk Sets and Tie Methods

For row `i`, start time `a_i`, stop time `b_i`, event indicator `delta_i`, and
stratum `s_i`, the risk set for an event at `t` is

$$
R_s(t)=\{i : a_i < t \le b_i,\ s_i=s\}.
$$

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

With `penalty > 0`, the optimized objective is the partial log likelihood minus
`penalty * ||beta||^2`. Classical likelihood-ratio statistics and information
criteria are therefore not reported as if the penalized estimate were an
unconstrained maximum-likelihood estimate.

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
- `n_iter_`;
- `final_kkt_inf_`;
- `final_kkt_normalized_`.

The likelihood, gradient, Hessian, covariance, baseline hazard, and public
convergence state are evaluated from the final coefficient vector.

## Covariance and Inference

| `cov_type` | Meaning |
|---|---|
| `"nonrobust"` | Model-based covariance from observed information |
| `"hc0"` | Score-sandwich covariance |
| `"hc1"` | Score-sandwich covariance with finite-unit correction |
| `"cluster"` | Cluster-robust covariance; pass `cluster=` to `fit` |

For Breslow and Efron ties, strict robust inference uses statgpu's internal exact
counting-process score residuals; it does not require statsmodels. Repeated rows
are summed by `subject_id` before forming HC0/HC1 meat, and cluster covariance is
summed by `cluster`.

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
- `full_host_transfer_performed_`.

For `CoxPHCV`, `full_host_transfer_performed_` describes the complete fit,
including host-orchestrated fold construction and selection. The more specific
`cv_full_host_transfer_performed_` and
`final_refit_full_host_transfer_performed_` attributes identify which phase
moved a full device-resident input to the host; `orchestration_device_` records
where CV orchestration ran.

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
cv_model = CoxPHCV(
    penalties=[0.0, 0.01, 0.1],
    cv=5,
    ties="efron",
    device="cpu",
).fit(
    X_rows,
    stop,
    event,
    start=start,
    strata=clinic,
    subject_id=patient_id,
)
```

## Prediction and Scoring

`predict`, `predict_risk_score`, `predict_hazard_ratio`, `predict_survival`, and
`score` execute on the fitted backend for array inputs. Stratified survival
prediction requires one known stratum label per prediction row. Survival curves
use log-domain baseline accumulation for numerical stability. Formula-fitted
models apply their saved design transformation before prediction.

## Outputs

- parameters: `coef_`, `hazard_ratios_`;
- inference: `_bse`, `_zvalues`, `_pvalues`, `_conf_int` when enabled;
- diagnostics: `log_likelihood`, `aic`, `bic`, `concordance_index` where defined;
- convergence: `converged_`, `termination_reason_`, `n_iter_`,
  `final_kkt_inf_`, `final_kkt_normalized_`;
- provenance: `inference_method_`, `inference_backend_`,
  `inference_approximate_`, `inference_fallback_reason_`,
  `full_host_transfer_performed_`.

`CoxPHCV` additionally exposes `cv_full_host_transfer_performed_`,
`final_refit_full_host_transfer_performed_`, and `orchestration_device_` so
data-movement audits do not confuse host CV selection with the final refit.
If a finite-input candidate returns non-finite fitted coefficients or
likelihood, `CoxPH` raises `CoxCandidateNumericalError` (a
`FloatingPointError` subclass); `CoxPHCV` excludes only that candidate while
letting input, allocator, CUDA, and unexpected runtime errors propagate.

## Validation

The PR #80 review through 2026-07-26 passed the local NumPy quick gate for ordinary
heavy ties, delayed entry, Exact ties, stratified start-stop data, inference,
subject-grouped CV, and statsmodels comparisons where the models are comparable.
The result schema passed with no local gate failures. The exact reviewed source
was then validated through Paramiko in an isolated remote `myconda` environment
on a Tesla P100-SXM2-16GB. The first physical-GPU run exposed 11 actionable
backend/test and scikit-learn 1.2.2 compatibility failures; after review and
fixes, all 15 failed and adjacent nodes passed. The final nested-Exact source
passed the complete 13-file physical-GPU matrix with **392 passed, 2 expected
skips, 0 failed**. Remote quick and full artifacts report
`validation_tier="remote-full"`, `schema_status="ok"`, and no gate failures.
The new regression coverage compares the nested-prefix path with the forced
normalized fallback on NumPy and Torch; the physical-GPU target also exercises
CuPy, the delayed-entry batched fallback, baseline parity on both GPU backends,
the extreme-predictor CuPy stability fallback, Torch channel-scan/native-scan
parity, and the channel-scan memory gate. The local 13-file matrix passed with
**297 passed, 97 skipped, 0 failed**; the skips are optional GPU/R availability
branches.

External Exact alignment then compared the same source with R 4.4.1
`survival::coxph(ties="exact")` from survival 3.8.9. Across the bounded scaling
cases and separate right-censored, delayed-entry, strata, and combined
start-stop/strata cases, every NumPy/CuPy/Torch fit converged and the artifact
reported zero gate failures. The maximum differences from R were `1.30e-09`
for a coefficient, `5.12e-09` for exact partial log likelihood, and `5.01e-12`
for model-based covariance.

Performance remains shape- and backend-dependent. On the Tesla P100 bounded-tie
right-censored workload (`p=4`, maximum tie size 8), median R/NumPy/CuPy/Torch
full-fit times were 0.0460/0.0354/0.0838/0.0571 s at `n=1,920`; the GPU paths
remain launch-bound at that small size. At `n=15,360`, the corresponding
medians were 0.295/0.273/0.0949/0.0558 s; at `n=61,440`,
1.323/1.465/0.1114/0.0662 s; and at `n=122,880`,
2.691/3.043/0.1430/0.1000 s. The Torch channel scans are 30.32x faster than
the prior native multidimensional-scan Torch result at the largest size. Torch
is 26.92x faster than R, 30.44x faster than NumPy, and 1.43x faster than CuPy
there; both GPU paths overtake R by the measured `n=15,360` point.

For three-stratum Exact fits, the optimized objective is now composed from one
bounded fast-path evaluation per stratum instead of a device/Python loop per
failure time. On the same P100 timing contract, R/NumPy/CuPy/Torch medians were
0.0180/0.0143/0.1742/0.0747 s at `n=160`,
0.258/0.2263/0.2181/0.1341 s at `n=15,360`, and
1.118/0.9874/0.2285/0.1384 s at `n=61,440`. Explicit GPU fits remain
launch-bound at the smallest size, overtake R by the measured `n=15,360`
point, and reach 4.89x CuPy and 8.08x Torch speedups over R at `n=61,440`.
See `results/benchmark_frontend_sources/coxph_exact_strata_pr80_20260726.json`
for source hashes, device metadata, convergence, and R-alignment errors.

Phase profiling at `n=61,440` identified baseline construction as the remaining
full-fit hotspot. Before the prefix change, NumPy/CuPy/Torch baseline phases
took 6.847/5.988/3.328 s; the same phases now take
0.0202/0.00701/0.00265 s while preserving R and cross-backend precision. In the
separate `n=160` delayed-entry case, which intentionally retains the normalized
fallback, R took 57.031 s while NumPy/CuPy/Torch took
0.182/0.594/0.345 s. These timings establish the measured shapes, not a
universal crossover. StatGPU timing includes input conversion and inference; R
timing covers the `coxph` call including inference but excludes process startup,
package loading, and CSV parsing.

Relevant validation entry points:

- `dev/tests/test_survival_risk_sets.py`;
- `dev/tests/test_cox_phase1_completion.py`;
- `dev/tests/test_cox_cv.py`;
- `dev/benchmarks/benchmark_survival_completion.py`;
- `dev/benchmarks/benchmark_exact_ties_scaling.py` (writes
  `results/exact_ties_scaling.json`).

## Limitations

- robust/cluster covariance for Exact ties is not implemented;
- Exact ties use combinatorial dynamic programming and are intended for modest
  tied-event groups rather than unrestricted large tie blocks;
- frailty/random-effect terms are not implemented;
- optional `torch.compile` acceleration requires compatible Triton-capable
  hardware and is not part of the portable correctness contract.

## References

- Cox, D. R. (1972). Regression models and life-tables. *JRSS B*, 34(2), 187?220.
- Breslow, N. (1974). Covariance analysis of censored survival data. *Biometrics*, 30(1), 89?99.
- Efron, B. (1977). The efficiency of Cox's likelihood function for censored data. *JASA*, 72(359), 557?565.
- Lin, D. Y., & Wei, L. J. (1989). The robust inference for the Cox proportional hazards model. *JASA*, 84(408), 1074?1078.
- R survival documentation: [`coxph`](https://stat.ethz.ch/R-manual/R-devel/library/survival/html/coxph.html).
