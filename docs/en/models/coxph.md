# CoxPH

> Language: English<br>
> Last updated: 2026-07-25<br>
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

`inference_mode="strict"` is the default. `inference_mode="approx"` is an
explicit opt-in to the legacy event-row Efron sandwich approximation when that
legacy path is used. Approximate inference is identified by the public
provenance fields and is never silently selected.

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
| `inference_mode` | `"strict"` | `"strict"` or explicit `"approx"` |
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

## Validation

The PR #80 review on 2026-07-25 passed the local NumPy quick gate for ordinary
heavy ties, delayed entry, Exact ties, stratified start-stop data, inference,
subject-grouped CV, and statsmodels comparisons where the models are comparable.
The result schema passed with no local gate failures. The exact reviewed source
was then validated through Paramiko in an isolated remote `myconda` environment
on a Tesla P100-SXM2-16GB. The first physical-GPU run exposed 11 actionable
backend/test and scikit-learn 1.2.2 compatibility failures; after review and
fixes, all 15 failed and adjacent nodes passed, followed by a complete result of
**379 passed, 2 expected skips, 0 failed**. Remote quick and full benchmark
artifacts both report `validation_tier="remote-full"`, `schema_status="ok"`,
and no gate failures; compatibility, inference, and subject-grouped CV pass on
NumPy, CuPy, and Torch. These current-source results, rather than earlier PR #79
results alone, validate the new PR #80 paths.

Relevant validation entry points:

- `dev/tests/test_survival_risk_sets.py`;
- `dev/tests/test_cox_phase1_completion.py`;
- `dev/tests/test_cox_cv.py`;
- `dev/benchmarks/benchmark_survival_completion.py`.

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
