# CoxPH

> Language: English
>
> Last updated: 2026-07-12
>
> This page: Model documentation
>
> Switch: [Chinese](../../cn/models/coxph.md)

## Overview

`CoxPH` fits Cox proportional-hazards models with native NumPy, CuPy, and
PyTorch implementations. It supports:

- `ties="breslow"`, `ties="efron"`, and exact tied partial likelihood with
  `ties="exact"`;
- right-censored, delayed-entry, and counting-process `(start, stop]` data;
- stratified risk sets and repeated rows identified by `subject_id`;
- model-based, HC0, HC1, and cluster-robust covariance; and
- coefficient, hazard-ratio, concordance, baseline-hazard, and survival
  prediction outputs.

Explicit `device="cuda"` and `device="torch"` requests never silently fall
back to NumPy. `device="auto"` is the only mode that selects an available
backend automatically.

Related estimators:

- `CoxPHCV` selects a scalar L2 penalty on a cross-validation grid and refits
  the final `CoxPH`. It accepts `start`, `strata`, `subject_id`, and all three
  tie methods.
- `PenalizedCoxPHModel` provides estimation-only L1, L2, ElasticNet, SCAD, and
  MCP fits. It has no intercept; SCAD/MCP use the FISTA-LLA path.

## Paths

```text
statgpu.survival.CoxPH
statgpu.survival.CoxPHCV
statgpu.linear_model.PenalizedCoxPHModel
```

## Objective Function

For row $i$, let $s_i$ be its start time, $t_i$ its stop time, and
$\delta_i$ its event indicator. Within stratum $g_i$, the risk set at an
event time $t$ is

$$
R_g(t)=\{j:g_j=g,\ s_j<t\leq t_j\}.
$$

`CoxPH` maximizes the partial log-likelihood. Breslow and Efron use their
respective tied-event denominators. `ties="exact"` sums over the relevant
tied-event subsets with an elementary-symmetric dynamic program; it is an
exact tied partial likelihood, not an approximation.

The coefficient estimate is obtained with Newton iterations. A scalar
`penalty` adds L2 regularization; `CoxPHCV` searches this same L2 parameter.

## Data Interfaces

The array interface is:

```python
model.fit(
    X, stop, event,
    entry=None,       # delayed entry
    cluster=None,     # covariance clusters
    start=None,       # counting-process alias for entry
    strata=None,      # independent risk sets/baselines
    subject_id=None,  # repeated-row subject identity
)
```

Pass only one of `entry` and `start`; each row must satisfy
`0 <= start < stop`. `subject_id` controls within-subject concordance pairs,
subject-preserving automatic CV folds, and the default independent unit for
HC0/HC1 covariance when a subject contributes repeated rows. It does not
replace `cluster`, which explicitly defines the units for cluster-robust
covariance.

The formula interface accepts both right-censored and start-stop responses:

```python
model.fit(formula="Surv(time, event) ~ age + treatment", data=df)
model.fit(formula="Surv(start, stop, event) ~ age + treatment", data=df_long)
```

Patsy removes rows with missing values in either the survival response or the
design terms during fitting, and auxiliary row-level arrays are aligned to the
retained rows. Prediction and scoring never silently apply this row removal:
missing formula covariates raise `ValueError`, preserving one output per input
row.

## Covariance and Inference

| `cov_type` | Meaning | Extra fit input |
|---|---|---|
| `"nonrobust"` | Inverse observed information | none |
| `"hc0"` | Score-residual sandwich covariance | none |
| `"hc1"` | HC1 finite-sample adjustment to HC0 | none |
| `"cluster"` | Cluster-summed score sandwich | `cluster=` |

`compute_inference=True` computes standard errors, z statistics, p-values,
confidence intervals, likelihood diagnostics, and baseline hazards.
`compute_inference=False` skips inference and baseline estimation; consequently,
`predict_survival()` is unavailable until the model is refit with inference
enabled.

Robust covariance is intentionally unsupported for `ties="exact"`. When
`compute_inference=True`, Exact-tie fits must use `cov_type="nonrobust"`;
requesting HC0, HC1, or cluster covariance raises `NotImplementedError`. With
`compute_inference=False`, `cov_type` is unused and does not block estimation.

For `penalty > 0`, covariance is based on penalized observed curvature and is
conditional on the chosen penalty. In particular, inference from the final
`CoxPHCV` refit is naive post-selection inference; it is not adjusted for the
CV search. Classical likelihood-ratio, AIC, and BIC diagnostics are therefore
reported only for an unpenalized fit.

## Baseline-Hazard Convention

Baseline hazards use one unified Breslow estimator for coefficients fitted
with Breslow, Efron, or Exact ties. A stratified model stores a separate
baseline for every stratum. Accordingly, `predict_survival(..., strata=...)`
requires a stratum label for each prediction row after a stratified fit.

This convention keeps survival predictions comparable across tie methods; it
does not change the tie method used to estimate the coefficients.

## Main Parameters

| Parameter | Default | Description |
|---|---:|---|
| `ties` | `"breslow"` | `"breslow"`, `"efron"`, or `"exact"` |
| `tol` | `1e-9` | Newton convergence tolerance |
| `max_iter` | `100` | Maximum Newton iterations |
| `device` | `"auto"` | `"cpu"`, `"cuda"`, `"torch"`, or `"auto"` |
| `compute_inference` | `True` | Compute inference and Breslow baselines |
| `compute_cindex` | `True` | Compute training concordance during fit |
| `cov_type` | `"nonrobust"` | `"nonrobust"`, `"hc0"`, `"hc1"`, or `"cluster"` |
| `penalty` | `0.0` | Scalar L2 penalty used by `CoxPH`/`CoxPHCV` |
| `gpu_memory_cleanup` | `False` | Best-effort cleanup after public prediction/scoring calls |

## CPU and GPU Examples

```python
from statgpu.survival import CoxPH

# NumPy with cluster-robust inference.
cpu = CoxPH(ties="efron", device="cpu", cov_type="cluster")
cpu.fit(X, stop, event, start=start, strata=strata, cluster=cluster_id)

# Native CuPy path.
cupy_model = CoxPH(ties="breslow", device="cuda")
cupy_model.fit(X_cupy, stop_cupy, event_cupy, entry=entry_cupy)

# Native PyTorch-CUDA Exact path. Exact supports nonrobust inference only.
torch_model = CoxPH(
    ties="exact", device="torch", cov_type="nonrobust"
)
torch_model.fit(X_torch, time_torch, event_torch)
```

For start-stop data, pass subject identity explicitly when rows repeat:

```python
model = CoxPH(ties="efron", device="cuda")
model.fit(
    X_long, stop, event,
    start=start, strata=strata, subject_id=subject_id,
)

survival, eval_times = model.predict_survival(
    X_new, times=[1.0, 2.0, 5.0], strata=new_strata
)
```

## Cross-Validation and Penalization

```python
import numpy as np
from statgpu.survival import CoxPHCV
from statgpu.linear_model import PenalizedCoxPHModel

# L2 grid search; subject rows remain together in automatically generated folds.
cv = CoxPHCV(
    penalties=np.geomspace(1.0, 1e-3, 20),
    ties="exact",
    cv=5,
    device="torch",
)
cv.fit(X_long, stop, event, start=start, strata=strata, subject_id=subject_id)

# Estimation-only sparse/non-convex Cox fit. y_surv has [time, event] columns.
y_surv = np.column_stack([time, event])
penalized = PenalizedCoxPHModel(
    penalty="scad", alpha=0.05, ties="efron",
    device="cuda", compute_inference=False,
)
penalized.fit(X, y_surv)

# The right-censored formula path supports categoricals, interactions,
# transforms, and Patsy NA removal. The intercept is removed automatically.
penalized_formula = PenalizedCoxPHModel(
    penalty="l2", alpha=0.05, ties="efron", device="cpu",
)
penalized_formula.fit(
    formula="Surv(time, event) ~ age * C(group) + np.log(marker)",
    data=frame,
)
```

Fold construction and diagnostics are orchestrated on the host. For explicit
CuPy or Torch devices, both candidate fitting and held-out partial-likelihood
scoring remain on the requested backend; `cv_results_` records the fitting,
scoring, and orchestration devices separately.

`PenalizedCoxPHModel` rejects `fit_intercept=True`. It also raises
`NotImplementedError` at fit time when `compute_inference=True`; use
unpenalized `CoxPH` when standard errors or confidence intervals are required.
The penalized formula interface accepts `Surv(time, event)` only; use `CoxPH`
for `Surv(start, stop, event)`, strata, or subject-level counting-process data.
Only the documented L1, L2/Ridge, ElasticNet, SCAD, MCP, and no-penalty choices
are accepted; adaptive and group penalties are rejected because their Cox paths
have not been validated. String names and the corresponding built-in penalty
objects are accepted. Penalty objects are revalidated before each fit and are
compatible with `sklearn.clone`. Non-finite regularization/solver controls are
rejected before optimization.

## Tie Methods and Strictness

- Breslow is the simplest tied-event approximation.
- Efron usually provides a closer approximation when tied failures are common.
- Exact evaluates the exact tied partial likelihood and is substantially more
  expensive as risk sets and tied-event multiplicities grow.

These are explicit statistical choices. The GPU backends do not replace one
method with another or fall back to a CPU approximation. The shared baseline
hazard remains Breslow by convention for every choice.

## Outputs

- Estimates: `coef_`, `hazard_ratios_`, `log_likelihood`; `aic` and `bic` for
  unpenalized fits
- Inference when enabled: `_bse`, `_zvalues`, `_pvalues`, `_conf_int`
- Diagnostics: `concordance_index`, convergence state, iteration count
- Predictions: `predict_risk_score`, `predict_hazard_ratio`,
  `predict_survival`, and `predict`
- CV: `penalty_`, `penalties_`, `cv_results_`, `best_score_`, and `estimator_`

## Performance and Validation

The audited 2026-07-12 artifacts define speedup as NumPy fit time divided by
backend fit time, so values above 1 mean the GPU backend was faster. Timings use
float64 on an NVIDIA RTX 5880 Ada Generation and include optimization,
inference, and baseline estimation, with transfer measured separately.

| Scenario | Scale | CuPy vs NumPy | Torch vs NumPy |
|---|---:|---:|---:|
| Delayed entry, Breslow | quick (`n=700`, `p=8`) | 0.647x | 0.959x |
| Delayed entry, Breslow | full (`n=2500`, `p=16`) | 1.044x | 1.374x |
| Stratified start-stop, Efron | full (`n=2400`, `p=16`) | 0.241x | 0.411x |

Exact-tie and standard heavy-tie target cases were also slower on GPU in these
runs. The artifacts do not establish a general crossover size: Exact dynamic
programming and small risk-set kernels expose launch/synchronization overhead,
and workload shape materially changes the result. Benchmark the intended data
shape instead of assuming a GPU speedup.

The same artifacts report coefficient, inference, likelihood, baseline, and
prediction parity across NumPy, CuPy, and Torch. The CV matrix selected the same
penalty on all three backends, with final-refit coefficient and standard-error
differences below `1e-16` in that run.

Auditable artifacts:

- `results/survival_completion_2026-07-12.json`
- `results/survival_completion_full_2026-07-12.json`

External Breslow/Efron checks use `statsmodels.duration.PHReg` with aligned
ties, entry, strata, and convergence settings. Exact is checked against
brute-force tied-risk-set references because PHReg does not supply that method.

## FAQ

- **Can I request robust inference with Exact ties?** No. Use
  `cov_type="nonrobust"`, or choose Breslow/Efron for robust covariance.
- **Are `subject_id` and `cluster` interchangeable?** No. `subject_id` describes
  repeated rows for concordance/CV grouping and is the HC0/HC1 aggregation unit
  for repeated-row data; `cluster` explicitly defines cluster-robust covariance
  units.
- **Why can `predict_survival()` fail after a successful fit?** Baseline
  estimation is skipped when `compute_inference=False`.
- **Does GPU always make Cox fitting faster?** No. Whether a crossover exists
  is workload- and hardware-dependent, especially for Exact and small kernels.

## References

- Cox, D. R. (1972). Regression models and life-tables. *Journal of the Royal Statistical Society: Series B*, 34(2), 187-220. [https://doi.org/10.1111/j.2517-6161.1972.tb00899.x](https://doi.org/10.1111/j.2517-6161.1972.tb00899.x)
- Breslow, N. (1974). Covariance analysis of censored survival data. *Biometrics*, 30(1), 89-99. [https://doi.org/10.2307/2529620](https://doi.org/10.2307/2529620)
- Efron, B. (1977). The efficiency of Cox's likelihood function for censored data. *Journal of the American Statistical Association*, 72(359), 557-565. [https://doi.org/10.1080/01621459.1977.10480613](https://doi.org/10.1080/01621459.1977.10480613)
- Lin, D. Y., & Wei, L. J. (1989). The robust inference for the Cox proportional hazards model. *Journal of the American Statistical Association*, 84(408), 1074-1078. [https://doi.org/10.1080/01621459.1989.10478874](https://doi.org/10.1080/01621459.1989.10478874)
