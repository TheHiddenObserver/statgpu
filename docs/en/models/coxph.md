# CoxPH

> Language: English  
> Last updated: 2026-07-24  
> This page: Model documentation  
> Switch: [Chinese](../../cn/models/coxph.md)

## Overview

`CoxPH` implements proportional-hazards regression with Breslow or Efron tie handling on NumPy, CuPy CUDA, and Torch CUDA backends. The public contract includes backend-native prediction, explicit optimizer termination state, optional robust or cluster-robust inference, delayed entry, and cross-validation through `CoxPHCV`.

Important behavior:

- explicit `device="cuda"` and `device="torch"` never silently fall back to CPU;
- `compute_inference=False` requests estimation only and leaves inference fields unset;
- robust inference is strict by default; approximate Efron inference requires an explicit opt-in;
- delayed-entry support depends on backend, penalty, covariance type, and whether inference is requested, as shown below.

## Path

```python
from statgpu.survival import CoxPH, CoxPHCV
```

## Objective Function

For covariates \(x_i\), event indicator δ_i, and risk set \(R_i\), the unpenalized model maximizes

$$
\ell(\beta)=\sum_{i:\delta_i=1}\left(x_i^\top\beta-\log\sum_{j\in R_i}\exp(x_j^\top\beta)\right),
$$

with Breslow or Efron tie handling. When `penalty > 0`, `CoxPH` applies an L2 penalty using the package's documented objective scaling.

## Optimization and Convergence

Newton iterations use line search and final-state KKT verification. A failed line search does not update coefficients and does not report convergence. Public fitted-state fields include:

- `converged_`;
- `termination_reason_`;
- `n_iter_`;
- `final_kkt_inf_`;
- `final_kkt_normalized_`.

The log likelihood, Hessian, covariance, and inference outputs are recomputed from the final coefficient vector rather than a stale intermediate iterate.

## Covariance and Inference

| `cov_type` | Meaning |
|---|---|
| `"nonrobust"` | Model-based covariance from observed information |
| `"hc0"` | Robust sandwich covariance |
| `"hc1"` | Robust sandwich covariance with finite-sample correction |
| `"cluster"` | Cluster-robust covariance; pass `cluster=` to `fit` |

`compute_inference=True` computes `_bse`, `_zvalues`, `_pvalues`, and `_conf_int`. `compute_inference=False` performs estimation only; the model may still be fitted with a robust `cov_type`, but no covariance or inferential fields are produced.

`inference_mode="strict"` is the default:

- exact Breslow score residuals are implemented internally;
- exact Efron robust residuals require the `survival` extra;
- `inference_mode="approx"` explicitly permits the event-row Efron sandwich fallback when exact residuals are unavailable.

Inference provenance is exposed through:

- `inference_method_`;
- `inference_backend_`;
- `inference_approximate_`;
- `inference_fallback_reason_`;
- `full_host_transfer_performed_`.

## Parameters

| Parameter | Default | Description |
|---|---:|---|
| `ties` | `"breslow"` | Tie handling: `"breslow"` or `"efron"` |
| `tol` | `1e-9` | Newton/KKT convergence tolerance |
| `max_iter` | `100` | Maximum iterations |
| `device` | `"auto"` | `"cpu"`, `"cuda"`, `"torch"`, or `"auto"` |
| `compute_inference` | `True` | Compute covariance and inferential outputs |
| `cov_type` | `"nonrobust"` | `"nonrobust"`, `"hc0"`, `"hc1"`, or `"cluster"` |
| `penalty` | `0.0` | Non-negative L2 penalty |
| `inference_mode` | `"strict"` | Robust-inference policy: `"strict"` or `"approx"` |
| `gpu_memory_cleanup` | `False` | Best-effort CuPy/Torch cache cleanup |

## Delayed-Entry Support Matrix

The key distinction is whether inference is requested.

| Entry | Penalty | Covariance | `compute_inference` | CPU | CuPy | Torch |
|---|---:|---|---:|---|---|---|
| no | any | supported `cov_type` | either | supported | supported | supported |
| yes | `0` | `nonrobust` | either | supported; CPU path requires `statgpu[survival]` | supported | supported |
| yes | `>0` | `nonrobust` | either | explicit `NotImplementedError` | supported | supported |
| yes | any | `hc0` / `hc1` / `cluster` | `True` | explicit `NotImplementedError` | explicit `NotImplementedError` | explicit `NotImplementedError` |
| yes | any | `hc0` / `hc1` / `cluster` | `False` | estimation supported; inference fields remain `None` | estimation supported; inference fields remain `None` | estimation supported; inference fields remain `None` |

Additional notes:

- install CPU delayed-entry and exact Efron robust support with `pip install "statgpu[survival]"`;
- both Breslow and Efron delayed-entry fitting follow the table above;
- `CoxPHCV` applies the same `compute_inference` guard during final refit;
- GPU delayed-entry CV currently supports `ties="breslow"` only;
- CPU delayed-entry CV supports an explicit unpenalized grid such as `penalties=[0.0]`; any nonzero delayed-entry CPU penalty candidate is rejected;
- `inference_mode` is forwarded to the final estimator;
- `predict` and `score` reuse the backend-native final estimator implementation.

## CPU and GPU Examples

```python
from statgpu.survival import CoxPH

# Exact Efron cluster-robust inference; requires statgpu[survival].
strict_model = CoxPH(
    device="cpu",
    ties="efron",
    cov_type="cluster",
    inference_mode="strict",
    compute_inference=True,
)
strict_model.fit(X, time, event, cluster=cluster_ids)

# Delayed-entry estimation with a robust covariance label but no inference.
estimation_only = CoxPH(
    device="cuda",
    ties="breslow",
    cov_type="hc0",
    compute_inference=False,
)
estimation_only.fit(X_gpu, time_gpu, event_gpu, entry=entry_gpu)
assert estimation_only._bse is None
assert estimation_only._conf_int is None

# Standard Torch CUDA fit with inference.
torch_model = CoxPH(
    device="torch",
    ties="efron",
    cov_type="nonrobust",
    compute_inference=True,
)
torch_model.fit(X_torch, time_torch, event_torch)
```

## Outputs

- parameters: `coef_`, `hazard_ratios_`;
- inference, when enabled: `_bse`, `_zvalues`, `_pvalues`, `_conf_int`;
- diagnostics: `log_likelihood`, `aic`, `bic`, `concordance_index`;
- convergence: `converged_`, `termination_reason_`, `n_iter_`, `final_kkt_inf_`, `final_kkt_normalized_`;
- provenance: `inference_method_`, `inference_backend_`, `inference_approximate_`, `inference_fallback_reason_`, `full_host_transfer_performed_`;
- backend-native prediction: `predict_risk_score`, `predict_hazard_ratio`, `predict_survival`, and `predict`.

## Validation

PR #79 validated maintained CoxPH behavior on NumPy, CuPy CUDA, and Torch CUDA. The exact-head GitHub Actions matrix covers Python 3.9–3.12, and the maintained physical-GPU suite passed on a Tesla P100. Canonical accuracy reports are generated only from clean exact-head validated artifacts; stale hard-coded PASS files are not authoritative.

See:

- `dev/reviews/pr79_physical_gpu_validation.md`;
- `dev/tests/test_pr79_physical_gpu.py`;
- `dev/benchmarks/pr79/`.

## Limitations

- delayed-entry robust or cluster covariance is not implemented when inference is requested;
- CPU delayed entry with a nonzero penalty is not implemented;
- strata, frailty, and time-varying covariates remain outside the current scope;
- `torch.compile`, when enabled, requires Triton-capable hardware; Tesla P100 is not supported for that optional path.

## References

- Cox, D. R. (1972). Regression models and life-tables. *JRSS B*, 34(2), 187–220.
- Breslow, N. (1974). Covariance analysis of censored survival data. *Biometrics*, 30(1), 89–99.
- Efron, B. (1977). The efficiency of Cox's likelihood function for censored data. *JASA*, 72(359), 557–565.
- Lin, D. Y., & Wei, L. J. (1989). The robust inference for the Cox proportional hazards model. *JASA*, 84(408), 1074–1078.
