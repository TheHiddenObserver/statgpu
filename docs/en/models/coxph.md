# CoxPH

> Language: English  
> Last updated: 2026-07-23
> This page: Model documentation  
> Switch: [Chinese](../../cn/models/coxph.md)

Language switch: [Chinese](../../cn/models/coxph.md)

## Overview

`CoxPH` implements proportional hazards regression with Breslow/Efron tie handling on CPU/GPU backends. Features vectorized Efron gradient/Hessian (no Python loops), multi-block CUDA kernels, and DLPack bridge for torch-CUDA.

Notes:

- **Efron optimization** (v0.2.1): prefix-sum vectorized path, 3-6x faster than statsmodels (n=5000); verified against statsmodels PHReg in CI.
- `PenalizedCoxRegression` supports SCAD/MCP penalties via proximal Newton solver.
- Delayed entry (`entry`) is available on all three backends subject to the
  explicit support matrix below.
- Explicit `device='cuda'` and `device='torch'` do not silently fall back to CPU. Use `device='cpu'` for the CPU implementation.
- `CoxPHCV` is trainable for penalty search + final refit.

## Path

`statgpu.survival.CoxPH`

## Objective Function

Estimate coefficients by maximizing the Cox partial log-likelihood:
$$
\ell(\beta)=\sum_{i:\delta_i=1}\left(x_i^\top\beta-\log\sum_{j\in R_i}\exp(x_j^\top\beta)\right)
$$
with tie handling determined by `ties`.

## Estimating Equation

Solve score equations \(\partial \ell(\beta)/\partial \beta = 0\) using Newton-Raphson iterations (`tol`, `max_iter`). Tie handling uses Breslow or Efron approximation within risk-set terms.

## Covariance/Inference

- `cov_type="nonrobust"`: model-based covariance from observed information.
- `cov_type="hc0"|"hc1"`: robust covariance variants.
- `cov_type="cluster"`: cluster-robust covariance; pass `cluster=` in `fit`.
- `compute_inference=True` enables `_bse`, `_zvalues`, `_pvalues`, `_conf_int`.
- Inference follows large-sample z-statistic conventions.
- `inference_mode="strict"` is the default. Exact Breslow score residuals are
  internal; exact Efron robust residuals require the `survival` extra. The
  event-row Efron approximation is used only with `inference_mode="approx"`.
- Inference records `inference_method_`, `inference_backend_`,
  `inference_approximate_`, and `inference_fallback_reason_`.

## Parameters

| Parameter | Default | Description |
|---|---:|---|
| `ties` | `"breslow"` | Tie handling: `breslow` / `efron` |
| `tol` | `1e-9` | Newton-Raphson convergence tolerance |
| `max_iter` | `100` | Max iterations |
| `device` | `"auto"` | `cpu` / `cuda` / `torch` / `auto` |
| `compute_inference` | `True` | Whether to compute inference and diagnostics |
| `cov_type` | `"nonrobust"` | `nonrobust` / `hc0` / `hc1` / `cluster` |
| `penalty` | `0.0` | Non-negative L2 penalty |
| `inference_mode` | `"strict"` | Robust inference policy: `strict` / `approx` |
| `gpu_memory_cleanup` | `False` | Best-effort CuPy pool cleanup after each fit |

## Entry and Device Notes

| Entry | Penalty | Covariance | CPU | CuPy | Torch |
|---|---:|---|---|---|---|
| no | any | supported `cov_type` | supported | supported | supported |
| yes | `0` | `nonrobust` | supported; requires statsmodels | supported | supported |
| yes | `>0` | `nonrobust` | explicit `NotImplementedError` | supported | supported |
| yes | any | `hc0` / `hc1` / `cluster` | explicit `NotImplementedError` | explicit `NotImplementedError` | explicit `NotImplementedError` |

- Both Breslow and Efron delayed-entry fitting follow this matrix.
- Install CPU delayed-entry and exact Efron robust support with
  `pip install "statgpu[survival]"`.
- `device='cuda'` requires a working CuPy CUDA backend.
- `device='torch'` requires `torch.cuda.is_available() == True`.
- `CoxPHCV`:
  - GPU `entry` currently supports `ties='breslow'` only
  - CPU delayed-entry CV rejects any nonzero penalty candidate; an explicit
    `penalties=[0.0]` unpenalized run is supported with `statgpu[survival]`
  - delayed-entry robust/cluster covariance follows the same explicit
    `NotImplementedError` contract as `CoxPH`
  - `inference_mode` is forwarded to the final estimator, and `predict`/`score`
    reuse its backend-native implementation
  - `gpu_memory_cleanup=True` forwards cleanup to the final `CoxPH` estimator and exposes best-effort CuPy/Torch cleanup hooks
- `torch.compile` (if enabled) requires Triton-capable GPUs (Compute Capability >= 7.0), e.g., A30/RTX 4090. Tesla P100 (CC 6.0) is not supported.

## CPU+GPU Examples

```python
from statgpu.survival import CoxPH

# Exact Efron cluster-robust covariance (requires statgpu[survival])
m_cpu = CoxPH(
    device="cpu", ties="efron", cov_type="cluster",
    inference_mode="strict", compute_inference=True,
)
m_cpu.fit(X, time, event, cluster=cluster_ids)

# GPU with standard covariance
m_gpu = CoxPH(device="cuda", ties="breslow", compute_inference=True, gpu_memory_cleanup=True)
m_gpu.fit(X_gpu, time_gpu, event_gpu)
```

## strict/approx difference

This switch controls robust score-residual inference, not tie handling.

- `strict` (default): never silently substitutes an approximate covariance.
  Internal exact Breslow residuals are available; exact Efron residuals require
  statsmodels from `statgpu[survival]`.
- `approx`: permits the event-row Efron sandwich fallback when exact residuals
  are unavailable. Inspect `inference_approximate_` and
  `inference_fallback_reason_` before reporting results.
- Delayed-entry robust/cluster covariance is not implemented and always raises,
  independent of this switch or `compute_inference`.

## Outputs

- Parameters: `coef_`, `hazard_ratios_`
- Inference: `_bse`, `_zvalues`, `_pvalues`, `_conf_int` (if enabled)
- Diagnostics: `log_likelihood`, `aic`, `bic`, `concordance_index`
- Fit state: `converged_`, `termination_reason_`, `n_iter_`,
  `final_kkt_inf_`, `final_kkt_normalized_`
- Inference provenance: `inference_method_`, `inference_backend_`,
  `inference_approximate_`, `inference_fallback_reason_`,
  `full_host_transfer_performed_`
- Prediction methods return arrays native to the estimator backend:
  `predict_risk_score`, `predict_hazard_ratio`, `predict_survival`, `predict`
- Fit method: `fit(X, time, event, entry=None)`

## FAQ

- Should I use `breslow` or `efron`? Prefer `efron` when ties are common; differences are usually small when ties are rare.
- Why might CPU/GPU C-index differ slightly? Numeric and approximation paths can vary; report both in strict reproducibility settings.
- Is full advanced survival modeling included? Not yet; strata/frailty/time-varying covariates remain out of current scope.

## External Validation

- Internal consistency and regression testing are maintained in `dev/tests/`.
- Survival benchmarking scripts are maintained in `dev/benchmarks/`.

## References

- Cox, D. R. (1972). Regression models and life-tables. *Journal of the Royal Statistical Society: Series B*, 34(2), 187-220. [https://doi.org/10.1111/j.2517-6161.1972.tb00899.x](https://doi.org/10.1111/j.2517-6161.1972.tb00899.x)
- Breslow, N. (1974). Covariance analysis of censored survival data. *Biometrics*, 30(1), 89-99. [https://doi.org/10.2307/2529620](https://doi.org/10.2307/2529620)
- Efron, B. (1977). The efficiency of Cox's likelihood function for censored data. *Journal of the American Statistical Association*, 72(359), 557-565. [https://doi.org/10.1080/01621459.1977.10480613](https://doi.org/10.1080/01621459.1977.10480613)
- Lin, D. Y., & Wei, L. J. (1989). The robust inference for the Cox proportional hazards model. *Journal of the American Statistical Association*, 84(408), 1074-1078. [https://doi.org/10.1080/01621459.1989.10478874](https://doi.org/10.1080/01621459.1989.10478874)
