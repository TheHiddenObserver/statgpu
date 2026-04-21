# CoxPH

> Language: English  
> Last updated: 2026-04-21  
> This page: Model documentation  
> Switch: [Chinese](../../models/coxph.md)

Language switch: [Chinese](../../models/coxph.md)

## Overview

`CoxPH` implements proportional hazards regression with Breslow/Efron tie handling on CPU/GPU backends. It focuses on the standard Cox path with optional robust/cluster covariance, and currently does not provide full strata/frailty/time-varying covariate workflows.

Note: `CoxPHCV` (cross-validated CoxPH) is now trainable (penalty search + final refit), but `entry` and `cluster` arguments are not yet supported in `CoxPHCV.fit()`.

## Path

`statgpu.survival.CoxPH`

## Objective Function

Estimate coefficients by maximizing the Cox partial log-likelihood:
\[
\ell(\beta)=\sum_{i:\delta_i=1}\left(x_i^\top\beta-\log\sum_{j\in R_i}\exp(x_j^\top\beta)\right)
\]
with tie handling determined by `ties`.

## Estimating Equation

Solve score equations \(\partial \ell(\beta)/\partial \beta = 0\) using Newton-Raphson iterations (`tol`, `max_iter`). Tie handling uses Breslow or Efron approximation within risk-set terms.

## Covariance/Inference

- `cov_type="nonrobust"`: model-based covariance from observed information.
- `cov_type="hc0"|"hc1"`: robust covariance variants.
- `cov_type="cluster"`: cluster-robust covariance; pass `cluster=` in `fit`.
- `compute_inference=True` enables `_bse`, `_zvalues`, `_pvalues`, `_conf_int`.
- Inference follows large-sample z-statistic conventions.

## Parameters

| Parameter | Default | Description |
|---|---:|---|
| `ties` | `"breslow"` | Tie handling: `breslow` / `efron` |
| `tol` | `1e-9` | Newton-Raphson convergence tolerance |
| `max_iter` | `100` | Max iterations |
| `device` | `"auto"` | `cpu` / `cuda` / `auto` |
| `compute_inference` | `True` | Whether to compute inference and diagnostics |
| `cov_type` | `"nonrobust"` | `nonrobust` / `hc0` / `hc1` / `cluster` |
| `gpu_memory_cleanup` | `False` | Best-effort CuPy pool cleanup after each fit |

## CPU+GPU Examples

```python
from statgpu.survival import CoxPH

# CPU with cluster-robust covariance
m_cpu = CoxPH(device="cpu", ties="efron", cov_type="cluster", compute_inference=True)
m_cpu.fit(X, time, event, cluster=cluster_ids)

# GPU with standard covariance
m_gpu = CoxPH(device="cuda", ties="breslow", compute_inference=True, gpu_memory_cleanup=True)
m_gpu.fit(X_gpu, time_gpu, event_gpu)
```

## strict/approx difference

For ties, `efron` is typically the stricter and more accurate approximation when ties are frequent, while `breslow` is usually faster. Both are supported in the release path.

## Outputs

- Parameters: `coef_`, `hazard_ratios_`
- Inference: `_bse`, `_zvalues`, `_pvalues`, `_conf_int` (if enabled)
- Diagnostics: `log_likelihood`, `aic`, `bic`, `concordance_index`
- Prediction methods: `predict_risk_score`, `predict_hazard_ratio`, `predict_survival`, `predict`
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
