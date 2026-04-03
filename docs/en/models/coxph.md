# CoxPH

> Language: English  
> Last updated: 2026-04-02  
> This page: Model documentation  
> Switch: [中文](../../models/coxph.md)

Language switch: [中文](../../models/coxph.md)

Path: `statgpu.survival.CoxPH`

## Parameter Table

| Parameter | Default | Description |
|---|---:|---|
| `ties` | `"breslow"` | Tie handling: `breslow` / `efron` |
| `tol` | `1e-9` | Newton-Raphson convergence tolerance |
| `max_iter` | `100` | Max iterations |
| `device` | `"auto"` | `cpu` / `cuda` / `auto` |
| `compute_inference` | `True` | Whether to compute inference and diagnostics |
| `cov_type` | `"nonrobust"` | `nonrobust` / `hc0` / `hc1` / `cluster` |
| `gpu_memory_cleanup` | `False` | Best-effort CuPy pool cleanup after each fit |

## Example

```python
from statgpu.survival import CoxPH

m = CoxPH(device="cuda", ties="efron", compute_inference=True)
m.fit(X, time, event)
```

When `cov_type="cluster"`, pass cluster ids in `fit`:

```python
m = CoxPH(device="cpu", cov_type="cluster")
m.fit(X, time, event, cluster=cluster_ids)
```

## Outputs

- Parameters: `coef_`, `hazard_ratios_`
- Inference: `_bse`, `_zvalues`, `_pvalues`, `_conf_int` (if enabled)
- Diagnostics: `log_likelihood`, `aic`, `bic`, `concordance_index`

## Returns and Properties

- `fit(X, time, event, entry=None)` returns `self`

## Notes

- Supports both Breslow and Efron tie handling.
- `strata/frailty/time-varying covariates` are planned but not complete yet.
