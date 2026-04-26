# LogisticRegression

> Language: English  
> Last updated: 2026-04-17  
> This page: Model documentation  
> Switch: [Chinese](../../models/logistic-regression.md)

Language switch: [Chinese](../../models/logistic-regression.md)

## Overview

`LogisticRegression` implements binary logit with IRLS fitting on CPU/GPU, robust covariance options, and integrated classification metrics. Current scope is binary classification with L2 regularization path; multiclass and elastic-net are not part of this API.

## Path

`statgpu.linear_model.LogisticRegression`

## Objective Function

Estimate binary log-likelihood (with L2 penalty controlled by `C`):
\[
\max_\beta \sum_i \left[y_i\log p_i + (1-y_i)\log(1-p_i)\right] - \lambda\|\beta\|_2^2
\]
where \(p_i = \sigma(x_i^\top\beta)\) and larger `C` means weaker regularization.

## Estimating Equation

The model is solved by IRLS/Newton-style updates to satisfy score equations:
\[
\sum_i x_i(y_i - p_i)=0
\]
under convergence controls `max_iter` and `tol`.

## Covariance/Inference

- `cov_type="nonrobust"`: information-matrix covariance.
- `cov_type="hc0"|"hc1"|"hc2"|"hc3"`: robust sandwich covariance variants.
- `cov_type="hac"`: Newey-West (Bartlett) covariance with optional `hac_maxlags`.
- Inference outputs use z-statistic conventions: `_bse`, `_zvalues`, `_pvalues`, `_conf_int`.
- `compute_inference=True` is required for inference fields.

## Parameters

| Parameter | Default | Description |
|---|---:|---|
| `fit_intercept` | `True` | Whether to fit an intercept |
| `C` | `1.0` | Inverse regularization strength |
| `max_iter` | `100` | Max IRLS iterations |
| `tol` | `1e-4` | Convergence tolerance |
| `device` | `"auto"` | `cpu` / `cuda` / `torch` / `auto` |
| `compute_inference` | `True` | Whether to compute inference stats |
| `cov_type` | `"nonrobust"` | `nonrobust` / `hc0` / `hc1` / `hc2` / `hc3` / `hac` |
| `hac_maxlags` | `None` | Max lag for `cov_type="hac"`; default follows Newey-West style heuristic |
| `gpu_memory_cleanup` | `False` | Best-effort CuPy pool cleanup after each fit |

## CPU+GPU Examples

```python
from statgpu.linear_model import LogisticRegression

# CPU near-unregularized logit
m_cpu = LogisticRegression(device="cpu", C=1e10, cov_type="hc1", compute_inference=True)
m_cpu.fit(X, y_binary)

# GPU with HAC covariance
m_gpu = LogisticRegression(
    device="cuda",
    C=1e10,
    cov_type="hac",
    hac_maxlags=4,
    compute_inference=True,
    gpu_memory_cleanup=True,
)
m_gpu.fit(X_gpu, y_gpu)
```

## strict/approx difference

No separate approx inference mode is exposed in this API. Robust covariance choice (`hc*`/`hac`) is the main practical trade-off between assumptions and computational cost.

## Outputs

- Coefficients: `intercept_`, `coef_`, `n_iter_`
- Inference: `_bse`, `_zvalues`, `_pvalues`, `_conf_int`
- Fit/metrics: `aic`, `bic`, `pseudo_rsquared`, `accuracy`, `precision`, `recall`, `f1`, `auc`, `average_precision`
- Prediction methods: `predict_proba`, `predict`, `predict_with_threshold`
- Evaluation methods: `confusion_matrix`, `classification_table`, `roc_curve`, `roc_auc_score`, `precision_recall_curve`, `average_precision_score`, `evaluate_classification`
- Plot helpers: `plot_roc_curve`, `plot_precision_recall_curve` (`matplotlib` optional dependency)

## FAQ

- How do I approximate unregularized MLE? Use a large `C` value (for example, `1e10`).
- Why are inference statistics z-values instead of t-values? Logistic regression inference follows large-sample normal approximation.
- Are evaluation methods GPU-native? Core prediction and metrics stay on the selected supported backend; plotting converts to NumPy for rendering.

## External Validation

- `dev/tests/test_external_consistency.py`
  - `test_logistic_robust_covariance_matches_statsmodels`
  - `test_logistic_robust_covariance_gpu_matches_statsmodels`
- Cross-backend artifact for `hc2/hc3/hac`:
  - `results/remote_covariance_full_compare_2026-04-10.json`

## References

- McCullagh, P., & Nelder, J. A. (1989). *Generalized Linear Models* (2nd ed.). Chapman & Hall/CRC.
- Hosmer, D. W., Lemeshow, S., & Sturdivant, R. X. (2013). *Applied Logistic Regression* (3rd ed.). Wiley.
