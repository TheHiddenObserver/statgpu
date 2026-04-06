# LogisticRegression

> Language: English  
> Last updated: 2026-04-02  
> This page: Model documentation  
> Switch: [中文](../../models/logistic-regression.md)

Language switch: [中文](../../models/logistic-regression.md)

Path: `statgpu.linear_model.LogisticRegression`

## Parameter Table

| Parameter | Default | Description |
|---|---:|---|
| `fit_intercept` | `True` | Whether to fit an intercept |
| `C` | `1.0` | Inverse regularization strength |
| `max_iter` | `100` | Max IRLS iterations |
| `tol` | `1e-4` | Convergence tolerance |
| `device` | `"auto"` | `cpu` / `cuda` / `auto` |
| `compute_inference` | `True` | Whether to compute inference stats |
| `cov_type` | `"nonrobust"` | `nonrobust` / `hc0` / `hc1` |
| `gpu_memory_cleanup` | `False` | Best-effort CuPy pool cleanup after each fit |

## Example

```python
from statgpu.linear_model import LogisticRegression

m = LogisticRegression(device="cuda", cov_type="hc1", compute_inference=True)
m.fit(X, y_binary)
proba = m.predict_proba(X)
```

## Robust Covariance (HC0/HC1)

- `cov_type="nonrobust"`: classical information-matrix covariance
- `cov_type="hc0"`: White/sandwich robust covariance
- `cov_type="hc1"`: HC0 with DOF correction `n/(n-k)`

## Outputs

- Coefficients: `intercept_`, `coef_`, `n_iter_`
- Inference: `_bse`, `_zvalues`, `_pvalues`, `_conf_int`
- Metrics: `aic`, `bic`, `pseudo_rsquared`, `accuracy`, `precision`, `recall`, `f1`, `auc`, `average_precision`

## Classification Evaluation APIs

- `predict_with_threshold(X, threshold=0.5)`: threshold-tuned class labels
- `confusion_matrix(X, y, threshold=0.5)`: returns `[[TN, FP], [FN, TP]]`
- `classification_table(X, y, threshold=0.5)`: compact dict with accuracy/precision/recall/F1/specificity
- `roc_curve(X, y)`: returns `fpr, tpr, thresholds`
- `roc_auc_score(X, y)`: returns ROC AUC
- `precision_recall_curve(X, y)`: returns `precision, recall, thresholds`
- `average_precision_score(X, y)`: returns Average Precision (AP)
- `evaluate_classification(X, y, threshold=0.5, include_curves=True)`: one-shot batch output for confusion/table/ROC/PR/AUC/AP (single probability pass)
- `statgpu.evaluation.evaluate_binary_classification(y_true, y_score, threshold=0.5, include_curves=True, backend='auto')`: one-shot batch evaluation for external probabilities (model-agnostic)
- `plot_roc_curve(X, y, ax=None, label=None)`: plots ROC curve and returns matplotlib axes
- `plot_precision_recall_curve(X, y, ax=None, label=None)`: plots PR curve and returns matplotlib axes

External-probability example:

```python
from statgpu import evaluate_binary_classification

# y_true: 0/1 labels; y_score: positive-class probabilities
out = evaluate_binary_classification(
  y_true,
  y_score,
  threshold=0.5,
  include_curves=True,
  backend="auto",  # auto / numpy / cupy / torch
)

print(out["classification_table"])
print(out["roc_auc"], out["average_precision"])
```

When `device="cuda"`, `confusion_matrix` / `classification_table` / `roc_curve` /
`roc_auc_score` / `precision_recall_curve` / `average_precision_score` run on GPU by
default without GPU-to-CPU array transfers (plotting APIs convert to NumPy for rendering).

> Plotting APIs require optional dependency `matplotlib`.

## Returns and Properties

- `fit(X, y)` returns `self`
- `predict_proba(X)` returns class probabilities
- `predict(X)` returns labels
- `predict_with_threshold(X, threshold)` returns labels with a custom threshold

## External Consistency

- `dev/tests/test_external_consistency.py`
  - `test_logistic_robust_covariance_matches_statsmodels`
  - `test_logistic_robust_covariance_gpu_matches_statsmodels`
