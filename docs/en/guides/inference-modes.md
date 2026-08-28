# Inference Modes

> Language: English  
> Last updated: 2026-08-28  
> This page: Guide  
> Switch: [Chinese](../../cn/guides/inference-modes.md)

Language switch: [Chinese](../../cn/guides/inference-modes.md)

## Gaussian linear-model inference

For the shared Gaussian inference path used by squared-error L2/Ridge consumers,
numerical covariance and reference-distribution inference run on the backend that
executed the fitted model: NumPy, CuPy, or Torch. The numerical path includes the
bread/covariance calculation, standard errors, test statistics, p-values, and
confidence-interval critical values.

The established public reporting contract is intentionally unchanged: after all
numerical inference is complete, the inference result and estimator reporting
attributes (`_params`, `_bse`, `_tvalues`, `_pvalues`, `_conf_int`) take a final
NumPy snapshot. This is a reporting boundary, not a CPU inference fallback.
`_inference_result.metadata` records `numerical_backend`, `numerical_device`,
`reporting_backend="numpy"`, and
`reporting_boundary="post_numerical_inference"` for this shared path.

Explicit `device="cuda"` and `device="torch"` requests do not silently downgrade
Gaussian L2 inference to NumPy. Missing or invalid executed-backend provenance
fails closed. `device="auto"` retains the estimator's existing backend-selection
policy.

This statement is deliberately scoped to the migrated shared Gaussian inference
path. It does **not** imply that every inference implementation in statgpu has
been migrated to the same lifecycle.

Supported covariance choices on the Gaussian path are:

- `nonrobust` — classical covariance with Student-t reference inference.
- `hc0`, `hc1`, `hc2`, `hc3` — heteroskedasticity-consistent sandwich covariance
  with normal reference inference.
- `hac` — Bartlett-kernel HAC covariance with normal reference inference.

The backend-native reference helper also preserves the maintained stable
Student-t identities at one and two residual degrees of freedom, avoiding
subtractive cancellation or an avoidable `t**2` overflow in representable
extreme tails.

## Lasso inference methods

`Lasso.inference_method` options:

- `cpu_ols_inference` (default)
- `gpu_ols_inference`
- `bootstrap`

Backward-compatible aliases:

- `naive_ols` -> `cpu_ols_inference`
- `gpu_naive_ols` -> `gpu_ols_inference`

Recommended usage:

```python
from statgpu.linear_model import Lasso

model = Lasso(
    alpha=0.1,
    device="cuda",
    solver="fista",
    stopping="kkt",
    compute_inference=True,
    inference_method="gpu_ols_inference",
)
model.fit(X, y)
```

Related robust covariance support:

- `LinearRegression(cov_type="nonrobust" | "hc0" | "hc1" | "hc2" | "hc3" | "hac")`
- `Ridge(cov_type="nonrobust" | "hc0" | "hc1" | "hc2" | "hc3" | "hac")`
- `LogisticRegression(cov_type="nonrobust" | "hc0" | "hc1" | "hc2" | "hc3" | "hac")`
