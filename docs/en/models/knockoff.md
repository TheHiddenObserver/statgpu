# Knockoff Feature Selection

> Language: English  
> Last updated: 2026-04-11  
> This page: Method documentation  
> Switch: [Chinese](../../models/knockoff.md)

Language switch: [Chinese](../../models/knockoff.md)

Path:
- statgpu.feature_selection.knockoff_filter
- statgpu.feature_selection.fixed_x_knockoff_filter
- statgpu.feature_selection.model_x_knockoff_filter
- statgpu.feature_selection.KnockoffSelector
- statgpu.feature_selection.FixedXKnockoffSelector

Top-level aliases:
- statgpu.knockoff_filter
- statgpu.fixed_x_knockoff_filter
- statgpu.model_x_knockoff_filter
- statgpu.KnockoffSelector
- statgpu.FixedXKnockoffSelector

## Overview

Current implementation supports two knockoff paths:

- fixed-X knockoff
  - For settings where design is treated as fixed.
  - Typical constraint: n >= 2p for valid fixed-X construction.
- model-X knockoff
  - Gaussian second-order approximation (covariance estimation + S-matrix).
  - Supports multi-draw W aggregation.

Use knockoff_filter as the unified entrypoint, with knockoff_type selecting fixed_x/model_x.

## Unified Entry Parameters (knockoff_filter)

| Parameter | Default | Description |
|---|---:|---|
| knockoff_type | fixed_x | fixed_x or model_x |
| q | 0.1 | Target FDR level in (0, 1) |
| method | corr_diff | W statistic: corr_diff / ols_coef_diff / lasso_coef_diff |
| fdr_control | knockoff_plus | Threshold rule: knockoff_plus or knockoff |
| random_state | None | Random seed |
| backend | auto | auto / numpy / cupy |
| Xk | None | User-provided knockoff matrix (must match X shape) |
| compat_mode | statgpu | statgpu or knockpy |
| lasso_cv_impl | auto | auto / statgpu / sklearn |
| lasso_fast_profile | off | Lasso fast-profile switch |
| modelx_covariance_shrinkage | 0.20 | Covariance shrinkage for model-X path |
| modelx_s_scale | 0.999 | S scaling for model-X path |
| modelx_draws | None | Number of model-X draws (auto if None) |
| modelx_shrinkage | ledoitwolf | Covariance estimator strategy for knockpy-compat path |
| modelx_smatrix_method | mvr | S-matrix method for knockpy-compat path |
| knockpy_sampler | None | Optional dispatch entry (gaussian/fx/metro/artk, etc.) |
| knockpy_sampler_method | None | Gaussian sub-method (mvr/sdp/maxent/equi/ci) |

## Return Object (KnockoffResult)

Core fields:

- selected_features: selected feature indices
- W: per-feature W statistics
- threshold: selected threshold at target q
- estimated_fdr: estimated FDR
- q_trajectory: threshold scan diagnostics
- metadata: run metadata (draw count, compat mode, Xk source, etc.)

## Quick Examples

### 1) fixed-X function call

```python
from statgpu import fixed_x_knockoff_filter

res = fixed_x_knockoff_filter(
    X,
    y,
    q=0.1,
    method="ols_coef_diff",
    backend="auto",
)

print(res.selected_features)
print(res.threshold, res.estimated_fdr)
```

### 2) model-X via unified API

```python
from statgpu import knockoff_filter

res = knockoff_filter(
    X,
    y,
    knockoff_type="model_x",
    q=0.1,
    method="lasso_coef_diff",
    compat_mode="knockpy",
    modelx_draws=3,
)
```

### 3) sklearn-style selector

```python
from statgpu import KnockoffSelector

selector = KnockoffSelector(
    knockoff_type="model_x",
    q=0.1,
    method="corr_diff",
    backend="auto",
)
selector.fit(X, y)
X_sel = selector.transform(X)
mask = selector.get_support()
```

## Unified Dispatch Interface (new)

A unified knockpy_sampler_dispatch interface is now wired into the model-X path:

- knockpy_sampler: gaussian / gaussian_mvr / gaussian_sdp / gaussian_maxent / gaussian_equi / gaussian_ci / fx / metro / artk
- knockpy_sampler_method: valid when sampler=gaussian (mvr/sdp/maxent/equi/ci)

Important notes:
- Dispatched sampler targets are currently placeholders (function body uses pass).
- Passing knockpy_sampler explicitly will raise NotImplementedError until those samplers are implemented.
- If you do not pass knockpy_sampler, stable existing logic is used.

## Constraints and Troubleshooting

- q must be in (0, 1).
- X must be 2D and y must align with X rows.
- If Xk is provided, shape must match X.
- fixed-X requires sufficient sample size/rank, otherwise an error is raised.
- cupy backend requires CuPy installed.

## Related Scripts

- dev/benchmarks/benchmark_knockoff_fixedx.py
- dev/benchmarks/benchmark_knockoff_vs_baselines.py
- dev/benchmarks/benchmark_knockoff_same_xk_parity.py
