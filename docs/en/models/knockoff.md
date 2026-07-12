# Knockoff Feature Selection

> Language: English  
> Last updated: 2026-07-12  
> This page: Method documentation  
> Switch: [Chinese](../../models/knockoff.md)

Language switch: [Chinese](../../models/knockoff.md)

## Overview

The knockoff module controls FDR for feature selection using feature-wise statistics \(W_j\) and data-adaptive thresholds. Two paths are provided: fixed-X knockoff (design treated as fixed) and model-X knockoff (Gaussian second-order construction). A unified `knockoff_filter` entry point switches between them.

## Path

- `statgpu.feature_selection.knockoff_filter`
- `statgpu.feature_selection.fixed_x_knockoff_filter`
- `statgpu.feature_selection.model_x_knockoff_filter`
- `statgpu.feature_selection.KnockoffSelector`
- `statgpu.feature_selection.FixedXKnockoffSelector`

Top-level aliases:
- `statgpu.knockoff_filter`
- `statgpu.fixed_x_knockoff_filter`
- `statgpu.model_x_knockoff_filter`
- `statgpu.KnockoffSelector`
- `statgpu.FixedXKnockoffSelector`

## Objective Function

Control false discovery rate at target `q` while maximizing stable power:
- Build knockoff variables \(\tilde X\) that mirror dependence structure.
- Compute antisymmetric statistics \(W_j\) (for example correlation or coefficient differences).
- Select features with \(W_j\) above knockoff threshold.

## Estimating Equation

The decision rule follows knockoff thresholding:
$$
T = \min \left\{ t>0 : \frac{1+\#\{j:W_j\le -t\}}{\max(1,\#\{j:W_j\ge t\})}\le q \right\}
$$
for knockoff+ (`fdr_control="knockoff_plus"`), with the standard knockoff variant available via `fdr_control="knockoff"`.

## Covariance/Inference

This method does not report coefficient covariance tables. Inference is selection-based FDR control:
- fixed-X path requires fixed-design assumptions and usually `n >= 2p`.
- model-X path uses covariance estimation plus S-matrix construction; optional multi-draw averaging is supported.
- `compat_mode="knockpy"` exposes compatibility controls for covariance/S-matrix behavior.

## Parameters

Key `knockoff_filter` parameters:

| Parameter | Default | Description |
|---|---:|---|
| `knockoff_type` | `fixed_x` | `fixed_x` or `model_x` |
| `q` | `0.1` | Target FDR in `(0, 1)` |
| `method` | `corr_diff` | `corr_diff` / `ols_coef_diff` / `lasso_coef_diff` |
| `fdr_control` | `knockoff_plus` | Threshold rule: `knockoff_plus` or `knockoff` |
| `backend` | `auto` | Compute backend: `auto` / `numpy` / `cupy` / `torch` |
| `Xk` | `None` | Optional external knockoff matrix (same shape as `X`) |
| `compat_mode` | `statgpu` | `statgpu` or `knockpy` |
| `lasso_cv_impl` | `auto` | `auto` / `statgpu` / `sklearn` |
| `modelx_covariance_shrinkage` | `0.20` | model-X covariance shrinkage factor |
| `modelx_s_scale` | `0.999` | model-X S-matrix scaling factor |
| `modelx_draws` | `None` | Strictly positive integer draw count; `None` uses the statistic-specific default |
| `modelx_shrinkage` | `ledoitwolf` | knockpy-compatible covariance strategy |
| `modelx_smatrix_method` | `mvr` | knockpy-compatible S-matrix method |
| `knockpy_sampler` | `None` | Optional dispatch target (`gaussian`, `fx`, `metro`, `artk`, ...) |
| `knockpy_sampler_method` | `None` | Gaussian submethod (`mvr`, `sdp`, `maxent`, `equi`, `ci`) |

## CPU+GPU Examples

```python
from statgpu import knockoff_filter

# CPU fixed-X
res_cpu = knockoff_filter(
    X,
    y,
    knockoff_type="fixed_x",
    q=0.1,
    method="ols_coef_diff",
    backend="numpy",
)

# GPU model-X
res_gpu = knockoff_filter(
    X_gpu,
    y_gpu,
    knockoff_type="model_x",
    q=0.1,
    method="lasso_coef_diff",
    backend="cupy",
    modelx_draws=3,
)

# GPU Torch fixed-X
import torch
X_torch = torch.from_numpy(X).to('cuda')
y_torch = torch.from_numpy(y).to('cuda')

res_torch = knockoff_filter(
    X_torch, y_torch,
    knockoff_type="fixed_x",
    q=0.1,
    method="lasso_coef_diff",
    backend="torch",
)

# GPU Torch model-X
res_torch_mx = knockoff_filter(
    X_torch, y_torch,
    knockoff_type="model_x",
    q=0.1,
    method="lasso_coef_diff",
    backend="torch",
    modelx_draws=3,
)
```

## strict/approx difference

- `fdr_control="knockoff_plus"` is the stricter, more conservative option and default.
- `fdr_control="knockoff"` is less conservative and may yield higher power.
- In model-X, higher `modelx_draws` usually improves stability at higher runtime cost.
- `knockpy_sampler` dispatch options are currently guarded; explicitly setting unsupported targets can raise `NotImplementedError` instead of silently falling back.

## Performance Boundary

Knockoff runtime depends strongly on `n`, `p`, statistic choice, draw count, and
backend launch/transfer costs. Historical benchmark scripts remain available, but no
current speedup factor is claimed until the physical-CUDA benchmark matrix is rerun.

## Outputs

`knockoff_filter` and selector wrappers return a `KnockoffResult`-style object with:
- `selected_features`
- `W`
- `threshold`
- `estimated_fdr`
- `q_trajectory`
- `metadata` (for example draw count, compatibility mode, and knockoff source)

## FAQ

- Why do I get an error for fixed-X? Check constraints (`q` in `(0,1)`, `X` is 2D, `Xk` shape matches `X`, and fixed-X rank/sample requirements are met).
- When should I use model-X? Use it when fixed-X construction is infeasible or when distribution-based knockoff construction is preferred.
- Is CuPy required for GPU? Yes, `backend="cupy"` requires CuPy in the environment.

## External Validation

- `dev/benchmarks/benchmark_knockoff_fixedx.py`
- `dev/benchmarks/benchmark_knockoff_vs_baselines.py`
- `dev/benchmarks/benchmark_knockoff_same_xk_parity.py`
- Result artifacts are stored under `results/benchmark_knockoff_*.json`.

## References

- Barber, R. F., & Candes, E. J. (2015). Controlling the false discovery rate via knockoffs. *Annals of Statistics*, 43(5), 2055-2085. [https://doi.org/10.1214/15-AOS1337](https://doi.org/10.1214/15-AOS1337)
- Candes, E., Fan, Y., Janson, L., & Lv, J. (2018). Panning for gold: Model-X knockoffs for high-dimensional controlled variable selection. *Journal of the Royal Statistical Society: Series B*, 80(3), 551-577. [https://doi.org/10.1111/rssb.12265](https://doi.org/10.1111/rssb.12265)
