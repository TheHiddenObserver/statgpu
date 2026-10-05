# Knockoff Feature Selection

> Language: English  
> Last updated: 2026-10-05  
> This page: Method documentation  
> Switch: [Chinese](../../cn/models/knockoff.md)

Language switch: [Chinese](../../cn/models/knockoff.md)

## Overview

The knockoff module implements feature-selection procedures designed for FDR control under their construction/statistic assumptions, using feature-wise statistics \(W_j\) and data-adaptive thresholds. Two paths are provided: fixed-X knockoff (design treated as fixed) and model-X knockoff (Gaussian second-order construction). A unified `knockoff_filter` entry point switches between them.

False discovery rate (FDR) is the expected fraction of selected features that are null, counting an empty selection as zero. It is not the probability that every selected feature is correct. Knockoffs act as matched negative controls: a feature must compete against an artificial counterpart with a similar dependence structure. Use this approach when selection error control is the goal and the construction assumptions are credible; for prediction-focused subset search, compare [stepwise selection](feature-selection.md) and validate on held-out data.

Complete function/selector signatures, all parameters, result fields and a self-contained CPU example are in the [feature-selection API reference](../reference/feature-selection-api.md).

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

The statistical goal is false discovery rate control at target `q`, subject to the assumptions and current tied-statistic limitation below:
- Build knockoff variables \(\tilde X\) that mirror dependence structure.
- Compute antisymmetric statistics \(W_j\) (for example correlation or coefficient differences).
- Select features with \(W_j\) above knockoff threshold.

## Estimating Equation

The decision rule follows knockoff thresholding:
$$
T = \min \left\{ t\in\{|W_j|:|W_j|>0\} : \frac{1+\#\{j:W_j\le -t\}}{\max(1,\#\{j:W_j\ge t\})}\le q \right\}
$$
for knockoff+ (`fdr_control="knockoff_plus"`), with the standard knockoff variant available via `fdr_control="knockoff"`.

### Tied-statistic limitation

The displayed threshold is the theoretical knockoff+ rule. The current implementation can disagree when absolute statistics tie: it evaluates partially accumulated tied groups, then selects the whole threshold group. For W=[8,8,-8] and q=0.5, it can report threshold 8 and estimated_fdr=0.5, although the full threshold ratio is (1+1)/2=1 and the theoretical rule selects none. Do not interpret current output as nominal knockoff+ FDR control when threshold ties occur. The same counting limitation also affects offset-0 `knockoff`. No coefficient refit, extra Monte Carlo draws, or change of device repairs it.

## Covariance/Inference

This method does not report coefficient covariance tables. Its inferential target is selection-based FDR control, subject to the stated assumptions and limitations:
- fixed-X path requires fixed-design assumptions and usually `n >= 2p`.
- model-X path uses covariance estimation plus S-matrix construction; optional multi-draw averaging is supported.
- `compat_mode="knockpy"` exposes covariance/S-matrix controls, but missing optional packages or S-matrix solver errors can substitute sample covariance or an equicorrelated S matrix. Inspect `metadata["modelx_covariance_estimator"]` and `metadata["modelx_smatrix_source"]`; the requested method name does not establish what ran. See [compatibility resolution](../reference/feature-selection-api.md#compatibility-resolution-and-fallbacks).

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

The following optional GPU variants assume `X`/`y` generated in the [self-contained example](../reference/feature-selection-api.md#runnable-fixed-x-example). CuPy/Torch require installed usable GPU backends.

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

import cupy as cp
X_gpu, y_gpu = cp.asarray(X), cp.asarray(y)

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

## Threshold rules and construction assumptions

- `fdr_control="knockoff_plus"` is the stricter, more conservative option and default.
- `fdr_control="knockoff"` uses offset 0 and has a different modified-FDR target under the relevant theory; it is not an approximate numerical version of knockoff+.
- In model-X, additional `modelx_draws` average feature statistics at greater computational cost. Reduced Monte Carlo variation does not establish an FDR theorem for the averaged statistic; an estimated Gaussian feature model also does not guarantee exchangeability for arbitrary feature distributions.
- `knockpy_sampler` dispatch options are currently guarded; explicitly setting unsupported targets can raise `NotImplementedError` instead of silently falling back.

## Performance Boundary

Knockoff runtime depends strongly on `n`, `p`, statistic choice, draw count, and
backend launch/transfer costs. Benchmark the target workload; no universal GPU
speedup follows from the algorithm name. Statistical error control depends on
the knockoff construction and feature-statistic assumptions.

## Outputs

Filter functions return `KnockoffResult`. Selector `fit` returns `self`; read its result through `result_` and selected indices through `selected_features_`. Result fields include:
- `selected_features`
- `W`
- `threshold`
- `estimated_fdr`
- `q_trajectory`
- `metadata` (for example draw count, compatibility mode, and knockoff source)

An empty `selected_features` array is a valid outcome. `estimated_fdr` is the threshold-rule estimate, not an observed error fraction or a per-feature p-value. Inspect `W` and `threshold` together, check the tied-statistic restriction above, and evaluate any downstream prediction model on data not used for selection.

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

