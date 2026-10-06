# Knockoff Feature Selection

> Language: English  
> Last updated: 2026-10-05  
> This page: Method documentation  
> Switch: [Chinese](../../cn/models/knockoff.md)

## Overview

The knockoff module implements feature-selection procedures designed for FDR control under their construction/statistic assumptions, with current automatic fixed-X centering and threshold-tie limitations described below, using feature-wise statistics \(W_j\) and data-adaptive thresholds. Two paths are provided: fixed-X knockoff (design treated as fixed) and model-X knockoff (Gaussian second-order construction). A unified `knockoff_filter` entry point switches between them.

False discovery rate (FDR) is the expected fraction of selected features that are null, counting an empty selection as zero. It is not the probability that every selected feature is correct. Knockoffs act as matched negative controls: a feature must compete against an artificial counterpart with a similar dependence structure. Use this approach when selection error control is the goal and the construction assumptions are credible; for prediction-focused subset search, compare [stepwise selection](feature-selection.md) and validate on held-out data.

Complete function/selector signatures, all parameters, result fields and a self-contained CPU example are in the [feature-selection API reference](../reference/feature-selection-api.md).

## Validate the target rate

Before any filter call or selector fit, check `np.isfinite(q) and 0 < q < 1`.
This applies to all three filter functions, both selector classes, and both
threshold rules. The current validation misses `q=np.nan`: it can return an
empty selection with `threshold=inf` and `estimated_fdr=0.0`, or an all-false
selector mask. That output is invalid, not evidence of no discoveries at a
meaningful target rate. Reject nonfinite q yourself; do not replace it with a
new target after inspecting results.

## A complete CPU example

This example uses 240 rows and 12 predictors, satisfying the sample-size
requirement for generated fixed-X knockoffs. It demonstrates the API, but the
[generated-design centering limitation](#generated-fixed-x-centering-limitation)
prevents a nominal FDR claim for this automatic construction. A centered
supplied-pair example is linked below. Only the first four predictors contribute
to the response. Choose q before examining the selection.

<!-- learner-example: knockoff-selection -->
```python
import numpy as np
from statgpu import fixed_x_knockoff_filter

rng = np.random.default_rng(42)
X = rng.normal(size=(240, 12))
y = X[:, :4] @ np.array([3.0, -2.5, 2.0, 1.5])
y += rng.normal(scale=0.5, size=240)
q = 0.25
if not (np.isfinite(q) and 0 < q < 1):
    raise ValueError("q must be finite and strictly between 0 and 1")
result = fixed_x_knockoff_filter(
    X, y, q=q, method="corr_diff", random_state=7, backend="numpy",
)
print("Selected columns:", result.selected_features.tolist())
print("Threshold:", round(result.threshold, 3))
print("Threshold ratio:", round(result.estimated_fdr, 3))
```

For this seed, the output is `Selected columns: [0, 1, 2, 3]`, threshold
`21.637`, and threshold ratio `0.25`. Selected indices refer to the original
columns. Positive W means a feature outscored its knockoff; the threshold ratio
is the rule's estimate, not the actual fraction of null features in this one
sample. Other samples can miss signals or select noise. No coefficient or
prediction model is fitted by this filter. If you subsequently assess prediction,
perform selection using training rows only and keep evaluation rows untouched.

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

The statistical goal is false discovery rate control at target `q`, subject to the response/construction assumptions and both the centering and tied-statistic limitations below:
- Build knockoff variables \(\tilde X\) that mirror dependence structure.
- Compute antisymmetric statistics \(W_j\) (for example correlation or coefficient differences).
- Select features with \(W_j\) above knockoff threshold.

## Estimating Equation

The decision rule follows knockoff thresholding:
$$
T = \min \left\{ t\in\{|W_j|:|W_j|>0\} : \frac{1+\#\{j:W_j\le -t\}}{\max(1,\#\{j:W_j\ge t\})}\le q \right\}
$$
Here q is the requested rate, # counts features, and W is the original-minus-
knockoff importance statistic. Select all j with W_j ≥ T; if the set defining T
is empty, select none. For `fdr_control="knockoff"`, replace the numerator's 1
with 0; its modified-FDR target differs from ordinary FDR.

### Tied-statistic limitation

The displayed threshold is the theoretical knockoff+ rule. The current implementation can disagree when absolute statistics tie: it evaluates partially accumulated tied groups, then selects the whole threshold group. For W=[8,8,-8] and q=0.5, it can report threshold 8 and estimated_fdr=0.5, although the full threshold ratio is (1+1)/2=1 and the theoretical rule selects none. Do not interpret current output as nominal knockoff+ FDR control when threshold ties occur. The same counting limitation also affects offset-0 `knockoff`. No coefficient refit, extra Monte Carlo draws, or change of device repairs it.

### Generated fixed-X centering limitation

Automatic fixed-X construction centers X but can generate Xk with nonzero
column means. The `corr_diff`, `ols_coef_diff`, and native non-knockpy Lasso
statistics center y. As a result, equality of raw Gram matrices need not survive
the response-centering projection P=I−11ᵀ/n: XᵀPX and XkᵀPXk can differ.
This breaks the usual null score-pair exchangeability argument independently
of threshold ties, including with full rank and n>2p. Increasing n alone does
not repair the current construction; avoiding ties is not enough to claim
nominal FDR control for automatically generated fixed-X output.

For example, the input X=[−1,1]ᵀ has Gram 2. After unit-norm standardization,
the working design is X_work=[−1,1]ᵀ/√2. With construction seed 7, its generated
knockoff is approximately [0.707,0.707]ᵀ. Both unprojected working Grams equal 1,
but the projected Grams are 1 and 0. The correlation-difference statistic is
positive for either sign of every nonconstant centered response. This illustrates lost null sign
symmetry, not a measured FDR exceedance: one-feature knockoff+ selects nothing.

A genuinely centered, valid externally supplied matched X/Xk pair can avoid
this geometry defect. Check the full matched-pair Gram conditions after any
intercept/nuisance projection, not just equal shapes or marginal norms. The
[centered orthogonal QR example](../reference/feature-selection-api.md#repeated-lasso-statistic-calls)
constructs a special design with n≥2p+1. It is not a repair for arbitrary X;
centering a generated Xk afterward can itself destroy its Gram constraints.
Response assumptions, statistic validity, threshold ties and cache limitations
still matter when a supplied pair has valid geometry.

### Response-model assumptions

The cited [fixed-X finite-sample result](https://arxiv.org/html/1404.5609v3)
assumes a Gaussian linear response, y=b1+Xβ+ε with independent homoskedastic
errors ε∼N(0,σ²I), plus a compatible knockoff design/statistic and nuisance
intercept treatment. Accepting finite y does not establish these assumptions.

[Model-X theory](https://arxiv.org/abs/1610.02351) instead requires feature-pair
exchangeability and knockoffs independent of y conditional on X; the response
relationship can be arbitrary. It is not simply a small-n substitute with the
same assumptions. Here the estimated Gaussian feature model and multi-draw
averaging retain the limitations stated below.

## Covariance/Inference

This method does not report coefficient covariance tables. Its inferential target is selection-based FDR control, subject to the stated assumptions and limitations:
- fixed-X construction usually requires `n >= 2p` and full column rank; these conditions alone do not ensure response-centering compatibility or the Gaussian response assumptions above.
- model-X path uses covariance estimation plus S-matrix construction; optional multi-draw averaging is supported.
- `compat_mode="knockpy"` exposes covariance/S-matrix controls, but missing optional packages or S-matrix solver errors can substitute sample covariance or an equicorrelated S matrix. Inspect `metadata["modelx_covariance_estimator"]` and `metadata["modelx_smatrix_source"]`; the requested method name does not establish what ran. See [compatibility resolution](../reference/feature-selection-api.md#compatibility-resolution-and-fallbacks).

## Parameters

Key `knockoff_filter` parameters:

| Parameter | Default | Description |
|---|---:|---|
| `knockoff_type` | `fixed_x` | `fixed_x` or `model_x` |
| `q` | `0.1` | Finite target FDR in `(0, 1)`; validate before calling because NaN currently passes the internal check. |
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

The following optional GPU variants assume `X`/`y` generated in the [self-contained example](../reference/feature-selection-api.md#runnable-fixed-x-example). They require installed usable CUDA backends. Here `backend="torch"` selects the Torch library: NumPy or Torch CPU inputs stay on CPU. Supply CUDA tensors, including any supplied Xk on the same device, for GPU execution. This differs from estimator `device="torch"`, which explicitly requests CUDA.

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

## Repeated Lasso-statistic calls

With `method="lasso_coef_diff"` and an integer `random_state`, changing X, y,
or Xk in place can silently reuse earlier statistics. A fresh selector object
alone does not prevent this. The same risk exists when memory from earlier
inputs is reused. For repeatable changed-data analyses, run each call in a fresh
Python process. With supplied float64 NumPy X/y/Xk, another option is to pass
fresh copies of all three arrays and keep every previous input alive and unchanged for
the duration of the analysis. Setting `random_state=None` avoids the seeded
reuse but gives up seed-based repeatability. See the [detailed limitation and
safe copy example](../reference/feature-selection-api.md#repeated-lasso-statistic-calls).

## Choosing a Lasso computational profile

`lasso_fast_profile="off"` keeps the default statistic-fitting settings.
`auto`, `moderate` and `aggressive` can change CV folds, candidate penalties,
iteration budgets or tolerance. They can therefore change W and the selected
features, not merely runtime. Choose the profile before inspecting discoveries
and hold it fixed when comparing backends or implementations.

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

An empty `selected_features` array is a valid outcome. `estimated_fdr` is the threshold-rule estimate, not an observed error fraction or a per-feature p-value. Inspect `W` and `threshold` together, check the generated-design centering and tied-statistic restrictions above, and evaluate any downstream prediction model on data not used for selection.

## FAQ

- Why do I get an error for fixed-X? Check constraints (finite `q` in `(0,1)`, `X` is 2D, `Xk` shape matches `X`, and fixed-X rank/sample requirements are met). Validate q yourself: NaN currently produces an invalid empty selection instead of an error.
- When should I use model-X? Use it when a credible feature-distribution construction is available, including settings where fixed-X is infeasible. Its feature assumptions differ from fixed-X response assumptions; choosing it alone does not validate an estimated feature model.
- Is CuPy required for GPU? Yes, `backend="cupy"` requires CuPy in the environment.

## External Validation

- `dev/benchmarks/benchmark_knockoff_fixedx.py`
- `dev/benchmarks/benchmark_knockoff_vs_baselines.py`
- `dev/benchmarks/benchmark_knockoff_same_xk_parity.py`
- Result artifacts are stored under `results/benchmark_knockoff_*.json`.

## References

- Barber, R. F., & Candes, E. J. (2015). Controlling the false discovery rate via knockoffs. *Annals of Statistics*, 43(5), 2055-2085. [https://doi.org/10.1214/15-AOS1337](https://doi.org/10.1214/15-AOS1337)
- Candes, E., Fan, Y., Janson, L., & Lv, J. (2018). Panning for gold: Model-X knockoffs for high-dimensional controlled variable selection. *Journal of the Royal Statistical Society: Series B*, 80(3), 551-577. [https://doi.org/10.1111/rssb.12265](https://doi.org/10.1111/rssb.12265)

