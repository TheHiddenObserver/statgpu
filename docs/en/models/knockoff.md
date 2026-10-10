# Knockoff Feature Selection

> Language: English  
> Last updated: 2026-10-09\
> This page: Method documentation  
> Switch: [Chinese](../../cn/models/knockoff.md)

<a id="overview"></a>

## What problem does knockoff selection solve?

Suppose you want to identify which predictors contribute to a response while
limiting false discoveries. False discovery rate (FDR) is the expected fraction
of selected features that are null, counting an empty selection as zero. It is
not the probability that every selected feature is correct.

Knockoffs act as matched negative controls: each feature competes against an
artificial counterpart with a similar dependence structure. A large positive
statistic W means the original feature outscored its knockoff. Comparing positive
and negative statistics supplies a threshold for selecting original columns.

Use fixed-X when a valid matched design and a Gaussian linear response model
with independent, equal-variance normal errors are credible. Model-X instead
requires a credible feature-distribution construction and permits arbitrary
response relationships. The current automatic constructions and threshold ties
have important [limitations](#tied-statistic-limitation); choosing an API path
does not verify its assumptions. For prediction-focused subset search, compare
[stepwise selection](feature-selection.md) and validate on held-out data.

Complete function/selector signatures, all parameters and result fields are in
the [feature-selection API reference](../reference/feature-selection-api.md).

<a id="validate-the-target-rate"></a>

## Plan the target rate before selection

Choose q and the statistic before looking at discoveries. For knockoff+, even
the most favorable threshold ratio cannot be smaller than 1/p. A target of
q=0.20 therefore requires at least five positive discoveries at an eligible
threshold; having 20 predictors makes this possible, but does not promise a
nonempty result. Choose q for the acceptable error rate, not to obtain a desired
number of features.

Before any filter call or selector fit, check `np.isfinite(q) and 0 < q < 1`.
This applies to all three filter functions, both selector classes, and both
threshold rules. The current validation misses `q=np.nan`: it can return an
empty selection with `threshold=inf` and `estimated_fdr=0.0`, or an all-false
selector mask. That output is invalid, not evidence of no discoveries at a
meaningful target rate. Reject nonfinite q yourself; do not replace it with a
new target after inspecting results.

<a id="centered-pair-example"></a>

<a id="a-complete-cpu-example"></a>

## A complete CPU example with a valid supplied pair

This controlled simulation creates X and Xk together, before generating y.
QR orthogonalization reserves its first column for the intercept; the remaining
columns form two centered, mutually orthogonal groups. Thus XᵀX=XkᵀXk=I and
XᵀXk=0, satisfying the fixed-X matched-pair conditions with S=I even after the
intercept projection. The construction needs n≥2p+1.

This is a special simulation design, not a way to repair arbitrary observed X.
For your own data, obtain and validate a matched X/Xk pair for that design,
including after any intercept/nuisance projection. Do not replace measured
predictors with this QR design or simply center an automatically generated Xk;
that can change the question or destroy the required Gram constraints. Passing
`Xk` bypasses automatic construction, which currently has a
[centering limitation](#generated-fixed-x-centering-limitation).

Here p=20, q=0.20, six nonzero coefficients, and `corr_diff` are chosen in
advance. The response has an intercept and independent Gaussian noise with
standard deviation 0.5, matching the fixed-X response assumptions.

Run these blocks in order. `X` and its supplied knockoff `Xk` will each have
shape `(240, 20)`; `y` will contain one response for each of the 240 rows.

<!-- learner-example: knockoff-selection -->
```python
import numpy as np
from statgpu import fixed_x_knockoff_filter
```

### Prespecify the design and target

Set the dimensions and target rate before generating any responses. The call
below will use the prespecified `corr_diff` statistic and knockoff+ rule.

```python
n, p = 240, 20
q = 0.20
if not (np.isfinite(q) and 0 < q < 1):
    raise ValueError("q must be finite and strictly between 0 and 1")
```

### Construct and verify the supplied pair

Reserve the constant direction for the intercept, then take two groups of
orthogonal columns. This is the special simulation design described above.

```python
rng = np.random.default_rng(42)
Q, _ = np.linalg.qr(np.column_stack([np.ones(n), rng.normal(size=(n, 2*p))]))
X, Xk = Q[:, 1:p+1], Q[:, p+1:2*p+1]
```

Check centering and the three Gram constraints before using the pair. These
checks establish the design properties for this simulation, not for arbitrary data.

```python
np.testing.assert_allclose(X.mean(axis=0), 0, atol=1e-14)
np.testing.assert_allclose(Xk.mean(axis=0), 0, atol=1e-14)
np.testing.assert_allclose(X.T @ X, np.eye(p), atol=1e-14)
np.testing.assert_allclose(Xk.T @ Xk, np.eye(p), atol=1e-14)
np.testing.assert_allclose(X.T @ Xk, 0, atol=1e-14)
```

### Generate the response and select features

Only the first six original columns affect the simulated response. Generate y
after the pair, with independent, equal-variance Gaussian noise.

```python
beta = np.zeros(p)
beta[:6] = [8, -7, 6, -5, 4, -3]
y = 2.0 + X @ beta + rng.normal(scale=0.5, size=n)
```

Pass both matrices explicitly to bypass automatic construction.

```python
result = fixed_x_knockoff_filter(
    X, y, Xk=Xk, q=q, method="corr_diff",
    fdr_control="knockoff_plus", backend="numpy",
)
```

Inspect the selected original-column indices and the threshold quantities; the
next section explains what they do and do not imply.

```python
print("Selected columns:", result.selected_features.tolist())
print("Threshold:", round(result.threshold, 3))
print("Threshold ratio:", round(result.estimated_fdr, 3))
```
<!-- example-end: knockoff-selection -->

## Read the selection and its uncertainty

For this seed, the output is `Selected columns: [0, 1, 2, 3, 4, 5, 15]`,
threshold `1.019`, and threshold ratio `0.143`. Indices are zero-based original
columns. We know the simulated truth: columns 0–5 are signals and column 15
is a false discovery. A target q does not promise that every selection is correct
or bound the false-discovery fraction in each individual sample.

For `corr_diff`, W_j=|X_jᵀ(y−ȳ)|−|Xk_jᵀ(y−ȳ)|. For example,
`result.W[0]` is about 6.386, favoring the original signal, whereas
`result.W[11]` is about −0.921, favoring its knockoff. This run has no tied
absolute statistics. At the selected threshold, seven W values are positive
and large enough, with none as negative as −T; the knockoff+ ratio is
(1+0)/7. `estimated_fdr` reports that rule's estimate, not the actual fraction
of null features in this sample or a per-feature p-value. Other samples can miss
signals or select more noise; one simulated result does not demonstrate FDR
control empirically.

An empty selection is also a valid outcome for valid inputs: no threshold met
the rule, so `threshold` is infinity and the implementation reports
`estimated_fdr=0.0`. That does not establish that all predictors are null.
Keep the prespecified q and investigate assumptions and power rather than
loosening q after seeing the result. The API reference gives a
[small example where emptiness is inevitable](../reference/feature-selection-api.md#runnable-fixed-x-example).

This filter does not fit a coefficient or prediction model. For prediction,
perform selection within each training fold, fit a separate model on its
selected columns, and keep evaluation rows untouched.

<a id="objective-function"></a>

## How the threshold works

The statistical goal is false discovery rate control at target `q`, subject to the response/construction assumptions and both the centering and tied-statistic limitations below:
- Build knockoff variables \(\tilde X\) that mirror dependence structure.
- Compute antisymmetric statistics \(W_j\) (for example correlation or coefficient differences).
- Select features with \(W_j\) above knockoff threshold.

<a id="estimating-equation"></a>

### The knockoff+ rule

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
[centered matched-pair example](#centered-pair-example)
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

The following optional GPU variants reuse X, y, Xk and q from the
[centered-pair CPU example](#centered-pair-example). They keep its supplied
fixed-X pair, statistic and target unchanged and require usable CUDA backends.
The QR pair is a fixed-X example, not an exchangeability construction for model-X.
For model-X, choose a construction appropriate to the feature distribution.

Here `backend="torch"` selects the Torch library, unlike estimator `device="torch"`, which requests CUDA. Native fixed-X and generated model-X construction follow X's device. Torch CPU input remains on CPU even when CUDA is available; CUDA input retains its GPU index. See [Torch device placement](../reference/feature-selection-api.md#torch-device-placement).

This device preservation applies to construction. Native Torch
`method="lasso_coef_diff"` tuning/fitting still requests CUDA, including for
CPU inputs, supplied Xk and fixed-X; it does not guarantee use of an input's
nondefault GPU. For Torch CPU statistics, choose `corr_diff` or `ols_coef_diff`
when appropriate; for CPU Lasso, use NumPy inputs with `backend="numpy"`.
See [Torch Lasso device routing](../reference/feature-selection-api.md#torch-lasso-device-routing).

For CPU model-X construction, use Torch CPU tensors with `backend="torch"`, or NumPy X/y with `backend="numpy"`. Keep X/y/Xk on the same intended device. A supplied, externally validated model-X Xk bypasses construction. Shape and device agreement do not establish exchangeability or conditional independence from y given X. These native construction rules apply to `compat_mode="statgpu"` with `Xk=None`, including `KnockoffSelector`; fixed-X has different construction and statistical assumptions.

Choose one backend below; both blocks also reuse `fixed_x_knockoff_filter`
from the CPU example.

### CuPy

```python
import cupy as cp

X_gpu, y_gpu, Xk_gpu = cp.asarray(X), cp.asarray(y), cp.asarray(Xk)
res_gpu = fixed_x_knockoff_filter(
    X_gpu, y_gpu, Xk=Xk_gpu, q=q, method="corr_diff",
    fdr_control="knockoff_plus", backend="cupy",
)
```

### Torch CUDA

```python
import torch

X_torch, y_torch, Xk_torch = [torch.from_numpy(a).to("cuda") for a in (X, y, Xk)]
res_torch = fixed_x_knockoff_filter(
    X_torch, y_torch, Xk=Xk_torch, q=q, method="corr_diff",
    fdr_control="knockoff_plus", backend="torch",
)
```

## Threshold rules and construction assumptions

- `fdr_control="knockoff_plus"` is the stricter, more conservative option and default.
- `fdr_control="knockoff"` uses offset 0 and has a different modified-FDR target under the relevant theory; it is not an approximate numerical version of knockoff+.
- In model-X, additional `modelx_draws` average feature statistics at greater computational cost. Reduced Monte Carlo variation does not establish an FDR theorem for the averaged statistic; an estimated Gaussian feature model also does not guarantee exchangeability for arbitrary feature distributions.
- `knockpy_sampler` dispatch options are currently guarded; explicitly setting unsupported targets can raise `NotImplementedError` instead of silently falling back.

## Reproducibility of generated Torch model-X

For `knockoff_type="model_x"`, `compat_mode="statgpu"`, and `Xk=None`, an integer
`random_state` controls construction randomness without advancing the global
Torch RNG. Repeating construction with the same inputs,
settings, backend, dtype, device and software environment is repeatable.
This also applies to `KnockoffSelector`. It does not guarantee cross-backend
or cross-GPU equality or empirical FDR control, and is separate from the
Lasso cache limitation below.

Native Torch/CuPy construction with `random_state=None` uses seed 0 for each
draw, so repeated draws use the same construction noise; use an integer
for repeatable, separately seeded draws. NumPy
construction with `None` remains unseeded. A valid externally generated `Xk`
bypasses construction; check the chosen statistic's repeatability separately.
A supplied matrix does not remove the statistical, threshold or Lasso-cache
limitations. Fixed-X keeps its own construction and statistical assumptions.

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
- Is CuPy required for GPU? `backend="cupy"` requires CuPy. Alternatively, `backend="torch"` accepts Torch CUDA tensors; keep inputs on the same device and follow the [Torch device-placement guidance](../reference/feature-selection-api.md#torch-device-placement). Installing Torch alone does not guarantee CUDA execution.

## Advanced reference

<a id="path"></a>

### Import paths

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

<a id="external-validation"></a>

For contributors reproducing implementation comparisons, see the optional
[Knockoff validation guide](../../../dev/references/model-validation.md#knockoff).

## References

- Barber, R. F., & Candes, E. J. (2015). Controlling the false discovery rate via knockoffs. *Annals of Statistics*, 43(5), 2055-2085. [https://doi.org/10.1214/15-AOS1337](https://doi.org/10.1214/15-AOS1337)
- Candes, E., Fan, Y., Janson, L., & Lv, J. (2018). Panning for gold: Model-X knockoffs for high-dimensional controlled variable selection. *Journal of the Royal Statistical Society: Series B*, 80(3), 551-577. [https://doi.org/10.1111/rssb.12265](https://doi.org/10.1111/rssb.12265)

