# Knockoff Feature Selection

> Language: English  
> Last updated: 2026-10-13  
> This page: Principles and method choice  
> Switch: [Chinese](../../../cn/models/feature-selection/knockoff.md)

<a id="overview"></a>

## What problem does knockoff selection solve?

Suppose many predictors might help explain a response, but selecting noise as a
signal would be costly. Knockoffs compare each predictor with an artificial
negative control. The goal is to select relevant original columns while
controlling the false discovery rate (FDR), subject to the method's assumptions.

If R is the number selected and V is the number of null features selected, then
$$
\operatorname{FDR}=\mathbb{E}\left[\frac{V}{\max(R,1)}\right].
$$
This is an average over repeated samples, counting an empty selection as zero.
It is not a per-feature correctness probability or a promise that every selected
feature is useful. A single sample can have a false-discovery fraction above q.

For prediction-focused subset search, compare [stepwise selection](README.md)
and measure held-out performance. Knockoff selection itself does not fit a
prediction model or provide coefficient p-values.

<a id="covarianceinference"></a>
<a id="faq"></a>
## Choose the assumptions before the API

| Approach | What must be credible? | Start here |
|---|---|---|
| Fixed-X | A matched design after nuisance/intercept projection, a compatible statistic, and a Gaussian linear response with independent equal-variance errors | [Fixed-X tutorial](knockoff-fixed-x.md) |
| Model-X | Feature-pair exchangeability and knockoffs independent of y conditional on X; the response relationship may be nonlinear | [Model-X tutorial](knockoff-model-x.md) |

The choice depends on which assumptions you can justify. Fixed-X uses a
matched design and a Gaussian linear response model; Model-X uses the joint
feature distribution and allows a broader response relationship. Both compute
knockoff statistics and apply a threshold. The distinction is statistical, not
a choice of speed or numerical precision, and a smaller sample alone is not a
reason to choose Model-X.

In particular, fitting a Gaussian distribution to arbitrary observed features
does not establish the exact Model-X conditions. Neither API checks all the
assumptions needed for FDR control.

### Response-model assumptions

Fixed-X uses y=b1+Xβ+ε with ε∼N(0,σ²I); a null feature has β_j=0 in that
model. Its design and statistic must remain
compatible with the intercept or other nuisance projection. Model-X instead
uses the feature distribution: swapping any subset of original/knockoff column
pairs must leave their joint law unchanged, and Xk must be independent of y
given X. Its null feature means y is conditionally independent of that feature
given all other original features. See the [sources](#references) and each
tutorial for the different constructions.

<a id="objective-function"></a>

## How the threshold works

1. Construct a valid paired matrix Xk without using response information in a way that violates the assumptions.
2. Compute one original-versus-knockoff statistic W_j per feature. Swapping a pair should reverse its sign; large positive values favor the original.
3. Use large negative statistics to estimate how often null original features might win the competition, choose a threshold T, and select original columns with W_j ≥ T.

For `corr_diff`, the statistic is the difference of absolute centered-response
inner products:
$$
W_j=|X_j^\mathsf{T}(y-\bar y)|-|Xk_j^\mathsf{T}(y-\bar y)|.
$$
It rewards marginal association and can miss pure nonlinear effects such as a
symmetric squared feature. `ols_coef_diff` and `lasso_coef_diff` use coefficient
magnitudes instead. A more elaborate statistic can change power, but cannot
repair invalid pairs or incorrect threshold counting.

<a id="validate-the-target-rate"></a>

## Plan the target rate before selection

Choose q, the hypothesis family, the statistic and any Lasso computational
profile before inspecting discoveries. Before every filter call or selector
fit, check `np.isfinite(q) and 0 < q < 1`. The current implementation misses
`q=np.nan` under both threshold rules and can return an invalid empty selection
with `threshold=inf` and `estimated_fdr=0.0`. Reject nonfinite q yourself.

For knockoff+, the smallest possible ratio is 1/p. With q=0.20, at least five
positive discoveries are needed at an eligible threshold; p=20 makes a nonempty
selection possible but does not ensure one. Do not increase q after seeing the
results or add unrelated predictors merely to obtain discoveries.

<a id="estimating-equation"></a>

### The knockoff+ rule

The theoretical rule considers the distinct positive values of |W| as candidate
thresholds:
$$
T = \min \left\{ t\in\{|W_j|:|W_j|>0\} : \frac{1+\#\{j:W_j\le -t\}}{\max(1,\#\{j:W_j\ge t\})}\le q \right\}.
$$
Count **all** features at each candidate threshold. If no threshold qualifies,
select none. `fdr_control="knockoff_plus"` uses the numerator offset 1 and is
the default. `fdr_control="knockoff"` uses offset 0 and has a different
modified-FDR target; it is not a numerical approximation to knockoff+.

### Tied-statistic limitation

The current implementation evaluates partially accumulated tied groups before
selecting the whole threshold group. For W=[8,8,-8] and q=0.5, it can report
threshold 8 and estimated_fdr=0.5, although the full threshold ratio is
(1+1)/2=1 and the theoretical rule selects none. Do not interpret tied-threshold
output as nominal knockoff+ FDR control. The same counting limitation affects
offset-0 `knockoff`. Extra draws, a coefficient refit or a different device do
not repair it. Check ties as well as construction validity.

<a id="outputs"></a>
## Read the selection and its uncertainty

`selected_features` contains zero-based original-column indices; `W` gives the
competition statistics and `threshold` the cutoff. `estimated_fdr` is the
threshold-rule estimate, not the realized false-discovery fraction or a
per-feature p-value. `q_trajectory` contains rank diagnostics, which do not
certify full tied-group counts. Inspect `metadata` to learn what construction
and draw count actually ran.

For valid inputs, an empty selection can mean that no threshold qualified.
Then `threshold` is infinity and the implementation reports
`estimated_fdr=0.0`; this does not prove that all predictors are null. The
[Fixed-X tutorial](knockoff-fixed-x.md) interprets a simulated selection with a
known false discovery. The [Model-X tutorial](knockoff-model-x.md) shows a
separate feature-distribution construction and a nonlinear response.

Selectors expose `result_`, `selected_features_`, `get_support()` and
`transform()`. Their `fit` returns self. They provide no prediction model:
re-select within each training fold, fit a separate estimator on the selected
training columns, and evaluate on untouched rows. After a failed refit, old
selection state can remain accessible; create a fresh selector and complete a
successful fit before treating its output as a result for the new data.

<a id="centered-pair-example"></a>
<a id="a-complete-cpu-example"></a>

<a id="construct-and-verify-the-supplied-pair"></a>
<a id="generate-the-response-and-select-features"></a>
<a id="prespecify-the-design-and-target"></a>
## A complete CPU example with a valid supplied pair

To put these principles into practice, follow the
[Fixed-X tutorial](knockoff-fixed-x.md): construct a controlled, centered QR
pair, verify all matched-pair Gram conditions, generate a Gaussian response,
and apply the filter and selector. The n≥2p+1 requirement belongs to this
special simulation; the QR design must not replace measured predictors.
The separate [Model-X example](knockoff-model-x.md) starts from a known
independent Gaussian feature distribution and allows a nonlinear response.

### Generated fixed-X centering limitation

Automatic fixed-X construction centers X but can leave Xk with nonzero column
means. For response-centered statistics, raw Gram equality then need not survive
P=I−11ᵀ/n: XᵀPX and XkᵀPXk can differ, losing the usual null score-pair
symmetry even without ties or with n>2p. Increasing n alone does not repair it.
Do not claim nominal FDR control from raw Grams alone. A valid external pair
must satisfy the complete conditions after projection; merely centering the
generated Xk afterward can destroy its Gram constraints. The
[Fixed-X construction explanation](knockoff-fixed-x.md) gives the details and
an explicit counterexample.

## Threshold rules and construction assumptions

For generated model-X, estimated Gaussian moments need not match the true
feature law. Additional `modelx_draws` average W across constructions; reduced
Monte Carlo variation does not establish an FDR theorem for that average.
With supplied Xk, it is evaluated once rather than supplying fresh draws.
The [Model-X tutorial](knockoff-model-x.md) explains these boundaries and how to
inspect compatibility metadata. Explicit `knockpy_sampler` dispatch is not
implemented and can raise `NotImplementedError`; it does not add a general
feature-distribution sampler.

<a id="cupy"></a>
<a id="torch-cuda"></a>
## CPU+GPU Examples

Optional CuPy and Torch CUDA variants extend the supplied-pair workflow in the
[Fixed-X tutorial](knockoff-fixed-x.md). `backend="torch"` selects a library:
native construction follows X's device, including Torch CPU. Native Torch
Lasso tuning/fitting instead requests CUDA, even with CPU inputs or supplied
Xk, and does not guarantee an input's nondefault GPU. Use `corr_diff` or
`ols_coef_diff` for appropriate Torch CPU statistics; use NumPy inputs and
`backend="numpy"` for CPU Lasso. See [device placement](../../reference/feature-selection-api.md#torch-device-placement)
and [Lasso routing](../../reference/feature-selection-api.md#torch-lasso-device-routing).

## Reproducibility of generated Torch model-X

An integer `random_state` makes native generated Torch model-X construction
repeatable within the same backend, dtype, device and software environment,
without advancing the global Torch RNG. Native Torch/CuPy construction with
`random_state=None` uses seed 0 for each draw, while NumPy remains unseeded.
This does not promise cross-backend equality or FDR control. See
[construction RNG details](../../reference/feature-selection-api.md#seeded-torch-model-x-construction).

## Repeated Lasso-statistic calls

Seeded Lasso calls can reuse stale statistics after X, y or Xk is changed in
place, or after input memory is recycled. A fresh selector alone is not a
remedy. Use a fresh Python process for each changed-data analysis. The
[cache reference](../../reference/feature-selection-api.md#repeated-lasso-statistic-calls)
explains two alternatives and their conditions: copy all supplied float64 NumPy
inputs and retain the old arrays, or use `random_state=None` and give up
seed-based repeatability.

## Choosing a Lasso computational profile

`lasso_fast_profile="off"` keeps default settings. Other profiles can change
folds, candidate penalties, iteration budgets and tolerance, hence W and the
selection. Prespecify the profile and hold it fixed in comparisons; see the
[complete API reference](../../reference/feature-selection-api.md).

<a id="external-validation"></a>
<a id="import-paths"></a>
<a id="parameters"></a>
<a id="path"></a>
<a id="performance-boundary"></a>
## Advanced reference

For all signatures, parameters, methods, result fields, dtype/device behavior,
RNG, cache and fallback details, use the
[feature-selection API reference](../../reference/feature-selection-api.md).
In particular, requested compatibility covariance/S-matrix methods can fall
back; inspect actual execution metadata rather than the requested method name.
Large problems may be dominated by construction or Lasso fitting, and choosing
a GPU backend does not guarantee every compatibility step stays on GPU.

For implementation comparisons, see the optional
[Knockoff validation guide](../../../../dev/references/model-validation.md#knockoff).

## References

- Barber, R. F., & Candes, E. J. (2015). Controlling the false discovery rate via knockoffs. *Annals of Statistics*, 43(5), 2055–2085. [Paper](https://doi.org/10.1214/15-AOS1337)
- Candes, E., Fan, Y., Janson, L., & Lv, J. (2018). Panning for gold: Model-X knockoffs for high-dimensional controlled variable selection. *JRSS B*, 80(3), 551–577. [Paper](https://doi.org/10.1111/rssb.12265)
