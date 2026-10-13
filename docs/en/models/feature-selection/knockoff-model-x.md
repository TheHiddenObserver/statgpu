# Model-X Knockoff Selection

> Language: English  
> Last updated: 2026-10-13  
> This page: Model-X tutorial  
> Switch: [Chinese](../../../cn/models/feature-selection/knockoff-model-x.md)

## Select variables when you can model their distribution

Model-X knockoffs ask which features carry information about y beyond the
other features. The null for feature j is conditional independence of y and
X_j given the remaining features. Each original feature competes against an
artificial counterpart; positive W favors the original. The goal is false
discovery rate (FDR) control, not prediction error or per-feature p-values.

The response may be nonlinear, discrete or otherwise non-Gaussian. In return,
you need to justify the knockoff construction for the joint feature distribution.
This tutorial explains that requirement and demonstrates it with a known
independent Gaussian feature law and a nonlinear response.

[Fixed-X knockoffs](knockoff-fixed-x.md) rely on a different assumption: a
Gaussian linear response model with a valid matched design. Sample size alone
does not determine which approach to use. See the [overview](knockoff.md) for
method choice and the [feature-selection API reference](../../reference/feature-selection-api.md)
for all functions, parameters, methods and result fields.

## What makes a Model-X knockoff valid?

Two requirements define the construction:

1. Swapping any subset of feature pairs X_j and Xk_j leaves the joint
   distribution of (X, Xk) unchanged. This is pairwise exchangeability.
2. Xk is independent of y conditional on X. Construct the knockoffs without
   using information from the response.

These are distributional requirements, not equality tests on one sample's Gram
matrices. Matching shapes, sample covariances or devices does not prove them.
A compatible antisymmetric statistic and valid threshold rule are needed too.

The CPU example uses a known feature law: every row and coordinate of X is an
independent standard Gaussian. A fresh matrix with independent draws from that
same law is a valid Xk here: swapping any pairs leaves all the independent
standard Gaussian entries identically distributed. The response is generated
from X and independent noise, so the second requirement also holds.

Independent resampling is not a general knockoff generator for correlated or
unknown features. Even an independent copy of an entire correlated feature
vector need not remain exchangeable under individual-coordinate swaps. Do not
replace your observed data with this simulation or standardize X and Xk
separately using their sample moments. For real data, justify a construction
for the actual feature law.

## From feature competition to a selection rule

For the supplied matrices in these examples, `corr_diff` computes
W_j=|X_jᵀ(y−ȳ)|−|Xk_jᵀ(y−ȳ)|, where X_j and Xk_j are the original and
knockoff columns and ȳ is the sample response mean. These are differences of
absolute inner products with the centered response, not normalized correlation
coefficients or p-values. A positive W_j favors the original feature; its
magnitude depends on the input scale and sample size.

The intended knockoff+ rule checks each distinct positive value of |W| as a
candidate T. Count **all** features with W≤−T and **all** with W≥T, including
every tie, and form (1+#{W≤−T})/max(1,#{W≥T}). Choose the smallest T whose
ratio is at most the prespecified FDR target q, then select the original features with W≥T. If no candidate
qualifies, select none. The extra 1 is the knockoff+ offset; offset-0 `knockoff`
removes it and has a different modified-FDR target. This defines the rule whose
current implementation limitations are discussed next; it does not resolve them.

## Before the first call: choose q and check threshold limitations

Prespecify the hypothesis family, q and statistic. For knockoff+, the minimum
possible threshold ratio is 1/p for p candidate features; q=0.20 needs at least five positive discoveries
at an eligible threshold. Twenty features make that possible, not certain.
Do not change q after seeing the selection to get more discoveries.

Before any filter call or selector fit, check
`np.isfinite(q) and 0 < q < 1`. Current validation misses `q=np.nan` and may
return an invalid empty selection with `threshold=inf`, `estimated_fdr=0.0`
or an all-false mask under either threshold rule.

<a id="tied-statistic-limitation"></a>

The current threshold implementation can mishandle tied absolute W values:
it counts a partial tied group, then selects the whole threshold group. For
W=[8,8,-8] and q=0.5, it can report 0.5 when the full knockoff+ ratio is
(1+1)/2=1. Do not interpret those outputs as nominal FDR control. Offset-0
`knockoff` is affected too. More draws, a refit or a different device do not fix
this counting problem. The example below has no tied absolute W values; see
the [shared threshold explanation](knockoff.md#tied-statistic-limitation).

## A complete CPU example with a known Gaussian feature law

Run these blocks in order. This simulation keeps the feature law known and
supplies Xk explicitly, so no covariance estimation is involved. Both matrices
have shape `(400, 20)`; y has 400 responses. The sine term makes the response
nonlinear, while a linear component gives the chosen marginal statistic power.

<!-- learner-example: knockoff-model-x -->
```python
import numpy as np
from statgpu import model_x_knockoff_filter

n, p = 400, 20
q = 0.20
if not (np.isfinite(q) and 0 < q < 1):
    raise ValueError("q must be finite and strictly between 0 and 1")
```

### Generate X and an independent knockoff from the known law

Draw both matrices before y. Every entry has the same standard Gaussian law;
neither matrix is centered or rescaled using sample estimates.

```python
rng = np.random.default_rng(73)
X = rng.normal(size=(n, p))
Xk = rng.normal(size=(n, p))
```

### Generate a nonlinear response

Only columns 0–5 enter the response. Its noise is independent of both matrices.
Gaussian noise is convenient for this simulation, but is not a Model-X response
requirement.

```python
beta = np.zeros(p)
beta[:6] = [3, -2.5, 2, -1.5, 1.2, -1]
y = 2.0 + X @ beta + 2 * np.sin(X[:, 0]) + rng.normal(scale=1.0, size=n)
```

### Select once using the supplied matrix

The requested `modelx_draws=7` deliberately illustrates the supplied-Xk rule:
there is only one supplied matrix, so the function evaluates it once rather
than creating seven new knockoffs. Read the actual count from the metadata.

```python
result = model_x_knockoff_filter(
    X, y, Xk=Xk, q=q, method="corr_diff",
    fdr_control="knockoff_plus", backend="numpy", modelx_draws=7,
)
print("Selected columns:", result.selected_features.tolist())
print("Threshold:", round(result.threshold, 3))
print("Threshold ratio:", round(result.estimated_fdr, 3))
print("Knockoff source:", result.metadata["xk_source"])
print("Actual draws:", result.metadata["n_modelx_draws"])
```

### Inspect the complete threshold counts

Use unrounded W and T for the counts. Checking all members of each threshold
group is important; the displayed decimal threshold is only for reading.

```python
positive_count = int(np.count_nonzero(result.W >= result.threshold))
negative_count = int(np.count_nonzero(result.W <= -result.threshold))
print("Positive / negative counts:", positive_count, negative_count)
print("W[0] / W[17]:", round(result.W[0], 3), round(result.W[17], 3))
assert np.unique(np.abs(result.W)).size == p
```
<!-- example-end: knockoff-model-x -->

## Interpret W, the threshold and the metadata

The output is `Selected columns: [0, 1, 2, 3, 4, 5]`, threshold `173.092`,
threshold ratio `0.167`, source `provided`, and actual draws `1`. The positive
and negative counts are `6 0`; W[0] and W[17] are approximately `1541.746` and
`-90.694`. The first favors an original signal, the second its knockoff.

For this statistic, W_j=|X_jᵀ(y−ȳ)|−|Xk_jᵀ(y−ȳ)|. These are differences of
absolute response-centered inner products, not normalized correlation
coefficients; their scale depends on the input scale and sample size. The six
large positive W values pass the knockoff+ ratio (1+0)/6. They happen to be
exactly the simulated signals in this run. The target q does not promise
perfect recovery, control the realized false-discovery fraction in every sample,
or establish validity from one successful simulation.

Knockoff+ selects the smallest positive |W| threshold with full ratio
(1+#{W≤−T})/max(1,#{W≥T})≤q. An empty selection means no threshold qualified;
`threshold=inf` and `estimated_fdr=0.0` are then reporting conventions, not proof
that all features are null. `estimated_fdr` is the threshold-rule estimate, not
a known error fraction or a p-value. Offset-0 `knockoff` has a different
modified-FDR target under the relevant theory.

The metadata confirms how the API used Xk, not its statistical validity. Here
that validity comes from the known independent feature law and the way y was
generated, not from the `provided` label or the absence of threshold ties.

## Apply the selection with KnockoffSelector

Continue with the same X, y, Xk and q. Choose `knockoff_type="model_x"`
explicitly because the unified selector defaults to fixed-X. Supply Xk to
`fit`, not to the constructor.

<!-- example-requires: knockoff-model-x -->
<!-- learner-example: knockoff-model-x-selector -->
```python
from statgpu import KnockoffSelector

selector = KnockoffSelector(
    knockoff_type="model_x", q=q, method="corr_diff",
    fdr_control="knockoff_plus", backend="numpy", modelx_draws=7,
)
assert selector.fit(X, y, Xk=Xk) is selector
support = selector.get_support()
selected = selector.transform(X[:10])
np.testing.assert_array_equal(selector.selected_features_, result.selected_features)
print("Support indices:", np.flatnonzero(support).tolist())
print("Transformed shape:", selected.shape)
```
<!-- example-end: knockoff-model-x-selector -->

The support indices are `[0, 1, 2, 3, 4, 5]` and the transformed shape is
`(10, 6)`. Diagnostics are in `selector.result_`; no prediction model has been
fitted. These rows illustrate the mask only, not held-out performance. For
prediction, do selection within each training fold, fit a separate model on
the selected training columns, then apply that mask to untouched evaluation
rows. A failed refit may leave a previous successful selection accessible;
use a fresh selector and complete a successful fit before using new-data results.

## What changes when Xk is generated automatically?

With `Xk=None`, the native Model-X path standardizes the observed features,
estimates a regularized Gaussian covariance and constructs Gaussian knockoffs.
`modelx_covariance_shrinkage` and `modelx_s_scale` affect this construction.
Estimated Gaussian moments do not give an exact exchangeability guarantee for
arbitrary feature distributions, or automatically recover the known-law
construction used above. A larger sample does not by itself certify the model.

For generated knockoffs, `modelx_draws` controls repeated construction and
statistic calculation. W is averaged across draws and the threshold is applied
once to that average; this is not a vote over selected sets. With `None`, the
current defaults are three draws for `corr_diff` and five for OLS/Lasso
coefficient differences. A supplied Xk uses one evaluation regardless of a
larger requested count. Inspect `metadata["n_modelx_draws"]` and
`metadata["xk_source"]` for what happened.

Averaging may reduce Monte Carlo variation, at extra cost, but does not by
itself establish an FDR theorem for the averaged statistic or repair an invalid
feature model. It also does not repair tied-threshold counts.

### Compatibility construction and metadata

`compat_mode="knockpy"` changes construction/statistic conventions and can
involve CPU work and optional libraries. With built-in generated Model-X
knockoffs, inspect `metadata["modelx_covariance_estimator"]` and
`metadata["modelx_smatrix_source"]`. The latter distinguishes `knockpy` from
`equicorrelated_fallback`; the requested `modelx_smatrix_method` label does not
prove that method executed. A missing package or an exception from the
S-matrix call, including an invalid method request, can trigger the fallback.
Covariance choices can also fall back, including `mle_fallback_no_sklearn` when
sklearn is unavailable. These labels do not certify knockoff validity or package
parity. See [compatibility resolution](../../reference/feature-selection-api.md#compatibility-resolution-and-fallbacks)
for the exact alternatives. Optional `knockpy_sampler` dispatch is used only
with `compat_mode="knockpy"` and `Xk=None`; other paths ignore it. Those dispatch
targets are currently unsupported and raise `NotImplementedError`. Leave that
setting at `None` for built-in construction.

## Choose statistics and computation deliberately

`corr_diff` is inexpensive but can miss features whose response relationship
has weak marginal correlation, such as a purely symmetric quadratic effect.
Allowing nonlinear responses does not make every statistic powerful against
every alternative. OLS/Lasso coefficient differences use joint fits but do not
validate an estimated feature law. Choose the statistic and any
`lasso_fast_profile` before inspecting discoveries. Faster profiles can change
CV folds, candidate penalties, iteration budgets and tolerance, changing W and
selected features as well as runtime.

### Lasso cache limitation

Seeded `lasso_coef_diff` can reuse stale statistics after in-place input changes
or earlier input memory is reused. A new selector does not isolate that cache.
Use a fresh Python process for repeatable changed-data analyses. For supplied
float64 NumPy X/y/Xk, fresh copies of all three also avoid stale reuse if every
previous input remains alive and unchanged; prefer process isolation for
automatically generated Xk. `random_state=None` disables seeded reuse but loses
seed-based repeatability. See the
[full Lasso warning](../../reference/feature-selection-api.md#repeated-lasso-statistic-calls).

### Torch construction and Lasso are different device choices

`backend="torch"` chooses the library, not CUDA. Native generated Model-X
construction follows X.device, including CPU and a nondefault CUDA index;
keep X/y/Xk on the same device for native statistics. However, native Torch
`lasso_coef_diff` tuning/fitting requests CUDA even with CPU inputs or supplied
Xk, and does not guarantee use of the input's GPU index. Explicit
`lasso_cv_impl="statgpu"` does not remove this restriction. For Torch CPU
statistics, use `corr_diff` or `ols_coef_diff` when appropriate; for CPU Lasso,
use NumPy inputs and `backend="numpy"`. See
[Torch placement](../../reference/feature-selection-api.md#torch-device-placement)
and [Lasso routing](../../reference/feature-selection-api.md#torch-lasso-device-routing).

<a id="reproducibility-of-generated-torch-model-x"></a>

### Reproducibility of generated Torch model-X

For native generated knockoffs (`compat_mode="statgpu"`, `Xk=None`), an integer
`random_state` gives repeatable Torch construction without advancing the global
Torch RNG, within the same backend, dtype, device and software environment.
This also applies through the unified filter and selector. It is neither a
cross-backend/cross-GPU equality guarantee nor a guarantee of empirical FDR
control; check statistic
repeatability separately, especially for Lasso.

With `random_state=None`, native Torch/CuPy construction uses seed 0 for each
draw, so repeated draws use the same construction noise. Use an integer for
repeatable, separately seeded draws. NumPy construction with `None` remains
unseeded. A supplied, externally validated Model-X Xk bypasses random
construction; the caller controls its randomness, as in the CPU example.
Validation must address exchangeability and conditional independence from y
given X, not merely shape or device agreement. Fixed-X has its own matched-design
and response assumptions. These seed rules are separate from the Lasso cache
limitation above. See the
[seed contract](../../reference/feature-selection-api.md#seeded-torch-model-x-construction).

## References and next steps

- Candès, E., Fan, Y., Janson, L., & Lv, J. (2018). Panning for gold: Model-X
  knockoffs for high-dimensional controlled variable selection. *JRSS B*,
  80(3), 551–577. [Paper](https://doi.org/10.1111/rssb.12265).
- [Fixed-X tutorial](knockoff-fixed-x.md) for matched-design geometry and a
  Gaussian linear response model.
- [Complete feature-selection API](../../reference/feature-selection-api.md).
- [Optional contributor validation guide](../../../../dev/references/model-validation.md#knockoff).
