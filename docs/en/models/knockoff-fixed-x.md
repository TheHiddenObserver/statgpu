# Fixed-X Knockoff Selection

> Language: English  
> Last updated: 2026-10-13  
> This page: Fixed-X tutorial  
> Switch: [Chinese](../../cn/models/knockoff-fixed-x.md)

## Select variables under a Gaussian linear response model

Fixed-X knockoffs compare each observed predictor with an artificial control
while treating the design matrix X as fixed. The aim is to control the false
discovery rate (FDR), the expected fraction of null features among discoveries,
with an empty selection counted as zero. A large positive statistic W favors
the original feature over its control; it is not a coefficient or a p-value.

Use this approach when a valid matched design and a Gaussian linear response
model are credible. [Model-X knockoffs](knockoff-model-x.md) use different
assumptions about the feature distribution and can allow nonlinear responses.
The [overview](knockoff.md) compares the two approaches. All signatures,
parameters, selector methods and result fields are documented in the
[feature-selection API reference](../reference/feature-selection-api.md).

## What makes a fixed-X pair valid?

For the design actually used by the statistic, write Σ=XᵀX. A matched pair
must satisfy XkᵀXk=Σ and XᵀXk=Σ−S, where S is a nonnegative diagonal matrix
and the resulting joint Gram matrix is positive semidefinite. Matching shapes
or column norms is not enough: the cross-Gram condition matters too.

An intercept introduces the projection P=I−11ᵀ/n. Check the pair after this
projection: replace each Gram above by XᵀPX, XkᵀPXk or XᵀPXk, respectively.
Other nuisance adjustments likewise need compatible projected geometry. The
[response-model assumption](#response-model-assumptions) is

y=b1+Xβ+ε, with independent equal-variance errors ε∼N(0,σ²I).

A compatible statistic and threshold rule are also required. The built-in
construction generally needs full column rank and n≥2p, but those dimensions
alone do not resolve its [centering limitation](#generated-fixed-x-centering-limitation).
The example below instead supplies a pair whose geometry can be checked directly.

## Before the first call: choose q and check threshold limitations

Choose the hypothesis family, q and statistic before examining discoveries.
Knockoff+ has a minimum possible threshold ratio of 1/p: q=0.20 needs at least
five positive discoveries at an eligible threshold. It does not guarantee a
nonempty selection, even with 20 predictors.

Before any filter call or selector fit, validate
`np.isfinite(q) and 0 < q < 1`. Current validation misses `q=np.nan`, which can
produce an invalid empty selection with `threshold=inf`, `estimated_fdr=0.0`
and an all-false selector mask. This warning applies to both threshold rules.

<a id="tied-statistic-limitation"></a>

Tied absolute statistics need particular care. The current implementation can
count only part of a tied group before selecting the whole group. For
W=[8,8,-8] and q=0.5, its reported ratio can be 0.5 although the full knockoff+
ratio is (1+1)/2=1. Do not interpret such output as nominal FDR control. This
also affects offset-0 `knockoff`; changing device, refitting or adding draws
does not repair it. The example below has no tied absolute W values. See the
[shared threshold explanation](knockoff.md#tied-statistic-limitation).

<a id="centered-pair-example"></a>
<a id="a-complete-cpu-example"></a>

## A complete CPU example with a supplied pair

This controlled simulation creates X and Xk together, before generating y.
QR orthogonalization reserves its first column for the intercept; the remaining
columns form two centered, mutually orthogonal groups. Thus XᵀX=XkᵀXk=I and
XᵀXk=0, satisfying the matched-pair conditions with S=I even after intercept
projection. This particular construction needs n≥2p+1.

This is a special simulation design, not a way to repair arbitrary observed X.
For your own data, obtain and validate a matched X/Xk pair for the observed
design, including after any intercept/nuisance projection. Do not replace
measured predictors with this QR design or simply center an automatically
generated Xk: that can change the question or destroy the Gram constraints.
Passing `Xk` bypasses automatic construction.

Run these blocks in order. Both matrices have shape `(240, 20)` and y has 240
responses. We prespecify six nonzero coefficients and `corr_diff`; the response
has an intercept and independent Gaussian noise with standard deviation 0.5.

<!-- learner-example: knockoff-selection -->
```python
import numpy as np
from statgpu import fixed_x_knockoff_filter
```

### Prespecify the design and target

Set the dimensions and error rate before generating responses.

```python
n, p = 240, 20
q = 0.20
if not (np.isfinite(q) and 0 < q < 1):
    raise ValueError("q must be finite and strictly between 0 and 1")
```

### Construct and verify the supplied pair

Reserve the constant direction for the intercept, then take two groups of
orthogonal columns.

```python
rng = np.random.default_rng(42)
Q, _ = np.linalg.qr(np.column_stack([np.ones(n), rng.normal(size=(n, 2*p))]))
X, Xk = Q[:, 1:p+1], Q[:, p+1:2*p+1]
```

Check centering and all three Gram constraints before using the pair. Because
both matrices are centered, their projected and unprojected Grams agree.

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

Inspect the original-column indices and the threshold quantities.

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
is a false discovery. A target q neither promises that every discovery is
correct nor bounds the false-discovery fraction in each individual sample.

For `corr_diff`, W_j=|X_jᵀ(y−ȳ)|−|Xk_jᵀ(y−ȳ)|. For example,
`result.W[0]` is about 6.386, favoring an original signal, whereas
`result.W[11]` is about −0.921, favoring its knockoff. There are no tied absolute
statistics. At the unrounded selected threshold, seven W values meet W≥T and none meet W≤−T.
The knockoff+ ratio is therefore (1+0)/7. `estimated_fdr` reports this threshold
ratio, not the realized fraction of null features or a per-feature p-value.
One simulated result does not demonstrate FDR control empirically.

More generally, knockoff+ chooses the smallest positive |W| threshold whose
full ratio (1+#{W≤−T})/max(1,#{W≥T}) is at most q. If none qualifies, selection
is empty, `threshold` is infinity and `estimated_fdr=0.0` is the implementation's
empty-result convention. This does not establish that all predictors are null.
Keep the prespecified q rather than loosening it to get discoveries.
`fdr_control="knockoff"` removes the numerator's 1 and has a different
modified-FDR target under the applicable theory.

## Use the same result in a selector workflow

Continue with X, y, Xk and q from the CPU example. Supply `Xk` to `fit`, not to
the constructor. The selector stores the result and applies a column mask;
it does not fit a response-prediction model.

<!-- example-requires: knockoff-selection -->
<!-- learner-example: knockoff-fixed-x-selector -->
```python
from statgpu import FixedXKnockoffSelector

selector = FixedXKnockoffSelector(
    q=q, method="corr_diff", fdr_control="knockoff_plus", backend="numpy",
)
assert selector.fit(X, y, Xk=Xk) is selector
support = selector.get_support()
selected = selector.transform(X[:10])
np.testing.assert_array_equal(selector.selected_features_, result.selected_features)
print("Support indices:", np.flatnonzero(support).tolist())
print("Transformed shape:", selected.shape)
```
<!-- example-end: knockoff-fixed-x-selector -->

The support indices are `[0, 1, 2, 3, 4, 5, 15]` and the transformed shape is
`(10, 7)`. Read diagnostics through `selector.result_`. These ten rows only
illustrate column selection; they are not held-out evaluation. For prediction,
select within each training fold, fit a separate estimator on the selected
training columns, and apply the same mask to untouched evaluation data.

A failed refit can leave an earlier successful selection accessible. After a
fit error, do not treat that state as a result for the new data; complete a
successful fit with a fresh selector. See the
[selector contract](../reference/feature-selection-api.md#selector-methods).

## Choose the statistic before seeing discoveries

- `corr_diff` compares response-centered inner products. It is simple and
  inexpensive; weak marginal relationships can reduce power.
- `ols_coef_diff` compares absolute coefficients from the joint original/
  knockoff regression. Consider the joint design and its numerical conditioning.
- `lasso_coef_diff` uses a regularized joint fit. CV implementation and numerical
  settings can change W. Keep `lasso_fast_profile="off"` for an initial comparison;
  faster profiles can change CV folds, candidate penalties, iteration budgets,
  tolerances, W and selected features.

Seeded `lasso_coef_diff` calls can reuse stale statistics after in-place input
changes or reuse of earlier arrays' memory. A new selector alone is not enough.
For repeatable changed-data calls, use a fresh Python process. With supplied
float64 NumPy X/y/Xk, fresh copies of all three also work if every previous
input remains alive and unchanged. `random_state=None` disables seeded reuse
but gives up seed-based repeatability. See the
[complete Lasso warning and copy example](../reference/feature-selection-api.md#repeated-lasso-statistic-calls).

<a id="generated-fixed-x-centering-limitation"></a>

## Why automatic construction needs caution

Automatic fixed-X construction centers X but can generate Xk with nonzero
column means. `corr_diff`, `ols_coef_diff`, and native non-knockpy Lasso
statistics center y. Equality of raw Gram matrices can therefore disappear
after P=I−11ᵀ/n: XᵀPX and XkᵀPXk can differ. This breaks the usual null
score-pair exchangeability argument independently of threshold ties, including
with full rank and n>2p. Increasing n alone does not repair it.

The input X=[−1,1]ᵀ has Gram 2. After standardization, the working design is
X_work=[−1,1]ᵀ/√2. Construction
seed 7 gives a knockoff approximately [0.707,0.707]ᵀ. The unprojected working
Grams both equal 1, but the projected Grams are 1 and 0. Correlation difference
then favors X for either sign of any nonconstant centered response. This shows
lost null sign symmetry, not measured FDR exceedance: one-feature knockoff+
selects nothing.

A valid centered supplied pair can avoid this geometry defect. Check all
matched-pair conditions after the intercept/nuisance projection. Simply
post-centering a generated Xk can destroy its required Gram relationships.
The [QR example](#centered-pair-example) is a special simulation with n≥2p+1,
not a replacement for observed X or a general arbitrary-X repair. Shape checks,
valid raw Grams and an absence of ties do not by themselves establish FDR control.

## Optional GPU versions of the same example

These blocks reuse the supplied pair and q from the CPU example and require a
usable CUDA backend. Choose one; neither changes the statistical construction.

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

`backend="torch"` selects the library, unlike estimator `device="torch"`,
which requests CUDA. Native construction follows X's device; keep X/y/Xk on
the same device. This does not extend to native Torch `lasso_coef_diff` tuning
and fitting: it requests CUDA even for CPU inputs or supplied Xk, and does not
guarantee the input's nondefault GPU. For Torch CPU statistics, use `corr_diff`
or `ols_coef_diff` when appropriate; for CPU Lasso, use NumPy inputs and
`backend="numpy"`. See [Torch device placement](../reference/feature-selection-api.md#torch-device-placement)
and [Torch Lasso routing](../reference/feature-selection-api.md#torch-lasso-device-routing).
Compatibility settings can also involve CPU work; a backend label alone does
not establish that every step runs on GPU.

<a id="response-model-assumptions"></a>

## Statistical scope and references

The fixed-X finite-sample result assumes a Gaussian linear response with
independent homoskedastic normal errors, a valid matched design, compatible
statistic and appropriate intercept treatment. Merely accepting finite y does
not establish these assumptions. The output does not include coefficient
covariances, confidence intervals or prediction methods. For arbitrary response
relationships with a credible feature-distribution construction, read the
[Model-X tutorial](knockoff-model-x.md).

- Barber, R. F., & Candès, E. J. (2015). Controlling the false discovery rate via
  knockoffs. *Annals of Statistics*, 43(5), 2055–2085.
  [Paper](https://doi.org/10.1214/15-AOS1337).
- [Complete feature-selection API](../reference/feature-selection-api.md).
- [Optional contributor validation guide](../../../dev/references/model-validation.md#knockoff).
