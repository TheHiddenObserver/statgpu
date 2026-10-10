# Spline Basis Functions

> Language: English  
> Last updated: 2026-10-09
> This page: Model documentation  
> Switch: [Chinese](../../cn/models/splines.md)

## Turn a curved predictor effect into reusable features

A spline basis expands one numeric predictor into several overlapping, local
curves. A later regression combines these columns to learn a nonlinear effect;
the basis itself does not fit a response. Use `SplineTransformer` when training
and new observations must share the same knots and feature definition. Use
[GAM](semiparametric.md) when you also want an additive least-squares model and
automatic smoothing selection. More knots allow more detail, but do not by
themselves control overfitting.

Fit the transformer only on training rows, then call `transform` on validation
or test rows. In a cross-validation pipeline, learn knots inside each training
fold. Otherwise quantile knots can use information from held-out observations.

## Fit once, then reuse the same features

This small CPU example fits an ordinary least-squares response model on learned
spline features. It separates the feature transform from the response fit and
reuses the training knots at new points. Dropping one basis column with
`include_bias=False` leaves room for the explicit intercept in the later model.
The synthetic curve is noiseless; this demonstration is not a validation score.

<!-- example: spline-transformer-reuse-cpu -->
```python
import numpy as np
from statgpu.nonparametric.splines import SplineTransformer
```

### Prepare one feature and its response

`X_train` is a `(41, 1)` array with one time point per row; `y_train` has shape `(41,)` and holds the noiseless response. `[:, None]` preserves the two-dimensional observations-by-features layout. Run this section's blocks in order.

```python
X_train = np.linspace(-2.0, 2.0, 41)[:, None]
y_train = np.sin(1.5 * X_train[:, 0])
```

### Learn spline features

`fit_transform` learns knots from training points and expands the feature into spline basis columns; this step does not use `y_train`.

```python
transformer = SplineTransformer(
    n_knots=6, degree=3, include_bias=False,
    extrapolation="constant", device="cpu",
)
B_train = transformer.fit_transform(X_train)
```

### Fit a response model on those features

Add an intercept column to `B_train`, then estimate coefficients by least squares. Fitting the response is separate from constructing the features.

```python
train_design = np.column_stack([np.ones(len(X_train)), B_train])
coefficients = np.linalg.lstsq(train_design, y_train, rcond=None)[0]
```

### Predict using the training knots

`X_query` contains three new points, shape `(3, 1)`. Call `transform` on the fitted object rather than fitting again on these points.

```python
X_query = np.array([[-1.5], [0.0], [1.5]])
B_query = transformer.transform(X_query)
query_design = np.column_stack([np.ones(len(X_query)), B_query])
prediction = query_design @ coefficients
print(B_train.shape, B_query.shape)
print(prediction.round(3))
```

The output shapes are `(41, 7)` and `(3, 7)`, followed by approximately
`[-0.778, 0.000, 0.778]`. Seven columns come from `6 + 3 - 2`; they are basis
features, not seven separately observed predictors. `transformer.predict`
returns those same features. Only multiplying the later regression design by
its fitted coefficients produces response predictions.

### Inspect feature names and boundary behavior

Continue with this section's `transformer`, `X_query`, and `B_query`. Names identify expanded basis columns; `predict` on this transformer is only an alias for `transform`.

```python
names = transformer.get_feature_names_out(["time"])
assert len(names) == B_query.shape[1]
assert np.allclose(transformer.predict(X_query), B_query)
assert np.allclose(transformer.transform([[2.5]]), transformer.transform([[2.0]]))
```
<!-- example-end: spline-transformer-reuse-cpu -->

The last assertion demonstrates the chosen constant extrapolation: a query at
2.5 gets the feature values at the upper training boundary 2.0. This is a
boundary rule, not evidence that the response is actually constant outside the
data. `linear` and `continue` make different assumptions and can grow quickly.
For noisy data, tune knot count and downstream regularization within training
folds and keep a separate test set for evaluating predictions.

## Overview

The splines module provides spline basis construction utilities. `bspline_basis` evaluates B-spline basis matrices using De Boor's recursive algorithm. `natural_cubic_spline_basis` projects a cubic basis using numerical constraints that approximate zero endpoint curvature. `cyclic_cubic_spline_basis` attempts a periodic projection, but its current boundary approximation is unreliable; see the limitation below before using it. `thin_plate_spline_basis` constructs multi-dimensional radial basis functions using the thin plate spline kernel. `SplineTransformer` wraps B-spline basis generation in an sklearn-compatible `fit`/`transform` API for use in pipelines. The functions accept NumPy, CuPy, or Torch through an explicit `xp`; `xp=None` uses NumPy, rather than inferring the input backend.

For the Generalized Additive Model (GAM) which uses these basis functions, see [GAM](semiparametric.md).

## Path

```
statgpu.nonparametric.splines.bspline_basis
statgpu.nonparametric.splines.natural_cubic_spline_basis
statgpu.nonparametric.splines.cyclic_cubic_spline_basis
statgpu.nonparametric.splines.thin_plate_spline_basis
statgpu.nonparametric.splines.SplineTransformer
```

## Objective Function

**B-spline basis** is computed via the De Boor recursion. The degree-0 basis functions are

$$
B_{i,0}(x) = \begin{cases} 1 & \text{if } t_i \le x < t_{i+1} \\ 0 & \text{otherwise} \end{cases}
$$

For degree $k \ge 1$:

$$
B_{i,k}(x) = \frac{x-t_i}{t_{i+k}-t_i}B_{i,k-1}(x)
+\frac{t_{i+k+1}-x}{t_{i+k+1}-t_{i+1}}B_{i+1,k-1}(x).
$$

A summand with a zero denominator is defined as zero, including at repeated boundary knots. The rightmost boundary is included by its limiting basis value rather than the half-open degree-0 convention.

**Natural cubic spline** basis: a cubic B-spline basis is projected onto the null space of numerical boundary second-derivative constraints, approximating $f'' = 0$ at the two endpoints. Two independent constraints reduce the basis dimension by 2; in general the reduction equals their numerical rank.

**Cyclic cubic spline** ideally imposes

$$
f(a)=f(b),\qquad f'(a)=f'(b),\qquad f''(a)=f''(b),
$$

at the evaluation-range endpoints $a=\min(x)$ and $b=\max(x)$, with knots strictly inside. Three independent constraints would reduce the cubic basis dimension by 3. The current function estimates derivatives using points outside the range, where its B-spline basis is zero. The returned basis can therefore fail the true one-sided derivative constraints and can have a different column count. **Do not rely on this function for periodic continuity.** For periodic predictors, a manually constructed sine/cosine basis is an alternative when that model is appropriate.

**Thin plate spline** basis: for input dimensionality $d$ and penalty order $m$, the radial basis functions are

$$
\phi(r) = \begin{cases} r^{2m-d} \log(r) & \text{if } d \text{ is even} \\ r^{2m-d} & \text{if } d \text{ is odd} \end{cases}
$$

where $r = \|x - \xi_j\|$ is the Euclidean distance to knot $\xi_j$. For 1-D data with $m=2$, this gives $\phi(r) = r^3$. For 2-D data with $m=2$, this gives $\phi(r) = r^2 \log(r)$. The basis always includes polynomial terms $[1, x_1, \ldots, x_d]$, regardless of `penalty_order`. This is the polynomial block for order 2; it is not the full order-specific block for other orders. The function creates features, not a fitted smoother, coefficient side constraints, or a penalty matrix. It requires `2 * penalty_order > d`.

**SplineTransformer**: an sklearn-compatible transformer that generates B-spline basis features for each input feature. Knots are placed using either a `'uniform'` or `'quantile'` strategy. Output dimension per feature is `n_knots + degree - 1` (with bias) or `n_knots + degree - 2` (without bias).

## Estimating Equation

Evaluation is a direct recursive computation; no linear system is solved. For `cyclic_cubic_spline_basis`, the null space of the periodicity constraint matrix is computed via SVD. `SplineTransformer` evaluates each feature with its own backend-native Cox–de Boor recurrence and explicit extrapolation semantics.

## Covariance / Inference

Spline basis functions are deterministic computational utilities. They do not produce inference outputs (no standard errors, p-values, or confidence intervals). For fitting additive smooths, see [GAM](semiparametric.md), which selects smoothing by GCV but also does not provide coefficient inference or confidence bands.

## Backend execution and extrapolation boundary

`SplineTransformer.fit()` learns knots and `transform()` constructs the basis
using NumPy/CuPy/Torch recurrence for `error`, `constant`, `linear`, and polynomial
`continue` modes.

Current device selection has an exception: a supplied Torch tensor takes priority
over `device`. Torch CPU inputs can leave both fitted knots and transformed features
on CPU even with explicit `device="torch"` or `device="cuda"`. With NumPy inputs, an
unavailable explicit accelerator can instead raise. Inspect each array in `knots_`
and the returned basis, not just `model.device`: Torch `.device` and `.is_cuda`
show placement, CuPy `.device` identifies the GPU, and NumPy arrays are on CPU.
If CUDA execution is required, reject CPU outputs before using them. For a
predictable CPU path, use NumPy inputs with `device="cpu"`. See the
[device guide](../guides/device-and-memory.md) for further placement checks.

Backend choice does not itself establish numerical accuracy or speed on your workload; validate the basis and boundary behavior required by your analysis.

`thin_plate_spline_basis` validates evaluation points, knots, and penalty order
before basis construction and returns an array on the supplied `xp` backend.

### Natural-basis scale sensitivity

`natural_cubic_spline_basis` uses an absolute finite-difference step for its
boundary constraints. Changing measurement units can substantially change the
represented function space: on a range of length `1e6`, the returned basis can
fail to represent even a constant, while very small ranges can have large
nonzero endpoint curvature. Finite output and the expected column count do not
prove natural boundary conditions. Centering and scaling points and knots to a
unit range reduces these demonstrated errors but does not make the constraints
exact. If natural endpoint conditions are essential, use an independently
verified natural-spline construction and check analytic boundary derivatives.

### Transformer input and output shapes

`SplineTransformer` takes a nonempty finite real `(n,p)` design and calculates
float64 basis values. A vector at fit time means one feature. At transformation,
a vector means many points for a single feature or one multivariate point;
for Torch, the latter currently raises `TypeError`, so always pass an explicit
`(q,p)` matrix, including `(1,p)` for one point. Outputs have shape
`(q,n_features_out_)`. Preserve feature order and use the fitted knots. For one
feature, custom knots may also be a length-`n_knots` vector.

<a id="strict--approx-difference"></a>

## Numerical accuracy and boundary conditions

Spline basis computation has no strict/approx mode. Explicit backend selection does not change the documented numerical limitations of the natural/cyclic boundary projections.

## Complete model-specific calls

Import these names from `statgpu.nonparametric.splines`. The call signatures
below complement the parameter and output descriptions on this page. The
[shared estimator helpers](../reference/estimator-api.md) do not turn this
feature transform into a fitted response or coefficient-inference model.

```text
bspline_basis(x, knots, degree=3, xp=None, boundary_lo=None, boundary_hi=None)
natural_cubic_spline_basis(x, knots, xp=None)
cyclic_cubic_spline_basis(x, knots, xp=None)
thin_plate_spline_basis(x, knots, penalty_order=2, xp=None)
SplineTransformer(n_knots=5, degree=3, knots='uniform', include_bias=True, extrapolation='constant', device='auto', n_jobs=None)
SplineTransformer.fit(X, y=None, sample_weight=None)
SplineTransformer.transform(X)
SplineTransformer.fit_transform(X, y=None, sample_weight=None)
SplineTransformer.predict(X)
SplineTransformer.get_feature_names_out(input_features=None)
SplineTransformer.get_params(deep=True)
SplineTransformer.set_params(**params)
```

A raw B-spline call requires at least one interior knot. Reusing only interior
knots is insufficient for new query batches: also pass the original
`boundary_lo` and `boundary_hi`, or use the fitted transformer. Raw basis values
outside explicit boundaries are zero; the transformer's extrapolation policy is
a separate API. Custom transformer knots include their boundary knots and need
not span every training row. With `extrapolation="error"`, `fit` can therefore
succeed while `fit_transform` rejects out-of-range training rows during its
transformation step.

## Parameters

**bspline_basis**:

| Parameter | Default | Description |
|---|---:|---|
| `x` | required | Evaluation points, shape `(n,)` |
| `knots` | required | Interior knot locations (strictly increasing) |
| `degree` | `3` | Nonnegative integer spline degree |
| `boundary_lo`, `boundary_hi` | `None` | Optional fixed boundaries. Defaults cover the evaluation points and knots; interior knots must be strictly inside them. Reuse training boundaries for a consistent basis at new points. |
| `xp` | `None` | Array module (`numpy`, `cupy`, or `torch`); uses NumPy if `None` |

**natural_cubic_spline_basis**:

| Parameter | Default | Description |
|---|---:|---|
| `x` | required | Evaluation points, shape `(n,)` |
| `knots` | required | Interior knot locations (strictly increasing) |
| `xp` | `None` | Array module; uses NumPy if `None` |

**cyclic_cubic_spline_basis**:

| Parameter | Default | Description |
|---|---:|---|
| `x` | required | Evaluation points, shape `(n,)` |
| `knots` | required | Interior knot locations (strictly increasing) |
| `xp` | `None` | Array module; uses NumPy if `None` |

**thin_plate_spline_basis**:

| Parameter | Default | Description |
|---|---:|---|
| `x` | required | Evaluation points, shape `(n,)` or `(n, d)` |
| `knots` | required | Knot positions, shape `(m,)` or `(m, d)`; must match dimensionality of `x` |
| `penalty_order` | `2` | Penalty order $m$; controls smoothness |
| `xp` | `None` | Array module; uses NumPy if `None` |

**SplineTransformer**:

| Parameter | Default | Description |
|---|---:|---|
| `n_knots` | `5` | Integer at least 3; number of knots including boundary knots. Quantile knots must be distinct. |
| `degree` | `3` | Spline degree (3 = cubic) |
| `knots` | `'uniform'` | Knot placement: `'uniform'`, `'quantile'`, or an array of shape `(n_knots, n_features)` |
| `include_bias` | `True` | If `True`, include all basis functions (including the redundant one from partition-of-unity) |
| `extrapolation` | `'constant'` | `'error'`, `'constant'` (clamp), `'linear'` (boundary tangent), or `'continue'` (continue the boundary polynomial piece) |
| `device` | `'auto'` | Requested device; Torch tensor input can override it, as described above. |
| `n_jobs` | `None` | Shared estimator option; does not parallelize basis construction. |

## Advanced: construct basis matrices directly

Use the low-level functions when you need explicit knots or want to compare basis families. This standalone computational grid has 500 one-dimensional evaluation points and 10 interior knots. It is separate from the response-model example: no target or model fit is involved. Run this section's blocks in order.

<!-- example: spline-raw-bases-cpu -->
```python
import numpy as np
from statgpu.nonparametric.splines import bspline_basis
```

<a id="raw-spline-setup"></a>

### Fix the grid, knots, and boundaries

```python
x = np.linspace(0, 1, 500)
knots = np.linspace(0.1, 0.9, 10)
boundary_lo, boundary_hi = 0.0, 1.0
```

Cubic B-splines return 14 columns, from `10 + 3 + 1`. To evaluate the same basis at new points, reuse these knots and both boundaries.

```python
B = bspline_basis(
    x, knots, degree=3, xp=np,
    boundary_lo=boundary_lo, boundary_hi=boundary_hi,
)
print(B.shape)  # (500, 14)
```

### Compare boundary projections

Reuse this section's `x` and `knots`. The natural basis approximates zero endpoint curvature; the cyclic basis does not reliably satisfy true one-sided derivative constraints. These calls illustrate returned matrices, not a validation of their boundary conditions.

```python
from statgpu.nonparametric.splines import (
    natural_cubic_spline_basis, cyclic_cubic_spline_basis,
)

B_nat = natural_cubic_spline_basis(x, knots, xp=np)
B_cyc = cyclic_cubic_spline_basis(x, knots, xp=np)
print(B_nat.shape, B_cyc.shape)  # (500, 12), (500, 12) on this grid
```

### Construct radial thin-plate features

Continue with this section's one-dimensional `x` and `knots`. The order-2 thin-plate basis has 10 radial columns plus an intercept and a linear term, for 12 columns. It does not itself supply a smoothing penalty or response fit.

```python
from statgpu.nonparametric.splines import thin_plate_spline_basis

B_tp = thin_plate_spline_basis(x, knots, penalty_order=2, xp=np)
print(B_tp.shape)  # (500, 12)
```

For two-dimensional input, each row is a point in a plane and the knots must also have two columns. This small grid and five knots produce eight columns: five radial terms and `[1, x1, x2]`.

```python
u, v = np.meshgrid(np.linspace(0, 1, 20), np.linspace(0, 1, 10))
xy = np.column_stack([u.ravel(), v.ravel()])
knots_2d = np.array([[0.1, 0.1], [0.1, 0.9], [0.5, 0.5],
                     [0.9, 0.1], [0.9, 0.9]])
B_tp2 = thin_plate_spline_basis(xy, knots_2d, penalty_order=2, xp=np)
print(B_tp2.shape)  # (200, 8)
```
<!-- example-end: spline-raw-bases-cpu -->

### Optional: evaluate the same B-spline basis on GPU

First complete [Fix the grid, knots, and boundaries](#raw-spline-setup) in this section, reusing `x`, `knots`, `boundary_lo`, `boundary_hi`, and `bspline_basis`. Choose one snippet for your installation, pass the array module as `xp`, and place both inputs on its GPU. The result is still a `(500, 14)` basis matrix, not response predictions.

CuPy/CUDA:

<!-- example-requires: spline-raw-bases-cpu -->
<!-- example: spline-raw-cupy -->
```python
import cupy as cp

B_gpu = bspline_basis(
    cp.asarray(x), cp.asarray(knots), degree=3, xp=cp,
    boundary_lo=boundary_lo, boundary_hi=boundary_hi,
)
print(B_gpu.shape)
```
<!-- example-end: spline-raw-cupy -->

Torch CUDA:

<!-- example-requires: spline-raw-bases-cpu -->
<!-- example: spline-raw-torch -->
```python
import torch

B_t = bspline_basis(
    torch.as_tensor(x, device="cuda"), torch.as_tensor(knots, device="cuda"),
    degree=3, xp=torch, boundary_lo=boundary_lo, boundary_hi=boundary_hi,
)
print(B_t.shape)
```
<!-- example-end: spline-raw-torch -->

Other low-level functions follow the same backend-array and matching-`xp` convention. Backend selection does not remove the natural/cyclic boundary limitations above; measure GPU benefit at your actual scale. For multi-feature `SplineTransformer`, keep the opening fit/transform workflow: each input feature is expanded separately, with column counts given in the parameter and output reference.

## Outputs

**bspline_basis**: returns a basis matrix $B$ of shape `(n, n_knots + degree + 1)`.

**natural_cubic_spline_basis**: normally returns `(n, n_knots + 2)` when the two boundary constraints are independent; in general, width is `n_knots + 4 - numerical_rank`. SVD chooses an arbitrary basis orientation, so the first column is not a dedicated intercept. The function recomputes its boundaries and projection from each supplied evaluation grid: do not fit coefficients on one grid and assume the same knots alone reproduce that basis on another. It has no saved training state or linear-extrapolation API; use `SplineTransformer` when a reusable fitted transform is needed.

**cyclic_cubic_spline_basis**: returns `(n, n_knots + 4 - numerical_rank)`; the illustrated grid returns `n_knots + 2`, not the ideal `n_knots + 1`. A returned basis does not establish periodic boundary continuity.

**thin_plate_spline_basis**: returns a basis matrix $B$ of shape `(n, m + d + 1)` where $m$ is the number of knots and $d$ is the input dimensionality. Includes $m$ radial basis function columns plus $d + 1$ polynomial columns (intercept + linear terms).

**SplineTransformer fitted attributes**:

| Attribute | Shape | Description |
|---|---|---|
| `knots_` | list of arrays | Knot positions for each feature |
| `boundary_lo_` | `(n_features,)` | Lower boundary per feature |
| `boundary_hi_` | `(n_features,)` | Upper boundary per feature |
| `n_features_in_` | int | Number of input features |
| `n_features_out_` | int | Number of output features |

**SplineTransformer methods**:

| Method | Description |
|---|---|
| `fit(X, y=None, sample_weight=None)` | Learn knot positions from training data. Returns `self`. `y` and `sample_weight` are unused; weights do not change quantile knots. |
| `transform(X)` | Return `(n_query,n_features_out_)` basis features on the resolved backend. Check actual placement under the device exception above. |
| `fit_transform(X, y=None, sample_weight=None)` | Fit and transform in one step, with the same unused arguments. |
| `predict(X)` | Alias for `transform(X)`; no response is predicted. |
| `get_feature_names_out(input_features=None)` | Return a list of `n_features_out_` strings. Optional input names must match `n_features_in_`. |
| `get_params(deep=True)`, `set_params(**params)` | Read/change constructor controls; refit after changing parameters. |

## FAQ

- **Natural vs regular B-spline?** The natural basis approximates zero endpoint curvature. This can restrict boundary wiggles, but does not guarantee less overfitting or provide a reusable linear extrapolation rule.
- **When to use cyclic cubic splines?** Periodic structure, such as day-of-year or angle, calls for a periodic model. The current function does not reliably enforce periodic derivatives; use a verified periodic basis instead.
- **When to use thin plate splines?** Thin plate splines are designed for multi-dimensional smoothing. Unlike B-splines, which are inherently 1-D, thin plate splines naturally handle $d$-dimensional inputs using radial basis functions.
- **SplineTransformer vs calling bspline_basis directly?** `SplineTransformer` provides an sklearn-compatible API that handles multiple features, automatic knot placement, and pipeline integration. Use it when building preprocessing pipelines or when you need `fit`/`transform` semantics.
- **GPU speedup for splines?** Speedup depends on sample size, degree, knot count, backend, and actual array placement. Check placement as described above and measure your workload.

## Validation guidance

Compare B-spline values and boundary derivatives with `scipy.interpolate.BSpline` using identical augmented knots and degree. For a transformer, align knot placement, `include_bias`, degree, and extrapolation before comparing with scikit-learn. Compare represented functions or subspaces for natural bases; SVD column signs and orientations need not agree. Check true one-sided boundary derivatives for periodicity, rather than reusing the implementation's outside-range finite differences. A CPU check does not establish GPU accuracy or speed.

## References

- De Boor, C. (1978). *A Practical Guide to Splines*. Springer.
- Eilers, P. H. C., & Marx, B. D. (1996). Flexible smoothing with B-splines and penalties. *Statistical Science*, 11(2), 89-121.
- Wahba, G. (1990). *Spline Models for Observational Data*. SIAM.
- Duchon, J. (1977). Splines minimizing rotation-invariant semi-norms in Sobolev spaces. In *Constructive Theory of Functions of Several Variables*, Springer.
