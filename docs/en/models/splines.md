# Spline Basis Functions

> Language: English  
> Last updated: 2026-07-14  
> This page: Model documentation  
> Switch: [Chinese](../../cn/models/splines.md)

Language switch: [Chinese](../../cn/models/splines.md)

## Overview

The splines module provides spline basis construction utilities. `bspline_basis` evaluates B-spline basis matrices using De Boor's recursive algorithm. `natural_cubic_spline_basis` constructs natural cubic spline bases with boundary constraints (zero second derivative at boundary knots). `cyclic_cubic_spline_basis` builds periodic cubic spline bases enforcing value, first-derivative, and second-derivative continuity at the boundaries. `thin_plate_spline_basis` constructs multi-dimensional radial basis functions using the thin plate spline kernel. `SplineTransformer` wraps B-spline basis generation in an sklearn-compatible `fit`/`transform` API for use in pipelines. All functions support CPU, CuPy, and Torch backends.

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
B_{i,k}(x) = w_1 \, B_{i,k-1}(x) + (1 - w_2) \, B_{i+1,k-1}(x)
$$

where

$$
w_1 = \frac{x - t_i}{t_{i+k} - t_i}, \qquad w_2 = \frac{x - t_{i+1}}{t_{i+k+1} - t_{i+1}}
$$

with the convention $0/0 = 0$.

**Natural cubic spline** basis: a cubic B-spline basis is projected onto the null space of boundary second-derivative constraints ($f'' = 0$ at the two boundary knots). This reduces the basis dimension by 2 compared to the corresponding regular B-spline basis.

**Cyclic cubic spline** basis: a cubic B-spline basis is projected onto the null space of three periodicity constraints at the boundary knots $a = \min(\text{knots})$, $b = \max(\text{knots})$:

$$
f(a) = f(b), \quad f'(a) = f'(b), \quad f''(a) = f''(b)
$$

This reduces the basis dimension by 3 compared to the standard B-spline basis and ensures smooth periodic behavior.

**Thin plate spline** basis: for input dimensionality $d$ and penalty order $m$, the radial basis functions are

$$
\phi(r) = \begin{cases} r^{2m-d} \log(r) & \text{if } d \text{ is even} \\ r^{2m-d} & \text{if } d \text{ is odd} \end{cases}
$$

where $r = \|x - \xi_j\|$ is the Euclidean distance to knot $\xi_j$. For 1-D data with $m=2$, this gives $\phi(r) = r^3$. For 2-D data with $m=2$, this gives $\phi(r) = r^2 \log(r)$. The basis includes polynomial terms $[1, x_1, \ldots, x_d]$ to ensure completeness.

**SplineTransformer**: an sklearn-compatible transformer that generates B-spline basis features for each input feature. Knots are placed using either a `'uniform'` or `'quantile'` strategy. Output dimension per feature is `n_knots + degree - 1` (with bias) or `n_knots + degree - 2` (without bias).

## Estimating Equation

Evaluation is a direct recursive computation; no linear system is solved. For `cyclic_cubic_spline_basis`, the null space of the periodicity constraint matrix is computed via SVD. For `thin_plate_spline_basis`, pairwise distances are computed via vectorized broadcasting. `SplineTransformer` evaluates each feature with its own backend-native Cox–de Boor recurrence and explicit extrapolation semantics.

## Covariance / Inference

Spline basis functions are deterministic computational utilities. They do not produce inference outputs (no standard errors, p-values, or confidence intervals). For statistical inference using splines, see the [GAM](semiparametric.md) model which wraps penalized splines with GCV-based smoothing parameter selection.

## Backend execution and extrapolation boundary

`SplineTransformer.fit()` learns knots on the selected backend and `transform()`
constructs the full basis there; it no longer transfers the complete input to SciPy.
`error`, `constant`, `linear`, and polynomial `continue` modes share the same
NumPy/CuPy/Torch recurrence. Moving a fitted transformer to another backend transfers
only knot metadata.

NumPy/Torch-CPU extrapolation parity is covered by CI. Physical CuPy CUDA and Torch
CUDA memory/runtime validation remains pending.

`thin_plate_spline_basis` also uses device-aware allocation and scalar-safe radial
operations across NumPy/CuPy/Torch; x, knots, and penalty order are validated before
basis construction. The QR fallback for natural splines allocates its identity matrix
on the same device as the constraint matrix.

## strict / approx Difference

Spline basis computation has no strict/approx mode. The same recurrence is used across NumPy, CuPy, and Torch. NumPy/Torch-CPU parity is tested at tight tolerance; physical CUDA parity and performance remain pending.

## Parameters

**bspline_basis**:

| Parameter | Default | Description |
|---|---:|---|
| `x` | required | Evaluation points, shape `(n,)` |
| `knots` | required | Interior knot locations (strictly increasing) |
| `degree` | `3` | Spline degree |
| `xp` | `None` | Array module (`numpy`, `cupy`, or `torch`); inferred from `x` if `None` |

**natural_cubic_spline_basis**:

| Parameter | Default | Description |
|---|---:|---|
| `x` | required | Evaluation points, shape `(n,)` |
| `knots` | required | Interior knot locations (strictly increasing) |
| `xp` | `None` | Array module; inferred from `x` if `None` |

**cyclic_cubic_spline_basis**:

| Parameter | Default | Description |
|---|---:|---|
| `x` | required | Evaluation points, shape `(n,)` |
| `knots` | required | Interior knot locations (strictly increasing) |
| `xp` | `None` | Array module; inferred from `x` if `None` |

**thin_plate_spline_basis**:

| Parameter | Default | Description |
|---|---:|---|
| `x` | required | Evaluation points, shape `(n,)` or `(n, d)` |
| `knots` | required | Knot positions, shape `(m,)` or `(m, d)`; must match dimensionality of `x` |
| `penalty_order` | `2` | Penalty order $m$; controls smoothness |
| `xp` | `None` | Array module; inferred from `x` if `None` |

**SplineTransformer**:

| Parameter | Default | Description |
|---|---:|---|
| `n_knots` | `5` | Number of knots (including boundary knots) |
| `degree` | `3` | Spline degree (3 = cubic) |
| `knots` | `'uniform'` | Knot placement: `'uniform'`, `'quantile'`, or an array of shape `(n_knots, n_features)` |
| `include_bias` | `True` | If `True`, include all basis functions (including the redundant one from partition-of-unity) |
| `extrapolation` | `'constant'` | `'error'`, `'constant'` (clamp), `'linear'` (boundary tangent), or `'continue'` (continue the boundary polynomial piece) |
| `device` | `'auto'` | Computation device |

## CPU+GPU Examples

```python
from statgpu.nonparametric.splines import (
    bspline_basis, natural_cubic_spline_basis,
    cyclic_cubic_spline_basis, thin_plate_spline_basis,
    SplineTransformer,
)
import numpy as np

x = np.linspace(0, 1, 500)
knots = np.linspace(0.1, 0.9, 10)

# CPU: B-spline basis
B = bspline_basis(x, knots, degree=3, xp=np)
print(f"Basis shape: {B.shape}")  # (500, 14)

# CPU: Natural cubic spline basis
B_nat = natural_cubic_spline_basis(x, knots, xp=np)
print(f"Natural basis shape: {B_nat.shape}")  # (500, 12)

# CPU: Cyclic (periodic) cubic spline basis
B_cyc = cyclic_cubic_spline_basis(x, knots, xp=np)
print(f"Cyclic basis shape: {B_cyc.shape}")  # (500, 11)

# CPU: Thin plate spline basis (1-D)
B_tp = thin_plate_spline_basis(x, knots, penalty_order=2, xp=np)
print(f"Thin plate basis shape: {B_tp.shape}")  # (500, 12)

# CPU: Thin plate spline basis (2-D)
xy = np.column_stack([np.linspace(0, 1, 200), np.linspace(0, 1, 200)])
knots_2d = np.column_stack([np.linspace(0.1, 0.9, 5), np.linspace(0.1, 0.9, 5)])
B_tp2 = thin_plate_spline_basis(xy, knots_2d, penalty_order=2, xp=np)
print(f"Thin plate 2D basis shape: {B_tp2.shape}")  # (200, 8)

# CPU: SplineTransformer (sklearn-compatible API)
X = np.random.randn(500, 3)
st = SplineTransformer(n_knots=10, degree=3, knots='quantile')
X_spline = st.fit_transform(X)
print(f"Transformed shape: {X_spline.shape}")  # (500, 30)
```

**CuPy (GPU)**:

```python
import cupy as cp

x_gpu = cp.asarray(x)
knots_gpu = cp.asarray(knots)

B_gpu = bspline_basis(x_gpu, knots_gpu, degree=3, xp=cp)
print(f"GPU basis shape: {B_gpu.shape}")  # (500, 14)

B_nat_gpu = natural_cubic_spline_basis(x_gpu, knots_gpu, xp=cp)
print(f"GPU natural basis shape: {B_nat_gpu.shape}")  # (500, 12)

B_cyc_gpu = cyclic_cubic_spline_basis(x_gpu, knots_gpu, xp=cp)
print(f"GPU cyclic basis shape: {B_cyc_gpu.shape}")  # (500, 11)

B_tp_gpu = thin_plate_spline_basis(x_gpu, knots_gpu, penalty_order=2, xp=cp)
print(f"GPU thin plate basis shape: {B_tp_gpu.shape}")  # (500, 12)
```

**PyTorch (GPU)**:

```python
import torch

x_t = torch.tensor(x, device='cuda')
knots_t = torch.tensor(knots, device='cuda')

B_t = bspline_basis(x_t, knots_t, degree=3, xp=torch)
print(f"Torch basis shape: {B_t.shape}")  # (500, 14)

B_cyc_t = cyclic_cubic_spline_basis(x_t, knots_t, xp=torch)
print(f"Torch cyclic basis shape: {B_cyc_t.shape}")  # (500, 11)

B_tp_t = thin_plate_spline_basis(x_t, knots_t, penalty_order=2, xp=torch)
print(f"Torch thin plate basis shape: {B_tp_t.shape}")  # (500, 12)
```

## Outputs

**bspline_basis**: returns a basis matrix $B$ of shape `(n, n_knots + degree + 1)`.

**natural_cubic_spline_basis**: returns a basis matrix $B$ of shape `(n, n_knots + 1)`.

**cyclic_cubic_spline_basis**: returns a basis matrix $B$ of shape `(n, n_knots + degree + 1 - 3)`. The dimension reduction of 3 corresponds to the three periodicity constraints.

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
| `fit(X, y=None)` | Learn knot positions from training data. Returns `self`. |
| `transform(X)` | Transform data to B-spline basis features. |
| `fit_transform(X, y=None)` | Fit and transform in one step. |
| `get_feature_names_out(input_features=None)` | Get output feature names. |

## FAQ

- **Natural vs regular B-spline?** Natural splines enforce linearity at the boundaries, reducing overfitting at the edges of the data range. Use natural splines when boundary behavior matters.
- **When to use cyclic cubic splines?** Use cyclic splines when the data has a periodic structure (e.g., day-of-year, angle). The basis enforces that the fitted function and its first two derivatives match at the period boundaries.
- **When to use thin plate splines?** Thin plate splines are designed for multi-dimensional smoothing. Unlike B-splines, which are inherently 1-D, thin plate splines naturally handle $d$-dimensional inputs using radial basis functions.
- **SplineTransformer vs calling bspline_basis directly?** `SplineTransformer` provides an sklearn-compatible API that handles multiple features, automatic knot placement, and pipeline integration. Use it when building preprocessing pipelines or when you need `fit`/`transform` semantics.
- **GPU speedup for splines?** The recurrence is vectorized over observations and remains on-device, but speedup depends on sample size, degree, knot count, and backend. No general speedup claim is made until the current CUDA benchmark pass is completed.

## External Validation

- B-spline basis values validated against `scipy.interpolate.BSpline`; relative error < 1e-15.
- Natural cubic spline accuracy: excellent (< 1e-10) for $n \le 500$; fair (~1.5e-6) for $n = 5000$ due to SVD conditioning in the boundary constraint projection.
- `SplineTransformer` output validated against `sklearn.preprocessing.SplineTransformer` for uniform and quantile knot strategies.
- Constant, linear, and continue extrapolation are checked for NumPy/Torch-CPU parity; optional CuPy tests require a physical CUDA runtime.
- `cyclic_cubic_spline_basis` periodicity verified: $f(a) \approx f(b)$, $f'(a) \approx f'(b)$, $f''(a) \approx f''(b)$ to within SVD tolerance.
- `thin_plate_spline_basis` validated against hand-computed $\phi(r) = r^2 \log(r)$ values for 2-D inputs.

## References

- De Boor, C. (1978). *A Practical Guide to Splines*. Springer.
- Eilers, P. H. C., & Marx, B. D. (1996). Flexible smoothing with B-splines and penalties. *Statistical Science*, 11(2), 89-121.
- Wahba, G. (1990). *Spline Models for Observational Data*. SIAM.
- Duchon, J. (1977). Splines minimizing rotation-invariant semi-norms in Sobolev spaces. In *Constructive Theory of Functions of Several Variables*, Springer.
