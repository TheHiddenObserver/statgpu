# Spline Basis Functions

> Language: English  
> Last updated: 2026-05-28  
> This page: Model documentation  
> Switch: [Chinese](../../models/splines.md)

Language switch: [Chinese](../../models/splines.md)

## Overview

The splines module provides spline basis construction utilities. `bspline_basis` evaluates B-spline basis matrices using De Boor's recursive algorithm. `natural_cubic_spline_basis` constructs natural cubic spline bases with boundary constraints (zero second derivative at boundary knots). Both support CPU, CuPy, and Torch backends.

For the Generalized Additive Model (GAM) which uses these basis functions, see [GAM](semiparametric.md).

## Path

`statgpu.nonparametric.splines.bspline_basis`, `statgpu.nonparametric.splines.natural_cubic_spline_basis`

## Objective Function

**B-spline basis** is computed via the De Boor recursion. The degree-0 basis functions are

\[
B_{i,0}(x) = \begin{cases} 1 & \text{if } t_i \le x < t_{i+1} \\ 0 & \text{otherwise} \end{cases}
\]

For degree $k \ge 1$:

\[
B_{i,k}(x) = w_1 \, B_{i,k-1}(x) + (1 - w_2) \, B_{i+1,k-1}(x)
\]

where

\[
w_1 = \frac{x - t_i}{t_{i+k} - t_i}, \qquad w_2 = \frac{x - t_{i+1}}{t_{i+k+1} - t_{i+1}}
\]

with the convention $0/0 = 0$.

**Natural cubic spline** basis: a cubic B-spline basis is projected onto the null space of boundary second-derivative constraints ($f'' = 0$ at the two boundary knots). This reduces the basis dimension by 2 compared to the corresponding regular B-spline basis.

## Estimating Equation

Evaluation is a direct recursive computation; no linear system is solved.

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

## CPU+GPU Examples

```python
from statgpu.nonparametric.splines import bspline_basis, natural_cubic_spline_basis
import numpy as np

x = np.linspace(0, 1, 500)
knots = np.linspace(0.1, 0.9, 10)

# CPU: B-spline basis
B = bspline_basis(x, knots, degree=3, xp=np)
print(f"Basis shape: {B.shape}")  # (500, 14)

# CPU: Natural cubic spline basis
B_nat = natural_cubic_spline_basis(x, knots, xp=np)
print(f"Natural basis shape: {B_nat.shape}")  # (500, 12)
```

## Outputs

**bspline_basis**: returns a basis matrix $B$ of shape `(n, n_knots + degree + 1)`.

**natural_cubic_spline_basis**: returns a basis matrix $B$ of shape `(n, n_knots + 1)`.

## FAQ

- **Natural vs regular B-spline?** Natural splines enforce linearity at the boundaries, reducing overfitting at the edges of the data range. Use natural splines when boundary behavior matters.
- **GPU speedup for splines?** The B-spline basis construction is vectorized over all sample points. For large $n$ (5000+), expect 2-3x speedup on GPU.

## External Validation

- B-spline basis values validated against `scipy.interpolate.BSpline`; relative error < 1e-15.
- Natural cubic spline accuracy: excellent (< 1e-10) for $n \le 500$; fair (~1.5e-6) for $n = 5000$ due to SVD conditioning in the boundary constraint projection.

## References

- De Boor, C. (1978). *A Practical Guide to Splines*. Springer.
