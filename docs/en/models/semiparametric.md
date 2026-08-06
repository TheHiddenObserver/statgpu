# GAM (Generalized Additive Model)

> Language: English  
> Last updated: 2026-05-28  
> This page: Model documentation  
> Switch: [Chinese](../../cn/models/semiparametric.md)

Language switch: [Chinese](../../cn/models/semiparametric.md)

## Overview

`GAM` fits a Generalized Additive Model using penalized B-splines with automatic smoothing parameter selection via Generalized Cross-Validation (GCV). The model is:

$$
y = \alpha + \sum_j f_j(x_j) + \epsilon
$$

where each $f_j$ is represented as a penalized B-spline. GAM is a semiparametric model: it has a parametric intercept and nonparametric smooth functions for each feature.

For the underlying B-spline basis utilities, see [Spline Basis Functions](splines.md).

## Path

`statgpu.semiparametric.GAM`

## Objective Function

GAM fits a penalized least-squares model:

$$
\min_{\beta} \|y - B\beta\|_2^2 + \lambda \, \beta^\top S \, \beta
$$

where $B$ is the column-wise concatenation of spline basis matrices for each feature (plus an intercept column), $S$ is a block-diagonal difference penalty matrix, and $\lambda$ is the smoothing parameter. The default penalty order is 2 (second differences), which penalizes curvature.

## Estimating Equation

The first-order condition of the penalized objective yields the system

$$
(B^\top B + \lambda S) \, \hat\beta = B^\top y
$$

solved via Cholesky factorization.

**GCV for lambda selection** (when `lam=None`):

$$
\text{GCV} = \frac{n \cdot \text{RSS}}{(n - \text{edf})^2}
$$

where the effective degrees of freedom is

$$
\text{edf} = \text{tr}\!\left((B^\top B + \lambda S)^{-1} B^\top B\right)
$$

Lambda is selected by minimizing GCV over a log-spaced grid.

## Covariance/Inference

- `edf_`: effective degrees of freedom of the fitted model.
- `gcv_score_`: GCV score (available when lambda is auto-selected).
- `lam_`: smoothing parameter used for the final fit.
- No coefficient-level standard errors or p-values are produced; the GAM is a smoother, not a parametric inference tool.

## Parameters

| Parameter | Default | Description |
|---|---:|---|
| `n_splines` | `20` | Number of spline basis functions per feature |
| `degree` | `3` | Spline degree |
| `lam` | `None` | Smoothing parameter; auto-selected via GCV if `None` |
| `penalty_order` | `2` | Order of the difference penalty matrix |
| `device` | `"auto"` | `cpu` / `cuda` / `auto` |

## CPU+GPU Examples

```python
from statgpu.semiparametric import GAM
import numpy as np

X = np.random.randn(500, 3)
y = np.sin(X[:, 0] * 3) + 0.1 * np.random.randn(500)

# GAM (CPU)
gam = GAM(n_splines=20, device='cpu')
gam.fit(X, y)
print(f"EDF: {gam.edf_:.1f}, GCV: {gam.gcv_score_:.6f}")
y_pred = gam.predict(X)

# GAM (GPU)
gam_gpu = GAM(n_splines=20, device='cuda')
gam_gpu.fit(X, y)
y_pred_gpu = gam_gpu.predict(X)
```

## strict/approx difference

- When `lam=None` (default), GAM uses GCV over a log-spaced grid (1e-10 to 1e10, 100 points) to select the smoothing parameter. This is the approximate path; the grid is coarse and may miss the optimal lambda in narrow valleys.
- When `lam` is specified manually, the exact penalized least-squares solution is computed for that single value. This is the exact path.
- For fine-tuning beyond the default grid, pass a custom `lam` value obtained from a narrower search or domain knowledge.

## Outputs

**GAM fitted attributes**:

| Attribute | Type | Description |
|---|---|---|
| `coef_` | array, shape `(1 + sum(n_basis_j),)` | Concatenated spline coefficients (including intercept) |
| `intercept_` | float | Intercept term |
| `edf_` | float | Effective degrees of freedom |
| `gcv_score_` | float | GCV score at the selected lambda |
| `lam_` | float | Smoothing parameter used |
| `knots_` | list of arrays | Knot locations per feature |
| `n_features_` | int | Number of input features |

**Methods**: `fit(X, y)`, `predict(X)`, `summary()`.

## FAQ

- **How many knots should I use?** `n_splines=20` is a good default. More knots give more flexibility but increase effective degrees of freedom and risk overfitting.
- **What penalty order should I use?** `penalty_order=2` (second differences) is standard for smooth functions. Use `penalty_order=1` for piecewise-linear fits.
- **GPU speedup?** The GAM solve is dominated by the Cholesky factorization, which benefits from GPU acceleration for large basis dimensions.

## External Validation

- GAM predictions validated against pyGAM on standard test datasets.

## References

- Hastie, T., & Tibshirani, R. (1990). *Generalized Additive Models*. Chapman & Hall.
- Wood, S. N. (2017). *Generalized Additive Models: An Introduction with R* (2nd ed.). Chapman & Hall/CRC.
