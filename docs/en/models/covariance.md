# Covariance

> Language: English
> Last updated: 2026-05-28
> This page: Model documentation
> Switch: [Chinese](../../models/covariance.md)

Language switch: [Chinese](../../models/covariance.md)

## Overview

The `covariance` module provides covariance matrix estimation with three estimators: `EmpiricalCovariance` (sample covariance), `LedoitWolf` (Ledoit & Wolf 2004 shrinkage), and `OAS` (Oracle Approximating Shrinkage, Chen et al. 2010). All three support CPU, CuPy, and PyTorch backends with automatic device detection. `LedoitWolf` and `OAS` extend `EmpiricalCovariance` with analytically optimal shrinkage toward a scaled identity target, producing well-conditioned covariance estimates even when the number of features approaches or exceeds the number of samples.

## Path

- `statgpu.covariance.EmpiricalCovariance`
- `statgpu.covariance.LedoitWolf`
- `statgpu.covariance.OAS`

## Objective Function

**EmpiricalCovariance** computes the maximum-likelihood sample covariance:

\[
\hat{S} = \frac{1}{n} X^\top X
\]

where \(X\) is the centered data matrix (mean subtracted column-wise unless `assume_centered=True`).

**LedoitWolf** and **OAS** both produce a shrunk covariance of the form:

\[
\hat{\Sigma} = (1 - \alpha)\,\hat{S} + \alpha\,\mu\,I
\]

where \(\mu = \operatorname{tr}(\hat{S})/p\) is the average eigenvalue of the sample covariance. The two estimators differ only in how they compute the optimal shrinkage intensity \(\alpha\).

**Ledoit-Wolf shrinkage intensity** (Ledoit & Wolf 2004):

\[
\alpha = \operatorname{clip}\!\left(\frac{\beta}{\delta},\; 0,\; 1\right)
\]

with

\[
\beta = \frac{1}{n^2}\left[\sum_{k=1}^{n} \|x_k\|_2^4 - n\,\|\hat{S}\|_F^2\right], \qquad
\delta = \|\hat{S} - \mu I\|_F^2 = \|\hat{S}\|_F^2 - \frac{\operatorname{tr}(\hat{S})^2}{p}
\]

**OAS shrinkage intensity** (Chen et al. 2010):

\[
\alpha = \operatorname{clip}\!\left(\frac{\overline{S^2} + \mu^2}{(n+1)\!\left(\overline{S^2} - \mu^2/p\right)},\; 0,\; 1\right)
\]

where \(\overline{S^2} = \frac{1}{p^2}\sum_{i,j} S_{ij}^2\) is the mean of the squared elements of \(\hat{S}\).

## Estimating Equation

All three estimators use direct computation rather than iterative optimization:

- **EmpiricalCovariance**: The sample covariance \(\hat{S} = X^\top X / n\) is computed directly. The precision matrix \(\hat{S}^{-1}\) is obtained via jitter-stabilized matrix inversion (progressive diagonal augmentation if the matrix is near-singular).
- **LedoitWolf**: The analytical Ledoit-Wolf formula for \(\alpha\) is evaluated in closed form from the centered data, then the shrunk covariance and its inverse are computed.
- **OAS**: Same closed-form approach as Ledoit-Wolf but with the OAS shrinkage formula, which is derived under a Gaussian assumption and is asymptotically optimal when \(n > p\).

## Covariance/Inference

All estimators produce the following fitted attributes after `fit()`:

- `covariance_`: the estimated covariance matrix \(\hat{\Sigma}\) (shape `(n_features, n_features)`).
- `precision_`: the inverse covariance matrix \(\hat{\Sigma}^{-1}\) (shape `(n_features, n_features)`), computed with jitter stabilization for numerical robustness.
- `location_`: the estimated mean vector (shape `(n_features,)`); zeros if `assume_centered=True`.
- `shrinkage_`: the shrinkage intensity \(\alpha\) as a float in \([0, 1]\) (LedoitWolf and OAS only).

The `score()` method computes the average Gaussian log-likelihood per observation:

\[
\ell = -\frac{1}{2}\!\left(p \log(2\pi) + \log\det(\hat{\Sigma}) + \frac{1}{n}\sum_{k=1}^{n}(x_k - \hat{\mu})^\top \hat{\Sigma}^{-1}(x_k - \hat{\mu})\right)
\]

## Parameters

| Parameter | Default | Description |
|---|---:|---|
| `assume_centered` | `False` | If `True`, skip mean estimation and centering; data is assumed already centered |
| `device` | `"auto"` | Computation device: `"cpu"`, `"cuda"`, `"torch"`, or `"auto"` (auto-detects from input array type) |
| `n_jobs` | `None` | Number of parallel jobs (reserved for future use, not currently active) |

These parameters are shared by `EmpiricalCovariance`, `LedoitWolf`, and `OAS`.

## CPU+GPU Examples

```python
from statgpu.covariance import EmpiricalCovariance, LedoitWolf, OAS
import numpy as np

X = np.random.randn(500, 10)

# --- CPU ---

# Empirical covariance
emp = EmpiricalCovariance(device="cpu")
emp.fit(X)
print(f"Covariance shape: {emp.covariance_.shape}")  # (10, 10)
print(f"Location shape:   {emp.location_.shape}")     # (10,)

# Ledoit-Wolf shrinkage
lw = LedoitWolf(device="cpu")
lw.fit(X)
print(f"Shrinkage: {lw.shrinkage_:.4f}")              # e.g. 0.1234

# OAS shrinkage
oas = OAS(device="cpu")
oas.fit(X)
print(f"OAS shrinkage: {oas.shrinkage_:.4f}")

# Scoring (average log-likelihood)
ll = lw.score(X)
print(f"Log-likelihood: {ll:.4f}")

# Mahalanobis distances
dists = lw.mahalanobis(X[:5])
print(f"Mahalanobis distances: {dists}")

# --- GPU (CuPy) ---

lw_gpu = LedoitWolf(device="cuda")
lw_gpu.fit(X)
print(f"GPU shrinkage: {lw_gpu.shrinkage_:.4f}")
print(f"GPU covariance shape: {lw_gpu.covariance_.shape}")

# --- GPU (PyTorch) ---

import torch
X_torch = torch.randn(500, 10, device="cuda", dtype=torch.float64)
lw_torch = LedoitWolf(device="cuda")
lw_torch.fit(X_torch)
print(f"Torch shrinkage: {lw_torch.shrinkage_:.4f}")
```

## strict/approx difference

The covariance estimators do not have separate strict or approx modes. All three estimators use direct analytical formulas with no iterative solver, so there is no convergence tolerance to tune.

`LedoitWolf` and `OAS` provide different shrinkage intensity formulas. Choose based on your use case:

- **LedoitWolf**: More general; performs well across a wide range of \(n/p\) ratios. This is the standard recommendation for shrinkage covariance estimation.
- **OAS**: Derived under a Gaussian assumption; asymptotically optimal when \(n > p\) and often achieves lower mean squared error than Ledoit-Wolf in that regime.

## Outputs

### Fitted attributes

| Attribute | Shape | Description |
|---|---|---|
| `covariance_` | `(n_features, n_features)` | Estimated covariance matrix |
| `precision_` | `(n_features, n_features)` | Inverse covariance (precision) matrix |
| `location_` | `(n_features,)` | Estimated mean vector |
| `n_samples_` | scalar | Number of training samples |
| `n_features_` | scalar | Number of features |
| `shrinkage_` | scalar (float) | Shrinkage intensity in [0, 1] (LedoitWolf/OAS only) |

### Methods

| Method | Returns | Description |
|---|---|---|
| `fit(X)` | `self` | Fit the covariance model to data matrix X |
| `predict(X)` | `ndarray (n_samples,)` | Mahalanobis distances for observations in X |
| `score(X)` | `float` | Average Gaussian log-likelihood per observation |
| `mahalanobis(X)` | `ndarray (n_samples,)` | Squared Mahalanobis distances for observations in X |

## FAQ

**When should I use LedoitWolf vs OAS?**
OAS is recommended when \(n > p\) (more samples than features) because it is derived under a Gaussian assumption and is asymptotically optimal in that setting. LedoitWolf is more general and is the safer default when you are unsure or when \(n\) and \(p\) are close. In practice the difference is often small.

**What does `score()` return?**
The average log-likelihood per observation under a multivariate Gaussian with the fitted covariance and mean. Higher values indicate a better fit. This can be used for model comparison between estimators.

**What happens when the covariance matrix is singular?**
The precision matrix computation uses jitter-stabilized inversion: progressively larger diagonal increments are added until a stable inverse is found. If you encounter persistent singularity warnings, consider using LedoitWolf or OAS instead of EmpiricalCovariance, as shrinkage guarantees a well-conditioned estimate.

**Can I pass CuPy or PyTorch arrays directly?**
Yes. If you pass a CuPy ndarray or a PyTorch tensor, the backend is detected automatically from the input type. You can also set `device="cuda"` or `device="torch"` explicitly with NumPy input to force GPU computation.

## External Validation

All three estimators are validated against their scikit-learn counterparts:

- `sklearn.covariance.EmpiricalCovariance`
- `sklearn.covariance.LedoitWolf`
- `sklearn.covariance.OAS`

Fitted `covariance_`, `precision_`, `location_`, and `shrinkage_` values match to relative error < 1e-15 on test datasets. Consistency checks are maintained in `dev/tests/test_external_consistency.py`.

## References

- Ledoit, O., & Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. *Journal of Multivariate Analysis*, 88(2), 365-411. [https://doi.org/10.1016/S0047-259X(03)00096-4](https://doi.org/10.1016/S0047-259X(03)00096-4)
- Chen, Y., Wiesel, A., Eldar, Y. C., & Hero, A. O. (2010). Shrinkage algorithms for MMSE covariance estimation. *IEEE Transactions on Signal Processing*, 58(10), 5297-5307. [https://doi.org/10.1109/TSP.2010.2053029](https://doi.org/10.1109/TSP.2010.2053029)
