# Covariance

> Language: English
> Last updated: 2026-07-14
> This page: Model documentation
> Switch: [Chinese](../../models/covariance.md)

Language switch: [Chinese](../../models/covariance.md)

## Overview

The `covariance` module provides covariance matrix estimation with seven estimators: `EmpiricalCovariance` (sample covariance), `LedoitWolf` (Ledoit & Wolf 2004 shrinkage), `OAS` (Oracle Approximating Shrinkage, Chen et al. 2010), `ShrunkCovariance` (generic shrinkage with user-specified intensity), `MinCovDet` (robust Minimum Covariance Determinant via FAST-MCD), `GraphicalLasso` (sparse inverse covariance via graphical lasso), and `GraphicalLassoCV` (cross-validated graphical lasso). All support CPU, CuPy, and PyTorch backends with automatic device detection. `LedoitWolf` and `OAS` extend `EmpiricalCovariance` with analytically optimal shrinkage toward a scaled identity target. `ShrunkCovariance` allows manual control of the shrinkage intensity. `MinCovDet` provides robust covariance estimation resistant to outliers. `GraphicalLasso` and `GraphicalLassoCV` estimate sparse precision matrices using L1 regularization.

## Path

- `statgpu.covariance.EmpiricalCovariance`
- `statgpu.covariance.LedoitWolf`
- `statgpu.covariance.OAS`
- `statgpu.covariance.ShrunkCovariance`
- `statgpu.covariance.MinCovDet`
- `statgpu.covariance.GraphicalLasso`
- `statgpu.covariance.GraphicalLassoCV`

## Objective Function

**EmpiricalCovariance** computes the maximum-likelihood sample covariance:

$$
\hat{S} = \frac{1}{n} X^\top X
$$

where \(X\) is the centered data matrix (mean subtracted column-wise unless `assume_centered=True`).

**LedoitWolf** and **OAS** both produce a shrunk covariance of the form:

$$
\hat{\Sigma} = (1 - \alpha)\,\hat{S} + \alpha\,\mu\,I
$$

where \(\mu = \operatorname{tr}(\hat{S})/p\) is the average eigenvalue of the sample covariance. The two estimators differ only in how they compute the optimal shrinkage intensity \(\alpha\).

**Ledoit-Wolf shrinkage intensity** (Ledoit & Wolf 2004):

$$
\alpha = \operatorname{clip}\!\left(\frac{\beta}{\delta},\; 0,\; 1\right)
$$

with

$$
\beta = \frac{1}{n^2}\left[\sum_{k=1}^{n} \|x_k\|_2^4 - n\,\|\hat{S}\|_F^2\right], \qquad
\delta = \|\hat{S} - \mu I\|_F^2 = \|\hat{S}\|_F^2 - \frac{\operatorname{tr}(\hat{S})^2}{p}
$$

**OAS shrinkage intensity** (Chen et al. 2010):

$$
\alpha = \operatorname{clip}\!\left(\frac{\overline{S^2} + \mu^2}{(n+1)\!\left(\overline{S^2} - \mu^2/p\right)},\; 0,\; 1\right)
$$

where \(\overline{S^2} = \frac{1}{p^2}\sum_{i,j} S_{ij}^2\) is the mean of the squared elements of \(\hat{S}\).

**ShrunkCovariance** uses the same shrinkage formula as LedoitWolf and OAS but with a user-specified shrinkage intensity \(\alpha\):

$$
\hat{\Sigma} = (1 - \alpha)\,\hat{S} + \alpha\,\mu\,I
$$

where \(\alpha\) is the `shrinkage` parameter (default 0.1), set manually rather than computed from data.

**MinCovDet** finds the subset of \(h = \lceil 0.5(n + p + 1) \rceil\) observations whose covariance matrix has the smallest determinant, using the FAST-MCD algorithm (Rousseeuw & Van Driessen 1999). The raw covariance estimate is corrected by a consistency factor \(c_\alpha\) (Croux & Haesbroeck 1999):

$$
c_\alpha = \frac{\alpha}{F_{\chi^2_{p+2}}(q_\alpha)}, \qquad q_\alpha = F^{-1}_{\chi^2_p}(\alpha)
$$

A reweighting step then uses observations within the 97.5th percentile of the \(\chi^2_p\) distribution, applying a second consistency correction at \(\alpha = 0.975\).

**GraphicalLasso** solves the following convex optimization problem:

$$
\max_{\Theta \succ 0}\; \log\det(\Theta) - \operatorname{tr}(S\Theta) - \alpha\|\Theta\|_{1,\mathrm{off}}
$$

using the block coordinate descent algorithm of Friedman, Hastie & Tibshirani (2008). Each outer iteration cycles over all \(p\) features, solving an L1-regularized regression for each column of the precision matrix via soft-thresholding. Convergence is checked by the maximum absolute covariance update between outer iterations; the precision diagonal is not L1-penalized.

**GraphicalLassoCV** selects the regularization parameter \(\alpha\) by K-fold cross-validation. A grid of candidate \(\alpha\) values is evaluated by fitting `GraphicalLasso` on each training fold and scoring the held-out log-likelihood. The \(\alpha\) with the highest mean cross-validated log-likelihood is selected for the final model.

## Estimating Equation

Most estimators use direct computation rather than iterative optimization:

- **EmpiricalCovariance**: The sample covariance \(\hat{S} = X^\top X / n\) is computed directly. The precision matrix \(\hat{S}^{-1}\) is computed by exact inversion first; progressive diagonal jitter is used only when the exact inverse fails or is non-finite.
- **LedoitWolf**: The analytical Ledoit-Wolf formula for \(\alpha\) is evaluated in closed form from the centered data, then the shrunk covariance and its inverse are computed.
- **OAS**: Same closed-form approach as Ledoit-Wolf but with the OAS shrinkage formula, which is derived under a Gaussian assumption and is asymptotically optimal when \(n > p\).
- **ShrunkCovariance**: Same as LedoitWolf/OAS but with a user-supplied \(\alpha\); no iterative optimization.
- **MinCovDet**: The FAST-MCD algorithm uses multi-stage C-steps (concentration steps). For \(n \le 500\), 30 random subsets are drawn, each refined by 2 C-steps, the top 10 are refined to convergence, and the best is kept. For larger data, 50 seeded random starts are used; candidate subsets are refined by backend-native C-steps and the best positive-definite support is retained. After finding the raw MCD estimate, reweighting and consistency correction are applied.
- **GraphicalLasso**: Block coordinate descent iterates over features, solving an L1-regularized regression per column via cyclical coordinate descent with soft-thresholding (up to 1000 inner iterations). Outer convergence is checked by the maximum absolute covariance update.
- **GraphicalLassoCV**: K-fold cross-validation over a grid of \(\alpha\) values, each fit using `GraphicalLasso`. The final model is refitted on all data with the best \(\alpha\).

## Covariance/Inference

All estimators produce the following fitted attributes after `fit()` (additional estimator-specific attributes are listed in the Outputs section below):

- `covariance_`: the estimated covariance matrix \(\hat{\Sigma}\) (shape `(n_features, n_features)`).
- `precision_`: the inverse covariance matrix \(\hat{\Sigma}^{-1}\) (shape `(n_features, n_features)`), computed with jitter stabilization for numerical robustness.
- `location_`: the estimated mean vector (shape `(n_features,)`); zeros if `assume_centered=True`.
- `shrinkage_`: the shrinkage intensity \(\alpha\) as a float in \([0, 1]\) (LedoitWolf, OAS, and ShrunkCovariance).

The `score()` method computes the average Gaussian log-likelihood per observation:

$$
\ell = -\frac{1}{2}\!\left(p \log(2\pi) + \log\det(\hat{\Sigma}) + \frac{1}{n}\sum_{k=1}^{n}(x_k - \hat{\mu})^\top \hat{\Sigma}^{-1}(x_k - \hat{\mu})\right)
$$

## Parameters

| Parameter | Default | Description |
|---|---:|---|
| `assume_centered` | `False` | If `True`, skip mean estimation and centering; data is assumed already centered |
| `device` | `"auto"` | Computation device: `"cpu"`, `"cuda"`, `"torch"`, or `"auto"` (auto-detects from input array type) |
| `n_jobs` | `None` | Number of parallel jobs (reserved for future use, not currently active) |

These parameters are shared by all seven estimators.

**ShrunkCovariance additional parameters:**

| Parameter | Default | Description |
|---|---:|---|
| `shrinkage` | `0.1` | Shrinkage intensity in [0, 1] |

**MinCovDet additional parameters:**

| Parameter | Default | Description |
|---|---:|---|
| `support_fraction` | `None` | Fraction of observations for MCD. Default: `ceil(0.5 * (n + p + 1)) / n` |
| `random_state` | `None` | Random seed for initial subset selection |

**GraphicalLasso additional parameters:**

| Parameter | Default | Description |
|---|---:|---|
| `alpha` | `0.01` | L1 regularization parameter |
| `max_iter` | `100` | Maximum number of outer iterations |
| `tol` | `1e-4` | Convergence tolerance on the dual gap |

**GraphicalLassoCV additional parameters:**

| Parameter | Default | Description |
|---|---:|---|
| `alphas` | `4` | Number of alpha values (int) or explicit array of alpha values |
| `cv` | `5` | Number of cross-validation folds |
| `max_iter` | `100` | Maximum number of GLasso iterations per alpha |
| `tol` | `1e-4` | Convergence tolerance |

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

```python
from statgpu.covariance import ShrunkCovariance, MinCovDet, GraphicalLasso, GraphicalLassoCV
import numpy as np

X = np.random.randn(500, 10)

# --- ShrunkCovariance (manual shrinkage) ---

sc = ShrunkCovariance(shrinkage=0.3, device="cpu")
sc.fit(X)
print(f"Covariance shape: {sc.covariance_.shape}")   # (10, 10)
print(f"Shrinkage used:   {sc.shrinkage_}")            # 0.3

# --- MinCovDet (robust MCD) ---

mcd = MinCovDet(random_state=42, device="cpu")
mcd.fit(X)
print(f"Robust covariance shape: {mcd.covariance_.shape}")
print(f"Support size:            {mcd.support_.sum()}")
print(f"Raw covariance shape:    {mcd.raw_covariance_.shape}")

# Mahalanobis distances (useful for outlier detection)
dists = mcd.dist_
print(f"Mahalanobis distances (first 5): {dists[:5]}")

# --- GraphicalLasso (sparse precision) ---

gl = GraphicalLasso(alpha=0.1, max_iter=100, device="cpu")
gl.fit(X)
print(f"Precision shape:  {gl.precision_.shape}")
print(f"Iterations:       {gl.n_iter_}")
# Sparsity: count near-zero entries
sparsity = np.mean(np.abs(np.asarray(gl.precision_)) < 1e-8)
print(f"Sparsity (fraction near-zero): {sparsity:.2%}")

# --- GraphicalLassoCV (cross-validated alpha) ---

glcv = GraphicalLassoCV(alphas=4, cv=5, device="cpu")
glcv.fit(X)
print(f"Best alpha:       {glcv.alpha_:.4f}")
print(f"Precision shape:  {glcv.precision_.shape}")

# Inspect CV results
for r in glcv.cv_results_:
    print(f"  alpha={r['alpha']:.4f}  mean_score={r['mean_score']:.4f}")
```

## Backend execution and validation boundary

`GraphicalLasso` and `GraphicalLassoCV` keep centering, covariance updates,
coordinate descent, inversion, fold fitting, and held-out scoring on the selected
NumPy, CuPy, or Torch backend. `MinCovDet` keeps C-steps, Mahalanobis distances,
sorting, support masks, reweighting, and final covariance/precision on the selected
backend. Only seeded integer indices, convergence/CV scalars, and chi-square scalar
CDF/quantile calculations cross the CPU boundary.

NumPy/Torch-CPU parity and output-backend preservation are covered by regression
tests. Physical CuPy CUDA and Torch CUDA convergence, memory, runtime, and repeated-fit
validation remains `PARTIAL_REMOTE_PENDING`.

Empirical and shrinkage covariance estimators validate a non-empty feature dimension
and finite input values on the selected backend before centering or inversion, avoiding
misleading singular-covariance errors for NaN/Inf data.

## strict/approx difference

The shrinkage estimators (`EmpiricalCovariance`, `LedoitWolf`, `OAS`, `ShrunkCovariance`) do not have separate strict or approx modes. They use direct analytical formulas with no iterative solver, so there is no convergence tolerance to tune.

`MinCovDet` uses iterative C-steps internally but the number of iterations is not user-configurable; convergence is determined by the algorithm.

`GraphicalLasso` and `GraphicalLassoCV` have two convergence-related parameters: `max_iter` (outer iterations) and `tol` (maximum absolute covariance-update tolerance). The inner coordinate descent uses a 1000-iteration cap and tolerance `min(1e-8, 0.1 * tol)`.

`LedoitWolf` and `OAS` provide different shrinkage intensity formulas. Choose based on your use case:

- **LedoitWolf**: More general; performs well across a wide range of \(n/p\) ratios. This is the standard recommendation for shrinkage covariance estimation.
- **OAS**: Derived under a Gaussian assumption; asymptotically optimal when \(n > p\) and often achieves lower mean squared error than Ledoit-Wolf in that regime.
- **ShrunkCovariance**: Use when you already know the desired shrinkage intensity (e.g., from domain knowledge or prior cross-validation).
- **MinCovDet**: Use when the data may contain outliers or contamination. The MCD estimate has a high breakdown point (up to 50%).
- **GraphicalLasso**: Use when you expect the precision matrix to be sparse (many conditional independencies). The `alpha` parameter controls sparsity.
- **GraphicalLassoCV**: Use when you want automatic selection of the `alpha` regularization parameter via cross-validation.

## Outputs

### Fitted attributes

| Attribute | Shape | Description |
|---|---|---|
| `covariance_` | `(n_features, n_features)` | Estimated covariance matrix |
| `precision_` | `(n_features, n_features)` | Inverse covariance (precision) matrix |
| `location_` | `(n_features,)` | Estimated mean vector |
| `n_samples_` | scalar | Number of training samples |
| `n_features_` | scalar | Number of features |
| `shrinkage_` | scalar (float) | Shrinkage intensity in [0, 1] (LedoitWolf/OAS/ShrunkCovariance) |

**MinCovDet additional attributes:**

| Attribute | Shape | Description |
|---|---|---|
| `support_` | `(n_samples,)` of bool | Boolean mask of observations in the support set |
| `raw_covariance_` | `(n_features, n_features)` | Raw covariance before reweighting |
| `raw_location_` | `(n_features,)` | Raw location before reweighting |
| `dist_` | `(n_samples,)` | Mahalanobis distances of training observations |

**GraphicalLasso additional attributes:**

| Attribute | Shape | Description |
|---|---|---|
| `n_iter_` | scalar | Number of iterations performed |

**GraphicalLassoCV additional attributes:**

| Attribute | Shape | Description |
|---|---|---|
| `alpha_` | scalar (float) | Best alpha selected by cross-validation |
| `cv_results_` | list of dict | Per-alpha CV results: `{alpha, mean_score, scores}` |

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

**When should I use MinCovDet vs EmpiricalCovariance?**
Use MinCovDet when your data may contain outliers or come from a heavy-tailed distribution. The MCD estimator has a breakdown point of up to 50%, meaning it remains valid even if nearly half the observations are contaminated. EmpiricalCovariance (and the shrinkage variants) are more efficient when the data is clean and approximately Gaussian.

**How does GraphicalLassoCV choose the alpha grid?**
If `alphas` is an integer, it generates that many log-spaced values between 0.01 and 1.0. You can also pass an explicit list of alpha values to search over.

**What does the `support_` attribute of MinCovDet mean?**
It is a boolean array indicating which training observations are considered "clean" (not outliers) after the reweighting step. Observations with Mahalanobis distances exceeding the 97.5th percentile of the \(\chi^2_p\) distribution are excluded from the final covariance estimate.

## External Validation

All estimators are validated against their scikit-learn counterparts:

- `sklearn.covariance.EmpiricalCovariance`
- `sklearn.covariance.LedoitWolf`
- `sklearn.covariance.OAS`
- `sklearn.covariance.ShrunkCovariance`
- `sklearn.covariance.MinCovDet`
- `sklearn.covariance.GraphicalLasso`
- `sklearn.covariance.GraphicalLassoCV`

Empirical and shrinkage estimators are compared with scikit-learn at tight numerical tolerances. `MinCovDet` is checked through robust-location/covariance and support invariants, while `GraphicalLasso` is checked against reference solutions and covariance/precision structural identities. NumPy/Torch-CPU parity is covered in `dev/tests/test_three_backend_native_followup.py`; physical CUDA parity is not yet claimed.

## References

- Ledoit, O., & Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. *Journal of Multivariate Analysis*, 88(2), 365-411. [https://doi.org/10.1016/S0047-259X(03)00096-4](https://doi.org/10.1016/S0047-259X(03)00096-4)
- Chen, Y., Wiesel, A., Eldar, Y. C., & Hero, A. O. (2010). Shrinkage algorithms for MMSE covariance estimation. *IEEE Transactions on Signal Processing*, 58(10), 5297-5307. [https://doi.org/10.1109/TSP.2010.2053029](https://doi.org/10.1109/TSP.2010.2053029)
- Rousseeuw, P. J., & Van Driessen, K. (1999). A fast algorithm for the minimum covariance determinant estimator. *Technometrics*, 41(3), 212-223. [https://doi.org/10.1080/00401706.1999.10485670](https://doi.org/10.1080/00401706.1999.10485670)
- Croux, C., & Haesbroeck, G. (1999). Influence function and efficiency of the minimum covariance determinant scatter matrix estimator. *Journal of Multivariate Analysis*, 71(2), 161-190. [https://doi.org/10.1006/jmva.1999.2848](https://doi.org/10.1006/jmva.1999.2848)
- Friedman, J., Hastie, T., & Tibshirani, R. (2008). Sparse inverse covariance estimation with the graphical lasso. *Biostatistics*, 9(3), 432-441. [https://doi.org/10.1093/biostatistics/kxm045](https://doi.org/10.1093/biostatistics/kxm045)
