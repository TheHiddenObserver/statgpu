# Covariance Estimation

> Language: English  
> Last updated: 2026-10-10\
> Switch: [Chinese](../../cn/models/covariance.md)

## Overview

Covariance describes how variables vary together; its inverse, the precision
matrix, describes conditional relationships in a Gaussian model. Use empirical
covariance for a well-sampled baseline, shrinkage for greater stability, robust
covariance for multivariate outliers, or Graphical Lasso for a sparse precision
matrix. The first example uses `LedoitWolf`, which chooses its shrinkage
strength from the data.

The `statgpu.covariance` module provides seven covariance and precision-matrix
estimators:

- `EmpiricalCovariance`
- `LedoitWolf`
- `OAS`
- `ShrunkCovariance`
- `MinCovDet`
- `GraphicalLasso`
- `GraphicalLassoCV`

The public estimators expose NumPy, CuPy, and Torch execution paths.

## Objectives

### Empirical covariance

For centered observations $X\in\mathbb R^{n\times p}$,

$$
\hat S = \frac{1}{n}X^\top X.
$$

Unless `assume_centered=True`, the column mean is estimated and removed before
forming the covariance matrix.

### Shrinkage estimators

`LedoitWolf`, `OAS`, and `ShrunkCovariance` use

$$
\hat\Sigma=(1-\alpha)\hat S+\alpha\mu I,
\qquad
\mu=\frac{\operatorname{tr}(\hat S)}{p}.
$$

`LedoitWolf` and `OAS` estimate $\alpha$ analytically. `ShrunkCovariance` uses the
user-supplied `shrinkage` value.

### Minimum covariance determinant

`MinCovDet` searches for a concentrated subset with a small covariance
determinant, applies FAST-MCD concentration steps, and then reweights observations
using robust Mahalanobis distances. It is intended for covariance estimation in
the presence of multivariate outliers.

### Graphical Lasso

`GraphicalLasso` estimates a sparse precision matrix $\Theta$ by solving

$$
\max_{\Theta\succ0}
\left\{
\log\det(\Theta)-\operatorname{tr}(S\Theta)
-\alpha\lVert\Theta\rVert_{1,\mathrm{off}}
\right\}.
$$

The precision diagonal is not L1-penalized. `GraphicalLassoCV` evaluates an alpha
grid by cross-validation and refits the selected model on the complete dataset.

<a id="covariance-basic"></a>

## Estimate a covariance matrix on CPU

Run these blocks in order. There is no response vector: `X` has one observation
per row and one measured variable per column. First import the estimator.

<!-- example: covariance-basic -->
```python
import numpy as np
from statgpu.covariance import LedoitWolf
```

Create 500 observations of 10 independent standard-normal variables, so `X` has
shape `(500, 10)`. The population covariance in this simulation is the identity.

```python
rng = np.random.default_rng(42)
X = rng.normal(size=(500, 10))
```

Fit the shrinkage estimate. The default estimates and subtracts column means;
you do not need to center `X` manually.

```python
lw = LedoitWolf(device="cpu").fit(X)
```

Inspect matrix size, shrinkage strength and the Gaussian score on these rows.

```python
print(lw.covariance_.shape, lw.shrinkage_)
print(lw.score(X))
```
<!-- example-end: covariance-basic -->

The covariance has shape `(10, 10)`: diagonal entries estimate variances and
off-diagonal entries estimate covariances. Here `shrinkage_` is `1.0`, so the
estimate uses the scaled-identity target completely. This matches the simple
simulation but is not a value to expect for every dataset. The score is about
`-14.175`; it is a Gaussian log-likelihood score, not an accuracy percentage.
Evaluate competing models on the same held-out rows when comparing generalization.

Numeric pandas `DataFrame` inputs and ordinary array-like inputs, such as nested
lists, can be used for fitting and for `score`, `predict`, and `mahalanobis`.
A one-dimensional numeric input, including a pandas `Series`, is treated as one
feature during fitting. Column labels do not align features automatically:
keep evaluation columns in the same order as training columns. Missing and
infinite values must be removed or handled before fitting or evaluation.

## Estimation Algorithms

- `EmpiricalCovariance` computes the sample covariance directly and obtains a
  precision matrix by inversion, using stabilization only when the exact inverse
  fails or is non-finite.
- `LedoitWolf` and `OAS` evaluate closed-form shrinkage intensities and then invert
  the shrunk covariance.
- `ShrunkCovariance` follows the same direct path with a fixed intensity.
- `MinCovDet` uses repeated initial subsets, concentration steps, consistency
  correction, and reweighting.
- `GraphicalLasso` uses block coordinate updates with soft-thresholded inner
  regressions.
- `GraphicalLassoCV` fits the Graphical Lasso across folds and candidate alpha
  values before the final refit.

## Common Parameters

| Parameter | Default | Description |
|---|---:|---|
| `assume_centered` | `False` | Treat input as already centered |
| `device` | `"auto"` | `"cpu"`, `"cuda"` (CuPy), `"torch"`, or `"auto"` |
| `n_jobs` | `None` | Reserved for API compatibility where not implemented |

Estimator-specific parameters include:

| Estimator | Parameters |
|---|---|
| `ShrunkCovariance` | `shrinkage` |
| `MinCovDet` | `support_fraction`, `random_state` |
| `GraphicalLasso` | `alpha`, `max_iter`, `tol` |
| `GraphicalLassoCV` | `alphas`, `cv`, `max_iter`, `tol`, `random_state` |

Consult class docstrings for the exact accepted type and range of each parameter.

## Fitted Attributes and Outputs

Common fitted attributes include:

| Attribute | Description |
|---|---|
| `covariance_` | Estimated covariance matrix |
| `precision_` | Estimated inverse covariance or sparse precision matrix |
| `location_` | Estimated mean vector; zero when centered input is assumed |
| `n_samples_` | Number of fitted observations |
| `n_features_` | Number of fitted features |

Additional attributes include:

- `shrinkage_` for shrinkage estimators;
- `support_`, `raw_location_`, `raw_covariance_`, and robust distances for
  `MinCovDet`;
- `n_iter_` for iterative sparse estimators;
- `alpha_`, CV scores, and the refitted model state for `GraphicalLassoCV`.

Where exposed, `score(X)` evaluates the fitted Gaussian covariance model and
`mahalanobis(X)` returns squared Mahalanobis distances under the fitted location
and precision.

## Robust and sparse alternatives

These examples answer different questions from shrinkage estimation. Both
reuse `X` from [the CPU example](#covariance-basic).

### Robust support

Use `MinCovDet` when outliers make ordinary covariance unreliable. The clean
simulation is only an API illustration; no observation is known to be an outlier.

<!-- example-requires: covariance-basic -->
<!-- example: covariance-robust -->
```python
from statgpu.covariance import MinCovDet

mcd = MinCovDet(random_state=42, device="cpu").fit(X)
print(mcd.support_.sum())
```
<!-- example-end: covariance-robust -->

`support_` is a Boolean mask over training rows. Its sum counts rows in the
final robust support; exclusion is not proof that an observation is erroneous.

### Sparse precision with cross-validation

Use `GraphicalLassoCV` to choose an L1 penalty and refit a sparse precision model.
In a Gaussian model, zero off-diagonal precision entries encode conditional
independence, rather than zero marginal covariance or a causal relationship.

<!-- example-requires: covariance-basic -->
<!-- example: covariance-sparse -->
```python
from statgpu.covariance import GraphicalLassoCV

glcv = GraphicalLassoCV(alphas=4, cv=5, device="cpu").fit(X)
print(glcv.alpha_)
```
<!-- example-end: covariance-sparse -->

`alpha_` is the selected penalty, not a significance level. The fitted
`covariance_` and `precision_` come from the final fit on all rows of `X`.

<a id="cpu-and-gpu-examples"></a>

## Optional GPU execution

Reuse `X` and `LedoitWolf` from [the CPU example](#covariance-basic). Choose
one of these blocks when its CUDA runtime is available.

### CuPy

```python
import cupy as cp

X_cupy = cp.asarray(X)
model_cupy = LedoitWolf(device="cuda").fit(X_cupy)
print(model_cupy.covariance_.shape)
```

### Torch CUDA

```python
import torch

X_torch = torch.as_tensor(X, device="cuda", dtype=torch.float64)
model_torch = LedoitWolf(device="torch").fit(X_torch)
print(model_torch.covariance_.shape)
```

The device selects computation independently of the input array type:

- `device="cpu"` converts NumPy, CuPy, or Torch inputs to NumPy CPU arrays.
- `device="cuda"` converts inputs to CuPy CUDA arrays. It requires a working
  CuPy CUDA runtime even when the input is a Torch tensor.
- `device="torch"` converts inputs to Torch CUDA tensors. A Torch CPU input
  is moved to CUDA; it never satisfies an explicit Torch GPU request by itself.
- `device="auto"` inherits `statgpu.set_device(...)`. When both settings are
  automatic, native CuPy/Torch input keeps its backend and device, including
  Torch CPU input. Other input types use CuPy CUDA, then Torch CUDA, then NumPy
  according to availability. An explicit global setting overrides this native
  input selection; an explicit estimator setting overrides the global setting.

When NumPy computation is selected, supported real Torch inputs are detached
from autograd, transferred to CPU, and normalized to `float64` before checking
for NaN or infinity. The original input is not modified. This includes
`bfloat16`, `float8_e5m2`, and real-valued views of dense, non-quantized tensors.
Transferring before conversion also allows CPU computation from MPS tensors,
whose source device does not support `float64`.

For Torch computation, dense `float8_e5m2` values are checked after widening to
`float64` on their source device; native CUDA input stays on CUDA. This works
without a raw float8 finite-check kernel, but still requires Torch support for
conversion to `float64`. Other float8 formats, sparse or quantized tensors, and
complex inputs do not gain support from this conversion. The estimators do not
guarantee autograd compatibility.

The seven estimators share this policy. `GraphicalLassoCV` keeps its initial
backend and device through all folds, scoring, and the final refit. Fitted arrays
remain on that backend. `score`, `mahalanobis`, and `predict` convert new inputs
to the fitted backend and device, even if the global policy subsequently changes.
`score` returns a Python float; `mahalanobis` and `predict` return NumPy arrays.

The estimator device values do not accept indices such as `"cuda:1"`. An
already-native GPU input retains its CUDA index when that backend is selected;
CPU inputs moved to a GPU use that library's current CUDA device. Scoring uses
the fitted array's index. See the [device guide](../guides/device-and-memory.md)
for global configuration.

## Covariance, Precision, and Inference Semantics

These classes estimate covariance or precision matrices; they do not generally
expose coefficient-level standard errors or regression p-values. Numerical
uncertainty should be assessed with a method appropriate to the covariance
estimator and application, such as resampling or a downstream model with a
specified inference contract.

A singular or nearly singular empirical covariance may require stabilization for
precision computation. Stabilization is a numerical safeguard and does not turn
a rank-deficient covariance into fully identified information in every direction.

## Backend and Execution Boundaries

Centering, covariance updates, matrix products, linear algebra, FAST-MCD
concentration steps, and Graphical Lasso coordinate updates remain on the
selected numerical backend where implemented. Some control and scalar
distribution calculations use CPU, so do not assume the entire workflow is GPU-resident.

Input validation for empty feature dimensions and NaN/Inf values occurs before
centering or inversion so invalid data is not misreported as a singular covariance
problem.

## Strict and Approximate Behavior

There is no global strict/approximate switch shared by all covariance estimators.
Each estimator uses its documented algorithm. Numerical inversion stabilization,
robust subset search, and CV selection are explicit parts of the corresponding
algorithm rather than silent backend fallbacks.

## Limitations and Failure Modes

- `EmpiricalCovariance` can be poorly conditioned when $p$ is large relative to
  $n$; shrinkage may be preferable.
- `LedoitWolf` and `OAS` shrink toward a scaled identity and may be inappropriate
  when a different structural target is required.
- `MinCovDet` is more expensive than direct covariance estimators and requires
  enough observations for a meaningful support subset.
- `GraphicalLasso` assumes a sparse precision representation and may fail to
  converge for unsuitable alpha or tolerance settings.
- `GraphicalLassoCV` multiplies the fitting cost by the number of folds and alpha
  candidates.
- Explicit GPU requests fail when the requested runtime is unavailable; they do
  not silently execute on CPU.

<a id="external-validation"></a>

## Comparing estimates

Use the same observations, feature order, centering convention, and estimator
when comparing covariance estimates. Match the covariance normalization
(`1/n` for the empirical estimate above) and any fixed shrinkage or Graphical
Lasso penalty. For iterative fits, also align convergence tolerances and
iteration budgets. Compare Gaussian scores on the same held-out rows rather
than treating training scores as generalization performance.

For contributor details, see the [validation reference](../../../dev/reviews/pr168-model-validation-provenance.md#covariance).

## FAQ

### Which estimator should I use when $p$ is close to $n$?

A shrinkage estimator such as `LedoitWolf` or `OAS` is usually more stable than the
unregularized empirical covariance.

### Does `MinCovDet` remove observations?

It identifies a robust support and returns robust covariance estimates. Inspect
`support_` and robust distances rather than assuming every observation contributes
equally to the final estimate.

### Why is the Graphical Lasso precision sparse but the covariance dense?

The L1 penalty is applied to off-diagonal precision entries. The inverse of a
sparse precision matrix need not be sparse.

### Can I pass a Torch CUDA tensor with `device="cuda"`?

Yes, if CuPy CUDA is available: the tensor is converted to CuPy and computation
uses CuPy. Use `device="torch"` to request Torch CUDA computation instead.
Input type does not override an explicit backend request.

## Paths

```python
from statgpu.covariance import (
    EmpiricalCovariance,
    LedoitWolf,
    OAS,
    ShrunkCovariance,
    MinCovDet,
    GraphicalLasso,
    GraphicalLassoCV,
)
```

## References

- Ledoit, O., & Wolf, M. (2004). A well-conditioned estimator for
  large-dimensional covariance matrices.
- Chen, Y., Wiesel, A., Eldar, Y. C., & Hero, A. O. (2010). Shrinkage algorithms
  for MMSE covariance estimation.
- Rousseeuw, P. J., & Van Driessen, K. (1999). A fast algorithm for the minimum
  covariance determinant estimator.
- Friedman, J., Hastie, T., & Tibshirani, R. (2008). Sparse inverse covariance
  estimation with the graphical lasso.
