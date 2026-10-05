# Kernel Methods

> Language: English  
> Last updated: 2026-10-05  
> Switch: [Chinese](../../cn/models/kernel-methods.md)

## Overview

The kernel-methods module provides:

- `KernelRidge`
- `KernelRidgeCV`
- `KernelPCA`
- `Nystroem`
- `pairwise_kernels`
- RBF, polynomial, linear, Laplacian, sigmoid, cosine, and chi-squared kernels

The public implementations expose NumPy, CuPy, and Torch execution paths where
supported by the selected estimator and kernel.

## Paths

```text
statgpu.nonparametric.kernel_methods.KernelRidge
statgpu.nonparametric.kernel_methods.KernelRidgeCV
statgpu.nonparametric.kernel_methods.KernelPCA
statgpu.nonparametric.kernel_methods.Nystroem
statgpu.nonparametric.kernel_methods.pairwise_kernels
```

Individual kernel functions are also importable from
`statgpu.nonparametric.kernel_methods`.

## Kernel Ridge Regression

Given a training kernel matrix $K$, kernel ridge regression solves the dual
system

$$
(K+\alpha I)c=y.
$$

For a positive-semidefinite kernel, the RKHS-penalized objective is

$$
\min_c \lVert y-Kc\rVert_2^2+\alpha c^\top Kc.
$$

The loss is a sum, not an average. For positive-definite $K$, its first-order condition gives the displayed linear system. With singular positive-semidefinite $K$, the same system (for $\alpha>0$) selects a coefficient representation of the minimizing fitted function. A penalty on $\lVert c\rVert^2$ alone would define a different estimator. Indefinite kernels do not have this convex RKHS interpretation.

Predictions for test observations are

$$
\hat y_{test}=K(X_{test},X_{train})c.
$$

`KernelRidge` solves the regularized linear system directly. If the solve raises a linear-algebra error, it retries with increasing diagonal jitter; that stabilized result need not solve the original displayed system exactly. Prefer a positive, well-scaled `alpha`. Multi-output responses are supported with a compatible response matrix.

`fit(X, y, sample_weight=None)` currently ignores supplied `sample_weight`; it does not fit a weighted objective. Omit that argument and use this estimator only for an unweighted analysis. Removing zero-weight rows is useful when the intended operation is simply exclusion, but does not implement arbitrary weighting.

## Kernel Ridge Cross-Validation

`KernelRidgeCV` evaluates a grid of regularization parameters across CV folds and
refits the selected value on the complete dataset. Backend-specific
implementations may reuse a kernel eigendecomposition or vectorize the alpha
sweep rather than solving every system independently.

Selection minimizes mean validation MSE, averaged equally over folds and response columns. `alpha_` is the chosen value; `best_score_` is the mean fold R-squared at that value, not the MSE used for selection. `cv_results_` contains `alphas`, `mean_mse`, `mse_table`, `mean_r2`, `r2_table`, `best_alpha`, and `best_score`. Tables have shape `(n_alphas, n_folds, n_targets)`; means have shape `(n_alphas, n_targets)`.

With a singular training-fold kernel, an explicit zero alpha can produce nonfinite CV scores and still be selected. Use strictly positive candidates and check that `mean_mse` and `best_score_` are finite before interpreting the selection. A returned estimator alone does not establish a valid search. The automatic grid uses positive values.

## Kernel PCA

For the centered kernel matrix $\widetilde K$, Kernel PCA eigendecomposes

$$
\widetilde K = V\Lambda V^\top.
$$

The leading eigenvectors define nonlinear components. Transforming new data
requires computing the test-to-training kernel, applying the training centering
quantities, and projecting onto the retained components.

## Nystroem Approximation

Nystroem selects $m$ landmark observations and forms an explicit approximate
feature map. If the landmark kernel is

$$
K_{mm}=V\Lambda V^\top,
$$

then the transformed features have the form

$$
Z=K_{nm}V\Lambda^{-1/2}V^\top.
$$

This is the symmetric inverse-square-root feature orientation used for a positive-semidefinite landmark kernel. The implementation actually computes $K_{mm}=U\Sigma V^\top$ by NumPy SVD and uses $U\max(\Sigma,10^{-12}I)^{-1/2}V^\top$ as its normalization. It keeps all $m$ selected landmarks; `eigenvalues_` contains the floored singular values, including for an indefinite kernel. Such a kernel does not imply an exact Euclidean feature representation.

This replaces a full $n\times n$ kernel representation with an $n\times m$
feature matrix when $m\ll n$.

## Built-In Kernels

| Kernel | Definition |
|---|---|
| RBF | $\exp(-\gamma\lVert x-y\rVert_2^2)$ |
| Polynomial | $(\gamma x^\top y+c_0)^d$ |
| Linear | $x^\top y$ |
| Laplacian | $\exp(-\gamma\lVert x-y\rVert_1)$ |
| Sigmoid | $\tanh(\gamma x^\top y+c_0)$ |
| Cosine | $x^\top y/(\lVert x\rVert\lVert y\rVert)$ |
| Chi-squared | $\exp\{-\gamma\sum_j (x_j-y_j)^2/(x_j+y_j)\}$ |

The chi-squared kernel requires non-negative input features. A callable kernel
may be supplied where accepted by the estimator; it must return an array on the
requested backend and obey the expected pairwise-kernel shape.

## Complete estimator API

All four classes are imported from `statgpu.nonparametric.kernel_methods`.
Constructors and defaults are:

```text
KernelRidge(alpha=1.0, kernel='rbf', gamma=None, degree=3, coef0=1, kernel_params=None, device='auto', n_jobs=None)
KernelRidgeCV(alphas=None, cv=5, kernel='rbf', gamma=None, degree=3, coef0=1, kernel_params=None, random_state=None, device='auto', n_jobs=None)
KernelPCA(n_components=2, kernel='rbf', gamma=None, degree=3, coef0=1, alpha=1.0, eigen_solver='auto', device='auto', n_jobs=None)
Nystroem(kernel='rbf', n_components=100, gamma=None, degree=3, coef0=1, random_state=None, device='auto', n_jobs=None)
```

### Constructor controls

| Argument | Meaning and restrictions |
|---|---|
| `kernel` | A built-in name or a matrix-valued callable; see the function API below. Use the lowercase canonical names with estimator-specific controls. |
| `gamma` | RBF/Laplacian/polynomial/sigmoid coefficient; `None` uses `1 / n_features`. Chi-squared defaults to `1.0`. KernelRidge/CV currently ignore this constructor argument for chi-squared; use `kernel_params={"gamma": value}` instead. |
| `degree`, `coef0` | Polynomial degree and additive constant; sigmoid uses `coef0`. Supply these only for applicable kernels. |
| `kernel_params` | KernelRidge/CV only: extra built-in or callable kernel arguments. Explicit applicable nondefault constructor controls can override dictionary entries. Pass callable-specific controls in this dictionary. |
| `alpha` | KernelRidge: finite nonnegative strength in the summed-loss objective. KernelPCA: finite nonnegative diagonal shift used during eigendecomposition; the shift is removed from reported eigenvalues and does not regularize an inverse transform. |
| `alphas` | KernelRidgeCV: finite nonempty nonnegative candidates; prefer strictly positive values. `None` builds 100 positive log-spaced values from the training kernel spectrum. |
| `cv` | KernelRidgeCV: integer between 2 and the number of rows; shuffled row-wise K-fold splitting. No custom/grouped-fold argument is available. |
| `n_components` | Positive integer. KernelPCA retains at most this many positive eigenvalue directions and can return fewer; Nystroem selects `min(n_components, n_training_rows)` landmarks without replacement. |
| `eigen_solver` | KernelPCA: `"auto"` or `"dense"`; both currently use dense symmetric eigendecomposition. |
| `random_state` | Integer seed or `None` for CV row shuffling or Nystroem landmark selection. |
| `device` | `"cpu"`, `"cuda"` (CuPy), `"torch"` (Torch CUDA), or `"auto"`. See backend boundaries below. |
| `n_jobs` | Shared estimator setting; these estimators do not use it to parallelize fitting. |

### Methods and shapes

Use a finite numeric design `X` of shape `(n,p)` with the same feature order at prediction. A one-dimensional `X` is interpreted as multiple rows of one feature, not a multivariate query. Estimator calculations convert numeric input to float64. Let `q` be the number of query rows and `r` the response-column count.

| Class | Complete model-specific method calls | Return / restrictions |
|---|---|---|
| KernelRidge | `fit(X, y, sample_weight=None)`, `predict(X)`, `score(X, y)` | `fit` returns `self`; `y` is `(n,)` or `(n,r)`. Weights are currently ignored. Predictions are `(q,)` for one target, including a one-column training response, otherwise `(q,r)`. |
| KernelRidgeCV | `fit(X, y)`, `predict(X)`, `score(X, y)` | Same response/prediction shapes; `fit` returns `self`. No sample-weight argument. Prediction and scoring delegate to `estimator_`. |
| KernelPCA, Nystroem | `fit(X, y=None)`, `transform(X)`, `fit_transform(X, y=None)`, `predict(X)` | `fit` returns `self`; `y` is unused. Other calls return `(q,k)` features on the selected backend; `predict` is a transform alias, not a response prediction. |

Ridge `score` is a Python float: per-target R-squared uniformly averaged over response columns. Constant targets use 1 for effectively exact predictions and 0 otherwise. KernelPCA/Nystroem have no model-specific `score` or inverse-transform method. All four also expose `get_params(deep=True)` and `set_params(**params)` plus the [shared inference utilities](../reference/estimator-api.md). Refit after parameter changes; KernelPCA/Nystroem can retain previous fitted arrays until refitting, so do not rely on automatic invalidation. These utilities do not provide kernel-coefficient inference.

### Fitted attributes and outputs

| Class | Fields and interpretation |
|---|---|
| KernelRidge | `X_fit_`: backend array `(n,p)`; `dual_coef_`: backend array `(n,r)`, including `(n,1)` for a vector response; `n_features_in_`: integer. These are kernel coefficients, not raw-feature slopes. |
| KernelRidgeCV | `alpha_`, `best_score_`, `cv_results_` as described above; `estimator_`: final KernelRidge; `dual_coef_` and `X_fit_`: its arrays. Read `n_features_in_` on `estimator_`. CV result arrays are NumPy. |
| KernelPCA | NumPy `lambdas_` `(k,)`: retained positive centered-kernel eigenvalues; NumPy `alphas_` `(n,k)`: eigenvectors divided by their square roots; NumPy `X_fit_` `(n,p)`; integers `n_samples_`, `n_features_in_`. Only eigenvalues above the numerical cutoff are retained; no positive directions raises an error. |
| Nystroem | NumPy `components_` `(m,p)`, `component_indices_` `(m,)`, `normalization_` `(m,m)`, `eigenvalues_` `(m,)`; integer `n_features_in_`. The misleading historical name `eigenvalues_` stores floored singular values. `m=min(n_components,n_training_rows)`. |

## Complete pairwise-function API

All functions below are public exports from the same module:

```text
pairwise_kernels(X, Y=None, metric='rbf', xp=None, **params)
rbf_kernel(X, Y=None, gamma=None, xp=None)
polynomial_kernel(X, Y=None, degree=3, gamma=None, coef0=1, xp=None)
linear_kernel(X, Y=None, xp=None)
laplacian_kernel(X, Y=None, gamma=None, xp=None)
sigmoid_kernel(X, Y=None, gamma=None, coef0=1, xp=None)
cosine_kernel(X, Y=None, xp=None)
chi2_kernel(X, Y=None, gamma=1.0, xp=None)
```

`X` and `Y` are compatible finite backend arrays `(n,p)` and `(q,p)`; `Y=None` means `Y=X`. The result is the pairwise matrix `(n,q)`, or `(n,n)` when `Y` is omitted. `xp=None` means NumPy, not automatic input-backend detection. Set `xp` to the matching NumPy/CuPy/Torch module; this argument alone does not transfer inputs to a GPU. The low-level functions do not share a uniform dtype-conversion or input-validation policy; prepare valid arrays explicitly. Chi-squared requires nonnegative values and finite nonnegative `gamma`.

Names are case/whitespace normalized by `pairwise_kernels`; aliases are `gaussian` for `rbf`, `poly` for `polynomial`, and `chi-squared` for `chi2`. A callable receives the complete `X, Y` arrays, additional `params`, and `xp` if its signature accepts that keyword. It must handle `Y=None` when used that way and return the complete pairwise matrix, not one scalar per pair. An unknown name raises `ValueError`.

The cosine implementation adds `1e-10` to its norm-product denominator, so zero-vector pairs return zero and tiny-norm inputs differ from exact cosine normalization. For chi-squared, zero/zero feature contributions are zero on NumPy; CuPy/Torch use a denominator floor of `1e-10`, which can change results for very small nonnegative features.

Implementation references: [KernelRidge](../../../statgpu/nonparametric/kernel_methods/_krr.py), [KernelRidgeCV](../../../statgpu/nonparametric/kernel_methods/_krr_cv.py), [KernelPCA](../../../statgpu/nonparametric/kernel_methods/_kpca.py), [Nystroem](../../../statgpu/nonparametric/kernel_methods/_nystroem.py), and [pairwise kernels](../../../statgpu/nonparametric/kernel_methods/_kernels.py).

## CPU and GPU Examples

### NumPy

```python
import numpy as np
from statgpu.nonparametric.kernel_methods import (
    KernelRidge,
    KernelRidgeCV,
    KernelPCA,
    Nystroem,
)

rng = np.random.default_rng(42)
X = rng.normal(size=(500, 10))
y = X[:, 0] - 0.5 * X[:, 1] + rng.normal(scale=0.1, size=500)

kr = KernelRidge(alpha=1.0, kernel="rbf", device="cpu").fit(X, y)
print(kr.score(X, y))

kr_cv = KernelRidgeCV(kernel="rbf", cv=5, device="cpu").fit(X, y)
print(kr_cv.alpha_)

kpca = KernelPCA(n_components=3, kernel="rbf", device="cpu")
X_kpca = kpca.fit_transform(X)

nystroem = Nystroem(kernel="rbf", n_components=50, random_state=42, device="cpu")
X_features = nystroem.fit_transform(X)
```

### CuPy

```python
import cupy as cp
from statgpu.nonparametric.kernel_methods import KernelRidgeCV

X = cp.random.randn(500, 10, dtype=cp.float64)
y = X[:, 0] - 0.5 * X[:, 1]
model = KernelRidgeCV(kernel="rbf", cv=5, device="cuda").fit(X, y)
```

### Torch CUDA

```python
import torch
from statgpu.nonparametric.kernel_methods import KernelRidgeCV

X = torch.randn(500, 10, device="cuda", dtype=torch.float64)
y = X[:, 0] - 0.5 * X[:, 1]
model = KernelRidgeCV(kernel="rbf", cv=5, device="torch").fit(X, y)
```

`device="cuda"` selects CuPy; `device="torch"` selects Torch.

## Backend and Execution Boundaries

KernelRidge/CV kernel construction and solves use the selected backend. KernelPCA normally decomposes on that backend, but a linear-algebra exception triggers a NumPy CPU eigendecomposition without a separate fallback indicator. Its stored training data and projection coefficients are also NumPy arrays. Do not assume a fully device-resident KernelPCA fit. Nystroem is a hybrid implementation: it copies selected landmarks to NumPy, constructs the landmark kernel and performs its SVD on CPU, then constructs the query-to-landmark kernel and returns features on the selected backend. Custom Nystroem kernels must therefore also accept NumPy landmark inputs. Landmark indices, normalization, and singular values are host arrays. Do not interpret a GPU Nystroem fit as an entirely device-resident decomposition. Explicit unavailable device requests raise.

`KernelPCA` and `Nystroem` reject NaN/Inf during fitting and transformation on
the public validation paths. Kernel-specific domain checks, such as non-negative
inputs for the chi-squared kernel, fail explicitly.

## Inference Semantics

Kernel methods do not currently expose coefficient-level standard errors,
hypothesis tests, or confidence intervals. Model quality is evaluated through
prediction scores, cross-validation loss, embedding properties, reconstruction
or approximation diagnostics, and application-specific validation.

There is no strict/approximate inference mode distinction in this module.
Nystroem is an explicit low-rank kernel approximation, not a silent fallback for
an exact kernel estimator.

## Complexity and Performance Notes

- Exact kernel methods materialize an $n\times n$ training kernel and therefore
  have quadratic memory cost.
- Direct kernel-ridge solves and dense eigendecompositions have cubic worst-case
  arithmetic cost in the number of training observations.
- `KernelRidgeCV` may reuse decompositions across alpha values, but CV still
  multiplies work across folds.
- `Nystroem` reduces kernel storage to $O(nm)$ plus landmark linear algebra.
- GPU speed depends on sample size, dtype, kernel, synchronization, and available
  memory; small problems may be faster on CPU.

## Limitations and Failure Modes

For RBF kernels, subtract the same training-derived offset from training and query features when coordinates have a large common offset. Current squared-distance arithmetic can otherwise lose the differences between nearby points: coordinates near `1e9` can produce an almost all-ones kernel and badly changed predictions. Centering preserves the intended RBF kernel; it is not a general preprocessing rule for every kernel, especially chi-squared.


- Kernel matrices can become poorly conditioned; increase `alpha` or adjust the
  kernel scale when necessary.
- RBF-like kernels are sensitive to `gamma`.
- Chi-squared kernels require non-negative inputs.
- Dense exact kernel methods may exhaust device memory for large $n$.
- User-supplied kernels are responsible for backend, dtype, shape, and symmetry
  contracts required by the selected estimator.
- `KernelRidgeCV` can be expensive for large fold and alpha grids.

## Validation guidance

For kernel ridge, verify the regularized linear system or compare with scikit-learn using the same kernel, alpha, data, and unweighted objective. For CV, inspect finite per-candidate MSE and verify the final refit uses the selected alpha. For Nystroem, align selected landmarks and the SVD singular-value floor before comparing normalization or features. CPU checks do not establish GPU accuracy or performance; validate those on the intended device.

## FAQ

### Should I use KernelRidge or Nystroem plus a linear model?

Use exact Kernel Ridge when the training kernel fits comfortably in memory and the
exact kernel representation is important. Use Nystroem when a controlled low-rank
feature approximation is preferable for scale or downstream reuse.

### Why can a GPU kernel method be slower than CPU?

Kernel construction and linear algebra must be large enough to amortize device
launch, synchronization, and memory-transfer overhead.

### Does `device="auto"` silently change an explicit request?

No. `"auto"` is itself an automatic selection request. An explicit `"cuda"` or
`"torch"` request fails if that backend is unavailable.

### Are KernelPCA components directly comparable across separate fits?

Eigenvector signs and bases within repeated or nearly repeated eigenspaces are not
uniquely identified. Compare represented subspaces or downstream quantities when
that ambiguity matters.

## References

- Schölkopf, B., Smola, A., & Müller, K.-R. (1998). Nonlinear component analysis
  as a kernel eigenvalue problem.
- Williams, C. K. I., & Seeger, M. (2001). Using the Nystroem method to speed up
  kernel machines.
- Shawe-Taylor, J., & Cristianini, N. (2004). *Kernel Methods for Pattern Analysis*.
